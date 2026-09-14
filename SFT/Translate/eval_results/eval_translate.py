#!/usr/bin/env python3
"""对比模型 zero-shot 翻译评测(生成阶段)。

在容器内 NPU 上跑: 用模型自带 chat template + 我们训练用的同款指令模板,
对 FLORES-200 dev (997 句) 做双向 zero-shot 翻译, 输出预测 JSONL。
指标计算在宿主机用 sacrebleu 另算(见 compute_metrics.py)。

用法: python eval_translate.py --model <本地模型目录> --out <输出.jsonl> [--limit N]
"""
import argparse
import json
import time

import torch
import torch_npu  # noqa: F401  (注册 NPU)
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL_DIR = "/mnt/models/CODE/mzy/datasets/sft_translate/eval/flores200_dataset/dev"
INS_EN2ZH = "Translate the following text from English to Simplified Chinese."
INS_ZH2EN = "请将以下简体中文翻译成英文。"


def load_pairs(eval_dir=None):
    d = eval_dir or EVAL_DIR
    en = [x.rstrip("\n") for x in open(f"{d}/eng_Latn.dev", encoding="utf-8")]
    zh = [x.rstrip("\n") for x in open(f"{d}/zho_Hans.dev", encoding="utf-8")]
    assert len(en) == len(zh)
    return en, zh


HYMT_TPL = "将以下文本翻译为{tgt}，注意只需要输出翻译后的结果，不要额外解释：\n\n{src}"


def build_prompts(tok, items, thinking=False, native_hymt=False, plain=False):
    prompts = []
    for ins, src in items:
        if native_hymt:
            # HY-MT 官方模板: 目标是中文还是英语由指令语言区分
            tgt = "英语" if "简体中文翻译成英文" in ins else "中文"
            content = HYMT_TPL.format(tgt=tgt, src=src)
        else:
            content = f"{ins}\n{src}"
        if plain:
            # 远端 fast_preprocess_v2 训练格式: 无 think 块的裸 im_start 格式
            prompts.append(f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")
            continue
        msgs = [{"role": "user", "content": content}]
        kw = dict(add_generation_prompt=True, tokenize=False)
        try:
            prompts.append(tok.apply_chat_template(msgs, enable_thinking=thinking, **kw))
        except TypeError:
            prompts.append(tok.apply_chat_template(msgs, **kw))
        except Exception:
            # 无 chat template 的模型(minimind 等): 用 im_start/im_end 手工格式兜底
            prompts.append(f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")
    return prompts


def generate(model, tok, prompts, batch_size, max_new):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        enc.pop("token_type_ids", None)  # minimind 的 tokenizer 会带, 模型不收
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new,
                                 pad_token_id=tok.pad_token_id)
        for j, g in enumerate(gen):
            outs.append(tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True))
        if (i // batch_size) % 10 == 0:
            print(f"  {i}/{len(prompts)}", flush=True)
    return outs


def load_tokenizer(path, trc):
    """优先 AutoTokenizer(use_fast); 失败(如 haidass-sft 标着 LlamaTokenizer 却只有 tokenizer.json)
    时回退到 PreTrainedTokenizerFast 直读, 并手动补 eos/pad 和 chat_template.jinja。"""
    try:
        return AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=trc)
    except Exception as e:
        print(f"AutoTokenizer 失败({type(e).__name__}), 回退 PreTrainedTokenizerFast", flush=True)
        import os
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(path, "tokenizer.json"))
        cfg = json.load(open(os.path.join(path, "tokenizer_config.json"), encoding="utf-8"))
        for k in ("eos_token", "pad_token", "bos_token", "unk_token"):
            if cfg.get(k):
                setattr(tok, k, cfg[k])
        ct = os.path.join(path, "chat_template.jinja")
        if os.path.exists(ct):
            tok.chat_template = open(ct, encoding="utf-8").read()
        return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="调试时只跑前 N 句")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="自定义架构模型(minimind 等)需要")
    ap.add_argument("--native-hymt", action="store_true",
                    help="用 HY-MT 官方翻译模板(中文/英语), 而非我们的训练指令")
    ap.add_argument("--plain-prompt", action="store_true",
                    help="绕过 chat template, 用无 think 块的裸 im_start 格式(远端 8M 训练格式)")
    ap.add_argument("--eval-dir", default=None,
                    help="覆盖 EVAL_DIR(目录内需有 eng_Latn.dev 和 zho_Hans.dev)")
    args = ap.parse_args()

    trc = args.trust_remote_code
    tok = load_tokenizer(args.model, trc)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="npu:0",
        trust_remote_code=trc).eval()

    en, zh = load_pairs(args.eval_dir)
    n = args.limit or len(en)
    en, zh = en[:n], zh[:n]

    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        for direction, ins, srcs, refs in (
            ("en-zh", INS_EN2ZH, en, zh),
            ("zh-en", INS_ZH2EN, zh, en),
        ):
            print(f"[{direction}] {n} prompts", flush=True)
            prompts = build_prompts(tok, [(ins, s) for s in srcs],
                                    native_hymt=args.native_hymt, plain=args.plain_prompt)
            hyps = generate(model, tok, prompts, args.batch_size, args.max_new)
            for i, (s, r, h) in enumerate(zip(srcs, refs, hyps)):
                f.write(json.dumps({"id": i, "direction": direction, "src": s,
                                    "ref": r, "hyp": h.strip()}, ensure_ascii=False) + "\n")
    print(f"done in {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
