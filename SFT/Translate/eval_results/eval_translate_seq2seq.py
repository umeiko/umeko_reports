#!/usr/bin/env python3
"""seq2seq 专用翻译模型评测(生成阶段)——NLLB / M2M-100 / OPUS-MT。

这类模型没有指令概念, 不用 chat template:
源句直接进 encoder, generate 时用 forced_bos 指定目标语言(OPUS-MT 单方向模型连这个都不用)。
输出 JSONL 与 eval_translate.py 同构, 指标用同一个 compute_metrics.py。

用法:
  python eval_translate_seq2seq.py --model <dir> --kind nllb|m2m|opusmt \
      --out <输出.jsonl> [--directions both|en-zh|zh-en] [--limit N]
opusmt 是单方向模型: en-zh 方向用 opus-mt-en-zh, zh-en 方向用 opus-mt-zh-en。
"""
import argparse
import json
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

EVAL_DIR = "/mnt/models/CODE/mzy/datasets/sft_translate/eval/flores200_dataset/dev"
LANG_CODE = {
    "nllb": {"en": "eng_Latn", "zh": "zho_Hans"},
    "m2m": {"en": "en", "zh": "zh"},
}


def load_pairs():
    en = [x.rstrip("\n") for x in open(f"{EVAL_DIR}/eng_Latn.dev", encoding="utf-8")]
    zh = [x.rstrip("\n") for x in open(f"{EVAL_DIR}/zho_Hans.dev", encoding="utf-8")]
    assert len(en) == len(zh)
    return en, zh


def generate(model, tok, srcs, batch_size, max_new, forced_bos=None):
    outs = []
    for i in range(0, len(srcs), batch_size):
        enc = tok(srcs[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(model.device)
        enc.pop("token_type_ids", None)
        kw = dict(do_sample=False, max_new_tokens=max_new)
        if forced_bos is not None:
            kw["forced_bos_token_id"] = forced_bos
        with torch.no_grad():
            gen = model.generate(**enc, **kw)
        outs += [tok.decode(g, skip_special_tokens=True) for g in gen]
        if (i // batch_size) % 10 == 0:
            print(f"  {i}/{len(srcs)}", flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=["nllb", "m2m", "opusmt"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--directions", default="both", choices=["both", "en-zh", "zh-en"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="npu:0").eval()

    en, zh = load_pairs()
    n = args.limit or len(en)
    en, zh = en[:n], zh[:n]

    def bos_for(tgt_lang):
        if args.kind == "opusmt":
            return None
        code = LANG_CODE[args.kind][tgt_lang]
        if args.kind == "nllb":
            return tok.convert_tokens_to_ids(code)
        return tok.lang_code_to_id[code]  # m2m

    plan = {"en-zh": (en, zh, "en", "zh"), "zh-en": (zh, en, "zh", "en")}
    dirs = ["en-zh", "zh-en"] if args.directions == "both" else [args.directions]

    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        for direction in dirs:
            srcs, refs, sl, tl = plan[direction]
            if args.kind in ("nllb", "m2m"):
                tok.src_lang = LANG_CODE[args.kind][sl]
            print(f"[{direction}] {n} sentences", flush=True)
            hyps = generate(model, tok, srcs, args.batch_size, args.max_new,
                            forced_bos=bos_for(tl))
            for i, (s, r, h) in enumerate(zip(srcs, refs, hyps)):
                f.write(json.dumps({"id": i, "direction": direction, "src": s,
                                    "ref": r, "hyp": h.strip()}, ensure_ascii=False) + "\n")
    print(f"done in {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
