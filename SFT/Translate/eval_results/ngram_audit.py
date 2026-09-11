#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ngram_audit.py — 训练集 vs FLORES-200 dev 的 n-gram 泄漏审计（CPU，多进程）

判定口径（业界常用去污染标准）：
- 英文侧：归一化（小写、去标点）后按词切分，8-gram 命中即算污染
- 中文侧：归一化（去标点、去空白）后按字，10-gram 命中即算污染
- 每条训练样本把 instruction+output 拼起来，按文字（拉丁/CJK）分开提 n-gram，
  分别与评测集 n-gram 集合求交；命中任意一个即计入 contaminated
- 另报每条样本的最大覆盖率（命中 n-gram 数 / 样本该侧 n-gram 总数），
  用于发现"凑不够一个完整 n-gram 但高度相似"的近似重复

用法:
  python ngram_audit.py --eval-en eng_Latn.dev --eval-zh zho_Hans.dev \
      --train '/mnt/data1/mzy/v2_data_raw/train_8m_shard_*.json' \
      --workers 32 --out audit_report.json
训练文件为 alpaca json（list[dict]，键 instruction/input/output/direction）。
"""
import argparse
import glob
import json
import re
import string
from multiprocessing import Pool

PUNCT = set(string.punctuation) | set("，。！？；：""''（）《》【】、—…·「」『』¡¿")
CJK_LO, CJK_HI = "一", "鿿"


def norm_en(t: str) -> list[str]:
    t = t.lower()
    t = "".join(" " if (ch in PUNCT or ch.isspace()) else ch for ch in t)
    return t.split()


def norm_zh(t: str) -> str:
    return "".join(ch for ch in t if CJK_LO <= ch <= CJK_HI)


def word_ngrams(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def char_ngrams(s, n):
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def split_scripts(text):
    """拉丁段提英文词序列；CJK 段提中文字符串"""
    en_chars, zh_chars = [], []
    for ch in text:
        if CJK_LO <= ch <= CJK_HI:
            zh_chars.append(ch)
        else:
            en_chars.append(ch)
    return norm_en("".join(en_chars)), norm_zh("".join(zh_chars))


def build_eval_ngrams(en_file, zh_file, n_en, n_zh):
    en_set, zh_set = set(), set()
    with open(en_file, encoding="utf-8") as f:
        for line in f:
            en_set |= word_ngrams(norm_en(line), n_en)
    with open(zh_file, encoding="utf-8") as f:
        for line in f:
            zh_set |= char_ngrams(norm_zh(line), n_zh)
    return en_set, zh_set


def audit_shard(path, en_set, zh_set, n_en, n_zh, topk):
    n = dirty = 0
    en_hit = zh_hit = 0
    examples = []  # (coverage, side, text_snippet)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for it in data:
        text = (it.get("instruction", "") + " " + it.get("input", "") + " " + it.get("output", ""))
        words, zhs = split_scripts(text)
        ng_en = word_ngrams(words, n_en)
        ng_zh = char_ngrams(zhs, n_zh)
        hit_en = ng_en & en_set
        hit_zh = ng_zh & zh_set
        n += 1
        if hit_en:
            en_hit += 1
            cov = len(hit_en) / max(1, len(ng_en))
            examples.append((cov, "en", text[:160]))
        if hit_zh:
            zh_hit += 1
            cov = len(hit_zh) / max(1, len(ng_zh))
            examples.append((cov, "zh", text[:160]))
        if hit_en or hit_zh:
            dirty += 1
    examples.sort(key=lambda x: -x[0])
    return path, n, dirty, en_hit, zh_hit, examples[:topk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-en", required=True)
    ap.add_argument("--eval-zh", required=True)
    ap.add_argument("--train", required=True, help="glob of training alpaca json shards")
    ap.add_argument("--n-en", type=int, default=8)
    ap.add_argument("--n-zh", type=int, default=10)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--topk", type=int, default=20, help="每片保留的最高覆盖率样例数")
    ap.add_argument("--out", default="audit_report.json")
    args = ap.parse_args()

    en_set, zh_set = build_eval_ngrams(args.eval_en, args.eval_zh, args.n_en, args.n_zh)
    print(f"eval n-grams: en={len(en_set)} ({args.n_en}-gram), zh={len(zh_set)} ({args.n_zh}-gram)", flush=True)

    files = sorted(glob.glob(args.train))
    print(f"train shards: {len(files)}", flush=True)
    total = dirty = en_hit = zh_hit = 0
    all_examples = []
    with Pool(args.workers) as pool:
        for path, n, d, eh, zh_, ex in pool.starmap(
            audit_shard, [(p, en_set, zh_set, args.n_en, args.n_zh, args.topk) for p in files]
        ):
            total += n; dirty += d; en_hit += eh; zh_hit += zh_
            all_examples += ex
            print(f"{path}: n={n} dirty={d} (en={eh} zh={zh_})", flush=True)
    all_examples.sort(key=lambda x: -x[0])
    report = {
        "eval_ngrams": {"en": len(en_set), "zh": len(zh_set), "n_en": args.n_en, "n_zh": args.n_zh},
        "train_samples": total,
        "contaminated": dirty,
        "contamination_rate": dirty / max(1, total),
        "en_side_hits": en_hit,
        "zh_side_hits": zh_hit,
        "top_examples": [
            {"coverage": round(c, 4), "side": s, "text": t} for c, s, t in all_examples[:50]
        ],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "top_examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
