#!/usr/bin/env python3
"""zero-shot 翻译评测(指标阶段, 宿主机跑)。

读 eval_translate.py 产出的预测 JSONL, 用 sacrebleu 算 corpus BLEU + chrF++:
  en 参考: BLEU(13a) / chrF++(word_order=2)
  zh 参考: BLEU(tokenize=zh) / chrF++(word_order=2)
用法: python compute_metrics.py pred1.jsonl pred2.jsonl ...
"""
import json
import sys
from collections import defaultdict

import sacrebleu


def main(paths):
    for p in paths:
        preds = defaultdict(lambda: {"hyp": [], "ref": []})
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            preds[d["direction"]]["hyp"].append(d["hyp"])
            preds[d["direction"]]["ref"].append(d["ref"])
        print(f"== {p}")
        for direction, dd in sorted(preds.items()):
            tgt_zh = direction.endswith("zh")
            bleu = sacrebleu.corpus_bleu(dd["hyp"], [dd["ref"]],
                                         tokenize="zh" if tgt_zh else "13a")
            chrf = sacrebleu.corpus_chrf(dd["hyp"], [dd["ref"]], word_order=2)
            print(f"  {direction}: n={len(dd['hyp'])} BLEU={bleu.score:.2f} chrF++={chrf.score:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
