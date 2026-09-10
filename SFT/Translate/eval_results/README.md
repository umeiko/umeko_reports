# 翻译 SFT 评测结果（FLORES-200 dev，中英双向）

- 总表与分析：`ZEROSHOT_RESULTS.md`（成绩总表 + 指标定义 + 读表要点 + 样例抽查）
- 逐句预测：`predictions/pred_<模型>_flores_dev.jsonl`，每行 `{direction, src, ref, hyp}`，
  可配合 `compute_metrics.py` 复算 BLEU / chrF++
- 评测脚本：`eval_translate.py`（LLM，chat 模板）、`eval_translate_seq2seq.py`（NLLB/M2M/OPUS-MT）

当前结论速览（2026-09-10）：Haidass1.5-143M 翻译 SFT 随数据规模 en→zh 16.65→20.10→23.64
单调爬升（1M/4M/8M），zh→en 稳定 14.4 平台期；8M v1 崩塌事故（zh→en 7.34）根因为
通用数据未清洗 + 打包缺文档间注意力隔离，v2 修复后全面恢复。
