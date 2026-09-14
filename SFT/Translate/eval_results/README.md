# 翻译 SFT 评测结果（FLORES-200 dev + FLORES+ devtest，中英双向）

- 总表与分析：`ZEROSHOT_RESULTS.md`（双 benchmark 成绩总表 + 指标定义 + 读表要点 + 样例抽查）
- 逐句预测：`predictions/pred_<模型>_flores_dev.jsonl` / `pred_<模型>_floresplus_devtest.jsonl`，
  每行 `{direction, src, ref, hyp}`，可配合 `compute_metrics.py` 复算 BLEU / chrF++
- 评测脚本：`eval_translate.py`（LLM，chat 模板）、`eval_translate_seq2seq.py`（NLLB/M2M/OPUS-MT），
  均支持 `--eval-dir` 切换评测集

当前结论速览（2026-09-14）：

- **双 benchmark 交叉验证**：全部 20 行模型在 FLORES+ devtest（1012 句，与 dev 零重叠）复测，
  两个 benchmark 的实测成绩如实并列总表、逐句预测公开可复核；我们的模型两集合差
  +1.2/-1.0 以内（8M-trans-v2：dev 23.89/14.65 → devtest 25.06/13.68），口径稳定
- **drafter-8M（8M 规模随机初始化对照）**：en→zh 12.04 / zh→en 5.47（基座版 23.89/14.65），
  数据 1M→8M 仅 +1.2/+0.3——基座预训练的语言知识不是堆 SFT 平行语料能替代的
- Haidass1.5-143M 翻译 SFT 随数据规模 en→zh 16.65→20.10→23.64→23.89 单调爬升（1M/4M/8M），
  zh→en 稳定 14.4 平台期；8M v1 崩塌事故（zh→en 7.34）根因为通用数据未清洗 +
  打包缺文档间注意力隔离，v2 修复后全面恢复
