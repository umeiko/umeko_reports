# Haidass：当前进展与实验报告

更新：2026-09-10。已有 EXPS 五档英文质量实验、43M→143M 双语补实验均已完成并纳入当前论文，不需要重新跑一遍。

## 只看这三个入口

1. [中文进展总览：已经做了什么、结果是什么意思、论文还差什么](reports/current/PROGRESS_CN.md)
2. [IID 50:50 后续方案：哪些只需评测，哪些才是新训练](reports/current/IID50_EXPERIMENT_PLAN_CN.md)
3. [独立模拟审稿、作者回复、复评与 AC：为什么被质疑](reports/current/ICLR_SIMULATION_CN.md)

[当前论文 PDF（唯一当前稿，位于论文仓库）](https://github.com/umeiko/ICLR-2027-Conference-Submissions/blob/main/output/pdf/haidass_iclr2027.pdf) · [论文源文件](https://github.com/umeiko/ICLR-2027-Conference-Submissions/blob/main/haidass_iclr2027.tex)

## 本轮结论

英文质量扫描的最终十任务 raw 平均分在五档规模均提高，幅度为 +0.39～+1.06 个百分点，但任务和总分口径会影响结论。双语顺序实验在两个规模都支持 IID 作为已测试条件中的强基线；参数和训练 token 同时放大，不是单独参数效应或 scaling law。

143M/803M 额外 20B 种子明确延期。拟议 43M 四格是英文质量×中文质量的新增桥接，不是重跑已有 IID 或 Scale。当前没有启动新训练。

## 查证入口，不必全部阅读

- [已有英文 Scale 技术报告](reports/exps_quality_scale/QUALITY_SCALE_ANALYSIS_CN.md)；[本轮指标敏感性复算](reports/current/QUALITY_METRIC_SENSITIVITY_CN.md)。
- [已有双语 Scale 技术报告](reports/scale_analysis_43m_143m/SCALE_ANALYSIS_CN.md)；[中文评测逐题审计](reports/current/CHINESE_EVAL_AUDIT_CN.md)。
- [集群 43M 复现记录](reports/p0_43m_npu_repro/EXPERIMENT_REPORT_CN.md)；[143M 扩展记录](reports/p1_143m_proportional_npu/EXPERIMENT_REPORT_CN.md)。
- [本地 2080 Ti 初探中文解释](reports/p0_scaled_100m_all_sources_2080ti/EXPERIMENT_REPORT_BEGINNER_CN.md)；[训练数据仓库与内容](reports/HAIDASS_DATASET_INVENTORY_CN.md)。
- [翻译 SFT 准备方案（未执行）](SFT/Translate/README.md)。
- [图表导出清理清单](reports/current/CLEANUP_MANIFEST.json)：删除重复 PDF 导出，保留 PNG、表格、脚本和 Git 历史；历史 manifest 记录当时产物，不代表每个旧 PDF 仍留在当前树。
- [旧的大预算计划（延期）](FUTURE_EXPERIMENT_PLAN.md)：不要按旧文中的 143M/803M 补种子和大矩阵直接开工。

公开仓库存放报告、配置、哈希和分析代码，不包含全部原始训练数据与 checkpoint。模拟评分不是正式 ICLR 评审，不能作为接收概率。
