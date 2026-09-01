# Haidass1.5-143M 中英翻译 SFT 实验包

本目录是一份可执行的实验准备方案，不包含尚未实际跑出的分数。方案调研日期为 **2026-09-01**。

## 先说结论

- 第一版目标是把原始基座模型训练成一个稳定的、双向的简体中文↔英文翻译模型。
- 主训练方法采用 **全参数 SFT + 只对译文计算损失**，不是把提示词和原文也当作答案学习。
- 先用 100 万条唯一平行句对做 4090 试验；通过质量门槛后，再扩展到 800 万条唯一句对上集群。
- 每条平行句对生成两个训练样本：英→中和中→英。因此 100 万句对对应约 200 万个方向样本，800 万句对对应约 1600 万个方向样本。
- FLORES+、历年 WMT、WMT24++、NTREX-128 等只用于评测，绝不进入训练。
- TowerBlocks-v0.2 暂时整库禁用，因为它包含 FLORES、NTREX 和历年 WMT 等评测材料；除非以后完成逐来源白名单过滤和许可证审计。
- WMT26 正式提交已于 2026-07-02 截止。现在可以做公开测试集上的离线对比；下一次正式参赛要等待 WMT27 公告。

## 从哪里开始看

如果你第一次做模型训练，按这个顺序阅读：

1. [小白实验准备报告](BEGINNER_EXPERIMENT_PREP_CN.md)：解释为什么做、数据里是什么、如何训练、分数怎么看。
2. [技术实验方案](TECHNICAL_PLAN_CN.md)：供训练和论文实验负责人执行。
3. [评测协议](EVALUATION_PROTOCOL_CN.md)：规定哪些集合能调参、哪些只能最终测试，以及如何报告指标。
4. [数据格式](DATA_FORMAT_CN.md)：说明每条数据应长什么样，如何划分和防止泄漏。
5. [榜单与对比方案](LEADERBOARD_AND_BASELINES_CN.md)：哪些对手需要自己跑，哪些可直接采用官方公开输出。
6. [实验矩阵](EXPERIMENT_MATRIX.csv)：要跑的基线、消融及优先级。
7. [数据登记表](DATASET_REGISTRY.json)：训练候选集、评测集、许可证状态和污染风险。
8. `configs/`：4090 试验和集群主实验的预注册参数。

## 当前最务实的执行顺序

1. 下载评测集并建立污染 denylist，先于下载训练语料。
2. 从候选语料抽样，每个来源人工检查至少 500 对，校准语言识别和对齐阈值。
3. 生成嵌套的 `250k ⊂ 1M ⊂ 4M ⊂ 8M` 训练子集；划分后再生成反向样本。
4. 在 4090 上做 1,000 step 冒烟测试和 100 万句对主试验。
5. 冻结提示词、解码参数、验证集和模型选择规则。
6. 上集群跑 4M/8M、双向与单向、句子与文档指令阶段等消融。
7. 只在全部决策冻结后运行最终测试集；保存逐句输出、日志和工具版本。

## 目录边界

`data/`、`runs/`、`checkpoints/` 和 `cache/` 默认忽略，不应直接提交大文件或模型权重。可复现实验应提交清单、哈希、脚本、配置和汇总结果；数据本身按各自许可证获取。

## 官方入口

- [Haidass1.5-143M 模型卡](https://huggingface.co/DALabCommunity/Haidass1.5-143M)
- [WMT26 General MT](https://www2.statmt.org/wmt26/translation-task.html)
- [WMT26 数据与提交仓库](https://github.com/wmt-conference/wmt26-general-mt)
- [mtdata](https://github.com/thammegowda/mtdata)
- [OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
- [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus)
- [SacreBLEU](https://github.com/mjpost/sacreBLEU)
- [COMET](https://github.com/Unbabel/COMET)
