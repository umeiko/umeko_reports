# 中英翻译 SFT 技术实验方案

版本：v0.1，调研冻结日期：2026-09-01。

## 1. 研究问题与边界

基础模型 [Haidass1.5-143M](https://huggingface.co/DALabCommunity/Haidass1.5-143M) 是 143M 参数、64K 中英词表、最大上下文 4096 的双语基座模型，尚未做指令微调。翻译 SFT 本身不是足够强的论文创新，因此本项目将“获得一个翻译模型”和“回答可发表的研究问题”分开。

工程目标：获得可复现的简体中文↔英文翻译模型，能够在公开句级、文档级和多领域测试上稳定运行。

研究问题：

1. 143M 双语基座在多少高质量平行数据下开始获得有效翻译能力？收益在 250K、1M、4M、8M 何处趋于饱和？
2. 双向联合训练相对两个单向专家，是产生正迁移还是容量竞争？这种效应在两个方向是否对称？
3. 相同数据预算下，严格过滤的小数据是否优于含噪的大数据？
4. 极小模型能否学习文档上下文、术语表和结构化输出约束，还是只会句级直译？

不作为创新点：4090/昇腾平台、常规全参数 SFT、使用 BLEU/COMET、简单堆叠公开语料。硬件只作为复现信息。

## 2. 数据集到底用哪些

### 2.1 训练候选语料及内容

| 数据来源 | 内容是什么 | 优点 | 主要风险 | 800 万池上限 |
|---|---|---|---|---:|
| OPUS-100 `en-zh` train | 多个 OPUS 来源汇总的英中平行句，单语对训练上限约 100 万 | 易获取、覆盖面广，适合首轮 | 来源许可证混合；dev/test 不得训练 | 700K |
| News Commentary 18.1 | 新闻和评论文章的人工/编辑平行文本 | 相对干净，适合新闻测试 | 规模较小，文体单一 | 250K |
| TED2020 / TEDTalks | 演讲字幕及其翻译 | 口语、演讲和解释性文本 | 字幕切分与上下文依赖 | 350K |
| WikiMatrix / WikiTitles | 维基百科句子对齐与页面标题 | 主题广 | 自动对齐有噪声 | 700K |
| MultiUN / UNPC | 联合国正式文件 | 术语稳定、长句和正式文体 | 重复多，法律外交领域偏置 | 1,000K |
| Wikimedia / QED / ELRC health / HRW | 百科、教育演讲、健康和人权文本 | 增加领域覆盖 | 各来源许可和切分要单独记录 | 500K |
| CCMatrix | 网页挖掘平行句 | 量大、领域广 | 噪声、错配和网页模板多 | 1,200K |
| CCAligned | 对齐网页文本 | 量大，可能有连续文档 | 噪声高，去重成本高 | 800K |
| ParaCrawl v9 bonus | 网页爬取平行文本 | 扩规模 | 噪声和基准污染风险最高 | 1,000K |
| OpenSubtitles | 电影/视频字幕平行句 | 对话和口语 | 主语省略、错位、短句过多 | 700K |
| PHP / Ubuntu / InfoPankki / Tanzil 等 | 软件本地化、论坛、公共信息和宗教文本 | 补充特定表达 | 领域偏置，不能占比过大 | 300K |
| 训练数据派生的格式/术语样本 | 从安全训练对构造 JSON、HTML、CSV、术语表和相邻上下文指令 | 训练约束遵循 | 自动构造质量需人工抽检 | 500K |

上表合计 8,000K 个唯一句对。它是“清洗后上限配额”，不是从原始数据直接随机抽 800 万条。若某来源达不到质量或许可要求，不允许为了凑数降低门槛；空缺从已经通过审计的来源按比例补充，并记录变更。

WMT26 官方通过 [mtdata](https://github.com/thammegowda/mtdata) 提供 News Commentary、CCAligned、ParaCrawl、WikiMatrix、CCMatrix、MultiUN、NLLB、QED、TED、OpenSubtitles、OPUS-100、UNPC 等英中资源的可复现获取清单。这个清单只负责“从哪里获取”，不能替代许可证审计、质量过滤和测试污染检查。

### 2.2 两个规模

#### 4090 试验池：1M 唯一句对

| 组成 | 数量 |
|---|---:|
| OPUS-100 train | 300K |
| WikiMatrix/WikiTitles | 200K |
| MultiUN/UNPC | 150K |
| News Commentary | 100K |
| TED2020/TEDTalks | 100K |
| OpenSubtitles/QED | 100K |
| Wikimedia/ELRC health/HRW/其他已审计小语料 | 50K |
| 合计 | 1,000K |

这批数据优先验证方法，不引入最大、最脏的网页池。每个句对生成两个方向，训练约 2M 个样本。

#### 集群主池：8M 唯一句对

按上表配额构建，生成约 16M 个方向样本。`250K ⊂ 1M ⊂ 4M ⊂ 8M` 必须是嵌套集合，以免规模实验把“数据不同”和“数量不同”混在一起。

### 2.3 明确不能训练的集合

- FLORES+ 的 `dev` 和 `devtest`。
- SacreBLEU/mtdata 提供的所有 WMT 历年开发和测试集。
- WMT24++、NTREX-128，以及用于最终报告的 TICO-19/IWSLT 测试切分。
- OPUS-100 的 `validation` 和 `test`；若把它们用于报告，就连相同文档/近重复也必须从训练中剔除。
- TowerBlocks-v0.2 整库默认禁用。其数据卡列出 WMT14–22、FLORES dev、NTREX 等评测来源，直接混训会导致虚假的高分。

### 2.4 许可门禁

“能下载”不等于“能重新发布或商用”。OPUS-100 和 WMT 清单包含多来源数据，必须追溯到原始子语料的许可证。`DATASET_REGISTRY.json` 中 `license_status=verify` 的来源，在审核完成前只能做内部研究样本，不得打包发布；任何 `blocked` 来源不得进入训练。

## 3. 清洗、划分与污染隔离

处理顺序固定如下：

1. 冻结来源版本、URL、下载日期、文件 SHA-256、原始行号和许可证。
2. 先载入所有评测源句和参考译文，建立 exact hash、字符 13-gram/词 8-gram 与 MinHash denylist。
3. 用语言识别检查两端语言。初始置信度建议 0.80，但每个来源人工检查至少 500 对后再校准，不把一个阈值机械套给全部语料。
4. 删除空文本、乱码、网页模板、复制对、明显非中英、极端长度和错位样本。建议句级长度为每侧 3–512 tokenizer tokens，长度比不超过 3；阈值同样以人工审计调整。
5. 用 LaBSE/LASER3 或已有 QE 分数检查语义对齐。0.75 只能作为初始候选阈值，必须画质量—保留率曲线并人工校准。
6. NFKC 只用于生成去重键，不替换训练原文。先跨来源精确去重，再做 MinHash 近重复去重。
7. 以原始 `pair_id/document_id` 为单位划分 train/dev；同一对的反向样本不能落到另一切分，同一文档不能跨切分。
8. 通过污染扫描后才生成两个方向的 SFT 样本。

内部验证集固定为 10,000 个唯一句对，按来源与领域分层，两个方向均评测。它只用于训练监控和早停，不进入最终排行榜表。

## 4. 样本格式与损失

基础模型没有经过指令微调，因此采用短、稳定、无聊天角色依赖的纯文本模板。训练时只使用 2–4 个语义相同的模板，评测固定一个模板。

英→简中：

```text
Translate the following text from English to Simplified Chinese.
English: {source}
Simplified Chinese: {target}<eos>
```

简中→英：

```text
请将以下简体中文翻译成英文。
简体中文：{source}
英文：{target}<eos>
```

只对 `{target}<eos>` 计算交叉熵；提示和源文 token 的 label 设为 `-100`。这样优化目标直接对应“给定原文，生成译文”。打包多个样本时，每个样本都要有独立 EOS、attention boundary 和 loss mask，不能让后一条看到前一条的答案。

文档阶段只在存在真实 `document_id` 和原始顺序时拼接相邻句。随机拼接不相关句子不能称为文档翻译。

## 5. 训练设置

### 5.1 4090 先导实验

采用 `configs/pilot_4090.json`：

- 全参数 BF16 SFT；序列长度 2048。
- micro batch 4、梯度累积 16；逻辑 batch 为 64 个打包序列。真实有效 batch 还应报告非 padding token 数。
- AdamW：`beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.01`，梯度裁剪 1.0。
- 默认峰值学习率 `3e-5`，先做 `[1e-5, 3e-5, 1e-4]` 三点短程扫描。
- 3% warmup，cosine 衰减，主实验 1 epoch；最多 2 epoch，仅在预注册的验证规则下早停。
- 动态 packing，Flash Attention 可用则开启；显存不足才开启 gradient checkpointing 或降低 micro batch。

143M 模型在 24GB 4090 上做全参训练是合理起点，但不能事先杜撰吞吐和耗时。先运行 1,000 step 校准，记录实际 tokens/s、峰值显存和数据加载占比，再按
`预计小时 = 计划非 padding token 总数 / 实测 tokens/s / 3600`
估算全程。

### 5.2 集群主实验

采用 `configs/main_cluster.json`。句级阶段长度 2048、1 epoch；之后用 5%–10% 的安全训练数据做文档/指令阶段，长度 4096、较低学习率 `1e-5`、0.25 epoch。第二阶段数据只能由训练来源和真实训练文档派生，不得利用最终评测题构造模板。

### 5.3 checkpoint 选择

每 500 updates 运行内部验证并保存关键 checkpoint。主选择指标为内部 dev 两方向 COMET 的平均值；BLEU/chrF++ 用于检查退化，若 COMET 差异小于预注册最小差异，则选择更早的 checkpoint。FLORES+ `dev` 只允许用于最后一次确认，不得反复挑 checkpoint；`devtest` 永远只在设置冻结后运行。

## 6. 评测组合

| 层级 | 数据集 | 方向 | 用途 |
|---|---|---|---|
| 训练监控 | 内部 10K clean dev | 双向 | loss、早停、错误分析 |
| 公共开发 | FLORES+ dev | 双向 | 一次性确认跨领域泛化 |
| 核心最终 | FLORES+ devtest（1012 句） | 双向 | 通用、多主题可比性 |
| 核心最终 | WMT22、WMT23 | 双向（按官方可用方向） | 较新新闻/通用对比 |
| 仅英→简中 | WMT24、WMT25、WMT26 | 英→简中 | 新近文档、领域与指令能力 |
| 多领域文档 | WMT24++ `en-zh_CN`，移除 `is_bad_source=true` | 英→简中 | news/social/speech/literary |
| 新闻扩展 | NTREX-128 | 英→中为原生；反向需明确标为反向使用 | 1997 句、带文档边界 |
| 医疗压力测试 | TICO-19 保留测试 | 英→中；反向另行标注 | 医疗和疫情术语 |
| 可选口语压力 | IWSLT transcript→translation | 按任务可用方向 | 这是语音转写文本翻译，不与纯文本主榜混报 |

FLORES+ 官方明确规定不应作为训练数据；其中 dev 为 997 条、devtest 为 1012 条。WMT24++ 每个语言配置约 998 行，包含 news、social、speech、literary 和 canary；评测时按官方建议移除坏源文本。NTREX-128 是英文新闻源到 128 种语言的参考翻译并保留文档边界，许可为 CC BY-SA 4.0。

WMT26 正式提交截止日是 2026-07-02，且本届中英有关方向只有英文→简体中文和英文→台湾繁体中文，没有中文→英文。当前模型第一版锁定 `zh-Hans`，繁体作为未来带显式 locale 标签的独立扩展，不能把简繁参考混作一个目标。

## 7. 指标与统计

主报告同时给出：

- SacreBLEU BLEU，并保存完整 signature；中文目标使用官方/基准约定的 `zh` tokenizer，英文目标通常使用 `13a`，以每个测试集官方要求为准。
- chrF++（word order 2），对中文分词差异更稳健。
- COMET `Unbabel/wmt22-comet-da` 作为主要语义指标；固定模型版本和哈希。
- XCOMET-XL 仅做代表性错误定位，MetricX-24 作为资源允许时的第二神经指标；MetricX 是误差分，通常越低越好，不能和 COMET 方向混淆。
- 语言正确率、空输出率、复制源文率、截断率、数字/实体保持率、格式合法率。

模型比较使用逐句 paired bootstrap 1,000 次，至少报告 95% 置信区间；关键主张再用 paired approximate randomization 10,000 次。不要因为 0.1 BLEU 的单次波动就声称改进。

最终每个方向抽 300–500 段做人评，20% 双人重复标注并仲裁。错误标签包括误译、漏译、增译、流畅性、术语、数字/专名、简繁体和格式。文档测试保留上下文。可以采用 MQM，或参考 WMT26 的 contrastive error-span 评价；自动指标不能代替人评。

## 8. 解码设置

主结果使用确定性 greedy decoding，温度和采样关闭。beam size 4 只作为一个预注册消融，length penalty 固定 1.0。按方向设置足够的 `max_new_tokens` 并记录所有截断；不使用测试集调 prompt、beam 或长度惩罚。

每次运行必须保存：模型 commit、checkpoint hash、数据 manifest/hash、prompt 版本、生成参数、逐句 hypothesis、参考、指标工具版本和 SacreBLEU signature。

## 9. 必做实验与消融

优先级详见 `EXPERIMENT_MATRIX.csv`。最低可发表证据链：

1. 原始 Haidass 零样本翻译基线。
2. 1M 双向、全参数、target-only loss 主基线。
3. target-only loss 对比整段 loss；其余设置相同。
4. 250K/1M/4M/8M 嵌套数据规模曲线。
5. clean-only 对比加入过滤 web 数据；固定总更新 token 或同时给出等 epoch 与等 token 两种解释。
6. 双向联合模型对比两个单向专家；报告两方向，不只报平均。
7. 句级模型对比加入文档/格式/术语阶段。
8. 全参对比 LoRA（建议 r=16 或 32）作为效率基线，报告可训练参数、峰值显存、tokens/s 和质量。
9. 至少 3 个随机种子用于 1M 主设置；8M 若成本受限，可 1 个主种子加 2 个较短复核，但必须如实写明。

## 10. 结果判定门槛

在上 8M 集群前，1M 试验必须满足：

- 两方向输出语言正确率均 ≥99%，空输出/源文复制/截断各 <1%。
- 相对原始基座，FLORES+ dev 的 BLEU、chrF++ 和 COMET 三者方向一致地改善。
- 500 条人工抽检中，没有系统性的反向翻译、简繁混淆或大量增译。
- 训练无 NaN，验证 loss 不持续恶化，数据抽检和污染报告齐全。

这些是工程门槛，不是论文显著性结论。论文结论必须以冻结测试集、置信区间和人评支持。

## 11. 论文可以诚实主张什么

只有数据支持时，才可形成如下主张：

- 在固定 143M 参数预算下，双语预训练模型的翻译 SFT 数据效率曲线。
- 双向联合训练在极小模型上的迁移—干扰边界及方向不对称。
- 数据质量、网页规模和文档/指令能力之间的可量化取舍。

不能提前写成结论，也不能把“在某公开榜单出现”本身当作创新。若实验不支持某个假设，应保留负结果并修改论点。

## 12. 主要官方资料

- [Haidass1.5-143M](https://huggingface.co/DALabCommunity/Haidass1.5-143M)
- [WMT26 General MT task](https://www2.statmt.org/wmt26/translation-task.html)
- [WMT26 repository](https://github.com/wmt-conference/wmt26-general-mt)
- [mtdata](https://github.com/thammegowda/mtdata)
- [OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
- [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus)
- [WMT24++](https://huggingface.co/datasets/google/wmt24pp)
- [NTREX-128](https://github.com/MicrosoftTranslator/NTREX)
- [TowerBlocks-v0.2](https://huggingface.co/datasets/Unbabel/TowerBlocks-v0.2)
- [SacreBLEU](https://github.com/mjpost/sacreBLEU)
- [COMET](https://github.com/Unbabel/COMET)
- [MetricX](https://github.com/google-research/metricx)
- [WMT MQM data/tools](https://github.com/google/wmt-mqm-human-evaluation)

