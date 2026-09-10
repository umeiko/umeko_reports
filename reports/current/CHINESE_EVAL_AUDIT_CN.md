# 已有中文评估工件核验

核验日期：2026-09-10。核验范围仅为 `eval_outputs/v15_zh_bf16/DALabCommunity__Haidass1.5-143M` 的现有汇总 JSON、121 个逐题 JSONL、相关本地模板和评估代码。没有启动训练、模型推理或新一轮评测；以下重算仅对保存的数字进行算术、计数和哈希核验。它是 R3 首轮之后的独立工件审计，不会倒算为锁定初稿已经提供的证据。

## 结论与可使用范围

**四项结果均有完整逐题工件支撑，可以加入当前进展报告，以及论文中“公开发布的 Haidass1.5-143M checkpoint：描述性中文评估”表。** 121 个文件合计 13,932 条记录；每个任务的逐题得分、保存的候选似然 argmax、题数和汇总正确率完全一致，没有非有限候选分数。这种一致性验证了保存结果的内部可核对性，不等于独立重跑模型。

结果属于 revision `d8a00d4943971088e6f0f4e08fb317dd5ba33ed1` 的发布模型，不属于随机初始化的 43M/143M 受控实验。不能用于证明某个质量过滤、语言配额或训练顺序造成改进，也不能代替中英成对正确率、迁移分解或污染排除后的因果分析。C-Eval/CMMLU 接近随机水平；XCOPA-ZH/XWinograd-ZH 在本次协议下的点估计高于二选一随机基线，支持有限的中文任务表现描述，不能概括成强中文知识或推理能力。

## 精确数值及不确定性

来源：[原始结果 JSON 第 1074–1113 行](https://github.com/umeiko/ICLR-2027-Conference-Submissions/blob/14c8fe216cc0853d2a58545d2515a6353be5d2d1/eval_outputs/v15_zh_bf16/DALabCommunity__Haidass1.5-143M/results_2026-08-30T00-12-02.015088.json#L1074)。下表的正确题数、Wilson 区间和学科宏平均是本次从已保存预测重算的数字；SE 原样保留已有工件的计算口径。

| 任务与 split | 子任务数 | 正确／总题数 | 已存正确率 (%) | 已存 SE (百分点) | 本次重算 Wilson 95% 区间 (%) | 均匀随机 (%) |
|---|---:|---:|---:|---:|---:|---:|
| C-Eval `val` | 52 | 349 / 1,346 | 25.92867756 | 1.19495310 | [23.6583, 28.3361] | 25 |
| CMMLU `test` | 67 | 2,949 / 11,582 | 25.46192367 | 0.40523120 | [24.6768, 26.2634] | 25 |
| XCOPA `zh/test` | 1 | 289 / 500 | 57.80000000 | 2.21090393 | [53.4277, 62.0534] | 50 |
| XWinograd `zh/test` 扩展版 | 1 | 300 / 504 | 59.52380952 | 2.18857373 | [55.1821, 63.7214] | 50 |

**聚合口径不能写错。** C-Eval 和 CMMLU 现有总分是按各学科题数加权，等于所有题目合并后的正确率。C-Eval group 配置（`.venv-eval/Lib/site-packages/lm_eval/tasks/ceval/_ceval-valid.yaml:1`，本地 lm-eval 0.4.12） 和 CMMLU group 配置（`.venv-eval/Lib/site-packages/lm_eval/tasks/cmmlu/_cmmlu.yaml:70`，本地 lm-eval 0.4.12） 均使用 `weight_by_size: true`。若正文坚持原计划中的“学科等权 macro”，无需重推理，可将已有学科准确率等权平均，得到 C-Eval **26.66673953%**、CMMLU **25.47889119%**。它们是不同指标，不能把现有 25.93/25.46 直接标为等权 macro；上表 SE 和区间也不能原样贴给新 macro。

**已有 `acc_stderr,none` 是标准误 SE，不是 SD、95% CI 或跨 seed 方差。** 对 XCOPA/XWinograd，数值精确等于 `sqrt(p*(1-p)/(n-1))`。C-Eval/CMMLU 使用 harness 的 pooled_sample_stderr（`.venv-eval/Lib/site-packages/lm_eval/api/metrics.py:590`，本地 lm-eval 0.4.12） 汇总学科内标准误；本次代入所有子任务重算后分别为 0.011949531007737630 与 0.004052312031024866，和原值在浮点误差内相同。它们不完全等于把全部题目视为同一 Bernoulli 样本后计算的 SE，差异有定义上的原因。

结果文件虽然记录 `bootstrap_iters=100000`，但均值准确率的 SE 由 mean_stderr 映射（`.venv-eval/Lib/site-packages/lm_eval/api/metrics.py:580`，本地 lm-eval 0.4.12） 解析计算，**不能据这个参数宣称这些结果做过十万次 bootstrap 并保存了区间**。本次 Wilson 区间使用 z=1.959963984540054 和正确／总题数计算，假设题目近似独立；它们没有覆盖训练随机性、学科或题型相关性、模板敏感性、数值精度及污染不确定性。正式表最简洁做法是报告点估计和 n；若展示区间，要保留“本次事后按题计算、非 seed CI”的说明。

## 模型、执行版本与模板

[JSON 第 9725–9768 行](https://github.com/umeiko/ICLR-2027-Conference-Submissions/blob/14c8fe216cc0853d2a58545d2515a6353be5d2d1/eval_outputs/v15_zh_bf16/DALabCommunity__Haidass1.5-143M/results_2026-08-30T00-12-02.015088.json#L9725) 记录：模型 `DALabCommunity/Haidass1.5-143M`，143,071,296 参数，revision/model_sha 均为 `d8a00d4943971088e6f0f4e08fb317dd5ba33ed1`；权重及运行类型 BF16；lm-eval 0.4.12，Transformers 4.51.0；最大长度 4096，batch64，无 `limit`，seed 为 0/1234/1234/1234。121 个 task config 都为 `num_fewshot=0`、repeats=1，且模型 revision 一致。没有 chat template、system instruction 或 CoT。日志开始时间为 2026-08-29 15:58:43 UTC（北京时间 23:58:43），保存文件名对应次日北京时间 00:12:02，总评测时长约 805.55 秒。

| 任务 | 保存的评分方式 | 模板关键点 | 与“统一长度归一化答案似然”的关系 |
|---|---|---|---|
| C-Eval | 四个答案字母的 log-likelihood；保存 `acc` 和 `acc_norm` | 中文学科说明＋题干＋A/B/C/D 全部选项＋`答案：`；续写 ` A` 等字母 | 四个字母相同长度，原始和归一化指标相同；不是生成完整答案或知识解释 |
| CMMLU | 同上 | 中文学科说明＋题干＋全部选项＋`答案：` | 同上 |
| XCOPA-ZH | 两个选项续写的**原始累计 log-likelihood**；仅保存 `acc` | premise 去掉末标点，接“因为／所以”，两个实际选项作为续写 | 本次没有 `acc_norm`，不能称这批分数使用了统一的长度归一化协议 |
| XWinograd-ZH | 两个替换了代词的不同前缀，预测同一后缀的 log-likelihood，比较候选上下文 | multiple-input 模式；不是把答案 A/B 当文本续写 | 保存的是 `acc`；应按任务实际实现描述 |

JSON 保存了具体题目、实际输入 arguments 和候选分数，足以核验这次的评分输入。XCOPA 的 helper 在 JSON 中部分序列化为函数地址，因此完整历史环境仍需代码快照。当前 XCOPA helper（`.venv-eval/Lib/site-packages/lm_eval/tasks/xcopa/utils.py:1`，本地 lm-eval 0.4.12） 和 XWinograd task 说明（`.venv-eval/Lib/site-packages/lm_eval/tasks/xwinograd/README.md:1`，本地 lm-eval 0.4.12） 与上述输入结构相符。

## 逐题审计：偏置、并列及重复记录

四组全部通过：文件数/任务数与 52/67/1/1 一致；有效题数匹配；逐题准确率重算误差 0；每子任务汇总误差 0；候选分数非有限值 0。C-Eval、XCOPA、XWinograd 的 `doc_hash` 在各自集合内唯一；CMMLU 的 11,582 条记录中有 11,579 个不同 `doc_hash`，即三个各重复两次的题目哈希，六条记录均答错。本次保留原始基准全题口径，没有自行删题；去重敏感性可在补充材料另报，不能静默改变 n。

| 任务 | 预测选项计数（A/B/C/D 或 1/2） | 正确标签计数 | 最高分并列题数 | 均匀打破最高分并列的期望准确率 (%) |
|---|---|---|---:|---:|
| C-Eval | 222 / 134 / 844 / 146 | 310 / 339 / 344 / 353 | 152 (11.29%) | 25.7615 |
| CMMLU | 2683 / 725 / 6521 / 1653 | 2926 / 2883 / 2877 / 2896 | 1338 (11.55%) | 25.4893 |
| XCOPA-ZH | 273 / 227 | 250 / 250 | 5 (1.00%) | 57.7000 |
| XWinograd-ZH | 261 / 243 | 247 / 257 | 45 (8.93%) | 59.4246 |

C 选项分别占 C-Eval 预测的 62.70%、CMMLU 的 56.30%，而标签接近均匀；因此“接近随机”不等于模型每道题真的在随机抽签。高选项偏好可能反映字母先验、模板适配或能力限制，这份工件本身不能区分其来源。并列按现有 `np.argmax` 选择最前选项；上表的均匀并列敏感性仅对保存分数重新计数，不是新推理，也不替换主结果。

当前 HF scorer 第 1506–1560 行（`.venv-eval/Lib/site-packages/lm_eval/models/huggingface.py:1506`，本地 lm-eval 0.4.12） 使用可选 `softmax_dtype`，默认 None；该次 model_args 没有显式设置 FP32 softmax。保存的 BF16 分数中确实存在并列，但不能仅凭现存工件证明所有并列都由 BF16 造成，或预言 FP32 重评会把分数提高多少。报告应标注 BF16；若将来需要比较小幅增益，FP32 log-softmax/累加及模板敏感性核查有价值，本轮未执行这些评测。

## 来源哈希与已保存的脱敏证据

已生成 [小型审计汇总 JSON](evidence/chinese_eval_audit_summary.json)，包含四项重算、模型/任务版本、并列敏感性、121 个源文件的文件名/SHA256/行数清单。只保留公开仓库标识、哈希和统计量；没有复制题目正文、完整 prompt、环境转储、缓存目录、配置来源绝对路径或凭据。导出前还检查了常见凭据串模式。没有复制大批逐题原文。

| 来源 | SHA256 |
|---|---|
| `results_2026-08-30T00-12-02.015088.json` 原始文件 | `97cceb9bbc89e9fc7ca59fc8dd97f204b1cc985c3cbe4cd7af0a26c7c67c0c0c` |
| XCOPA 500 条逐题 JSONL 原始文件 | `8b10ab6cac6dad780b4eccfbc8b3d5cae2d501c9eebe9820dc6a06b88d93353a` |
| XWinograd 504 条逐题 JSONL 原始文件 | `b734eb9e7a609abedea4f718c307e6921377c3fae25c69a1071f2fd6b930ca5c` |
| 全部 121 个逐题文件清单的规范化摘要 | `e1cff753356298c87b5e226b02377979d9d5261553008d12586c47ab901dc09b` |

最后一行摘要的精确定义为：按文件名排序，对每个源文件形成 `文件名 + TAB + 小写SHA256 + LF`，依次拼接为 UTF-8 字节串再取 SHA256。它覆盖文件集合和逐文件字节哈希，不是题目语义哈希。原始 JSON/JSONL 文件哈希对换行变化敏感。

## 尚存复现缺口与引文

1. **数据集 revision 有预期值，但历史执行证据不完整。** 当前 [预检脚本第 6–10 行](https://github.com/umeiko/ICLR-2027-Conference-Submissions/blob/14c8fe216cc0853d2a58545d2515a6353be5d2d1/eval/check_dataset_revisions.py#L6) 记录 C-Eval `617524a00b307ff6f9933702f724131fe12ca7ce`、CMMLU `efcc940752ea4a1ea94d2727f11f83858d64fc8e`、XCOPA `042f78955ba48e6404616762fa6e05e839c3907a`、XWinograd `90b619ef5278605cd4f572e3e8506ab16afd44f2`。[运行脚本第 23–24 行](https://github.com/umeiko/ICLR-2027-Conference-Submissions/blob/14c8fe216cc0853d2a58545d2515a6353be5d2d1/eval/run_p0_zh.ps1#L23) 设计为先检查远端 HEAD 再运行。但这不是 task loader 的显式 `revision=` 锁定；结果中没有 dataset_kwargs/revision，也没有本次预检日志的不可变关联。因此这里将它们称作“预检预期 revision”，没有伪称已独立验证实际载入的每个数据文件来自该版本。已有逐题哈希与原文可用于后续离线对齐核验。
2. **环境版本较完整，代码和权重的历史字节级锁定仍缺。** 结果中的 harness `git_hash`、`upper_git_hash` 均为 null；model_sha 是仓库 revision，不是另算的权重张量哈希。当前启动脚本指向 `.venv-eval-zh`，结果 config_source 记录的是 `.venv-eval`；核验两处当前 `huggingface.py` 与 XCOPA helper 的对应文件 SHA256 相同，但不能据此恢复整个历史环境。正式附件应保存已用 package lock、关键 task YAML/helper、模型文件哈希与执行命令。
3. **污染状态未解决。** 全部 121 个 task config 的 `should_decontaminate=false`。这证明该 harness 没有做其内置去污染流程，不证明上游训练一定污染，也不证明任何外部污染审计已经完成。发布模型完整 400B 语料不可见时，只能标注污染状态 unresolved；不能把现有准确率标成“无污染成绩”。
4. **XWinograd-ZH 的 504 题是扩展集合。** [数据维护者说明](https://huggingface.co/datasets/Muennighoff/xwinograd) 明确为原 XWinograd 的 16 道中文题，加 488 道 `clue/cluewsc2020` 题。建议表中写 `XWinograd-ZH (Muennighoff, 504)`，文中交代扩展来源，并引用 Tikhonov & Ryabinin 的 [原研究](https://arxiv.org/abs/2106.12066) 和 Muennighoff et al. 的 [Crosslingual Generalization through Multitask Finetuning](https://arxiv.org/abs/2211.01786)。不能把此集合写成 504 道英文—中文平行题，也不应只引用原 16 题的数据来源。
5. **其余建议引文和 split 名称。** C-Eval 用 [作者官方仓库](https://github.com/hkust-nlp/ceval)／Huang et al. 2023，明确是 1,346 题 validation，而非官方隐藏标签 test 排行榜；CMMLU 用 [作者官方仓库](https://github.com/haonan-li/CMMLU)／Li et al. 2023，67 学科 test。XCOPA-ZH 用 [Ponti et al. 2020](https://aclanthology.org/2020.emnlp-main.185/) 和 [官方数据说明](https://github.com/cambridgeltl/xcopa)。本次只有中文分数，无需自行翻译；未来若做中英配对，必须额外冻结原英文 COPA 与 XCOPA 的题号、标签及 changed 项映射。上述论文对应 C-Eval、CMMLU、XCOPA 的 BibTeX 已在现有论文 bib 中；XWinograd 扩展版还应补对应来源，而不是把扩展数据误标成原始版本。

## 可直接使用的表注与正文

推荐表名：“发布模型 Haidass1.5-143M 的描述性中文零样本评估”。推荐表注：“模型 revision d8a00d49；lm-eval 0.4.12，BF16；采用各任务保存的原始似然准确率。C-Eval/CMMLU 为按题数加权准确率；C-Eval 使用 validation，其余使用 test。XWinograd-ZH 为含 CLUE 补充题的 504 题版本。结果来自单个发布 checkpoint，训练污染状态未确定，不用于估计受控训练干预效应。”

推荐正文：“在现有发布模型的零样本评估中，C-Eval 和 CMMLU 的按题准确率分别为 25.93% 和 25.46%，接近四选一随机水平；XCOPA-ZH 与扩展版 XWinograd-ZH 分别为 57.80% 和 59.52%。这些结果描述该发布 checkpoint 在指定中文任务和模板下的表现，不将其归因于某个语言配额或课程设计。”

如论文选择学科等权 macro，应在另列中使用 26.67%/25.48% 并更新表注；不要混用两个聚合定义。本轮未将这些数值写入论文正文或原始结果文件。

## BPB 问题：共同分母何时抵消，哪些问题仍保留

此前发现短程 BPB 实现对 1024-token packed sequence 只预测位置 1…1023，但 bytes 统计整个解码序列；EOS 进入 NLL 而特殊 token 被解码时跳过。因此原始数值不是严格对齐预测内容的逐文档 BPB。

**对于同一语言、完全相同的 eval 集与边界规则，如果只是修正所有模型共享的字节分母，语言内相对差值会精确抵消这个共同分母。** 设 `BPB_A=N_A/(B ln2)`、`BPB_IID=N_IID/(B ln2)`，则 `(BPB_A-BPB_IID)/BPB_IID=(N_A-N_IID)/N_IID`。将共同 B 换成正确的共同 B' 不改变这个比值，也不改变同语言终点排序。论文“先对每种语言相对 IID，再等权”的平衡指标因此不因**单独的共同分母修正**自动失效。两个规模若也使用同一语言的相同 eval，语言内相对规模收益及同单位差分之比同理。

但不能把这个抵消性质扩大成全部问题已解决：

- 若修复时同时去掉 EOS NLL、补算首 token、改变上下文重置、截断或 padding mask，分子也会变化，不能保证相对效应原样保留。
- 绝对 BPB、绝对遗忘量会随 B 修正而变；直接平均 EN/ZH 原始 BPB 的语言权重也会受两种语言不同的修正比例影响。原始绝对量上的跨语言“谁更怕忘”没有因此获得可比性。
- 逐序列汇总总 NLL/总 bytes 不等于已经保存逐文档结果；同一文档可能跨多个 packed sequences，不能把相邻 sequence 都当独立文档重抽样。现存汇总没有补出文档级 bootstrap。
- 即使 bytes 与预测文本严格对齐，UTF-8、语言文本内容与来源分布仍不同。BPB 更适合相同语言/相同 held-out 集内部比较，不能将不同语言原始 BPB 当作相同难度的语义能力量尺。

因此，当前最准确的处理是保留有共同 eval 支持的语言内相对探索结果，披露原始 BPB 的边界/EOS定义和统计单位；绝对量与跨语言解释维持限制。中文下游表本身采用候选似然准确率，与这一 BPB 分母问题是不同评估链条，不能因 BPB 问题直接否定四项已保存准确率，也不能拿四项准确率反向证明 BPB 口径正确。
