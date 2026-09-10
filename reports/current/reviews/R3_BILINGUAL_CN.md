# R3：双语评估、可复现性与低成本补实验（首轮）

本文件是用户要求的独立多智能体模拟评审，不是 ICLR 官方评审或录用预测。评审对象严格锁定为 `.repo-sync` 中 `git show 14c8fe2:haidass_iclr2027.tex`；下文 `M:Lx` 均指该提交的 TeX 源文件行号，不指后续工作树。未读取其他模拟审稿人的评分。计划、执行清单和稿外评估文件不计作初稿已经报告的结果。

本轮参考 [ICLR 2027 Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines) 对严谨、开放和聚焦决定性问题的要求；[Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) 鼓励可复现性声明，但不意味着列出未来将提供的附件就已完成复现。本文件的 0/2/4/6/8/10 推荐分是本次模拟约定，不称为官方评分表。

## Summary

论文把小型中英模型的数据选择、语言配额和训练顺序放在同一研究框架中。已完成的证据包括五个规模的英文质量过滤扫描，以及 43.46M/100.66M tokens 与 143.07M/331.35M tokens 两个成比例规模上的双种子顺序实验。后者显示，在所测试的四种双语调度中，固定 50:50 的随机序列混合具有最低的平衡 BPB，单次切换语言会显著损害先训练的语言。

这些结果有实用价值，且修订后的规模分析相当克制。不过，初稿将语言配额、迁移分解和成对双语行为列为已完成贡献，而对应结果仍是占位符。当前证据支持“这组短程训练中随机混合优于所测试的长块调度”，尚不支持已经建立一个经中文下游和等曝光迁移分析验证的双语训练配方。我的主要顾虑是证据与论文主张之间的距离，而不是模型必须更大、必须再训练 143M/803M。

## Strengths

1. **50:50 顺序比较的基本匹配有实际支持。** [顺序生成代码](https://github.com/umeiko/umeko_reports/blob/cdd9ac071bb89fd718d09a346c1c0ccc1e64095a/scripts/npu/make_order_143m.py#L38) 第 38–101 行先固定各语言数量，再以相同 seed 生成语言内索引；四个双语条件具有相同的语言内序列流。[顺序清单](https://github.com/umeiko/umeko_reports/blob/cdd9ac071bb89fd718d09a346c1c0ccc1e64095a/reports/p1_143m_proportional_npu/provenance/order_manifest.json#L47) 第 47–129 行记录一致的逐来源数量；[初始化清单](https://github.com/umeiko/umeko_reports/blob/cdd9ac071bb89fd718d09a346c1c0ccc1e64095a/reports/p1_143m_proportional_npu/provenance/init_report.json#L2) 保存两个 seed 的参数哈希。各运行记录相同的 331,350,016 tokens。小白解释：这些条件用同一批教材、同一个起点和相同课时，主要改变上课顺序，比较比公开模型排行榜更可信。

2. **成比例扩展的结论及限制区分得较好。** M:L378–390 明确指出参数、token、更新数和数据池一起变化，不把结果称为纯参数效应。独立 [规模分析](https://github.com/umeiko/umeko_reports/blob/cdd9ac071bb89fd718d09a346c1c0ccc1e64095a/reports/scale_analysis_43m_143m/SCALE_ANALYSIS_CN.md#L64) 第 64–81 行同时报告语言别代价，并说明“中文更容易忘”只适用于当前设置。原始 P1 报告第 14、73–83 行的较强表述不应倒灌到论文中。

3. **拟议的成对行为指标比单一一致率更合理。** M:L283、464–481 区分双语都正确、回答一致和两个方向的不对称错误；M:L281 要求两种语言分别满足非劣条件。这些是合理的设计选择。只是设计本身尚不能替代数值结果。

## Weaknesses

### W1：双语主贡献尚未得到对应的已完成评估支持

M:L38–40、64–68、88–89 使用已经完成的措辞描述比例实验、迁移分解和成对双语评估；但比例图在 M:L399–408 是占位图，分解表 M:L419–426、主要 143M 表 M:L450–459、成对表 M:L473–481 都是 TBD。附录列出 C-Eval/CMMLU 的数据量（M:L554–555）不等于给出中文成绩。已完成的 [143M 报告](https://github.com/umeiko/umeko_reports/blob/cdd9ac071bb89fd718d09a346c1c0ccc1e64095a/reports/p1_143m_proportional_npu/EXPERIMENT_REPORT_CN.md#L87) 第 87 行明确写“无下游任务评测”。英文质量扫描与英文排行榜也不能补上这块证据。

已完成条件确实没有 q=.25 和 q=.75：[43M 配置](https://github.com/umeiko/umeko_reports/blob/cdd9ac071bb89fd718d09a346c1c0ccc1e64095a/configs/p0_scaled_100m.json#L53) 第 53–77 行和 [143M 配置](https://github.com/umeiko/umeko_reports/blob/cdd9ac071bb89fd718d09a346c1c0ccc1e64095a/configs/p1_143m_proportional.json#L287) 第 287–324 行都只有单语端点、IID50 和三个顺序。因此不能声称已找到配额 Pareto 前沿，或 50:50 是最佳比例。小白解释：同一比例下换顺序，回答不了“该给中文多少课时”。

### W2：已运行协议与正文主要协议需要可一眼核对的区分

M:L245–256 的计划为 43M/1B/ctx2048、143M/3B/ctx4096、全局 batch128、BF16；实际短程配置为 ctx1024、batch32、FP16，分别 3,072/10,112 更新，并且每语言测试池只有 512 个 packed sequences。M:L550–551 的每语言 10M tokens、逐文档内外部分布评估属于未来合同。M:L529 的“will contain”也不是已存档证据。

实际 [物化代码](https://github.com/umeiko/umeko_reports/blob/cdd9ac071bb89fd718d09a346c1c0ccc1e64095a/scripts/materialize_data.py#L472) 第 472 行明确记录短程实验只做精确跨源去重，正式 P1/P2 才另外要求 MinHash。作者应把已完成实验的精确协议、去重/污染检查状态、实际评估单位单列，而不是让读者将正式计划的更完整保证应用到旧实验。此问题主要可以通过如实写作和已有工件审计解决，不要求重训大模型。

### W3：BPB 有效用于固定语料内比较，但实现和跨语言解释仍有边界

M:L269 定义逐文档 BPB。实际 [评估代码](https://github.com/umeiko/umeko_reports/blob/cdd9ac071bb89fd718d09a346c1c0ccc1e64095a/scripts/npu/eval_bpb.py#L24) 第 24–50 行计算 packed sequences 的总 NLL/总 bytes，并只预测 token 位置 1…1023；[物化代码](https://github.com/umeiko/umeko_reports/blob/cdd9ac071bb89fd718d09a346c1c0ccc1e64095a/scripts/materialize_data.py#L206) 第 206–225 行却以完整 1024-token 序列解码计算 bytes，跳过 special tokens。因此分母包含未预测首 token 对应文本，而 EOS 进入 NLL、没有普通文本字节。应明确边界与 EOS 协议，核对预测文本和字节分母，并补充逐评估单位记录。

这不自动推翻已有相同评估集下的相对顺序结果：若每个模型使用同一语言的共同字节分母，仅修改这个共同分母会在语言内相对 BPB 中抵消。但绝对 BPB、跨语言原始遗忘量，以及“逐文档统计”的称谓需要修正。固定 tokenizer 下 BPB 减少分词单位依赖，也不能把不同内容、不同 UTF-8 编码的中英文变成相同难度的试卷。不能从“中文原始遗忘是英文的 2.4 倍”推导中文语言本身更易遗忘；当前稿的语言内相对差值更合适。

### W4：等曝光分解不能直接解释成已经识别出的语言迁移机制

分解恒等式本身成立，但“相同目标语言 token 数”不足以隔离另一个语言的作用。单语 C/2 checkpoint 位于半程学习率和优化状态，双语 C 终点已经历全程更新与退火。此外，顺序代码第 61–76 行的来源配额与最终打乱依赖 `required`；单语全量流的前半段并不保证就是双语条件的半量流。因此，同 seed、同 token 数不自动保证这个分解比较中的目标语言序列前缀相同。

作者可以计算现有 checkpoint 上的协议相关残差，但应把它明确称为“相对该单语学习轨迹的差异”，披露 checkpoint/学习率/数据前缀匹配状态。小白解释：两个学生读了同样多英文，并不意味着他们以相同节奏、相同教材和相同复习程度学过；其差值不能全算成中文带来的帮助或干扰。这里并不要求先补齐昂贵的确认性研究，才允许报告有边界的探索结果。

### W5：XCOPA 的英文来源未定义，配对协议不可直接执行

M:L283、553 以及 [未来计划](https://github.com/umeiko/umeko_reports/blob/cdd9ac071bb89fd718d09a346c1c0ccc1e64095a/FUTURE_EXPERIMENT_PLAN.md#L60) 第 60 行假定 `XCOPA en/zh` 有 500 个平行对。实际核对计划锁定的 [数据集 revision 的 README](https://huggingface.co/datasets/cambridgeltl/xcopa/blob/042f78955ba48e6404616762fa6e05e839c3907a/README.md)，配置包含 `zh`、`translation-zh`，没有 `en`。作者的 [XCOPA 官方仓库](https://github.com/cambridgeltl/xcopa) 说明它是原英文 COPA 的翻译与重新标注，`data-gmt` 是 Google Translate 的 translate-test 数据。英文原题 COPA、中文题目的机器回译以及自行翻译，代表不同评估条件。

应冻结英文 COPA 来源、对应 `idx`、cause/effect、选项顺序和两边标签，并对 `changed` 项逐题审计后决定主分析是否排除。2026-09-10 读取的 [官方中文 test 文件](https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/zh/test.zh.jsonl) 有 500 项、17 项 `changed=true`；这是本轮对当前上游的核查，正式使用仍须与最终锁定 revision 对齐。不能把模型自己翻译出的英文称为原生标准测试对，否则翻译误差、难度改变和模型熟悉自身措辞都可能影响结果。

相比之下，[Belebele 官方说明](https://github.com/facebookresearch/belebele) 确认它有每语言 900 题、488 篇文章且语言间完全平行。应以 `link + split + question_number` 配对，使用 `eng_Latn/zho_Hans`；文章中有共同题目，置信区间最好增加以文章为簇的重抽样。对这类小模型应同时报告随机基线、答案位置频率和正确答案概率，避免将固定选同一选项造成的一致率视为双语理解。

### W6：当前顺序实验识别一个整体调度效应，尚不能分离其机制

代码中的 IID 是固定各半标签后随机置换，更准确称为“精确配额的随机混合”，并非每次独立 Bernoulli 采样。相对于长块条件，它同时改变每个 batch 的语言组成、连续单语跨度、切换次数、各语言在学习率曲线上的位置，以及最后训练语言。这不破坏“这两个整体配方谁更好”的比较，但限制了“块越短越好”或特定课程机制的归因。只测一个八块交替方案，也不足以证明单调块长规律。成比例 scale 又同时改变每块 token 数与更新数；论文已声明这个限制，应保持该限定。

## Questions

1. 作者是否愿意将标题、摘要和贡献收束到“英文过滤的描述性扫描＋双语短程顺序诊断”，把未完成的比例、确认性迁移和成对结果移到未来工作？若维持现有核心主张，请指出初稿中支持它们的已完成结果。
2. 能否提供“已运行/仅计划”协议对照，以及现有两个 seed 的初始化映射、语言内序列哈希、实际 token 计数、评估字节口径和污染检查状态？这些证据比再承诺一个更大实验更能改变我的判断。
3. 能否先对已有 released checkpoint 和已有 43M/143M 终点做一次冻结的中英下游评估，明确 XCOPA/COPA 配对来源，并完整报告 near-chance 或负结果？只需前向推理，无需增加 143M/803M 训练种子。
4. 对迁移分解，作者能否核验单语匹配曝光 checkpoint 的数据前缀和学习率状态，并明确哪些残差仍包含优化过程差异？

### 在当前预算约束下，最有价值的小实验

| 优先级 | 实际工作 | 预计新增训练 | 能回答／不能回答 |
|---|---|---|---|
| 1 | 审计已有中文逐题输出；为现有终点补 Belebele 中英与经过核验的 COPA/XCOPA 配对推理；补共同外部文本的逐单位 NLL/bytes | 0 | 可检查 BPB 排序是否落到任务表现，补中文实证；不能估计训练种子方差或证明迁移机制 |
| 2 | 在 43M、100,663,296 tokens、50:50 和已有两 seed 上增加“batch 整体为 EN 或 ZH、batch 标签随机且总量平衡”的条件 | 2 个 43M 短程 run | 与现有序列混合比较，有助区分 batch 内混合和长程顺序；仍是一个明确的调度干预，不是完整机制证明 |
| 3 | 若确要回答比例问题，复用同一协议补 q=.25/.75，两 seed | 4 个 43M 短程 run | 补齐短程五点配额前沿；不能声称 143M/3B 或发布模型的最佳比例 |
| 4 | 选择一个明确的质量干预，在 43M 上补下面所述的质量×顺序小型设计 | 严格满足复用条件时 4 个新 run | 直接检查选定过滤效应是否随选定顺序变化；范围只到该语料、阈值、顺序和短程规模 |

以上按选择题理解，不是要求把全部项目做完。当前优先完成第 1 项和表述收束；不把增加 143M/803M seeds 当作修稿先决条件。

**50:50 下的质量 2×2 必须纠正一个设计逻辑。** [桥接实验建议](https://github.com/umeiko/umeko_reports/blob/cdd9ac071bb89fd718d09a346c1c0ccc1e64095a/reports/exps_quality_scale/PAPER_STORY_REFRAME_CN.md#L123) 第 123–138 行把四格定义为 EN full/filtered × ZH full/filtered，而且全部固定 IID，却希望判断“过滤和语言顺序是否存在交互”。该设计可以估计两个语言的过滤主效应、跨语言影响，以及 EN 过滤×ZH 过滤交互；**没有变化过顺序，便无法识别质量×顺序交互。**

若这一交互确实是问题，可改成质量 full/filtered × 顺序 IID/八块交替，语言比例始终 50:50，每格两 seed。预先定义每语言相同 held-out BPB 上的差中之差：

`I_l = (L_l[filtered, block] - L_l[full, block]) - (L_l[filtered, IID] - L_l[full, IID])`。

先决定质量干预是只过滤 EN，还是同时过滤两语言；二者回答不同问题。若原 full-IID/full-block 两格的数据规则、模型、token、初始化和全部训练设置与新实验完全一致，可复用已有四个 run，仅新增 filtered-IID/filtered-block × 两 seed＝四个 43M 短程 run（合计约 402.7M 新训练 tokens）。若现有混合语料不能对应所选 full 条件，就不能借“同为 43M”强行复用，应重做小型四格或只报告质量主效应。过滤后保持每语言 token 配额、固定来源权重与补样规则，并记录唯一文档数及重复曝光；各质量格内不同顺序必须使用相同序列集合。这个差中之差最多识别选定过滤操作与选定调度的交互，不等于发现普遍的“质量机制”。

### 稿外可复用结果：不计入首轮分数

本地 [已有中文评估 JSON](https://github.com/umeiko/ICLR-2027-Conference-Submissions/blob/14c8fe216cc0853d2a58545d2515a6353be5d2d1/eval_outputs/v15_zh_bf16/DALabCommunity__Haidass1.5-143M/results_2026-08-30T00-12-02.015088.json#L1074) 第 1074–1113 行已有发布模型的 XCOPA-ZH 57.8%、XWinograd-ZH 59.52%、C-Eval 25.93%、CMMLU 25.46%，并有逐题文件。它们未进入锁定初稿的结果表，也不是这批受控 replicas 的结果，故不据此加分。作者可先核对 harness/task 版本、few-shot 配置、聚合权重和预测文件，整理成真实的描述性中文结果；尤其后两项接近四选一随机水平，不能包装成强中文知识能力。这说明补齐“中文下游证据”可能主要是评估审计和写作工作，而非新训练工作。

## Recommendation

**2／10：拒绝（本次模拟的内部档位）。** 原因是两项核心双语贡献仍停留在方案与 TBD，发布模型的中文证据未报告，且实际短程协议、BPB 和配对评估合同之间还有影响解释的缺口。已经完成的顺序结果值得报告，我不要求作者靠更大模型或更多昂贵种子来获得资格。若作者将主张收束到已有探索证据，完成现有 checkpoint 的中文/成对评估与协议审计，并修正 2×2 的可识别问题，我会重新评估；仅追加未来实验承诺不足以改变本轮推荐。

## Confidence

**4／5。** 对初稿是否已提供结果、配置差异、顺序代码和数据集配置的判断有直接证据。未独立重跑 NPU 训练，也未验证每个实际 checkpoint 与执行日志，因此不把工件审计提升为完整端到端复现。对交互设计的判断来自标准因子识别逻辑；对训练机制的判断保留上述限制。
