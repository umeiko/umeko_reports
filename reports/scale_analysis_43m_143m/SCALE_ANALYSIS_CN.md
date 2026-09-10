# 43M→143M 成比例 Scale-up 独立分析

> 历史数值保留。EXPS 已成功拉取并完成另一个英文质量 Scale 分析，旧文访问失败状态不再适用。历史 BPB 存在目标范围/字节分母与 EOS 约定问题，现按打包块损失解释；相同语言内相对量与跨语言绝对量不能混用。当前边界和已完成清单见 [进展总览](../current/PROGRESS_CN.md)。

分析日期：2026-09-09。分析脚本：`scripts/analysis/analyze_scale_43m_143m.py`。

## 结论摘要

在这组**成比例放大**实验中，参数量从 43,461,504 增至 143,071,296（3.292 倍），每次运行的训练 token 从 100,663,296 增至 331,350,016（3.292 倍），token/参数比保持约 2.316。所有六种调度的最终 BPB 都下降，但调度决定了规模收益能否同时留在两种语言上。

最稳健的观察是：

1. `iid_50` 在两个规模上均取得最低的双语平均测试 BPB：1.5962→1.3403，相对下降 16.0%。
2. `alternating` 的预注册等语言相对惩罚几乎不随 scale 改变：相对 IID 为 +4.57%→+4.62%。它获得了与 IID 接近的规模收益，但始终没有超过 IID。
3. 一次性切换语言的长块调度随 scale 变得更不平衡：`en_first` 的等语言相对惩罚由 +12.01% 增至 +19.73%，`zh_first` 由 +23.89% 增至 +36.41%。
4. 更直观地说，扩大模型和训练量后，`en_first` 中先学英文只得到 IID 英文绝对规模收益的 29.3%；`zh_first` 中先学中文只得到 IID 中文绝对规模收益的 11.8%。后学语言得到了绝大部分收益，先学语言的收益被后续训练冲掉。
5. 终点遗忘与上述现象一致：`en_first` 的英文遗忘 0.3080→0.4568 BPB，`zh_first` 的中文遗忘 0.8268→1.0912 BPB。这个变化只能表述为“本次成比例 scale-up 中观察到更大遗忘”，不能归因成纯粹的模型参数效应。

![43M 到 143M 的 Scale 总结](figures/fig_scale_summary.png)

## 1. 分析了哪些实验

本分析只使用 `umeko_reports` 中可核验的两组 NPU 结果：

| 项目 | 43M | 143M |
|---|---:|---:|
| 参数量 | 43,461,504 | 143,071,296 |
| tokens/run | 100,663,296 | 331,350,016 |
| token/参数 | 2.3162 | 2.3160 |
| context | 1,024 | 1,024 |
| 条件 | 6 | 6 |
| seeds | 2027、2028 | 2027、2028 |
| 完整运行 | 12/12 | 12/12 |

六个条件为英文单语、中文单语、50:50 逐序列 IID 混合、先英文后中文、先中文后英文、八块交替。43M 和 143M 使用相同的 dev/test；训练池在来源配比上对齐，143M 池是 43M 池的前缀扩展。

143M 是与 Haidass1.5 相同架构规模的**受控随机初始化训练**，不是公开 Haidass1.5-143M checkpoint 的继续训练，也不是其 400B-token 原始预训练的复刻。

## 2. 完整性审计

- 两个 `report_manifest.json` 均登记 `status=complete`、12 个完整 run、0 个缺失 run。
- 两份逐 seed CSV 的 SHA-256 与 manifest 完全一致：43M 为 `e806d05f…ba0681`，143M 为 `c79b7014…f44e3`。
- 每个规模恰有 6 条件×2 seeds=12 行，指标无缺失；合计重算 24 个运行终点。
- 报告、JSON 和 Markdown 在 Windows checkout 下因 Git `autocrlf=true` 呈 CRLF，按 LF 规范化后的 SHA-256 与 manifest 一致；这是行尾转换，不是内容损坏。
- 公共仓库提供终点和汇总指标，但没有每个 held-out document 的 NLL/byte 记录，因此无法独立重做预注册的文档层 hierarchical bootstrap。

## 3. 规模收益

BPB 是“模型平均需要多少 bit 才能解释一个字节”，越低越好。下表给出两种语言原始 BPB 的等权平均，仅用于直观总览。

| 条件 | 43M 平均 BPB | 143M 平均 BPB | 相对下降 |
|---|---:|---:|---:|
| EN only | 2.2491 | 2.0412 | 9.24% |
| ZH only | 1.8920 | 1.6798 | 11.22% |
| IID 50:50 | **1.5962** | **1.3403** | **16.04%** |
| EN→ZH blocks | 1.7703 | 1.5783 | 10.85% |
| ZH→EN blocks | 2.0200 | 1.8820 | 6.83% |
| Alternating | 1.6690 | 1.4024 | 15.97% |

更大的模型配合更多训练 token 确实整体更好，但改善并不均匀。长块课程的先学语言只改善 3.68%（EN→ZH 中的英文）和 1.34%（ZH→EN 中的中文）；同样语言在 IID 中分别改善 15.27% 和 16.65%。

## 4. 相对 IID 的调度代价

论文 Eq. `Delta_curr` 是先分别计算两种语言相对 IID 的变化，再等权平均。这比直接平均原始 BPB 更符合预注册口径。

| 调度 | 43M 等语言相对惩罚 | 143M 等语言相对惩罚 | 143M 英文相对 IID | 143M 中文相对 IID |
|---|---:|---:|---:|---:|
| Alternating | +4.57% | +4.62% | +4.42% | +4.82% |
| EN→ZH blocks | +12.01% | +19.73% | +38.22% | +1.25% |
| ZH→EN blocks | +23.89% | +36.41% | −1.21% | +74.02% |

`zh_first` 的 143M 英文甚至比 IID 好 1.21%，但中文差 74.02%；这不是双语改进，而是明显的语言权衡。`en_first` 同理，最终中文接近 IID，但英文大幅退化。任何只展示后学语言或只展示平均排名的叙述都会掩盖这个问题。

## 5. 遗忘随 Scale 的描述性变化

| 先学后忘的语言 | 43M forgetting | 143M forgetting | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| EN→ZH 中的英文 | 0.3080 | 0.4568 | +0.1488 | +48.3% |
| ZH→EN 中的中文 | 0.8268 | 1.0912 | +0.2644 | +32.0% |
| Alternating 中的英文 | 0.0129 | 0.0381 | +0.0252 | — |
| Alternating 中的中文 | 0.0000 | 0.0375 | +0.0375 | — |

“中文遗忘值大于英文”只在当前 tokenizer、语料、调度和 BPB 口径下成立。不能据此声称中文天然更容易遗忘。中英文数据难度、字节编码、源构成和模型共享方式都可能影响数值。

## 6. 新洞见

### 洞见一：Scale 收益会被训练顺序定向分配

规模升级不是均匀地把两个语言都变好。最后训练的语言拿到绝大部分 scale dividend，先训练的语言收益被覆盖。这比笼统说“会遗忘”更具体，也更容易在下一轮实验中验证。

### 洞见二：短块交替保留规模收益，但仍有固定调度税

Alternating 在英文和中文上分别获得 IID 绝对 scale 收益的 105.8% 和 103.0%，说明它没有丢掉 scale dividend；但两个规模都比 IID 差约 4.6%。一个合理但尚未验证的解释是：大块内的局部遗忘能被后续块部分修复，却始终产生额外振荡。要验证这一机制，必须增加块长 sweep，而不是直接把解释写成事实。

### 洞见三：大模型不自动解决课程遗忘

至少在参数和训练量同步放大的这两个点上，容量增加没有自动消除顺序问题。后续最值得做的不是再跑一个同样的长块，而是固定 143M、固定总 token，系统扫描块长或语言切换频率，并在每个边界密集评测。

## 7. 论文可用表述

> In a proportional scale-up from 43.46M parameters/100.66M tokens to 143.07M parameters/331.35M tokens, sequence-level i.i.d. mixing remained the strongest bilingual baseline. The balanced relative penalty of eight-block alternation was stable (4.57% vs. 4.62%), whereas one-switch curricula became more imbalanced (12.01% to 19.73% for EN-then-ZH; 23.89% to 36.41% for ZH-then-EN). The first-trained language captured only 29.3% and 11.8% of the corresponding i.i.d. scale dividend. These two-seed results are exploratory because parameters, training tokens, update count, and pool size change together.

不应使用：

- “我们发现了 scaling law”——只有两个 scale 点，不能拟合可靠规律。
- “模型越大必然遗忘越严重”——参数和训练 token 同时变化，因果不可分。
- “143M 结果确认了论文主假设”——当前只有 2 seeds、331M tokens、ctx 1024，和预注册 3 seeds、3B tokens、ctx 4096 的 P2 不同。
- “Haidass1.5 本体证明了该机制”——这些是同架构的受控随机初始化 replica。

## 8. 下一轮最有价值的实验

1. **块长 sweep**：固定 143M、331M tokens 和 50:50 数据，比较 IID、1/2/4/8/16/32 块；每个块边界前后评测。
2. **纯参数 Scale**：固定相同训练 token 和样本顺序，只改 43M/143M，分离容量效应。
3. **纯训练量 Scale**：固定 143M，只改 101M/331M/1B/3B tokens，分离训练时长效应。
4. **补第三及以上 seed**：至少 3 seeds；公开逐文档 NLL/bytes 后执行预注册 hierarchical bootstrap。
5. **下游验证**：确认 BPB 的排序是否在 Belebele/XCOPA 成对中英正确率上保持。
6. **尾段补偿**：在长块结束后加入小比例先学语言，测最少多少 replay 可以收回遗忘；这应作为新实验，不从现有结果直接推断最优比例。

## 9. 数据缺口

用户指定的 `https://huggingface.co/DALabCommunity/EXPS` 在 2026-09-09 使用有效、已认证的账户 token 查询时，对 model、dataset 和 space API 均返回 404，组织可见仓库列表中也没有该名称。因此本报告没有使用 EXPS 数据。待仓库地址或权限修正后，应把其中更多参数规模点接入同一脚本，再决定是否能够拟合 scaling curve。
