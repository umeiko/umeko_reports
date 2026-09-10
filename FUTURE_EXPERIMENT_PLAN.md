# ICLR 2027 双语受控实验执行表

> 本文件为历史大预算计划，143M/803M 额外种子与 143M 3B 矩阵已延期。当前已完成 Scale/IID 实验和低预算增量工作以 [当前进展](reports/current/PROGRESS_CN.md) 与 [IID 后续方案](reports/current/IID50_EXPERIMENT_PLAN_CN.md) 为准，不要重复启动已完成项目。

版本：2026-08-30。本文档是上集群前的执行合同，不包含任何论文结果。硬件型号只进入运行清单，不构成论文贡献。

## 1. 唯一主问题

在固定模型参数、Tokenizer、总训练 Token、优化器更新次数和具体中英文样本流时：

1. 双语模型相对单语模型的差距，有多少来自目标语言数据减少（data dilution）？
2. 有多少来自另一语言带来的正迁移或负干扰？
3. 固定中英文各 50% 时，仅改变出现顺序，能否减少遗忘或改善双语 Pareto 折中？

不再研究 v1/v1.5 差异、昇腾扩展效率、350M 容量交互或通用合成数据调度。公开榜单只承担外部英文定位。

## 2. 数据合同

### 2.1 训练数据

| 语言 | 数据集 | 配置/切分 | 冻结 revision | 语言内比例 | 最终物化量 |
|---|---|---|---|---:|---:|
| 英文 | `openbmb/Ultra-FineWeb` | `default/en` | `02c85641e3d19a854be2e09139c25adaa9518063` | 30% | 0.960B Haidass tokens |
| 英文 | `mlfoundations/dclm-baseline-1.0-parquet` | baseline parquet | `817d6752765f6a41261085171dd546b104f60626` | 16% | 0.512B |
| 英文 | `HuggingFaceTB/finemath` | `finemath-4plus` | `e92b25a616738fe95dc186b64dfb19f9c8525594` | 16% | 0.512B |
| 英文 | `openbmb/Ultra-FineWeb-L3` | EN QA | `bc3b1ba986fcaef6871b9790a413b16267c2de0f` | 8% | 0.256B |
| 英文 | `openbmb/Ultra-FineWeb-L3` | EN Multi-Style | 同上 | 10% | 0.320B |
| 英文 | `HuggingFaceTB/cosmopedia` | 8 个公开 config，等配额 | `0ae6ec63f91742bd2d1eaef4f02232c55d719385` | 20% | 0.640B（每类 0.080B） |
| 中文 | `openbmb/Ultra-FineWeb` | `default/zh` | `02c85641e3d19a854be2e09139c25adaa9518063` | 60% | 1.920B |
| 中文 | `openbmb/Ultra-FineWeb-L3` | ZH QA | `bc3b1ba986fcaef6871b9790a413b16267c2de0f` | 15% | 0.480B |
| 中文 | `openbmb/Ultra-FineWeb-L3` | ZH Multi-Style | 同上 | 25% | 0.800B |

这些来源是 Haidass1.5 模型卡列出的五个主要训练仓库。模型卡只披露总量约 400B tokens，未披露源间比例、阶段边界、L3/Cosmopedia 子配置或原始 shard。上表因此是预注册的可复现受控重构，不是对原 400B-token 配方的精确复刻。英文比例覆盖 Cosmopedia 的 `auto_math_text`、`khanacademy`、`openstax`、`stanford`、`stories`、`web_samples_v1`、`web_samples_v2` 和 `wikihow`；L3 同时覆盖中英文 QA 与 Multi-Style。

选择流程必须按以下顺序执行：

1. 对每个仓库/配置独立枚举 Parquet 路径并计算 `SHA256(revision|config|path|2027)`；先下载哈希最小的 shard，过滤后不足该源配额时按相同顺序增加。不得用相邻数据集替代缺失源。
2. 使用发布版 Haidass1.5 tokenizer（必须记录 SHA256）计数。删除少于 200 或多于 8,192 tokens 的文档。
3. 文本仅为哈希执行 NFKC、大小写折叠（拉丁字符）和空白归一化；训练仍保留原文。
4. 先做规范化文本 SHA256 精确去重，再用 13-unit shingle MinHash 去重；一个 unit 为一个 CJK 字符或一个小写拉丁数字串，Jaccard 阈值为 0.8。
5. 先扫描全部评测集污染并删除 direct/ambiguous overlap，再划分 train/dev/test。
6. 只用 Ultra-FineWeb EN/ZH 作内部 dev/test，以 `SHA256("haidass-iclr2027"|normalized_text) mod 10000` 划分：`0..49` 为 dev，`50..99` 为 test，其余为 train。其他来源只进入 train，避免合成数据决定主评测分布。
7. 每个来源内再按独立盐值的 SHA256 排序，严格截取表中配额。每种语言训练合计 3.20B tokens；dev/test 各取 10M tokens。配额不足时失败，不循环补样本。
8. 各来源先独立打包，再按表中固定配额组成英文和中文不可变 sequence stream。相同 seed 的所有条件共享相同语言内且来源内的序列多重集；条件之间只改变两个语言流的交织。

每条数据清单必须保存：上游 repo/revision/config/split、Parquet 路径及文件 SHA256、上游行号、原始文档 ID（如有）、规范化文档 SHA256、MinHash cluster、语言、质量分、原始字节数、token 数、split、packed sequence IDs、许可证来源。

### 2.2 外部 held-out 文本

| 数据集 | 配置 | revision | 数量 | 用途 |
|---|---|---|---:|---|
| `wikimedia/wikipedia` | `20231101.en` | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | 10M tokens | 英文 OOD BPB |
| `wikimedia/wikipedia` | `20231101.zh` | 同上 | 10M tokens | 中文 OOD BPB |

按文档哈希抽样，并与训练池做精确和 MinHash 交叉去重。Wikipedia 只进入 test，不参与超参、比例或课程选择。

### 2.3 下游评测

| 数据集 | revision | 切分/数量 | 指标 | 地位 |
|---|---|---:|---|---|
| Belebele `eng_Latn/zho_Hans` | `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2` | 全部 900 对 | EN/ZH accuracy、both-correct、same-answer、两类不对称错误、正确选项概率质量 | 主要平行评测 |
| XCOPA `en/zh` | `042f78955ba48e6404616762fa6e05e839c3907a` | 全部 test 500 对 | 同上 | 次要平行评测 |
| C-Eval | `617524a00b307ff6f9933702f724131fe12ca7ce` | validation 1,346 | 零样本宏平均 accuracy | 描述性中文附录 |
| CMMLU | `efcc940752ea4a1ea94d2727f11f83858d64fc8e` | labeled test 11,582 | 零样本宏平均 accuracy | 描述性中文附录 |
| Open SLM Leaderboard | `a7783862440048822c4a2316a98999978f2f71c4` | 100M--150M、Base、开放权重、五项齐全 | 榜单公开分数 | 外部英文定位 |

Belebele 另构造四个探索性输入：EN passage/EN question、EN/ZH、ZH/EN、ZH/ZH。它们不参与训练方案选择，也不承担主要结论。

## 3. 模型和训练超参

### 3.1 Proxy-43M

```text
architecture            Qwen3 causal decoder
parameters              43,461,504（实测）
vocab / tied embedding  64,000 / yes
layers                  12
hidden / FFN            384 / 1,024
heads / KV heads        6 / 2
head_dim                64
context                 2,048
rope_theta              100,000
global batch            128 sequences
tokens/update           262,144
updates                 3,815
tokens/run              1,000,079,360
precision               BF16
optimizer               AdamW, beta=(0.9,0.95), eps=1e-8
weight_decay            1e-5
gradient_clip           2.0
peak/min LR             1.5e-3 / 0
LR schedule             1% linear warmup + cosine
dropout                 0
seeds                   2027, 2028
```

每个 seed 只生成一次初始化，所有条件复制同一 initialization checkpoint。每约 10% updates 保存 checkpoint，并记录逐语言累计 tokens。

### 3.2 Target-143M

```text
parameters              143,071,296
layers                  30
hidden / FFN            576 / 1,536
heads / KV heads        9 / 3
head_dim                64
context                 4,096
global batch            128 sequences
tokens/update           524,288
updates                 5,722
tokens/run              2,999,975,936
其余优化器参数          与 Proxy-43M 相同
seeds                   2027, 2028, 2029
```

单张 4090 独立跑一个 run。建议从 `micro_batch=2, grad_accum=64` 起步；若 200-step dry run 证明 `micro_batch=4` 稳定，则统一改为 `4 × 32`。正式条件之间不得改变 global batch 或 LR。

## 4. 正式运行矩阵

### P0：2080 Ti 全来源本地预跑（不进入确证性主表）

- Proxy-43M、context 1,024，每条件 100,663,296 tokens；12 runs 合计 1,207,959,552 tokens。
- 五个上游仓库、16 个语言/子配置流全部进入本地物化池，按第 2.1 节的语言内比例分层抽样。
- 条件：EN-only、ZH-only、IID-50、EN-first、ZH-first、alternating；seeds 2027、2028，每条件两次。
- 验证 finite logits/loss/grad、初始化哈希、条件间语言/来源 sequence multiset 相等、manifest、checkpoint 和曲线生成。
- P0 只用于发现管线故障、估算吞吐和预览效应量；不用于确证性主张或 P1 课程选择。

### P1：4090 proxy 筛选（16 runs）

| ID | 中文比例 | 顺序 | Seeds | Runs |
|---|---:|---|---:|---:|
| `en_only` | 0% | 英文流 | 2027, 2028 | 2 |
| `iid_zh25` | 25% | sequence-level IID | 2027, 2028 | 2 |
| `iid_50` | 50% | sequence-level IID | 2027, 2028 | 2 |
| `iid_zh75` | 75% | sequence-level IID | 2027, 2028 | 2 |
| `zh_only` | 100% | 中文流 | 2027, 2028 | 2 |
| `en_first` | 50% | `EEEEZZZZ` 八个等 token block | 2027, 2028 | 2 |
| `zh_first` | 50% | `ZZZZEEEE` | 2027, 2028 | 2 |
| `alternating` | 50% | `EZEZEZEZ` | 2027, 2028 | 2 |

选课规则：只读取 internal dev 的 EN/ZH BPB 与 forgetting；在三个非 IID 顺序中最小化相对 IID 的等语言平均 BPB，平手时选择最差语言相对退化更小者，再平手按 `alternating > en_first > zh_first` 的预注册顺序。签署选择记录后才允许 P2 下游评测。

### P2：4090 target 确认（15 runs）

| ID | 中文比例 | 顺序 | Seeds | Runs | 用途 |
|---|---:|---|---:|---:|---|
| `en_only` | 0% | 英文流 | 2027--2029 | 3 | 英文单语学习曲线 |
| `zh_only` | 100% | 中文流 | 2027--2029 | 3 | 中文单语学习曲线 |
| `iid_50` | 50% | sequence-level IID | 2027--2029 | 3 | 主要基线 |
| `selected_curriculum` | 50% | P1 冻结方案 | 2027--2029 | 3 | 唯一主要处理 |
| `block_shuffled` | 50% | 每 seed 随机排列 4E+4Z blocks | 2027--2029 | 3 | 顺序控制 |

`block_shuffled` 使用 RNG `10000 + seed`，若排列等于 selected curriculum 则继续采样。所有 50:50 条件在 seed 内使用完全相同的英文和中文 sequence multiset。

## 5. 主要指标和判断门槛

### 5.1 BPB 与分解

```text
BPB = total next-token NLL / (UTF-8 bytes × ln 2)
```

中英文分别报告，不能直接比较 token perplexity。对每种语言：

```text
specialist gap = bilingual(C,q) - monolingual(C)
dilution       = monolingual(q_l*C) - monolingual(C)
transfer term  = bilingual(C,q) - monolingual(q_l*C)
```

`transfer term < 0` 才称正迁移，`> 0` 才称干扰；只有 final specialist gap 不足以判断干扰。

### 5.2 唯一 confirmatory estimand

```text
Delta_curr = 0.5 * sum_lang(
  (BPB_selected - BPB_iid50) / BPB_iid50
)
```

只有同时满足以下条件才写“课程改善双语建模”：

1. `Delta_curr` 的 95% CI 完全低于 0；
2. 每门语言相对 BPB 的 CI 上界不超过 +0.5%；
3. block-shuffled 不复现同等幅度的收益；
4. 三个 target seeds 的主效应方向一致。

否则按结果写成 null、语言 trade-off 或只改善 compression，不修改阈值。

### 5.3 遗忘

```text
F_lang = final BPB - min BPB over registered checkpoints
```

同时报告切换后最大退化和 BPB 曲线下面积，不能只看最终点。

### 5.4 统计

- Primary：seed × common document 的层次配对 bootstrap，10,000 次，seed 2027。
- 平行任务：按 aligned item ID 配对 bootstrap。
- 其余三个 schedule contrast 使用 Holm 校正。
- 每个 seed 单独列值，不能只给合并 CI。
- 失败、OOM、非有限 loss 和恢复运行全部进入运行日志。
- 任一 P2 cell 或逐文档 NLL 缺失时，拒绝生成主要结果。

## 6. 需要保存的输出

每个 run 至少保存：

- `manifest.json`：run ID、代码 commit、容器 digest、GPU/driver/CUDA/PyTorch、模型/Tokenizer/init hash、完整超参；
- `ordered_sequences.sha256` 与每 step 的 batch hash；
- 逐 step：总/英文/中文 seen tokens、LR、loss、grad norm、tokens/s、data wait、峰值显存；
- 十个中间 checkpoint、优化器和 sampler state；
- internal dev/test 与 Wikipedia 的逐文档 `nll_sum/token_count/utf8_bytes`；
- 所有下游任务逐样本选择和各选项 log-likelihood；
- 污染候选与人工 adjudication；
- 失败、重启位置、恢复后首 20 个 batch hash。

文件写入临时路径后必须原子 rename；checkpoint 写完立即计算 SHA256。

## 7. 上集群顺序

1. 冻结 tokenizer、代码、容器和 dataset revision。
2. 只下载候选 shard，生成数据 manifest、去重和污染报告。
3. 运行 P0，修完所有数值与 resume 问题。
4. 在 4090 上用 `iid_50` 做 200-step dry run；记录真实 tokens/s 后用 `tokens / tokens_per_second` 计算预算。
5. 并行运行 P1；不运行任何下游任务。
6. 生成并签署 `selected_curriculum.json`。
7. 展开 P2 的 15 个 run ID；每卡独立跑一个 run。
8. P2 全部完成且 manifest 验证通过后，统一运行 downstream 与统计脚本。
9. 替换论文占位表图；若 claim gate 不通过，删除对应主张而不是改指标。

## 8. Launch blockers

以下任一项未完成则不上 P2：

- 训练数据实际不足每语言 3.20B unique tokens；
- 许可证、revision 或 shard SHA256 缺失；
- 跨 split MinHash cluster 泄漏；
- 评测污染未审理；
- 相同 seed 的初始化或 per-language stream hash 不一致；
- resume 后样本次序改变；
- 200-step dry run 出现非有限 loss/grad、GPU 利用率低于 90% 或 data wait 超过 5%；
- P1 选课记录查看过 downstream test；
- P2 少于 15 个预注册 run IDs。

## 9. 务实资源估算

不要预填 4090 速度。Dry run 后按下式计算：

```text
hours_per_run = seen_tokens / measured_nonpadding_tokens_per_second / 3600
wall_hours    = sum(hours_per_run) / number_of_independent_GPUs × 1.15
```

`1.15` 为保存、评测、调度和故障余量。模型不做多卡切分；每张 4090 独立跑一个条件。两张卡是最低务实配置，四张卡主要缩短墙钟时间。
