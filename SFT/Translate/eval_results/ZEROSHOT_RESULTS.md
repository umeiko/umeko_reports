# 翻译基线成绩总表（FLORES-200 dev）

- **评测集**：FLORES-200 dev，997 句，双向（en→zh / zh→en）
- **指标**：sacrebleu corpus BLEU（zh 用 `tokenize=zh`，en 用 `13a`）+ chrF++（word_order=2）
- **生成**：NPU 910B3，贪心解码，max_new=512，bf16
- **脚本**：`sft_eval/eval_translate.py`（LLM，chat 模板）+ `sft_eval/eval_translate_seq2seq.py`（NLLB/M2M/OPUS-MT，forced-BOS）+ `compute_metrics.py`（指标）
- LLM 用与 pilot SFT 训练逐字一致的指令模板；HY-MT1.5 用其官方模板（见下）
- 日期：2026-09-04（Haidass1.5-143M-SFT pilot 行补测于 2026-09-07；8M-mix 行补测于 2026-09-09；4M-mix-v2 / 8M-mix-v2 行补测于 2026-09-10）
- **我们的 SFT pilot 配方**：97.7 万对清洗后中英平行语料（双向展开 195 万条，packed seq2048），
  8×910B3，GBS=256，lr 3e-5 cosine→3e-6，2 epochs=15270 步，终 loss 1.886；
  脚本 `pilot_sft/tune_haidass_translate_pilot.sh`，HF 权重 `pilot_sft/hf_ckpt/`（=iter_0015270）
- **8M-mix 配方**（16 卡机 110.120.0.3 训，16×910C）：783.7 万对清洗后翻译语料双向展开 1567 万条
  **+ STEP_FUN 142 万条（未清洗，占 8.3%）**，knapsack 真打包 seq2048（129 万条 packed），
  GBS=256，lr 3e-5，2 epochs=10080 步，终 loss 1.633，95 分钟；HF 权重 `pilot_sft_8m/hf_ckpt_final/`（=iter_0010080）。
  注意：该模型训练数据是**无 think 块的裸 im_start 格式**，评测须用 `--plain-prompt`（见"读表要点"第 8 条）
- **4M-mix-v2 配方**（本机 8×910B3）：391.6 万对清洗后翻译语料双向展开 783 万条
  **+ 清洗后 STEP_FUN 78,913 条（9.1% token 占比）**，MindSpeed 官方 `preprocess_data.py --pack --neat-pack`
  打包（锯齿注意力掩码，文档间隔离），341,321 条满 2048 packed 序列，
  GBS=256，MBS=1，lr 3e-5 cosine，2 epochs=2666 步，训练加 `--reset-attention-mask --neat-pack`；
  脚本 `pilot_sft_4m/tune_haidass_translate_4m.sh`，HF 权重 `pilot_sft_4m/hf_ckpt/`（=iter_0002666）。
  与 8M-mix(v1) 的差异只有两处：**STEP_FUN 清洗过** + **打包带文档间注意力隔离**，模板走官方 qwen3 模板（带 think 块）
- **8M-mix-v2 配方**（16 卡机 110.120.0.3 重训，16×910C）：783.7 万对清洗后翻译语料双向展开
  **+ 清洗后 STEP_FUN 157,593 条（9.1% token 占比）**，与 4M-mix-v2 同一批官方 `--pack --neat-pack`
  打包产物（683,227 条满 2048 packed 序列，~28 亿 token × 2 epochs，neat-pack padding 仅 0.37%），
  GBS=256，lr 3e-5 cosine，2 epochs=5337 步（1,366,272 行，99.99% 完整 2 轮），终 loss 1.788，
  全程无 NaN；HF 权重 `pilot_sft_8m/hf_ckpt_v2/`（=iter_0005337，本机转换）。
  即 4M-mix-v2 的 8M 数据版，同样带文档间注意力隔离，标准 chat template 评测（无 --plain-prompt）
- **1M-mix-v2 配方**（本机 8×910B3）：pilot 同款 97.7 万对双向展开 195 万条
  **+ 清洗后 STEP_FUN 19,644 条（9.1% token 占比，同 seed 选取，是 4M/8M 子集的前缀）**，
  官方 `--pack --neat-pack` 打包出 93,980 条满 2048 序列，GBS=256，MBS=1，lr 3e-5 cosine，
  2 epochs=734 步，终 loss 2.167；脚本 `pilot_sft_1m_mix/tune_haidass_translate_1m_mix.sh`，
  HF 权重 `pilot_sft_1m_mix/hf_ckpt/`（=iter_0000734）。
  ⚠️ 注意：与 pilot 的差别**不止 STEP_FUN 一个变量**——打包后优化步数只有 734 步
  （pilot 不打包有 15270 步，同 token 量差 21 倍），该行的偏低成绩不能单独归因于混通用数据
  （见"读表要点"第 12 条）
- **1M-mix-nopack 配方**（本机 8×910B3，**干净对照**）：与 pilot **唯一差别是混入清洗后
  STEP_FUN 19,644 条**（9.1% token 占比）——数据同样不打包逐条喂（1,973,982 条），
  MBS=16，`--no-pad-to-seq-lengths`，GBS=256，lr 3e-5 cosine，2 epochs=15421 步，
  优化步数与 pilot（15270）对齐；脚本 `pilot_sft_1m_mix_nopack/tune_1m_mix_nopack.sh`，
  HF 权重 `pilot_sft_1m_mix_nopack/hf_ckpt/`（=iter_0015421）

## 总表（按 en→zh BLEU 排序）

| 模型 | 类型 | 参数量 | en→zh BLEU | en→zh chrF++ | zh→en BLEU | zh→en chrF++ |
|---|---|---:|---:|---:|---:|---:|
| **HY-MT1.5-1.8B** | 专用翻译 LLM | 1.8B | **44.65** | **30.98** | **27.68** | **57.96** |
| OPUS-MT en-zh / zh-en | 专用 seq2seq | 78M×2 | 30.88 | 21.80 | 22.99 | 51.03 |
| Qwen3-0.6B | 通用指令 LLM | 0.6B | 30.94 | 21.10 | 20.21 | 48.62 |
| Qwen2.5-0.5B-Instruct | 通用指令 LLM | 0.5B | 28.96 | 19.65 | 18.09 | 45.85 |
| M2M-100-418M | 专用 seq2seq | 418M | 28.04 | 20.53 | 20.58 | 48.79 |
| NLLB-200-distilled-600M | 专用 seq2seq | 600M | 22.44 | 16.74 | 25.71 | 52.28 |
| MiniMind2 | 小聊天 LLM | 104M | 0.14 | 2.10 | 0.85 | 12.00 |
| minimind-3 | 小聊天 LLM | ~57M | 0.03 | 0.65 | 0.20 | 5.34 |
| MiniMind2-Small | 小聊天 LLM | 26M | 0.02 | 0.62 | 0.28 | 4.90 |
| MiniMind2-MoE | 小聊天 LLM | 39M 激活 | 0.01 | 0.39 | 0.03 | 3.96 |
| **Haidass1.5-143M-SFT 8M-mix-v2（8M翻译+清洗STEP_FUN 9.1%, 官方pack隔离, 2ep）** | SFT LLM | 0.14B | 23.64 | 17.21 | 14.35 | 40.13 |
| **Haidass1.5-143M-SFT 4M-mix-v2（4M翻译+清洗STEP_FUN 9.1%, 官方pack隔离, 2ep）** | SFT LLM | 0.14B | 20.10 | 15.98 | 13.88 | 38.79 |
| **Haidass1.5-143M-SFT 8M-mix（8M翻译+未清洗STEP_FUN, 2ep）** | SFT LLM | 0.14B | 17.18 | 15.22 | 7.34 | 31.92 |
| **Haidass1.5-143M-SFT（我们的翻译 SFT, pilot 2ep）** | SFT LLM | 0.14B | 16.65 | 13.72 | 14.49 | 39.75 |
| **Haidass1.5-143M-SFT 1M-mix-nopack（pilot+清洗STEP_FUN 9.1%, 不打包, 2ep）** | SFT LLM | 0.14B | 16.03 | 13.24 | 14.09 | 39.62 |
| **Haidass1.5-143M-SFT 1M-mix-v2（1M翻译+清洗STEP_FUN 9.1%, 官方pack隔离, 2ep）** ⚠️优化步数只有 pilot 的 1/21，见配方注 | SFT LLM | 0.14B | 14.86 | 13.07 | 8.79 | 33.65 |
| Haidass-sft-ckpt168000（通用 SFT 版） | 通用 SFT LLM | 136M | 10.12 | 10.32 | 2.85 | 11.78 |

预测文件：`sft_eval/pred_<模型>_flores_dev.jsonl`（含 src/ref/hyp，可人工抽查）。

## 指标定义

**BLEU**（Papineni et al., 2002）：看机器译文和参考译文有多少 **n-gram 重合**，
算的是**精确率**——译文里出现的 1~4 元词组，有多少比例能在参考译文里找到
（取几何平均），再乘上"短小惩罚"（brevity penalty：译文比参考短就扣分，防止只翻半句取巧）。
0~100 分，越高越好。本报告用 sacrebleu 标准实现：英文目标 `tokenize=13a`（默认分词），
中文目标 `tokenize=zh`（按中文字符切分）。BLEU 是 MT 领域几十年来的默认尺子，
审稿人必看，但对中文这类无空格语言的分词较敏感。

**chrF++**（Popović, 2015/2017）：不看词，看**字符 n-gram**（1~6 元）的 F2 分数
（精确率和召回率的调和平均，且更偏召回，β=2），"++" 表示再叠上**词 2-gram** 的统计
（本报告 `word_order=2`）。完全不需要分词，对中文、形态丰富的语言更公平；
和 BLEU 的相关性高但视角互补：BLEU 问"词组用得对不对"，chrF++ 问"字面上像不像"。
也是 WMT 官方近年主推的指标之一。

两个都看的原因：BLEU 对"翻了但措辞不同"罚得狠，chrF++ 更宽容；
两者同升才是真提升。注意 chrF++ 在中文/英文目标上的绝对值量纲不同
（中文按字符打分天然偏低），只能同方向内横比。

陷阱举例（haidass-sft-ckpt168000）：它的 zh→en chrF++（11.78）反而高于
en→zh（10.32），与 BLEU（2.85 vs 10.12）看似矛盾。原因：chrF 只数字符重叠，
破英文对英文参考能蹭到大量 `the/ing/tion` 类字符片段（同文字垃圾虚高），
而 en→zh 输出里 43% 是拉丁字母垃圾、对中文参考零贡献，外加输出冗长拖死精度。
即 chrF 对"同文字的垃圾"宽容、对"跨文字的垃圾"零容忍——两指标不矛盾，
合起来说明该模型两个方向都不会翻译。

## 读表要点

1. **HY-MT1.5-1.8B 是天花板**（腾讯混元 2025-12 开源，WMT25 冠军 7B 的小弟）：
   双向都比其他模型高一大截。它用**官方翻译模板**（`将以下文本翻译为{目标语言}…`）
   + 官方 chat 格式（`<｜hy_User｜>…<｜hy_Assistant｜>`）测的；
   早前用我们通用指令测的 zh→en 只有 6.28，是它把中文当中文处理导致的口径事故，已作废。
2. **Qwen3-0.6B zero-shot 已打平老牌科班 NLLB/M2M**——通用指令模型的时代确实来了。
3. **OPUS-MT 78M 都有 30.88**：中英方向"数据对口"比"模型大"重要——143M 的故事成立。
4. **MiniMind 系全军覆没**（BLEU<1）：做过聊天 SFT 但没训过翻译的小模型，
   zero-shot 翻译约等于零（复读原文/循环/跑偏）。
5. chrF++ 中英文侧量纲不同（中文按字符计分），只在同方向内横比。
6. **通用 SFT 版 Haidass（ckpt168000）只有 10.12 / 2.85**：通用指令微调给的翻译能力
   很有限，而且严重偏科——en→zh 还能译，zh→en 基本照抄中文原文（人工抽查确认，
   换指令措辞也没用）。
7. **翻译 SFT pilot 版（97.7 万对清洗后语料 × 2 epochs，8×910B3 训 2.4h）验证成立**：
   en→zh 16.65（+6.5）、zh→en 14.49（**+11.6，5 倍**），两个方向**照抄源文归零**
   （通用版分别是 159/414 条照抄），输出已是连贯译文。但仍远落后 Qwen3-0.6B
   （30.94/20.21）——预料之中：Qwen3 的底座见过海量中英预训练语料，我们底座 143M
   只靠 pilot 语料。en→zh 的 chrF++（13.72）偏低与输出偏短/措辞保守有关。
   抽查可见典型错误：专有名词编造（"killing 39 people"）、繁体字泄漏（"運動車"），
   说明语料里 zh 侧的规范性和实体一致性还有清洗/扩充空间（4M/8M 阶段重点）。
8. **8M-mix（16×910C 训完，2026-09-09 评）结果分化**：en→zh 17.18（比 pilot +0.5）略升，
   **zh→en 7.34 大幅倒退（pilot 14.49）**，chrF++ 同步倒退（31.92 vs 39.75）。抽查 zh→en
   输出可见退化循环（"country/regions of the country/regions…"、数字串死循环 "1000000…"）。
   嫌疑按序：① STEP_FUN 142 万条**未清洗**直接混入（本地清洗发现其中 60.4 万条跨文件重复、
   19.6 万条超长、2007 条撞评测集——重复数据 2 epochs 等于被看了 5 遍）；② knapsack 打包
   不重置文档间注意力（一个 2048 序列里多条样本互相可见）；③ 8.3% 的英文主导通用数据
   改变了输出分布。**结论：数据清洗不是可选项**，4M 本机复训（清洗后 STEP_FUN 子集 9.1%）
   将给出干净对照。
9. **模板口径事故记录**：8M-mix 首测 en→zh 只有 2.89——不是模型差，是评测脚本走了
   chat template 自动塞空 `<think></think>` 块，而该模型训练数据（远端 fast_preprocess_v2
   打包）是无 think 块的裸格式，模型看到未知块直接输出 EOS。用 `--plain-prompt` 复测
   才是真实水平。pilot/haidass-sft 行走的是 MindSpeed qwen3 模板（带 think 块），不受影响。
   **教训：评测 prompt 必须和训练格式逐字节一致。**
10. **4M-mix-v2（2026-09-10 评）：v1 事故的两项修复验证成立**。与 8M-mix(v1) 唯一差异是
    ①STEP_FUN 清洗后混入 ②官方 pack 文档间注意力隔离，结果 **zh→en 从 7.34 恢复到 13.88**
    （基本回到 pilot 的 14.49 水平），**en→zh 20.10 创我们的新高**（pilot 16.65 +3.45，8M v1 17.18 +2.92），
    chrF++ 四象限同步健康（15.98/38.79）。结论坐实：**8M v1 的 zh→en 崩塌是数据未清洗+打包
    无注意力隔离导致的，不是"混通用数据"或"scale 上去"本身有害**。zh→en 相对 pilot 仍 -0.61，
    是否由通用数据轻微稀释或 MBS=1 打包模式导致，留给 8M v2（远端重训）对照确认。
    抽查：高分句质量明显好于 pilot（见文末样例区），典型错误仍是实体张冠李戴
    （"斯洛伐克"→Slovenia）。
11. **8M-mix-v2（16×910C 重训，2026-09-10 评）：v2 配方在 8M 规模全面成立**。
    en→zh **23.64（我们的新高）**：pilot 16.65 → 4M v2 20.10 → 8M v2 23.64，
    随数据量单调爬升，chrF++ 同步（13.72→15.98→17.21）；zh→en **14.35/40.13**，
    彻底洗掉 v1 的 7.34 崩塌，回到 pilot 水平（14.49/39.75），第 10 条留下的
    "zh→en 轻微稀释"疑点在 8M 规模消失（4M v2 的 -0.61 属小数据噪声，非通用数据有害）。
    **但 zh→en BLEU 三个规模全部停在 14.4±0.6 的平台期**——更多同分布平行语料
    只推 en→zh，推不动 zh→en。zh→en 的下一步杠杆大概率不是"再堆数据"，而是
    英文侧数据质量/多样性（或更大底座）。与外部对照：en→zh 23.64 已越过
    NLLB-600M（22.44），距 Qwen3-0.6B（30.94）和 OPUS-MT（30.88）仍有明确差距。
12. **1M-mix-v2 的偏低成绩是"优化步数"变量污染，不能归因于 STEP_FUN**：该行与 pilot 的差别
    除了混 9.1% 通用数据，还有打包方式——官方 pack 把 195 万条样本压成 9.4 万条满 2048 序列，
    优化步数从 pilot 的 15270 步缩到 **734 步（同 token 量少 21 倍梯度更新）**，cosine 日程
    也等比缩短，模型明显欠训（终 loss 2.167 vs pilot 1.886；输出连贯无崩溃，只是错得多）。
    4M/8M 规模不受此影响是因为数据量给了足够步数（2666/5337 步）。
    本条也是对规模梯度解读的警告：1M/4M/8M 三档 v2 的 en→zh（14.86→20.10→23.64）
    同时混着数据量和优化步数两个变量。
13. **干净对照（1M-mix-nopack，2026-09-11 评）：1M 规模掺 9.1% 清洗后 STEP_FUN ≈ 无害略损**。
    与 pilot 唯一差别是混入通用数据（不打包、15421 步对齐），结果 en→zh 16.03（pilot -0.62）、
    zh→en 14.09（-0.40）、chrF++ 几乎持平（13.24/39.62 vs 13.72/39.75）——**轻微稀释，无崩塌**。
    与 8M v1 的 zh→en 7.34 崩塌对比，最终实锤：v1 事故的元凶是"未清洗 + 无注意力隔离"，
    通用数据本身在 9.1% 配比下对翻译能力至多造成 <1 BLEU 的稀释。代价结论：
    小规模纯翻译目标下可不掺；若希望模型兼顾通用对话，1 BLEU 以内的代价可接受。

## 未测清单（及原因）

- BananaMindBench 前排 8 个（BananaMind-2-Pro、SmolLM/2-135M 等）：**base 模型无指令微调**，
  zero-shot 不测；价值在后续同配方 SFT 对照。
- Gemma-3-1B-it / Llama-3.2-1B-Instruct：HF 门控，需 token。
- HY-MT1.5-7B / Hunyuan-MT-7B / Seed-X-7B：7B 档，超出"小模型"对比范围。

## 我们的 SFT pilot 样例抽查（iter_0015270）

zh→en 高分例：

> 源：巴西是世界上最大的罗马天主教国家，而该国的罗马天主教会向来反对同性婚姻合法化。
> 输出：Brazil is the largest Roman Catholic country in the world, and the Roman Catholic Church has long opposed the legalization of same-sex marriage.
> 参考：…on Earth…has consistently opposed…（句 chrF++ 77.7）

> 源：在这种共振中，磁场和无线电波导致原子发出微弱的无线电信号。
> 输出：In this resonance, the magnetic field and radio waves cause atoms to emit weak radio signals.（句 chrF++ 78.7）

en→zh 高分例：

> 源：…the most known person of this culture is Homer, the legendary blind poet…
> 输出：…或许最知名的人是希罗多德，一个传说中的盲人诗人…《伊利亚特》和《奥德赛》。（流畅但实体错：荷马→希罗多德）

> 源：…rules of origin and tariff concessions…start trading on July 1, 2020
> 输出：…尚未就原产地和关税协定达成一致…2020年7月1日将开始交易。（术语准确）

典型错误模式：实体编造（"killing 39 people"）、繁体字泄漏（"運動車"）、个别专名张冠李戴。
完整预测：`pred_haidass15-143M-translate-sft_flores_dev.jsonl`。

## 4M-mix-v2 样例抽查（iter_0002666）

en→zh 高分例：

> 源：The cause of death was announced as intrahepatic bile duct cancer.
> 输出：死因是肝内胆管癌。（句 chrF++ 61.8，术语准确、无冗余）

> 源：Italy's main goals were African countries. To capture those countries, they would need to have a troop launching pad…
> 输出：意大利的主要目标是非洲国家。为了夺取这些国家，他们需要有一个基地，以便士兵可以穿越地中海，入侵非洲。（句 chrF++ 61.2，长句结构完整）

> 源：Jerusalem has many historic, archeological and cultural sites, along with vibrant and crowded shopping centers, cafés…
> 输出：耶路撒冷有许多历史、考古和文化遗址，还有充满活力的购物中心、咖啡馆和餐厅。（句 chrF++ 59.1）

zh→en 高分例：

> 源：光子甚至比构成原子的物质还要小！
> 输出：Even photons are smaller than the stuff that makes up atoms!（句 chrF++ 84.1，仅句首词序小差异）

> 源：索福克勒斯（Sophocles）和阿里斯托芬（Aristophanes）仍然是受欢迎的剧作家…
> 输出：Sophocles and Aristophanes are still popular authors, and their plays are considered to be one of the greatest works in the world of literature.（句 chrF++ 77.0，"剧作家"译成 authors 稍泛化）

典型错误仍是实体张冠李戴："拉脱维亚和斯洛伐克都推迟了加入 ACTA 的进程" →
"Latvia and **Slovenia** have also delayed…"（Slovakia→Slovenia）。
相比 pilot 抽查，4M v2 高分句更干净（无繁体字泄漏、无数字编造），与 BLEU 提升一致。
完整预测：`pred_haidass-4M-v2mix_flores_dev.jsonl`。

## 8M-mix-v2 样例抽查（iter_0005337）

en→zh 高分例：

> 源：They are listed on the UNESCO World Heritage List.
> 输出：它们被列入联合国教科文组织世界遗产名录。（句 chrF++ 69.4，机构名全称准确）

> 源：The cause of death was announced as intrahepatic bile duct cancer.
> 输出：死因是肝内胆管癌。（句 chrF++ 61.8，与 4M v2 同样的术语命中）

zh→en 高分例：

> 源：土卫二 (Enceladus) 是太阳系中反射能力最强的物体，能反射约90％照射到它表面的太阳光。
> 输出：Enceladus is the most reflective object in the solar system, reflecting about 90% of the solar light that reaches it.（句 chrF++ 80.2，括号内专名正确取用）

> 源：这些代理人负责根据《巴基斯坦宪法》第 247 条提供政府和司法服务。
> 输出：These agents are responsible for providing government and judicial services under the Pakistan Constitution.（句 chrF++ 82.3；瑕疵：漏译"第 247 条"，数字类细节仍是弱项）

完整预测：`pred_haidass-8M-v2mix_flores_dev.jsonl`。
