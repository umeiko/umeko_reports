# 在 NPU 集群上复现 43M 双语训练实验 · 小白完全手册

> 这份文档假设你**什么都不知道**：不知道什么是 token、没听过 Megatron、没进过 docker 容器。
> 每一步都会先讲道理，再给可以直接复制粘贴的命令。
> 所有路径都是真实的，命令在我们这台机器上实测跑通过。

---

## 〇、一句话说明白我们在干嘛

我们有一张"考题"：一个 **4300 万个参数**的小语言模型（43M），用 **1 亿个中英 token** 的数据，
按 **6 种不同的语言顺序**各训练 2 次（共 12 次），看"先中文后英文"还是"中英混着来"对最终效果的影响。

之前这个实验在一块 2080Ti 显卡上跑过一遍（叫"初探"）。
现在我们把它搬到 **华为昇腾 NPU 集群**上，用 **MindSpeed 框架**复现——
复现的意思就是：**同样的数据、同样的顺序、同样的模型、同样的超参数，换一个框架和硬件，看结论还成不成立。**

---

## 一、先认识这些词（看到不懂的回来查）

| 词 | 大白话解释 |
|---|---|
| **token** | 模型眼里的最小单位。可以粗略理解为"字/词碎片"。中文一个字大约 1~1.5 个 token，英文一个词大约 1~2 个。 |
| **参数 / 43M** | 模型内部可调节的数字的个数。43M = 4346 万个。这些数字一开始是随机的，训练就是把它们慢慢调对。 |
| **训练一步（update/step）** | 给模型看一小批文本 → 模型猜下一个 token → 算猜错多少（loss）→ 把参数往"少错"的方向拧一点点。我们一共拧 **3072 次**。 |
| **global batch = 32** | 每一步给模型看 **32 条**文本，每条固定 **1024 个 token**。所以一步看 32×1024 = 32768 个 token。 |
| **micro batch** | 显卡一次性能吃下几条文本。初探的 2080Ti 显存小，一次只敢吃 4 条，吃 8 次凑够 32 条（这叫"梯度累加"）。NPU 显存大，我们一次直接吃 32 条。**凑数方式不同，但每一步学的内容完全一样。** |
| **物化（materialize）** | 这个词你之前问过。**物化 = 把"配方"变成"实物"。** 原始数据集散落在 HuggingFace 上的不同数据集里，有配额、有筛选规则、有 tokenizer 切分规则——这些合起来只是一份"配方"。物化就是照着配方把数据真正下载、筛选、切 token、补齐到规定长度，落成硬盘上的 `.npy` 文件。训练时读这些文件就行，不用每次重新下载和切分。 |
| **seed（种子）** | 随机数的起点。同一个 seed，随机打乱的结果就完全一样。我们用 2027 和 2028 两个种子各跑一遍，确认结论不是"碰巧"。 |
| **condition（条件）** | 数据的 6 种语言排列方式：`en_only`（纯英文）、`zh_only`（纯中文）、`iid_50`（中英一半一半随机混合）、`en_first`（先英后中）、`zh_first`（先中后英）、`alternating`（中英交替八块）。 |
| **BPB（bits per byte）** | 我们的"考试成绩"。模型对没见过的文本平均每个字节要花多少"比特"来编码，**越低越好**。用它而不用准确率，是因为中英文字符长度差异大，按字节算才公平。 |
| **遗忘（forgetting）** | 训练中途某语言曾达到的最好 BPB，到最后退步了多少。退步越大 = 遗忘越严重。 |
| **NPU** | 华为的 AI 加速卡，地位相当于 NVIDIA 的 GPU。我们这台服务器有 8 张，我们用其中第 6、7 两张。 |
| **CANN** | 昇腾的"显卡驱动+运行库"，相当于 NVIDIA 的 CUDA。**装错版本整个机器的训练都会挂，所以我们坚决不动它**（详见"坑"一节）。 |
| **MindSpeed / Megatron** | 华为的大模型训练框架，里面套着 NVIDIA 开源的 Megatron 训练骨架。我们写的训练脚本就是插在它的标准流程里跑的。 |
| **docker 容器** | 一个"套娃"系统：和宿主机共享硬件，但软件环境独立。同事在宿主机上有自己的容器，我们自己起一个，互不干扰。 |
| **fp16 / loss scale** | 训练时用 16 位小数（省显存、快），但 16 位小数能表示的范围小，梯度容易"小到变成 0"，所以先把 loss 放大 65536 倍再算梯度，算完再缩回去。这个放大倍数叫 loss scale，框架会自动调整它。 |

---

## 二、整个实验的流程（先看全景，再看细节）

```
 HuggingFace 原始数据
        │
        ▼  ① 下载+筛选+切token+补齐长度（物化）
 数据池：en/train_tokens.npy  zh/train_tokens.npy
 （英文 102400 条 + 中文 102400 条，每条 1024 token，存在硬盘上）
        │
        ▼  ② 按 6 种顺序 × 2 个种子生成"点菜单"
 顺序表：order_<条件>_s<种子>.npz（12 份）
  ★ 校验：这 12 份顺序表和 2080Ti 初探的逐字节一致（sha256 全部对上）
        │
        ▼  ③ 生成随机初始模型（每个种子一份）
 初始权重：init_hf/init-s2027, init-s2028  → 转换成 MindSpeed 格式 init_mg/
        │
        ▼  ④ 训练 12 次（两张卡，每张卡同时跑 2 个，共 3 波）
 每次：3072 步 × 32 条/步，约 25 分钟
 每 192 步存一次档（checkpoint）：step 192, 384, ..., 3072
        │
        ▼  ⑤ 离线评测：每个存档 + 第 0 步的初始模型
 在没见过的验证集（dev，256 条/语言）上算 BPB → 17 个点连成曲线
 最后一步在测试集（test，512 条/语言）上算最终成绩
        │
        ▼  ⑥ 汇总对比：和 2080Ti 初探的最终成绩并排列表
```

为什么要存档 17 次而不是只看结果？因为"遗忘"是发生在**过程中**的——
比如 `zh_first` 条件下，模型前半程中文学得很好，后半程猛学英文后中文就退步了。
只有沿途存档、逐点考试，才能画出这条"先进步后退步"的曲线。

---

## 三、一次训练步里到底发生了什么（有耐心版）

拿 `iid_50` 条件、第 100 步举例：

1. **点菜**：按顺序表取第 3200~3231 号共 32 条文本（比如 16 条英文 16 条中文，顺序表早就定好了）。
2. **喂模型**：每条 1024 个 token，模型拿着前 1023 个猜下一个，一共猜 1023 个位置。
3. **算 loss**：每个位置算一个"错得多离谱"的分数（交叉熵），32×1023 个位置取平均，得到这一步的 loss。
4. **反向传播**：算出"每个参数该往哪个方向拧、拧多少"（梯度）。
5. **拧参数前先刹车**：如果所有梯度加起来的总长度（grad norm）超过 2.0，就整体等比缩小——防止某一步拧太猛把车拧翻。
6. **拧参数（AdamW）**：一种成熟的拧法，带惯性、带自适应力度，还会让所有参数轻微地往 0 收（weight decay = 1e-5，防止参数膨胀）。
7. **学习率先热身后退坡**：前 31 步"拧的力度"（学习率）从 0 线性涨到 1.5e-3，之后按余弦曲线慢慢降回 0。热身是为了别让一开始还是随机数的模型被大力拧坏。

3072 步之后，学习率降到 0，训练结束，存最终档。

---

## 四、目录和文件地图（东西都在哪）

```
/mnt/models/CODE/mzy/
├── Haidass1.5-143M/                      ← tokenizer 用的（分词器，把文本切成 token）
├── umeko_reports/
│   ├── scripts/train_p0.py               ← 初探（2080Ti）的原始训练脚本，复现的"标准答案"
│   ├── data/materialized_100m_all_sources/  ← ① 物化好的数据池（en/ zh/ 各 6 个 .npy）
│   └── reports/p0_scaled_100m_all_sources_2080ti*  ← 初探的报告和对照表
├── bilingual_npu_cluster/
│   ├── data/raw_shards/                  ← 从 HuggingFace 下载切好的训练原料（物化的上游）
│   ├── data/eval/                        ← 下游评测题（XCOPA 等，本次复现暂没用到）
│   ├── README.md / configs/ / manifests/ ← 第一阶段数据准备的说明和清单
│   └── training/                         ← ★ 本次复现的全部代码和产物
│       ├── container_env.sh              ← 容器内一键配好环境变量（不改任何系统文件）
│       ├── make_order.py                 ← ② 生成 12 份顺序表 + sha256 校验
│       ├── order/                        ← 顺序表和校验结果（manifest.json 里全是 MATCH）
│       ├── gen_init.py                   ← ③ 生成随机初始模型
│       ├── init_hf/                      ← 初始模型（HF 格式）
│       ├── init_mg/                      ← 初始模型（MindSpeed/Megatron 格式）
│       ├── pretrain_bilingual_p0.py      ← ★ 训练主程序（插进 MindSpeed 流程）
│       ├── run_one.sh                    ← 跑一个 run（容器内用）
│       ├── run_wave.sh                   ← 一张卡跑 6 个 run（2 并发 × 3 波）
│       ├── eval_bpb.py                   ← 给一个模型存档算 BPB（逐字移植初探的评测）
│       ├── eval_run.sh                   ← 一个 run 的全部存档逐个转格式+评测
│       ├── compare_results.py            ← ⑥ 汇总 12 个 run，和初探并排对比
│       └── runs/<条件>-s<种子>/          ← 每个 run 的日志、存档、评测结果
└── TRAINING_REPRO_BEGINNER_GUIDE_CN.md   ← 本文档
```

MindSpeed-LLM 框架本体在 `/mnt/models/CODE/MindSpeed-LLM-v2.3.0`（只读使用，一个字节都没改过）。

---

## 五、完整命令清单（按顺序复制粘贴）

### 0. 大前提：容器

训练在名为 `mzy-mindspeed` 的 docker 容器里进行（用和同事 verl 项目相同的镜像，
只挂载了第 6、7 两张 NPU 和 /mnt 目录，**没有装、没有改任何 CANN/驱动**）。
容器已经起好了。下面所有命令，凡是以 `docker exec` 开头的，都是在**宿主机**终端里敲。

### 1. 物化数据池（已完成，重跑才需要）

```bash
# 宿主机上跑，用宿主机自己的 python 环境（需先 export HF_ENDPOINT=https://hf-mirror.com）
/mnt/models/CODE/mzy/bilingual_npu_cluster/.venv/bin/python \
    /mnt/models/CODE/mzy/umeko_reports/scripts/materialize_data.py \
    --config /mnt/models/CODE/mzy/umeko_reports/configs/p0_scaled_100m.json
# 产物：umeko_reports/data/materialized_100m_all_sources/{en,zh}/*.npy
```

### 2. 生成 12 份顺序表（已完成）

```bash
/mnt/models/CODE/mzy/bilingual_npu_cluster/.venv/bin/python \
    /mnt/models/CODE/mzy/bilingual_npu_cluster/training/make_order.py
# 会逐行打印 MATCH/MISMATCH，必须 12 个全 MATCH（说明数据池和初探逐比特一致）
```

### 3. 生成初始模型（已完成）

```bash
/mnt/models/CODE/mzy/bilingual_npu_cluster/.venv/bin/python \
    /mnt/models/CODE/mzy/bilingual_npu_cluster/training/gen_init.py
# 产物：training/init_hf/init-s2027 和 init-s2028
```

### 4. 初始模型转成 MindSpeed 格式（已完成）

```bash
docker exec mzy-mindspeed bash -c '
source /mnt/models/CODE/mzy/bilingual_npu_cluster/training/container_env.sh
cd /mnt/models/CODE/MindSpeed-LLM-v2.3.0
for S in 2027 2028; do
  $PY convert_ckpt_v2.py \
    --load-model-type hf --save-model-type mg \
    --load-dir /mnt/models/CODE/mzy/bilingual_npu_cluster/training/init_hf/init-s$S \
    --save-dir /mnt/models/CODE/mzy/bilingual_npu_cluster/training/init_mg/init-s$S \
    --model-type-hf qwen3 \
    --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1
done'
```

### 5. 冒烟测试（先小跑 64 步验证一切正常）

```bash
docker exec -e RUN_NAME=smoke-test mzy-mindspeed \
    bash /mnt/models/CODE/mzy/bilingual_npu_cluster/training/run_one.sh iid_50 2027 0 29601 64
# 参数依次：条件 种子 用第几张卡(0或1) 端口号 跑几步
# 看日志：loss 应该从 11.1 左右开始明显下降
```

### 6. 正式开跑：两张卡，每卡 6 个 run

```bash
# 卡 0 跑种子 2027 的 6 个条件（后台，约 75 分钟训练 + 25 分钟评测）
docker exec mzy-mindspeed bash /mnt/models/CODE/mzy/bilingual_npu_cluster/training/run_wave.sh 0 2027 29700

# 卡 1 跑种子 2028 的 6 个条件
docker exec mzy-mindspeed bash /mnt/models/CODE/mzy/bilingual_npu_cluster/training/run_wave.sh 1 2028 29710
```

跑的过程中随时看某个 run 的进度：

```bash
tail -5 /mnt/models/CODE/mzy/bilingual_npu_cluster/training/runs/iid_50-s2027/train.log
```

### 7. 汇总对比（全部跑完后）

```bash
/mnt/models/CODE/mzy/bilingual_npu_cluster/.venv/bin/python \
    /mnt/models/CODE/mzy/bilingual_npu_cluster/training/compare_results.py
# 产物：training/results/comparison.md（和 2080Ti 初探并排的表格）
```

---

## 六、踩过的坑（每一个都是真金白银的时间换来的）

按"如果你自己重来一遍，会按什么顺序撞上它们"排序：

### 坑 1：CANN 千万别动
这台机器上同事的环境（王浩的 verl 容器）依赖当前版本的 CANN 和驱动。
**不要 `pip install` 任何 ascend 相关包、不要升级降级 CANN、不要改 /usr/local/Ascend 下的任何东西。**
正确做法：用和同事**相同的镜像**起自己的容器（我们起了 `mzy-mindspeed`），
环境变量用 `container_env.sh` 里的 `source` 方式注入，这只影响当前 shell，不动系统。

### 坑 2：HuggingFace 直连不通
这台服务器直接连 huggingface.co 会超时。要设置镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
另外那个看着像代理的 `110.120.0.9:9981` 是**死的**，别用它，用了反而全挂。
pip 源用阿里云：`https://mirrors.aliyun.com/pypi/simple/`。

### 坑 3：bash 的 `set -u` 和昇腾环境脚本打架
昇腾的 `setenv.bash` 里引用了可能不存在的变量 `LD_LIBRARY_PATH`。
如果你的脚本开头写了 `set -euo pipefail`（严格模式），一 source 它就报
`LD_LIBRARY_PATH: unbound variable` 直接退出。解决：只用 `set -eo pipefail`，别加 `u`。

### 坑 4：容器里没有 `torchrun` 命令
torch 的命令行启动器没加到 PATH。用等价的写法：
```bash
$PY -m torch.distributed.run --nproc_per_node 1 ...   # 代替 torchrun ...
```

### 坑 5：容器里 root 写的文件，宿主机删不掉
容器内是 root，写出来的文件宿主机 admin 用户没权限删（sudo 又要密码）。
要删这些文件，让容器去删：
```bash
docker exec mzy-mindspeed rm -rf /mnt/models/CODE/mzy/某目录
```

### 坑 6：初探的初始化权重无法逐比特复现
初探是在 **Windows** 上跑的，它的随机数生成器和 Linux 的结果不同
（同样的种子 2027，两边生成的随机权重不一样）。我们验证过：同一台 Linux 上
不同 python/torch 版本生成的结果完全相同，说明不是我们的错，是平台差异。
**结论：接受这个差异。** 因为同一个种子下 6 个条件共享同一份初始化，
"6 个条件互相对比"这个实验的核心逻辑完全不受影响。报告里如实记录即可。

### 坑 7：Megatron 默认不给"细参数"做 weight decay，和初探不一致
Megatron 默认对偏置（bias）和归一化层（只有一维的参数）**不做** weight decay，
而初探用的 HuggingFace AdamW 对**所有**参数都做。差一个词表embedding和几十个小参数，
结果会有可见偏差。我们的解法是在训练脚本里打了一个补丁（monkeypatch），
强制所有参数都参与 weight decay，并单独写小程序验证过补丁生效。

### 坑 8：模型转换器不生成 config.json
`convert_ckpt_v2.py` 把 MindSpeed 格式转回 HuggingFace 格式时，**只存权重，不存配置文件**。
直接拿去评测会报 "Unrecognized model"。解法：转完手动把初始模型目录里的
`config.json` 和 `generation_config.json` 拷过去（`eval_run.sh` 里已经处理了）。

### 坑 9：micro batch 太小，NPU 在"磨洋工"
照搬初探的 micro batch=4 时，一张 NPU 每步要 1.08 秒，利用率只有约 3%——
时间全花在"启动几千个小计算任务"的调度开销上，而不是真正计算。
改成 micro batch=32（一次吃下整个 batch）+ 开启 NPU 融合算子后，每步 0.26 秒，**快了 4 倍**。
再让每张卡同时跑 2 个 run（小模型只占 17GB 显存，64GB 的卡绰绰有余），总时长从 6.5 小时压到约 1.5 小时。
**教训：小模型在大卡上，瓶颈几乎永远是"任务发射"，不是算力。batch 尽量给大。**

### 坑 10：Megatron 的学习率比直觉慢半拍
Megatron 在第 N 步打印的学习率，实际上是"初探公式里第 N-1 步"的值
（它的第 1 步学习率是 0，等于白走一步；最后少一步学习率为 0 的尾巴）。
我们把两边公式逐点核对过：**总的学习率剂量完全相同**（一个从 0 开始、一个以 0 结束，正好对称），
不影响结论，但看日志时别被"第 1 步 lr=0"吓到。

### 坑 11：`pkill -f` 会杀了自己
在 `docker exec bash -c '...'` 里写 `pkill -f run_card.sh` 时，
这条命令自己的命令行里也包含 "run_card.sh"，于是 pkill 先把自己杀了，后面的清理全没执行。
技巧：用括号让模式匹配不到自己：`pkill -f "run_car[d]"`。

### 坑 12：文件名里的前导零
checkpoint 目录叫 `iter_0000128`，shell 里做数字比较前记得把前导零去掉
（`sed 's/iter_0*//'`），否则 `"128" = "0000128"` 判定为假，最后一步的测试集评测会被悄悄跳过——我们真踩了。

---

## 七、怎么判断跑出来的结果是好的

训练时看 `runs/<条件>-s<种子>/train.log`：

- **loss**：第 2 步约 11.1（≈ ln 64000，随机瞎猜的理论值），100 步后约 7，最终应降到 4 以下。
- **grad norm**：一般在 0.3~1.5 之间，偶尔跳到 2 也正常（会被刹车）。如果持续超过 10，有问题。
- **number of skipped iterations**：偶尔 1 次（第一步）正常，持续增加说明 fp16 溢出严重。
- 每条日志的时间戳应该在动，超过 20 分钟没新行就是卡死了。

最终成绩和初探的参考值（两个种子的平均，BPB，越低越好）：

| 条件 | 英文 | 中文 | 说明 |
|---|---:|---:|---|
| en_only | 1.316 | 3.146 | 只学英文，中文当然不会 |
| zh_only | 2.137 | 1.641 | 反过来 |
| iid_50 | **1.419** | **1.789** | 混合训练，双语都好的基准 |
| en_first | 1.730 | 1.843 | 英文被遗忘拖累 |
| zh_first | 1.419 | 2.659 | 中文被严重遗忘（最惨） |
| alternating | 1.495 | 1.890 | 交替比分块好，但没超过混合 |

我们 NPU 复现的数值不会和这些**一模一样**（初始化不同、硬件不同、框架融合算子不同），
但应该落在相近范围（±0.05 内），而且**结论的排序和遗忘的方向**应该完全一致。

---

## 八、已知偏差清单（写报告时的诚实声明）

复现 ≠ 逐比特相同。以下偏差均已确认原因、评估过影响：

1. **初始权重不同**：初探在 Windows 上生成初始化，平台 RNG 差异，无法复刻（坑 6）。同种子内 6 条件共享初始化，对照关系完整。
2. **micro batch 4×8 → 32×1**：数学等价，仅 fp32 求和顺序差异（坑 9）。
3. **融合算子**：flash-attn / fused-rmsnorm / fused-swiglu 与初探的朴素实现存在 1e-3 量级数值差。
4. **fp16 loss scale 轨迹不同**：NPU 版首步溢出把 65536 降到 32768（发生在 lr=0 的第 1 步，无实际影响），初探全程 65536。缩放倍数是 2 的幂，对梯度数值无影响。
5. **硬件差异**：2080Ti (CUDA) vs 昇腾 NPU (CANN)，底层算子实现不同。

以上任何一条都不影响实验要回答的问题：**6 种数据顺序在相同初始化、相同数据下的相对表现。**

---

## 九、求助前自查

| 症状 | 大概率原因 | 怎么办 |
|---|---|---|
| `LD_LIBRARY_PATH: unbound variable` | 脚本写了 `set -u` | 去掉 `u`（坑 3） |
| `torchrun: command not found` | 没用 `$PY -m torch.distributed.run` | 坑 4 |
| 下载数据超时 | 没设 HF_ENDPOINT | `export HF_ENDPOINT=https://hf-mirror.com` |
| 容器里 import torch_npu 报 libhccl | 没 source 环境 | `source .../container_env.sh` |
| 宿主机删不动文件 | 文件是容器 root 写的 | `docker exec mzy-mindspeed rm -rf ...`（坑 5） |
| 评测报 Unrecognized model | 转换后缺 config.json | 坑 8 |
| 训练日志 20 分钟不动 | 可能 hang | 先 `docker exec mzy-mindspeed ps aux` 看进程还在不在 |
