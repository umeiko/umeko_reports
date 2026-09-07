# NPU 集群实操手册（P0 复现 / P1 143M 扩展的经验沉淀）

日期：2026-09-07 ｜ 适用范围：昇腾 8 卡 NPU 主机 + MindSpeed-LLM，复现/扩展本仓库的双语顺序实验
读者：要在**另一台服务器**上重跑或扩展这些实验的人。按顺序照做即可。

---

## 1. 环境长什么样

- 宿主机：8× Ascend NPU（64GB HBM/卡），数据盘是 NFS（`/mnt/models`）
- 容器镜像：`swr.cn-south-1.myhuaweicloud.com/ascendhub/verl_pt27_25rc3:a2-arm`
  （自带 CANN 25rc3、torch 2.7.1 + torch_npu；python 在 `/root/miniconda3/envs/ms/bin/python`，记为 `$PY`）
- MindSpeed-LLM v2.3.0 仓库：`/mnt/models/CODE/MindSpeed-LLM-v2.3.0`（宿主机只读挂载，容器内直接用）
- megatron-core 0.12.1、mindspeed 0.12.1、transformers 4.57.1（容器内）；宿主机另有 venv（transformers 4.51）用于物化/汇总

### 容器创建命令（8 卡全挂，按本机实测还原）

```bash
docker create --name mzy-mindspeed \
  -v /mnt:/mnt \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro -v /usr/sbin:/usr/sbin:ro \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  --ipc=host --network=host --shm-size=64m \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/verl_pt27_25rc3:a2-arm \
  /bin/bash -c "/mnt/models/CODE/mzy/sft_smoke/container_sshd_setup.sh; exec /bin/bash"
```

**铁律：不要动宿主机的 CANN/驱动/固件**，容器里 source 现成的 set_env 就行
（`training/container_env.sh` 已封装：driver/cann/nnal set_env + PYTHONPATH 指向 MindSpeed-LLM +
`CUDA_DEVICE_MAX_CONNECTIONS=1` + `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`）。

## 2. 网络（离线/半离线环境必看）

- HuggingFace 直连不通：一律 `export HF_ENDPOINT=https://hf-mirror.com`（物化、下载 tokenizer 都需要）
- pypi 用阿里云镜像：`pip install -i https://mirrors.aliyun.com/pypi/simple/`（或 uv 加 `UV_INDEX_URL`）
- 本机的 110.120.0.9:9981 代理是**死的**，别配置它
- 宿主机 venv 连 pip 模块都没有的话：`uv pip install --python <venv>/bin/python <包>`

## 3. 数据流水线（数据配方怎么变成 token 池）

```bash
# 1) 盘点上游仓库有哪些文件（可选，已提供 manifests/）
python scripts/inventory_haidass_data.py ...
# 2) 按 config 里的 selected_files 下载固定 shard（幂等，记录每个文件的 sha256）
python scripts/download_raw_shards.py --config configs/p1_143m_proportional.json
# 3) 物化：NFKC 清洗 → sha256 全局去重 → 200~8192 token 过滤 → 拼成 1024 序列
HF_ENDPOINT=https://hf-mirror.com python scripts/materialize_data.py \
    --config configs/p1_143m_proportional.json --raw-shards-root <shard目录>
```

- P0（43M）池：每语言 102,400 序列 ≈ 100M tokens/run；P1（143M）池：每语言 337,920 序列 ≈ 331M tokens/run
- dev/test 只从 ultrafineweb en/zh 按文档哈希 mod 10000 划分（0-49 dev、50-99 test），**池子放大时 dev/test 逐字节不变**（P1 已验证），跨规模可比
- 池子放大时把 P0 用的文件放每源 selected_files 第一个，可保持"同源文档流前缀超集"性质
- 验收三件事：各源 filled=配额；dev/test 与旧池 sha256 一致；每源块前缀与旧池一致（抽前 100 行对比）

## 4. 训练流水线（以 143M 为例，43M 同理换脚本名）

```bash
# 1) 生成确定性初始化（宿主机 venv，transformers 4.51！见坑#1）
python scripts/npu/gen_init_143m.py        # 产物 init_hf_143m/init-s{2027,2028}
# 2) 转 Megatron 格式（容器内，各 ~4s）
docker exec mzy-mindspeed bash -c 'source .../container_env.sh; cd /mnt/models/CODE/MindSpeed-LLM-v2.3.0; \
  $PY convert_ckpt_v2.py --load-model-type hf --save-model-type mg \
  --load-dir .../init_hf_143m/init-s2027 --save-dir .../init_mg_143m/init-s2027 \
  --model-type-hf qwen3 --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1'
# 3) 生成 6 条件×2 种子的顺序表（逐行移植初探调度逻辑）
python scripts/npu/make_order_143m.py
# 4) smoke：先跑 128 步确认参数量=143,071,296、loss 从 ~11.1 下降、记录显存峰值
docker exec -e RUN_NAME=smoke -e MICRO_BATCH=32 mzy-mindspeed \
  bash scripts/npu/run_one_143m.sh iid_50 2027 4 29801 128
# 5) 正式：每卡一条 wave（条件:种子 列表 + 唯一端口段）
docker exec mzy-mindspeed bash scripts/npu/run_wave_143m.sh 4 29820 iid_50:2027 en_only:2027 zh_only:2027 &
# 6) 评测（wave 会自动接着做）：17 个 dev 点 + 终点 test
# 7) 汇总：compare_results_143m.py → plot_143m_figures.py → make_repo_tables.py
```

## 5. 评测方法（验证协议）

- 指标 BPB = nll_sum / (utf8_bytes × ln2)，逐行移植初探 `evaluate()`；micro batch 4，fp16 autocast
- 评测点：step 0（随机初始化）+ 每 632 步共 17 个 dev 点 + 终点 test；遗忘量 = 终点 dev BPB − 全程最优 dev BPB
- Megatron checkpoint 转回 HF 用"symlink iter 目录 + 写 latest_checkpointed_iteration.txt"的 view 目录技巧（`eval_run_143m.sh` 里现成）；**转换后记得补拷 config.json/generation_config.json**（mg→hf 不生成）

## 6. 坑清单（每个都是实际踩过的）

1. **transformers 4.57 把 config 里的 `torch_dtype` 改名 `dtype`**，MindSpeed 转换器认旧名 → `KeyError: 'torch_dtype'`。初始化脚本用宿主机 venv（4.51）跑，权重哈希与容器内生成完全一致（已验证）。
2. 物化脚本忘加 `HF_ENDPOINT` → 卡在 tokenizer 下载报连接错误。
3. bash 脚本**别用 `set -u`**：昇腾 `setenv.bash` 引用未定义变量会直接炸。
4. 容器里没有 torchrun 入口，用 `$PY -m torch.distributed.run` 代替。
5. 容器内 root 写的文件宿主机 admin 删不掉：`docker exec mzy-mindspeed rm -rf <路径>`。
6. `pkill -f` 匹配模式要用 `"run_car[d]"` 括号 trick，否则把 pkill 自己所在的 shell 也杀掉。
7. host 侧重定向日志到不存在的目录会让 `docker exec` 直接失败（先 `mkdir -p`）。
8. Megatron 的 LR 调度：cosine 分母是 (decay_iters − warmup)，iteration N 施加第 N−1 步的 LR——与初探公式对齐要靠 `--lr-decay-iters = train_iters`、`--lr-warmup-iters = warmup`。
9. Megatron 默认对 1-D 参数不做 weight decay；要与初探一致（全参数 wd）需 monkeypatch `_get_param_groups`（`pretrain_bilingual_p0.py` 里现成）。
10. checkpoint 写 NFS 有可见开销（143M 每 632 步写 286MB，单步时间从 0.58s 漂到 ~0.8s），估时留余量。

## 7. 参考数字（64GB HBM/卡）

| 规模 | micro batch | 单 run 峰值显存 | 单步耗时 | 单 run 时长 | 结论 |
|---|---|---:|---:|---:|---|
| 43M | 32 | ~17.6GB | 0.26s（单跑）/0.47s（双并发） | ~15min | 每卡 2 并发安全 |
| 143M | 32 | ~32.5GB | 0.58s → 含 ckpt ~0.8s | ~2.2h | 每卡只能 1 个 |
| 143M | 16 ×2 并发 | 2×16.8GB | 1.07s | — | 仅快 7.6%，不值得 |

- 143M 吞吐 ~40-57k tokens/s/卡；12 runs（3.98G tokens）四卡总墙钟 ~7h（含评测）
- 143M 每 run 17 个 checkpoint ≈ 4.9GB；12 runs ≈ 58GB 磁盘

## 8. 换服务器时的搬运清单

1. 仓库本体（reports/configs/scripts/manifests 全在 git 里）
2. 大文件（不进 git，需单独拷贝）：
   - `data/raw_shards/`（30GB，50 个固定 shard；或按 `manifests/raw_shard_inventory.json` 的 sha256 重新下载校验）
   - 物化池 `data/materialized_*/`（864MB / 2.8GB；或用配置重新物化，更快）
   - `Haidass1.5-143M/` 模型目录（286MB；或从 HF 拉，`manifests/base_model_haidass1.5_143m.json` 有 sha256 校验）
   - checkpoint/日志（可选，纯存档）
3. MindSpeed-LLM v2.3.0 源码 + 昇腾容器镜像
