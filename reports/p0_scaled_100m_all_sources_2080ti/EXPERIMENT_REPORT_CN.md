# RTX 2080 Ti 全来源中英双语放大实验报告

状态：**已完成**。证据等级：**2 seed、本机 FP16 放大试验；仍不是集群确认性结果**。

## 实验设置

- 每条件 `100,663,296` tokens，`2` 个 seed，共 `12` 个计划运行。
- 模型参数量 `43,461,504`，context `1024`，全局 batch `32` 条序列。
- 主双语条件固定 50:50 EN:ZH；各 seed 内四个双语顺序条件使用完全相同的序列多重集，只改变呈现顺序。
- Haidass 模型卡没有披露原始五源配比、训练阶段边界以及 L3/Cosmopedia 具体 config；本实验使用公开、显式、可审计的受控配比，不声称复刻 400B-token 原训练。

## 本地物化来源

| Source | Language | Config | Train tokens in pool | Pinned revision |
|---|---|---|---:|---|
| `ultrafineweb_en` | en | `default/en` | 31,457,280 | `02c85641e3d19a854be2e09139c25adaa9518063` |
| `ultrafineweb_zh` | zh | `default/zh` | 62,914,560 | `02c85641e3d19a854be2e09139c25adaa9518063` |
| `dclm` | en | `baseline parquet` | 16,777,216 | `817d6752765f6a41261085171dd546b104f60626` |
| `finemath_4plus` | en | `finemath-4plus` | 16,777,216 | `e92b25a616738fe95dc186b64dfb19f9c8525594` |
| `l3_en_qa` | en | `Ultra-FineWeb-L3-en-QA-Synthetic` | 8,388,608 | `bc3b1ba986fcaef6871b9790a413b16267c2de0f` |
| `l3_en_multistyle` | en | `Ultra-FineWeb-L3-en-Multi-Style-Synthetic` | 10,485,760 | `bc3b1ba986fcaef6871b9790a413b16267c2de0f` |
| `l3_zh_qa` | zh | `Ultra-FineWeb-L3-zh-QA-Synthetic` | 15,728,640 | `bc3b1ba986fcaef6871b9790a413b16267c2de0f` |
| `l3_zh_multistyle` | zh | `Ultra-FineWeb-L3-zh-Multi-Style-Synthetic` | 26,214,400 | `bc3b1ba986fcaef6871b9790a413b16267c2de0f` |
| `cosmopedia_auto_math` | en | `auto_math_text` | 2,621,440 | `0ae6ec63f91742bd2d1eaef4f02232c55d719385` |
| `cosmopedia_khanacademy` | en | `khanacademy` | 2,621,440 | `0ae6ec63f91742bd2d1eaef4f02232c55d719385` |
| `cosmopedia_openstax` | en | `openstax` | 2,621,440 | `0ae6ec63f91742bd2d1eaef4f02232c55d719385` |
| `cosmopedia_stanford` | en | `stanford` | 2,621,440 | `0ae6ec63f91742bd2d1eaef4f02232c55d719385` |
| `cosmopedia_stories` | en | `stories` | 2,621,440 | `0ae6ec63f91742bd2d1eaef4f02232c55d719385` |
| `cosmopedia_web_v1` | en | `web_samples_v1` | 2,621,440 | `0ae6ec63f91742bd2d1eaef4f02232c55d719385` |
| `cosmopedia_web_v2` | en | `web_samples_v2` | 2,621,440 | `0ae6ec63f91742bd2d1eaef4f02232c55d719385` |
| `cosmopedia_wikihow` | en | `wikihow` | 2,621,440 | `0ae6ec63f91742bd2d1eaef4f02232c55d719385` |

## 结果（mean ± sample SD）

| Condition | Seeds | EN test BPB | ZH test BPB | Equal-language mean BPB | EN forgetting | ZH forgetting |
|---|---:|---:|---:|---:|---:|---:|
| `en_only` | 2 | 1.3158 ± 0.0094 | 3.1460 ± 0.0045 | 2.2309 ± 0.0070 | 0.0000 ± 0.0000 | 0.0031 ± 0.0044 |
| `zh_only` | 2 | 2.1371 ± 0.0119 | 1.6413 ± 0.0048 | 1.8892 ± 0.0083 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| `iid_50` | 2 | 1.4191 ± 0.0038 | 1.7885 ± 0.0043 | 1.6038 ± 0.0040 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| `en_first` | 2 | 1.7303 ± 0.0188 | 1.8430 ± 0.0013 | 1.7866 ± 0.0100 | 0.3040 ± 0.0138 | 0.0000 ± 0.0000 |
| `zh_first` | 2 | 1.4189 ± 0.0023 | 2.6592 ± 0.0073 | 2.0391 ± 0.0025 | 0.0000 ± 0.0000 | 0.8785 ± 0.0172 |
| `alternating` | 2 | 1.4954 ± 0.0021 | 1.8896 ± 0.0012 | 1.6925 ± 0.0016 | 0.0123 ± 0.0009 | 0.0000 ± 0.0000 |

## 等语言曝光分解（validation BPB）

这里把单语模型在中点（约 50.33M 该语言 tokens）与 IID 模型终点（也约 50.33M/语言）比较。`mixing delta = IID − 单语中点`；正值表示在相同目标语言曝光量下，混合训练的 BPB 更高。该差值是受控的混合效应描述，不直接等同于因果意义上的“参数干扰”。

| Language | Mono @ 50.33M | IID @ 50.33M/lang | Mixing delta | Extra mono 50.33M gain |
|---|---:|---:|---:|---:|
| EN | 1.4033 ± 0.0133 | 1.4144 ± 0.0045 | +0.0111 ± 0.0178 | +0.0894 ± 0.0050 |
| ZH | 1.7805 ± 0.0081 | 1.8077 ± 0.0056 | +0.0272 ± 0.0025 | +0.1166 ± 0.0014 |

## 运行完整性与计时边界

- 审计要求 12/12 状态完整、训练与模型哈希一致、无非有限 loss/gradient、FP16 scale 不下降、同 seed 共享初始化，以及四个双语条件共享完全相同的数据多重集。
- CSV 中的 `elapsed_wall_clock_minutes` 仅为运维记录：部分运行跨越主机挂起，不能用于性能比较；性能诊断应使用每次更新的 active-compute median tokens/s。
- 温控重跑：`alternating-s2028` (250 ms/update)。首次尝试在 GPU 触及 84°C 并报告软件温控降频后被人工停止并归档；重跑只在每次优化更新后休眠，不改变初始化、数据顺序或优化轨迹。

## 解释边界

双 seed 只用于检验方向是否在两次初始化下重复，并不足以支撑精确显著性结论。论文中的正式主张仍需预注册的多 seed 集群实验、外部双语任务和层级 bootstrap。
