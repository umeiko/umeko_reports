# Haidass1.5 训练数据调研与本地取样清单

依据：[Haidass1.5-143M 模型卡](https://huggingface.co/DALabCommunity/Haidass1.5-143M)。模型卡声明约 400B token，并列出五个主要数据仓库，但未公开各源比例、阶段边界、精确 shard 或随机种子。

## 上游规模

| Repository | Pinned revision | License tag | Files | Repository size | Local coverage |
|---|---|---|---:|---:|---|
| [openbmb/Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) | `02c85641e3d19a854be2e09139c25adaa9518063` | apache-2.0 | 64,669 | 10.211 TB | `default/en`, `default/zh` |
| [mlfoundations/dclm-baseline-1.0-parquet](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet) | `817d6752765f6a41261085171dd546b104f60626` | cc-by-4.0 | 27,940 | 7.420 TB | `baseline parquet` |
| [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | `e92b25a616738fe95dc186b64dfb19f9c8525594` | odc-by | 293 | 0.149 TB | `finemath-4plus` |
| [openbmb/Ultra-FineWeb-L3](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3) | `bc3b1ba986fcaef6871b9790a413b16267c2de0f` | apache-2.0 | 1,771 | 1.899 TB | `Ultra-FineWeb-L3-en-QA-Synthetic`, `Ultra-FineWeb-L3-en-Multi-Style-Synthetic`, `Ultra-FineWeb-L3-zh-QA-Synthetic`, `Ultra-FineWeb-L3-zh-Multi-Style-Synthetic` |
| [HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | `0ae6ec63f91742bd2d1eaef4f02232c55d719385` | apache-2.0 | 338 | 0.092 TB | `auto_math_text`, `khanacademy`, `openstax`, `stanford`, `stories`, `web_samples_v1`, `web_samples_v2`, `wikihow` |

五仓库当前快照合计约 **19.77 TB**；工作区所在磁盘可用约 **1.48 TiB**。因此全量镜像在本机不可行，也与约 1 亿 token/条件的实验预算不匹配。

## 本地执行口径

- 五个模型卡仓库全部进入主双语训练池，不以相邻数据集替代。
- FineMath 严格使用 `finemath-4plus`，不混入 3plus 或 InfiWebMath。
- 模型卡没有说明 L3 子类，因此覆盖 EN/ZH 的 QA 与 multi-style 四类；没有说明 Cosmopedia 子类，因此覆盖其全部八类 config。
- 每个上游仓库固定 revision；shard 由公开 salt 对路径做 SHA-256 排序后确定，实际路径写入物化 manifest。
- 本地池每种语言各约 104.86M token；主双语条件严格 50:50，并按来源分层抽样。
- 该设计是受控、可复现的全来源实验，不宣称复刻原始 400B-token 配方。

## 模型卡未披露项

- 五源 token 比例与各阶段变化；
- 多阶段训练的边界、重复采样次数与退火配方；
- Ultra-FineWeb-L3 和 Cosmopedia 的具体 config；
- 原始文档/shard 选择和去重实现。

## 本地原始 shard 归档

已按主实验配置中冻结的 revision 和路径，将 16 个入选 Parquet 原文件持久化到 `data/raw_shards_100m_all_sources/`。完整性汇总：

- 文件：16/16；
- 总字节：9,170,960,563（8.541 GiB）；
- 互异 SHA-256：16；
- 零字节文件：0；
- 残留临时文件：0。

每个文件的来源 ID、仓库、revision、上游路径、本地路径、字节数和 SHA-256 见 `data/raw_shards_100m_all_sources/raw_shard_inventory.json`。
