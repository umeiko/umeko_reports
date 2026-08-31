from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi

from common import ROOT, load_config, write_json


MODEL_CARD_URL = "https://huggingface.co/DALabCommunity/Haidass1.5-143M"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "p0_scaled_100m.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    sources = config["dataset"]["sources"]
    api = HfApi()
    repositories = {}
    grouped = defaultdict(list)
    for source in sources:
        grouped[source["repository"]].append(source)
    for repository, repository_sources in grouped.items():
        for attempt in range(1, 4):
            try:
                info = api.dataset_info(repository, revision=repository_sources[0]["revision"], files_metadata=True)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * attempt)
        total_bytes = sum((file.size or 0) for file in info.siblings)
        license_tags = [tag.split(":", 1)[1] for tag in info.tags if tag.startswith("license:")]
        repositories[repository] = {
            "revision": info.sha,
            "license_tags": license_tags,
            "file_count": len(info.siblings),
            "repository_bytes": total_bytes,
            "repository_decimal_tb": total_bytes / 1e12,
            "local_source_ids": [source["id"] for source in repository_sources],
            "covered_configs": [source.get("config") for source in repository_sources],
        }
    usage = shutil.disk_usage(ROOT)
    materialization_path = ROOT / "data" / config["materialized_subdir"] / "materialization_summary.json"
    materialization = json.loads(materialization_path.read_text(encoding="utf-8")) if materialization_path.exists() else None
    raw_inventory_path = ROOT / "data" / "raw_shards_100m_all_sources" / "raw_shard_inventory.json"
    raw_inventory = json.loads(raw_inventory_path.read_text(encoding="utf-8")) if raw_inventory_path.exists() else None
    inventory = {
        "model_card": MODEL_CARD_URL,
        "model_card_claim": "approximately 400B pretraining tokens from five named repositories; exact mixture and stage boundaries undisclosed",
        "repositories": repositories,
        "repository_total_bytes": sum(repository["repository_bytes"] for repository in repositories.values()),
        "repository_total_decimal_tb": sum(repository["repository_bytes"] for repository in repositories.values()) / 1e12,
        "workspace_drive_free_bytes": usage.free,
        "workspace_drive_free_tib": usage.free / 2**40,
        "full_mirror_feasible": sum(repository["repository_bytes"] for repository in repositories.values()) <= usage.free,
        "sampling_policy": config["dataset"]["sampling_salt"],
        "local_pool_tokens": {
            language: config["local_data"]["train_sequences_per_language"] * config["local_data"]["context_length"]
            for language in ("en", "zh")
        },
        "selected_files": {
            source_id: source["selected_files"]
            for source_id, source in materialization.get("sources", {}).items()
        } if materialization else None,
        "raw_shard_archive": raw_inventory,
        "reproducibility_boundary": {
            "explicit_in_model_card": ["Ultra-FineWeb en+zh", "DCLM baseline", "FineMath-4plus", "Ultra-FineWeb-L3 repository", "Cosmopedia repository"],
            "not_disclosed": ["five-source token ratios", "multi-stage boundaries", "Ultra-FineWeb-L3 config selection", "Cosmopedia config selection", "document/shard selection seed"],
            "local_control": "cover all four L3 configs and all eight Cosmopedia configs; use pinned revisions and deterministic shard ranking",
        },
    }
    output_json = ROOT / "reports" / "HAIDASS_DATASET_INVENTORY.json"
    write_json(output_json, inventory)

    lines = [
        "# Haidass1.5 训练数据调研与本地取样清单",
        "",
        f"依据：[Haidass1.5-143M 模型卡]({MODEL_CARD_URL})。模型卡声明约 400B token，并列出五个主要数据仓库，但未公开各源比例、阶段边界、精确 shard 或随机种子。",
        "",
        "## 上游规模",
        "",
        "| Repository | Pinned revision | License tag | Files | Repository size | Local coverage |",
        "|---|---|---|---:|---:|---|",
    ]
    for repository, record in repositories.items():
        lines.append(
            f"| [{repository}](https://huggingface.co/datasets/{repository}) | `{record['revision']}` | "
            f"{', '.join(record['license_tags']) or 'not tagged'} | {record['file_count']:,} | "
            f"{record['repository_decimal_tb']:.3f} TB | {', '.join(f'`{value}`' for value in record['covered_configs'])} |"
        )
    lines.extend([
        "",
        f"五仓库当前快照合计约 **{inventory['repository_total_decimal_tb']:.2f} TB**；工作区所在磁盘可用约 **{inventory['workspace_drive_free_tib']:.2f} TiB**。因此全量镜像在本机不可行，也与约 1 亿 token/条件的实验预算不匹配。",
        "",
        "## 本地执行口径",
        "",
        "- 五个模型卡仓库全部进入主双语训练池，不以相邻数据集替代。",
        "- FineMath 严格使用 `finemath-4plus`，不混入 3plus 或 InfiWebMath。",
        "- 模型卡没有说明 L3 子类，因此覆盖 EN/ZH 的 QA 与 multi-style 四类；没有说明 Cosmopedia 子类，因此覆盖其全部八类 config。",
        "- 每个上游仓库固定 revision；shard 由公开 salt 对路径做 SHA-256 排序后确定，实际路径写入物化 manifest。",
        "- 本地池每种语言各约 104.86M token；主双语条件严格 50:50，并按来源分层抽样。",
        "- 该设计是受控、可复现的全来源实验，不宣称复刻原始 400B-token 配方。",
        "",
        "## 模型卡未披露项",
        "",
        "- 五源 token 比例与各阶段变化；",
        "- 多阶段训练的边界、重复采样次数与退火配方；",
        "- Ultra-FineWeb-L3 和 Cosmopedia 的具体 config；",
        "- 原始文档/shard 选择和去重实现。",
    ])
    if raw_inventory:
        unique_hashes = len({record["sha256"] for record in raw_inventory["files"]})
        zero_byte = sum(record["bytes"] <= 0 for record in raw_inventory["files"])
        temporary_files = list((raw_inventory_path.parent).rglob("*.tmp"))
        lines.extend([
            "",
            "## 本地原始 shard 归档",
            "",
            "已按主实验配置中冻结的 revision 和路径，将 16 个入选 Parquet 原文件持久化到 `data/raw_shards_100m_all_sources/`。完整性汇总：",
            "",
            f"- 文件：{len(raw_inventory['files'])}/16；",
            f"- 总字节：{raw_inventory['total_bytes']:,}（{raw_inventory['total_bytes'] / 2**30:.3f} GiB）；",
            f"- 互异 SHA-256：{unique_hashes}；",
            f"- 零字节文件：{zero_byte}；",
            f"- 残留临时文件：{len(temporary_files)}。",
            "",
            "每个文件的来源 ID、仓库、revision、上游路径、本地路径、字节数和 SHA-256 见 `data/raw_shards_100m_all_sources/raw_shard_inventory.json`。",
        ])
    output_md = ROOT / "reports" / "HAIDASS_DATASET_INVENTORY_CN.md"
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_json} and {output_md}")


if __name__ == "__main__":
    main()
