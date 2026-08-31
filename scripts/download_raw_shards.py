import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import hf_hub_download


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Persist the exact raw Parquet shards selected for the all-source Haidass pilot.")
    parser.add_argument("--config", default="configs/p0_scaled_100m.json")
    parser.add_argument("--output", default="data/raw_shards_100m_all_sources")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    inventory_path = output_root / "raw_shard_inventory.json"
    inventory = {
        "status": "running",
        "config_path": str(config_path),
        "files": [],
        "total_bytes": 0,
        "started_unix": time.time(),
    }
    atomic_json(inventory_path, inventory)

    seen = set()
    for source in config["dataset"]["sources"]:
        for filename in source["selected_files"]:
            identity = (source["repository"], source["revision"], filename)
            if identity in seen:
                continue
            seen.add(identity)
            source_root = output_root / source["id"]
            source_root.mkdir(parents=True, exist_ok=True)
            print(f"download {source['id']}: {filename}", flush=True)
            local_path = Path(
                hf_hub_download(
                    repo_id=source["repository"],
                    repo_type="dataset",
                    revision=source["revision"],
                    filename=filename,
                    local_dir=source_root,
                )
            ).resolve()
            record = {
                "source_id": source["id"],
                "repository": source["repository"],
                "revision": source["revision"],
                "filename": filename,
                "local_path": str(local_path),
                "bytes": local_path.stat().st_size,
                "sha256": sha256_file(local_path),
            }
            inventory["files"].append(record)
            inventory["total_bytes"] = sum(item["bytes"] for item in inventory["files"])
            atomic_json(inventory_path, inventory)

    inventory["status"] = "complete"
    inventory["completed_unix"] = time.time()
    atomic_json(inventory_path, inventory)
    print(json.dumps({"status": "complete", "files": len(inventory["files"]), "total_bytes": inventory["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
