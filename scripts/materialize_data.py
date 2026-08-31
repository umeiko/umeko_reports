from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from numpy.lib.format import open_memmap
from tqdm import tqdm
from transformers import AutoTokenizer

from common import ROOT, environment_snapshot, load_config, sha256_file, write_json


REPO_FILE_CACHE: dict[tuple[str, str], list[str]] = {}


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def split_name(document_hash: str, dataset_config: dict) -> str:
    value = int(document_hash[:16], 16) % dataset_config["partition_modulus"]
    if dataset_config["dev_range"][0] <= value <= dataset_config["dev_range"][1]:
        return "dev"
    if dataset_config["test_range"][0] <= value <= dataset_config["test_range"][1]:
        return "test"
    return "train"


def tokenizer_artifact_hash(tokenizer) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tokenizer.get_vocab().items()):
        digest.update(name.encode("utf-8"))
        digest.update(int(value).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def resolve_source_files(source: dict, sampling_salt: str, raw_root: Path | None = None) -> tuple[list[str], list[str]]:
    if source.get("selected_files"):
        selected = source["selected_files"]
    else:
        cache_key = (source["repository"], source["revision"])
        if cache_key not in REPO_FILE_CACHE:
            for attempt in range(1, 4):
                try:
                    REPO_FILE_CACHE[cache_key] = HfApi().list_repo_files(
                        source["repository"], repo_type="dataset", revision=source["revision"]
                    )
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(5 * attempt)
        files = REPO_FILE_CACHE[cache_key]
        matches = sorted(path for path in files if fnmatch.fnmatchcase(path, source["file_pattern"]))
        if not matches:
            raise RuntimeError(f"{source['id']}: no files match {source['file_pattern']}")
        ranked = sorted(
            matches,
            key=lambda path: hashlib.sha256(f"{sampling_salt}|{source['id']}|{path}".encode()).hexdigest(),
        )
        selected = ranked[: min(source.get("selected_shards", 1), len(ranked))]
    if raw_root is not None:
        local_files = [(raw_root / source["id"] / path).resolve() for path in selected]
        missing = [str(path) for path in local_files if not path.is_file()]
        if missing:
            raise RuntimeError(f"{source['id']}: missing frozen local shards: {missing}")
        return selected, [str(path) for path in local_files]
    remote_files = [
        "https://huggingface.co/datasets/"
        f"{source['repository']}/resolve/{source['revision']}/{quote(path, safe='/')}"
        for path in selected
    ]
    return selected, remote_files


def materialize_multisource(config: dict, tokenizer, output_root: Path, raw_root: Path | None = None) -> dict:
    dataset_config = config["dataset"]
    local = config["local_data"]
    context = local["context_length"]
    targets = {
        "dev": local["dev_sequences_per_language"],
        "test": local["test_sequences_per_language"],
    }
    sources = dataset_config["sources"]
    if raw_root is not None:
        raw_root = raw_root.resolve()
        if not raw_root.is_dir():
            raise RuntimeError(f"raw shard root does not exist: {raw_root}")
    source_ids = [source["id"] for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("dataset source ids must be unique")
    for language in ("en", "zh"):
        language_total = sum(source["train_sequences"] for source in sources if source["language"] == language)
        if language_total != local["train_sequences_per_language"]:
            raise RuntimeError(
                f"{language}: source train quota sums to {language_total}, expected {local['train_sequences_per_language']}"
            )
        evaluation_sources = [source["id"] for source in sources if source["language"] == language and source.get("evaluation_source")]
        if len(evaluation_sources) != 1:
            raise RuntimeError(f"{language}: expected exactly one evaluation source, found {evaluation_sources}")

    arrays = {}
    byte_arrays = {}
    source_arrays = {}
    source_vocabulary = {source["id"]: index for index, source in enumerate(sources)}
    for language in ("en", "zh"):
        language_root = output_root / language
        language_root.mkdir(parents=True, exist_ok=True)
        arrays[language] = {
            "train": open_memmap(
                language_root / "train_tokens.npy",
                mode="w+",
                dtype=np.int32,
                shape=(local["train_sequences_per_language"], context),
            ),
            **{
                split: open_memmap(
                    language_root / f"{split}_tokens.npy", mode="w+", dtype=np.int32, shape=(count, context)
                )
                for split, count in targets.items()
            },
        }
        byte_arrays[language] = {
            split: open_memmap(
                language_root / f"{split}_bytes.npy", mode="w+", dtype=np.int32, shape=(count,)
            )
            for split, count in targets.items()
        }
        source_arrays[language] = open_memmap(
            language_root / "train_sources.npy",
            mode="w+",
            dtype=np.int16,
            shape=(local["train_sequences_per_language"],),
        )

    offsets = {"en": 0, "zh": 0}
    eval_filled = {language: defaultdict(int) for language in ("en", "zh")}
    seen_hashes: set[str] = set()
    source_summaries = {}
    manifests_root = output_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    for source in sources:
        source_id = source["id"]
        language = source["language"]
        train_target = source["train_sequences"]
        train_filled = 0
        buffers = {"train": [], "dev": [], "test": []}
        counters = defaultdict(int)
        selected_paths, selected_locations = resolve_source_files(source, dataset_config["sampling_salt"], raw_root)
        for attempt in range(1, 4):
            try:
                stream = load_dataset(
                    "parquet",
                    data_files={"train": selected_locations},
                    split="train",
                    streaming=True,
                )
                break
            except Exception:
                if attempt == 3:
                    raise
                delay = 5 * attempt
                print(f"{source_id}: remote parquet initialization failed; retry {attempt}/3 in {delay}s")
                time.sleep(delay)
        manifest_path = manifests_root / f"{source_id}.documents.jsonl"
        temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
        progress = tqdm(desc=f"materialize {source_id}", unit="doc")
        with temporary_manifest.open("w", encoding="utf-8") as manifest:
            for row_index, row in enumerate(stream):
                counters["rows_seen"] += 1
                text = row.get(source["text_field"])
                if not isinstance(text, str) or not text.strip():
                    counters["empty"] += 1
                    continue
                normalized = normalized_text(text)
                document_hash = hashlib.sha256(
                    (dataset_config["hash_salt"] + "|" + normalized).encode("utf-8")
                ).hexdigest()
                if document_hash in seen_hashes:
                    counters["exact_duplicate"] += 1
                    continue
                seen_hashes.add(document_hash)
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if not dataset_config["min_document_tokens"] <= len(token_ids) <= dataset_config["max_document_tokens"]:
                    counters["length_filtered"] += 1
                    continue
                partition = split_name(document_hash, dataset_config) if source.get("evaluation_source") else "train"
                if partition == "train" and train_filled >= train_target:
                    counters["unused_train_after_quota"] += 1
                    continue
                if partition in targets and eval_filled[language][partition] >= targets[partition]:
                    counters[f"unused_{partition}_after_quota"] += 1
                    continue
                token_ids.append(config["tokenizer"]["eos_token_id"])
                buffers[partition].extend(token_ids)
                emitted = 0
                if partition == "train":
                    while len(buffers[partition]) >= context and train_filled < train_target:
                        sequence = buffers[partition][:context]
                        del buffers[partition][:context]
                        index = offsets[language] + train_filled
                        arrays[language][partition][index] = np.asarray(sequence, dtype=np.int32)
                        source_arrays[language][index] = source_vocabulary[source_id]
                        train_filled += 1
                        emitted += 1
                else:
                    while len(buffers[partition]) >= context and eval_filled[language][partition] < targets[partition]:
                        sequence = buffers[partition][:context]
                        del buffers[partition][:context]
                        index = eval_filled[language][partition]
                        arrays[language][partition][index] = np.asarray(sequence, dtype=np.int32)
                        decoded = tokenizer.decode(sequence, skip_special_tokens=True)
                        byte_arrays[language][partition][index] = max(1, len(decoded.encode("utf-8")))
                        eval_filled[language][partition] += 1
                        emitted += 1
                counters["accepted_documents"] += 1
                counters[f"accepted_{partition}_tokens"] += len(token_ids)
                manifest.write(json.dumps({
                    "source_id": source_id,
                    "language": language,
                    "row_index": row_index,
                    "document_sha256": document_hash,
                    "split": partition,
                    "token_count_with_eos": len(token_ids),
                    "utf8_bytes": len(text.encode("utf-8")),
                    "emitted_sequences": emitted,
                }, ensure_ascii=False) + "\n")
                if counters["accepted_documents"] % 100 == 0:
                    progress.update(100)
                    progress.set_postfix({
                        "train": train_filled,
                        "dev": eval_filled[language]["dev"],
                        "test": eval_filled[language]["test"],
                    })
                evaluation_complete = not source.get("evaluation_source") or all(
                    eval_filled[language][split] >= targets[split] for split in targets
                )
                if train_filled >= train_target and evaluation_complete:
                    break
            else:
                raise RuntimeError(
                    f"{source_id}: selected stream ended before quotas were filled; "
                    f"train={train_filled}/{train_target}, eval={dict(eval_filled[language])}"
                )
        progress.close()
        os.replace(temporary_manifest, manifest_path)
        offsets[language] += train_target
        source_summaries[source_id] = {
            "repository": source["repository"],
            "revision": source["revision"],
            "config": source.get("config"),
            "language": language,
            "text_field": source["text_field"],
            "train_sequences": train_filled,
            "train_tokens": train_filled * context,
            "evaluation_source": bool(source.get("evaluation_source")),
            "selected_files": selected_paths,
            "manifest_sha256": sha256_file(manifest_path),
            "counters": dict(counters),
        }

    for language in ("en", "zh"):
        if offsets[language] != local["train_sequences_per_language"]:
            raise RuntimeError(f"{language}: incomplete train pool")
        if any(eval_filled[language][split] != targets[split] for split in targets):
            raise RuntimeError(f"{language}: incomplete evaluation pool {dict(eval_filled[language])}")
        for array in arrays[language].values():
            array.flush()
        for array in byte_arrays[language].values():
            array.flush()
        source_arrays[language].flush()
        files = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted((output_root / language).glob("*.npy"))
        }
        language_summary = {
            "language": language,
            "context_length": context,
            "filled_sequences": {
                "train": offsets[language],
                **{split: eval_filled[language][split] for split in targets},
            },
            "materialized_tokens": {
                "train": offsets[language] * context,
                **{split: targets[split] * context for split in targets},
            },
            "source_vocabulary": source_vocabulary,
            "files": files,
        }
        write_json(output_root / language / "summary.json", language_summary)

    return {
        "languages": {
            language: json.loads((output_root / language / "summary.json").read_text(encoding="utf-8"))
            for language in ("en", "zh")
        },
        "sources": source_summaries,
        "source_vocabulary": source_vocabulary,
        "unique_document_hashes": len(seen_hashes),
    }


def materialize_language(language: str, config: dict, tokenizer, output_root: Path) -> dict:
    dataset_config = config["dataset"]
    local = config["local_data"]
    context = local["context_length"]
    targets = {
        "train": local["train_sequences_per_language"],
        "dev": local["dev_sequences_per_language"],
        "test": local["test_sequences_per_language"],
    }
    language_root = output_root / language
    language_root.mkdir(parents=True, exist_ok=True)

    arrays = {
        split: open_memmap(language_root / f"{split}_tokens.npy", mode="w+", dtype=np.int32, shape=(count, context))
        for split, count in targets.items()
    }
    byte_arrays = {
        split: open_memmap(language_root / f"{split}_bytes.npy", mode="w+", dtype=np.int32, shape=(count,))
        for split, count in targets.items()
    }
    filled = defaultdict(int)
    buffers: dict[str, list[int]] = {split: [] for split in targets}
    seen_hashes: set[str] = set()
    counters = defaultdict(int)
    manifest_path = language_root / "documents.jsonl"
    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")

    stream = load_dataset(
        dataset_config["repository"],
        split=dataset_config["splits"][language],
        streaming=True,
        revision=dataset_config["revision"],
    )
    progress = tqdm(desc=f"materialize {language}", unit="doc")
    with temporary_manifest.open("w", encoding="utf-8") as manifest:
        for row_index, row in enumerate(stream):
            counters["rows_seen"] += 1
            text = row.get(dataset_config["text_field"])
            if not isinstance(text, str) or not text.strip():
                counters["empty"] += 1
                continue
            normalized = normalized_text(text)
            document_hash = hashlib.sha256(
                (dataset_config["hash_salt"] + "|" + normalized).encode("utf-8")
            ).hexdigest()
            if document_hash in seen_hashes:
                counters["exact_duplicate"] += 1
                continue
            seen_hashes.add(document_hash)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not dataset_config["min_document_tokens"] <= len(token_ids) <= dataset_config["max_document_tokens"]:
                counters["length_filtered"] += 1
                continue
            partition = split_name(document_hash, dataset_config)
            if filled[partition] >= targets[partition]:
                counters[f"unused_{partition}_after_quota"] += 1
                continue
            token_ids.append(config["tokenizer"]["eos_token_id"])
            buffers[partition].extend(token_ids)
            emitted = 0
            while len(buffers[partition]) >= context and filled[partition] < targets[partition]:
                sequence = buffers[partition][:context]
                del buffers[partition][:context]
                index = filled[partition]
                arrays[partition][index] = np.asarray(sequence, dtype=np.int32)
                decoded = tokenizer.decode(sequence, skip_special_tokens=True)
                byte_arrays[partition][index] = max(1, len(decoded.encode("utf-8")))
                filled[partition] += 1
                emitted += 1
            counters["accepted_documents"] += 1
            counters[f"accepted_{partition}_tokens"] += len(token_ids)
            manifest.write(json.dumps({
                "language": language,
                "row_index": row_index,
                "source": row.get("source"),
                "score": row.get("score"),
                "document_sha256": document_hash,
                "split": partition,
                "token_count_with_eos": len(token_ids),
                "utf8_bytes": len(text.encode("utf-8")),
                "emitted_sequences": emitted,
            }, ensure_ascii=False) + "\n")
            if counters["accepted_documents"] % 100 == 0:
                progress.update(100)
                progress.set_postfix({key: filled[key] for key in ("train", "dev", "test")})
            if all(filled[split] >= targets[split] for split in targets):
                break
        else:
            raise RuntimeError(f"{language}: stream ended before quotas were filled: {dict(filled)}")
    progress.close()
    os.replace(temporary_manifest, manifest_path)
    for array in arrays.values():
        array.flush()
    for array in byte_arrays.values():
        array.flush()

    files = {}
    for path in sorted(language_root.glob("*.npy")):
        files[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    files[manifest_path.name] = {"sha256": sha256_file(manifest_path), "bytes": manifest_path.stat().st_size}
    summary = {
        "language": language,
        "dataset": dataset_config["repository"],
        "revision": dataset_config["revision"],
        "upstream_split": dataset_config["splits"][language],
        "context_length": context,
        "filled_sequences": dict(filled),
        "materialized_tokens": {split: targets[split] * context for split in targets},
        "counters": dict(counters),
        "files": files,
    }
    write_json(language_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "p0_local.json"))
    parser.add_argument("--raw-shards-root")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = ROOT / "data" / config.get("materialized_subdir", "materialized")
    completion_path = output_root / "materialization_summary.json"
    if completion_path.exists() and not args.force:
        print(f"data already materialized: {completion_path}")
        return

    tokenizer_config = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config["repository"], revision=tokenizer_config["revision"], use_fast=True
    )
    if len(tokenizer) != tokenizer_config["expected_vocabulary"]:
        raise RuntimeError(f"tokenizer vocabulary changed: {len(tokenizer)}")
    if config["dataset"].get("sources"):
        raw_root = Path(args.raw_shards_root) if args.raw_shards_root else None
        multisource = materialize_multisource(config, tokenizer, output_root, raw_root)
        summaries = multisource["languages"]
    else:
        multisource = None
        summaries = {
            language: materialize_language(language, config, tokenizer, output_root)
            for language in ("en", "zh")
        }
    write_json(completion_path, {
        "status": "complete",
        "scientific_status": config["scientific_status"],
        "config_path": config["_config_path"],
        "config_sha256": sha256_file(Path(config["_config_path"])),
        "tokenizer_repository": tokenizer_config["repository"],
        "tokenizer_revision": tokenizer_config["revision"],
        "tokenizer_vocabulary_sha256": tokenizer_artifact_hash(tokenizer),
        "environment": environment_snapshot(),
        "languages": summaries,
        "sources": multisource["sources"] if multisource else None,
        "source_vocabulary": multisource["source_vocabulary"] if multisource else None,
        "unique_document_hashes": multisource["unique_document_hashes"] if multisource else None,
        "known_deviation_from_cluster_plan": "Local scaled pilot uses revision-and-salt-selected upstream shards and exact cross-source deduplication; formal P1/P2 additionally require MinHash deduplication. The Haidass model card does not disclose the original source mixture, stage boundaries, or exact source configs.",
    })
    print(f"wrote {completion_path}")


if __name__ == "__main__":
    main()
