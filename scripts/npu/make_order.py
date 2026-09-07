"""Regenerate the pilot P0 per-run training order and verify against the audit.

Reimplements the exact schedule logic of umeko_reports/scripts/train_p0.py
(language_schedule / sequence_indices / stratified_order, copied verbatim with
provenance) so each (condition, seed) gets the identical sequence order as the
2080 Ti pilot — provided the regenerated pools are bitwise identical. The pilot
run_audit.json stores schedule_sha256 per run; matching it proves the pools
and this schedule code reproduce the pilot exactly.

Outputs:
  training/order/<condition>_s<seed>.npz  (labels, indices int32 arrays)
  training/order/manifest.json            (per-run sha256 + audit comparison)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/mnt/models/CODE/mzy/umeko_reports/data/materialized_100m_all_sources")
AUDIT = Path("/mnt/models/CODE/mzy/umeko_reports/reports/p0_scaled_100m_all_sources_2080ti_run_audit.json")
OUT_DIR = ROOT / "training" / "order"

SEQUENCES_PER_RUN = 98304  # pilot training.sequences_per_run
CONTEXT = 1024
SEEDS = (2027, 2028)
CONDITIONS = ("en_only", "zh_only", "iid_50", "en_first", "zh_first", "alternating")
PILOT_RUN_PREFIX = "p0_scaled_100m_all_sources_2080ti"


# ---- begin verbatim pilot logic (umeko_reports/scripts/train_p0.py) ----

def language_schedule(condition: str, sequence_count: int, seed: int) -> np.ndarray:
    if condition == "en_only":
        return np.zeros(sequence_count, dtype=np.int8)
    if condition == "zh_only":
        return np.ones(sequence_count, dtype=np.int8)
    if sequence_count % 8:
        raise ValueError("bilingual sequence count must be divisible by eight")
    half = sequence_count // 2
    if condition == "iid_50":
        labels = np.concatenate([np.zeros(half, dtype=np.int8), np.ones(half, dtype=np.int8)])
        np.random.default_rng(seed + 9000).shuffle(labels)
        return labels
    block = sequence_count // 8
    patterns = {
        "en_first": [0, 0, 0, 0, 1, 1, 1, 1],
        "zh_first": [1, 1, 1, 1, 0, 0, 0, 0],
        "alternating": [0, 1, 0, 1, 0, 1, 0, 1],
    }
    if condition not in patterns:
        raise ValueError(f"unknown condition: {condition}")
    return np.repeat(np.asarray(patterns[condition], dtype=np.int8), block)


def stratified_order(source_codes: np.ndarray, required: int, seed: int) -> np.ndarray:
    unique, counts = np.unique(source_codes, return_counts=True)
    raw_quotas = required * counts / counts.sum()
    quotas = np.floor(raw_quotas).astype(int)
    remainder = required - int(quotas.sum())
    fractional_order = np.lexsort((unique, -(raw_quotas - quotas)))
    quotas[fractional_order[:remainder]] += 1
    selected = []
    for source_code, quota in zip(unique.tolist(), quotas.tolist()):
        source_indices = np.flatnonzero(source_codes == source_code)
        if quota > len(source_indices):
            raise RuntimeError(f"source {source_code} quota exceeds the materialized pool")
        rng = np.random.default_rng(seed + 1009 * (int(source_code) + 1))
        selected.extend(rng.permutation(source_indices)[:quota].tolist())
    selected = np.asarray(selected, dtype=np.int32)
    np.random.default_rng(seed + 7919).shuffle(selected)
    return selected


def sequence_indices(labels, available, seed, source_arrays=None) -> np.ndarray:
    required = {language: int((labels == language).sum()) for language in (0, 1)}
    if source_arrays:
        orders = {
            0: stratified_order(np.asarray(source_arrays["en"]), required[0], seed + 101),
            1: stratified_order(np.asarray(source_arrays["zh"]), required[1], seed + 202),
        }
    else:
        orders = {
            0: np.random.default_rng(seed + 101).permutation(available),
            1: np.random.default_rng(seed + 202).permutation(available),
        }
    positions = {0: 0, 1: 0}
    indices = np.empty(len(labels), dtype=np.int32)
    for offset, label_value in enumerate(labels.tolist()):
        language = int(label_value)
        position = positions[language]
        if position >= len(orders[language]):
            raise RuntimeError(f"language {language} sequence stream exhausted")
        indices[offset] = orders[language][position]
        positions[language] += 1
    return indices

# ---- end verbatim pilot logic ----


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))["runs"]
    source_arrays = {
        lang: np.load(DATA_ROOT / lang / "train_sources.npy", mmap_mode="r")
        for lang in ("en", "zh")
    }
    materialization = json.loads((DATA_ROOT / "materialization_summary.json").read_text(encoding="utf-8"))
    source_vocabulary = materialization["source_vocabulary"]
    train_sequences_per_language = materialization["languages"]["en"]["filled_sequences"]["train"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"sequences_per_run": SEQUENCES_PER_RUN, "context": CONTEXT, "runs": {}}
    all_match = True
    for seed in SEEDS:
        for condition in CONDITIONS:
            labels = language_schedule(condition, SEQUENCES_PER_RUN, seed)
            indices = sequence_indices(labels, train_sequences_per_language, seed, source_arrays)
            source_codes = np.asarray(
                [
                    source_arrays["en" if int(language) == 0 else "zh"][index]
                    for language, index in zip(labels.tolist(), indices.tolist())
                ],
                dtype=np.int32,
            )
            # pilot schedule.npy layout: columns [label, index, source_code]
            schedule_path = OUT_DIR / f"schedule-{condition}-s{seed}.npy"
            np.save(schedule_path, np.stack([labels.astype(np.int32), indices, source_codes], axis=1))
            schedule_hash = sha256_file(schedule_path)
            pilot_run_id = f"{PILOT_RUN_PREFIX}-{condition}-s{seed}"
            pilot_hash = audit[pilot_run_id]["schedule_sha256"]
            match = schedule_hash == pilot_hash
            all_match &= match
            np.savez(
                OUT_DIR / f"order_{condition}_s{seed}.npz",
                labels=labels.astype(np.int8),
                indices=indices,
            )
            manifest["runs"][f"{condition}-s{seed}"] = {
                "schedule_sha256": schedule_hash,
                "pilot_schedule_sha256": pilot_hash,
                "match": match,
                "source_sequence_counts": {
                    source_id: int((source_codes == code).sum())
                    for source_id, code in source_vocabulary.items()
                },
                "pilot_source_sequence_counts": audit[pilot_run_id]["source_sequence_counts"],
            }
            print(f"{condition}-s{seed}: schedule sha256 {'MATCH' if match else 'MISMATCH'}", flush=True)
    manifest["all_match_pilot"] = all_match
    tmp = OUT_DIR / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, OUT_DIR / "manifest.json")
    print(f"all_match_pilot={all_match}; wrote {OUT_DIR / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
