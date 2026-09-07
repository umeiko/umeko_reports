"""Generate the P1 143M per-run training orders (6 conditions x 2 seeds).

Identical schedule logic as make_order.py (verbatim pilot train_p0.py
language_schedule / sequence_indices / stratified_order), only the constants
change: SEQUENCES_PER_RUN = 323,584 (331,350,016 tokens = 2.3160 tok/param,
matching the pilot's 2.3162 ratio) drawn from the enlarged pool
materialized_331m_143m (337,920 train sequences per language = 3.3x P0).

There is no pilot audit for 143M; the manifest records sha256 + per-source
counts as the reference for future reproduction.

Outputs:
  training/order_143m/<condition>_s<seed>.npz   (labels, indices int32 arrays)
  training/order_143m/schedule-<condition>-s<seed>.npy  (columns [label, index, source_code])
  training/order_143m/manifest.json             (per-run sha256 + source counts)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/mnt/models/CODE/mzy/umeko_reports/data/materialized_331m_143m")
OUT_DIR = ROOT / "training" / "order_143m"

SEQUENCES_PER_RUN = 323584  # p1_143m_proportional.json training.sequences_per_run
CONTEXT = 1024
SEEDS = (2027, 2028)
CONDITIONS = ("en_only", "zh_only", "iid_50", "en_first", "zh_first", "alternating")


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
    source_arrays = {
        lang: np.load(DATA_ROOT / lang / "train_sources.npy", mmap_mode="r")
        for lang in ("en", "zh")
    }
    materialization = json.loads((DATA_ROOT / "materialization_summary.json").read_text(encoding="utf-8"))
    source_vocabulary = materialization["source_vocabulary"]
    train_sequences_per_language = materialization["languages"]["en"]["filled_sequences"]["train"]
    if train_sequences_per_language < SEQUENCES_PER_RUN:
        raise RuntimeError(
            f"pool has {train_sequences_per_language} sequences per language, "
            f"need at least {SEQUENCES_PER_RUN}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"sequences_per_run": SEQUENCES_PER_RUN, "context": CONTEXT, "runs": {}}
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
            np.savez(
                OUT_DIR / f"order_{condition}_s{seed}.npz",
                labels=labels.astype(np.int8),
                indices=indices,
            )
            manifest["runs"][f"{condition}-s{seed}"] = {
                "schedule_sha256": schedule_hash,
                "source_sequence_counts": {
                    source_id: int((source_codes == code).sum())
                    for source_id, code in source_vocabulary.items()
                },
            }
            print(f"{condition}-s{seed}: schedule sha256 {schedule_hash[:16]}...", flush=True)
    tmp = OUT_DIR / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, OUT_DIR / "manifest.json")
    print(f"wrote {OUT_DIR / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
