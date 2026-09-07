"""Aggregate the P1 143M NPU runs and compare against the 43M results.

Reads bilingual_npu_cluster/training/runs_143m/<condition>-s<seed>/evals/*.json,
computes the pilot metrics (test BPB per language, dev-curve forgetting =
final dev BPB - best dev BPB), averages over seeds (sample SD), and compares:
  1. vs the 43M NPU reproduction (training/runs, same machine/framework)
  2. vs the original 2080 Ti pilot (final_metrics_per_seed.csv) for reference

Outputs training/results/comparison_143m.json + comparison_143m.md
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path("/mnt/models/CODE/mzy/bilingual_npu_cluster/training")
RUNS_143M = ROOT / "runs_143m"
RUNS_43M = ROOT / "runs"
PILOT_CSV = Path(
    "/mnt/models/CODE/mzy/umeko_reports/reports/p0_scaled_100m_all_sources_2080ti"
    "/tables/final_metrics_per_seed.csv"
)
OUT = ROOT / "results"
CONDITIONS = ("en_only", "zh_only", "iid_50", "en_first", "zh_first", "alternating")
SEEDS = (2027, 2028)
METRICS = (
    ("en_test_bpb", "英文测试BPB"),
    ("zh_test_bpb", "中文测试BPB"),
    ("mean_test_bpb", "平均测试BPB"),
    ("en_forgetting_bpb", "英文遗忘"),
    ("zh_forgetting_bpb", "中文遗忘"),
)


def load_run(runs_root: Path, condition: str, seed: int) -> dict:
    evals = runs_root / f"{condition}-s{seed}" / "evals"
    dev = {}
    for path in sorted(evals.glob("dev_step*.json")):
        step = int(path.stem.replace("dev_step", ""))
        dev[step] = json.loads(path.read_text())
    test_files = sorted(evals.glob("test_step*.json"))
    test = json.loads(test_files[-1].read_text()) if test_files else None
    return {"dev": dev, "test": test}


def run_metrics(run: dict) -> dict:
    steps = sorted(run["dev"])
    final_step = steps[-1]
    metrics = {}
    for lang in ("en", "zh"):
        curve = [run["dev"][s][lang]["bits_per_byte"] for s in steps]
        final_dev = run["dev"][final_step][lang]["bits_per_byte"]
        metrics[f"{lang}_dev_final_bpb"] = final_dev
        metrics[f"{lang}_dev_best_bpb"] = min(curve)
        metrics[f"{lang}_forgetting_bpb"] = final_dev - min(curve)
        metrics[f"{lang}_test_bpb"] = run["test"][lang]["bits_per_byte"] if run["test"] else None
    if run["test"]:
        metrics["mean_test_bpb"] = (metrics["en_test_bpb"] + metrics["zh_test_bpb"]) / 2
    return metrics


def mean_sd(values):
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, sd


def main() -> None:
    pilot = {}
    with PILOT_CSV.open() as handle:
        for row in csv.DictReader(handle):
            pilot[(row["condition"], int(row["seed"]))] = {
                key: float(value) for key, value in row.items() if key not in ("condition", "seed")
            }

    per_seed_143, per_seed_43, missing = {}, {}, []
    for condition in CONDITIONS:
        for seed in SEEDS:
            try:
                per_seed_143[(condition, seed)] = run_metrics(load_run(RUNS_143M, condition, seed))
            except (FileNotFoundError, IndexError):
                missing.append(f"143m:{condition}-s{seed}")
            try:
                per_seed_43[(condition, seed)] = run_metrics(load_run(RUNS_43M, condition, seed))
            except (FileNotFoundError, IndexError):
                missing.append(f"43m:{condition}-s{seed}")

    OUT.mkdir(exist_ok=True)
    lines = [
        "| 条件 | 指标 | 43M 初探 (2080Ti) | 43M NPU | 143M NPU | 143M-43M(NPU) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    comparison = {}
    for condition in CONDITIONS:
        if (condition, SEEDS[0]) not in per_seed_143 or (condition, SEEDS[1]) not in per_seed_143:
            continue
        for metric, label in METRICS:
            v143 = [per_seed_143[(condition, s)][metric] for s in SEEDS]
            v43 = [per_seed_43[(condition, s)][metric] for s in SEEDS]
            ref = [pilot[(condition, s)][metric] for s in SEEDS]
            m143, sd143 = mean_sd(v143)
            m43, sd43 = mean_sd(v43)
            mp, sdp = mean_sd(ref)
            comparison[f"{condition}/{metric}"] = {
                "npu_143m_mean": m143, "npu_143m_sd": sd143,
                "npu_43m_mean": m43, "npu_43m_sd": sd43,
                "pilot_43m_mean": mp, "pilot_43m_sd": sdp,
                "delta_143m_minus_43m": m143 - m43,
            }
            lines.append(
                f"| {condition} | {label} | {mp:.4f} ± {sdp:.4f} "
                f"| {m43:.4f} ± {sd43:.4f} | {m143:.4f} ± {sd143:.4f} | {m143 - m43:+.4f} |"
            )

    result = {
        "missing_runs": missing,
        "per_seed_143m": {f"{c}-s{s}": m for (c, s), m in per_seed_143.items()},
        "comparison": comparison,
    }
    (OUT / "comparison_143m.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "comparison_143m.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if missing:
        print("missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
