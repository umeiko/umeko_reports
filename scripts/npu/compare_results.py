"""Aggregate the NPU reproduction runs and compare against the 2080 Ti pilot.

Reads bilingual_npu_cluster/training/runs/<condition>-s<seed>/evals/*.json,
computes the pilot's metrics (test BPB per language, dev-curve forgetting =
final dev BPB - best dev BPB), averages over seeds (sample SD), and writes a
side-by-side comparison with the pilot's final_metrics_per_seed.csv.

Outputs training/results/comparison.json + comparison.md
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path("/mnt/models/CODE/mzy/bilingual_npu_cluster/training")
RUNS = ROOT / "runs"
PILOT_CSV = Path(
    "/mnt/models/CODE/mzy/umeko_reports/reports/p0_scaled_100m_all_sources_2080ti"
    "/tables/final_metrics_per_seed.csv"
)
OUT = ROOT / "results"
CONDITIONS = ("en_only", "zh_only", "iid_50", "en_first", "zh_first", "alternating")
SEEDS = (2027, 2028)


def load_run(condition: str, seed: int) -> dict:
    evals = RUNS / f"{condition}-s{seed}" / "evals"
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


def main() -> None:
    pilot = {}
    with PILOT_CSV.open() as handle:
        for row in csv.DictReader(handle):
            pilot[(row["condition"], int(row["seed"]))] = {
                key: float(value) for key, value in row.items() if key not in ("condition", "seed")
            }

    OUT.mkdir(exist_ok=True)
    per_seed = {}
    missing = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            try:
                per_seed[(condition, seed)] = run_metrics(load_run(condition, seed))
            except (FileNotFoundError, IndexError):
                missing.append(f"{condition}-s{seed}")

    def mean_sd(values):
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
        return mean, sd

    lines = [
        "| 条件 | 指标 | 初探 (2080Ti) | NPU 复现 | 差值 |",
        "|---|---|---:|---:|---:|",
    ]
    comparison = {}
    for condition in CONDITIONS:
        keys = [f"{condition}-s{s}" for s in SEEDS if (condition, s) in per_seed]
        if len(keys) < 2:
            continue
        for metric, label in (
            ("en_test_bpb", "英文测试BPB"),
            ("zh_test_bpb", "中文测试BPB"),
            ("mean_test_bpb", "平均测试BPB"),
            ("en_forgetting_bpb", "英文遗忘"),
            ("zh_forgetting_bpb", "中文遗忘"),
        ):
            ours = [per_seed[(condition, s)][metric] for s in SEEDS]
            ref = [pilot[(condition, s)][metric] for s in SEEDS]
            o_mean, o_sd = mean_sd(ours)
            r_mean, r_sd = mean_sd(ref)
            comparison[f"{condition}/{metric}"] = {
                "npu_mean": o_mean, "npu_sd": o_sd,
                "pilot_mean": r_mean, "pilot_sd": r_sd,
                "delta": o_mean - r_mean,
            }
            lines.append(
                f"| {condition} | {label} | {r_mean:.4f} ± {r_sd:.4f} "
                f"| {o_mean:.4f} ± {o_sd:.4f} | {o_mean - r_mean:+.4f} |"
            )

    result = {"missing_runs": missing, "per_seed": {f"{c}-s{s}": m for (c, s), m in per_seed.items()}, "comparison": comparison}
    (OUT / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if missing:
        print("missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
