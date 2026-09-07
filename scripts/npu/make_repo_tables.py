"""Package the NPU experiments (P0 43M repro + P1 143M scale-up) into the
umeko_reports GitHub repo, following the pilot's reports/<experiment>/ layout.

For each experiment:
  tables/final_metrics_per_seed.csv   (pilot CSV schema)
  tables/final_metrics_aggregate.csv  (pilot CSV schema)
  report_manifest.json                (run ids + sha256 of tables/figures/reports)

Raw metrics come from bilingual_npu_cluster/training/{runs,runs_143m}; this
script only copies/computes small artifacts. Large data (pools, checkpoints,
logs) stays on the cluster, per the repo's data/ gitignore convention.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path

TRAIN = Path("/mnt/models/CODE/mzy/bilingual_npu_cluster/training")
REPO = Path("/mnt/models/CODE/mzy/umeko_reports")
CONDITIONS = ("en_only", "zh_only", "iid_50", "en_first", "zh_first", "alternating")
SEEDS = (2027, 2028)

EXPERIMENTS = {
    "p0_43m_npu_repro": {
        "runs": TRAIN / "runs",
        "tokens": 100663296,
        "run_prefix": "p0_43m_npu_repro",
    },
    "p1_143m_proportional_npu": {
        "runs": TRAIN / "runs_143m",
        "tokens": 331350016,
        "run_prefix": "p1_143m_proportional_npu",
    },
}

ITER_RE = re.compile(r"elapsed time per iteration \(ms\): ([0-9.]+)")
MEM_RE = re.compile(r"max allocated: ([0-9.]+)")
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def run_stats(run_dir: Path) -> dict:
    log = (run_dir / "train.log").read_text(errors="replace")
    iters = [float(m) for m in ITER_RE.findall(log)]
    iters = iters[1:] if iters and iters[0] > 3000 else iters  # drop first-step warmup outlier
    mems = [float(m) for m in MEM_RE.findall(log)]
    ts = TS_RE.findall(log)
    elapsed_min = None
    if len(ts) >= 2:
        t0 = datetime.strptime(ts[0], "%Y-%m-%d %H:%M:%S")
        t1 = datetime.strptime(ts[-1], "%Y-%m-%d %H:%M:%S")
        elapsed_min = (t1 - t0).total_seconds() / 60.0
    med_ms = sorted(iters)[len(iters) // 2] if iters else float("nan")
    return {
        "median_tokens_per_second": 32768.0 / (med_ms / 1000.0),
        "peak_memory_gib": max(mems) / 1024.0 if mems else float("nan"),
        "elapsed_wall_clock_minutes": elapsed_min,
    }


def run_metrics(runs_root: Path, condition: str, seed: int) -> dict:
    run_dir = runs_root / f"{condition}-s{seed}"
    evals = run_dir / "evals"
    dev = {}
    for path in sorted(evals.glob("dev_step*.json")):
        dev[int(path.stem.replace("dev_step", ""))] = json.loads(path.read_text())
    test = json.loads(sorted(evals.glob("test_step*.json"))[-1].read_text())
    steps = sorted(dev)
    out = {}
    for lang in ("en", "zh"):
        curve = [dev[s][lang]["bits_per_byte"] for s in steps]
        out[f"{lang}_test_bpb"] = test[lang]["bits_per_byte"]
        out[f"{lang}_forgetting_bpb"] = curve[-1] - min(curve)
    out["mean_test_bpb"] = (out["en_test_bpb"] + out["zh_test_bpb"]) / 2
    out.update(run_stats(run_dir))
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 23), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_sd(values):
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0


def main() -> None:
    for exp_name, cfg in EXPERIMENTS.items():
        exp_dir = REPO / "reports" / exp_name
        tables = exp_dir / "tables"
        tables.mkdir(parents=True, exist_ok=True)

        rows = []
        for condition in CONDITIONS:
            for seed in SEEDS:
                m = run_metrics(cfg["runs"], condition, seed)
                rows.append({
                    "condition": condition,
                    "seed": seed,
                    "tokens": cfg["tokens"],
                    **m,
                    "thermal_pacing_ms_per_update": 0.0,
                })

        per_seed_cols = ["condition", "seed", "tokens", "en_test_bpb", "zh_test_bpb",
                         "mean_test_bpb", "en_forgetting_bpb", "zh_forgetting_bpb",
                         "median_tokens_per_second", "peak_memory_gib",
                         "elapsed_wall_clock_minutes", "thermal_pacing_ms_per_update"]
        per_seed_path = tables / "final_metrics_per_seed.csv"
        with per_seed_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=per_seed_cols)
            writer.writeheader()
            writer.writerows(rows)

        metric_cols = ["en_test_bpb", "zh_test_bpb", "mean_test_bpb", "en_forgetting_bpb",
                       "zh_forgetting_bpb", "median_tokens_per_second", "peak_memory_gib"]
        agg_path = tables / "final_metrics_aggregate.csv"
        with agg_path.open("w", newline="") as fh:
            cols = ["condition", "n_seeds"] + [f"{c}_{s}" for c in metric_cols for s in ("mean", "sample_std")]
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for condition in CONDITIONS:
                crows = [r for r in rows if r["condition"] == condition]
                row = {"condition": condition, "n_seeds": len(crows)}
                for c in metric_cols:
                    vals = [r[c] for r in crows]
                    row[f"{c}_mean"] = sum(vals) / len(vals)
                    row[f"{c}_sample_std"] = sample_sd(vals)
                writer.writerow(row)

        manifest = {
            "status": "complete",
            "complete_run_ids": [f"{cfg['run_prefix']}-{c}-s{s}" for c in CONDITIONS for s in SEEDS],
            "incomplete_run_ids": [],
            "tables": {p.name: sha256_file(p) for p in sorted(tables.glob("*"))},
        }
        # figures + reports are copied separately; hash whatever exists now
        for sub, key in ((exp_dir / "figures", "figures"), (exp_dir, "reports")):
            if sub.exists():
                manifest.setdefault(key, {})
                for p in sorted(sub.glob("*")):
                    if p.is_file() and p.name != "report_manifest.json" and (key != "reports" or p.suffix == ".md"):
                        manifest[key][p.name] = sha256_file(p)
        (exp_dir / "report_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{exp_name}: wrote {per_seed_path.name}, {agg_path.name}, report_manifest.json")


if __name__ == "__main__":
    main()
