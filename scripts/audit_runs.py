from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import ROOT, load_config, sha256_file, write_json


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "p0_local.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    training = config["training"]
    seeds = training.get("seeds", [training.get("seed", 2027)])
    failures: list[str] = []
    summaries = {}
    schedules = {}
    run_audits = {}

    for seed in seeds:
      for condition in training["conditions"]:
        run_id = f"{config['experiment_name']}-{condition}-s{seed}"
        run_root = ROOT / "runs" / run_id
        summary_path = run_root / "summary.json"
        status_path = run_root / "status.json"
        metrics_path = run_root / "metrics.jsonl"
        schedule_path = run_root / "schedule.npy"
        require(summary_path.exists(), f"{run_id}: summary missing", failures)
        require(status_path.exists(), f"{run_id}: status missing", failures)
        require(metrics_path.exists(), f"{run_id}: metrics missing", failures)
        require(schedule_path.exists(), f"{run_id}: schedule missing", failures)
        if not all(path.exists() for path in (summary_path, status_path, metrics_path, schedule_path)):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        schedule = np.load(schedule_path)
        records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        train_records = [record for record in records if record["record_type"] == "train"]
        evaluation_records = [record for record in records if record["record_type"] == "evaluation"]
        require(status.get("status") == "complete", f"{run_id}: status is not complete", failures)
        require(summary.get("status") == "complete", f"{run_id}: summary is not complete", failures)
        require(summary.get("seed") == seed, f"{run_id}: seed mismatch", failures)
        require(summary.get("tokens") == training["tokens_per_run"], f"{run_id}: token count mismatch", failures)
        require(len(train_records) == training["optimizer_updates"], f"{run_id}: train record count mismatch", failures)
        expected_steps = {0, training["optimizer_updates"]}
        expected_steps.update(range(training["evaluation_interval_updates"], training["optimizer_updates"] + 1, training["evaluation_interval_updates"]))
        require({record["step"] for record in evaluation_records} == expected_steps, f"{run_id}: evaluation steps mismatch", failures)
        expected_columns = 3 if config["dataset"].get("sources") else 2
        require(schedule.shape == (training["sequences_per_run"], expected_columns), f"{run_id}: schedule shape mismatch", failures)
        require(sha256_file(schedule_path) == summary.get("schedule_sha256"), f"{run_id}: schedule hash mismatch", failures)
        require(sha256_file(metrics_path) == summary.get("metrics_sha256"), f"{run_id}: metrics hash mismatch", failures)
        require(all(math.isfinite(record["loss"]) and math.isfinite(record["grad_norm"]) for record in train_records), f"{run_id}: non-finite train metric", failures)
        scales = [float(record["scale"]) for record in train_records]
        require(
            all(math.isfinite(scale) and scale > 0 for scale in scales),
            f"{run_id}: invalid FP16 scale",
            failures,
        )
        require(
            all(current >= previous for previous, current in zip(scales, scales[1:])),
            f"{run_id}: FP16 scale decreased (an overflow would have skipped an optimizer update)",
            failures,
        )
        require(train_records[-1]["seen_tokens"] == training["tokens_per_run"], f"{run_id}: final seen tokens mismatch", failures)
        model_artifact = summary.get("model_artifact", {})
        model_path = Path(model_artifact.get("path", ""))
        require(model_path.exists(), f"{run_id}: final model artifact missing", failures)
        if model_path.exists():
            require(sha256_file(model_path) == model_artifact.get("sha256"), f"{run_id}: final model hash mismatch", failures)
        if expected_columns == 3:
            vocabulary = json.loads((ROOT / "data" / config["materialized_subdir"] / "materialization_summary.json").read_text(encoding="utf-8"))["source_vocabulary"]
            observed_source_counts = {
                source_id: int((schedule[:, 2] == source_code).sum())
                for source_id, source_code in vocabulary.items()
            }
            require(observed_source_counts == summary.get("source_sequence_counts"), f"{run_id}: source counts mismatch", failures)
            if condition in ("iid_50", "en_first", "zh_first", "alternating"):
                require(all(count > 0 for count in observed_source_counts.values()), f"{run_id}: a declared source is absent", failures)
        summaries[run_id] = summary
        schedules[(seed, condition)] = schedule
        run_audits[run_id] = {
            "train_records": len(train_records),
            "evaluation_records": len(evaluation_records),
            "schedule_sha256": summary["schedule_sha256"],
            "metrics_sha256": summary["metrics_sha256"],
            "initialization_sha256": summary["initialization_sha256"],
            "final_language_tokens": train_records[-1]["language_counts"],
            "initial_fp16_scale": scales[0],
            "minimum_fp16_scale": min(scales),
            "maximum_fp16_scale": max(scales),
            "nonfinite_records": 0,
            "model_sha256": model_artifact.get("sha256"),
            "source_sequence_counts": summary.get("source_sequence_counts"),
        }

    if summaries:
        require(len({summary["config_sha256"] for summary in summaries.values()}) == 1, "config hash differs across conditions", failures)
        for seed in seeds:
            seed_summaries = [summary for summary in summaries.values() if summary["seed"] == seed]
            require(len({summary["initialization_sha256"] for summary in seed_summaries}) == 1, f"seed {seed}: initialization hash differs across conditions", failures)

    bilingual = ["iid_50", "en_first", "zh_first", "alternating"]
    for seed in seeds:
        if all((seed, condition) in schedules for condition in bilingual):
            reference_sets = None
            for condition in bilingual:
                schedule = schedules[(seed, condition)]
                counts = {language: int((schedule[:, 0] == code).sum()) for language, code in (("en", 0), ("zh", 1))}
                require(counts == {"en": training["sequences_per_run"] // 2, "zh": training["sequences_per_run"] // 2}, f"seed {seed} {condition}: not exactly 50:50", failures)
                condition_sets = {
                    code: set(schedule[schedule[:, 0] == code, 1].tolist())
                    for code in (0, 1)
                }
                if reference_sets is None:
                    reference_sets = condition_sets
                else:
                    require(condition_sets == reference_sets, f"seed {seed} {condition}: bilingual sequence multiset differs", failures)

    output = {
        "status": "valid" if not failures else "invalid",
        "scientific_status": config["scientific_status"],
        "failures": failures,
        "run_count": len(summaries),
        "expected_run_count": len(training["conditions"]) * len(seeds),
        "bilingual_document_multiset_equal": not any("multiset" in failure for failure in failures),
        "common_initialization": not any("initialization" in failure for failure in failures),
        "runs": run_audits,
    }
    write_json(ROOT / "reports" / f"{config['experiment_name']}_run_audit.json", output)
    write_json(ROOT / "reports" / "run_audit.json", output)
    print(json.dumps(output, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
