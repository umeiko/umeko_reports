"""Audit and analyze the five-size Ultra-FineWeb quality-filtering sweep.

The source repository contains one baseline and one high-quality-filtered run
at each of five model sizes, evaluated repeatedly from about 1B to 20B tokens.
This script treats the sweep as exploratory: checkpoints from one run are not
independent replicates, and no seed-based confidence interval is available.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
SOURCE = WORKSPACE / "EXPS"
OUT = ROOT / "reports" / "exps_quality_scale"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

SIZES = [56, 143, 263, 469, 803]
CONDITIONS = {"base": "ultra-en", "hq": "ultra-en-hq"}
TASKS = [
    "mmlu",
    "arc_easy",
    "arc_challenge",
    "sciq",
    "openbookqa",
    "hellaswag",
    "piqa",
    "siqa",
    "winogrande",
    "csqa",
]
TASK_LABELS = {
    "mmlu": "MMLU",
    "arc_easy": "ARC-E",
    "arc_challenge": "ARC-C",
    "sciq": "SciQ",
    "openbookqa": "OpenBookQA",
    "hellaswag": "HellaSwag",
    "piqa": "PIQA",
    "siqa": "SIQA",
    "winogrande": "WinoGrande",
    "csqa": "CSQA",
}
ORIGINAL_AVERAGE_TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa"]
FAMILIES = {
    "knowledge_science": ["mmlu", "arc_easy", "arc_challenge", "sciq", "openbookqa"],
    "commonsense_social": ["hellaswag", "piqa", "siqa", "winogrande", "csqa"],
}
STAGES = {
    "1-5B": (0.0, 5.0),
    "5-10B": (5.0, 10.0),
    "10-15B": (10.0, 15.0),
    "15-20B": (15.0, 20.1),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(size: int) -> Path:
    return SOURCE / f"comparison_{size}M-ultra-en-vs-hq" / "comparison_results.json"


def macro(result: dict, metric: str, tasks: list[str] = TASKS) -> float:
    return float(np.mean([result[task][metric] for task in tasks]))


def fit_log_size(points: pd.DataFrame, value_col: str) -> tuple[float, float, float]:
    x = np.log(points["parameters_m"].to_numpy(dtype=float))
    y = points[value_col].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    denominator = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if denominator == 0 else 1 - np.sum((y - predicted) ** 2) / denominator
    return float(slope), float(intercept), float(r2)


def load_and_audit() -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    audit: dict = {
        "source_repository": "https://huggingface.co/DALabCommunity/EXPS",
        "source_commit": "",
        "input_sha256": {},
        "sizes_m": SIZES,
        "conditions": CONDITIONS,
        "tasks": TASKS,
        "warnings": [],
    }

    import subprocess

    audit["source_commit"] = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()

    checkpoint_counts: dict[str, dict[str, int]] = {}
    pair_tokens: dict[int, list[float]] = {}
    average_errors: list[float] = []
    missing_records: list[dict] = []

    for size in SIZES:
        path = source_path(size)
        if not path.exists():
            raise FileNotFoundError(path)
        audit["input_sha256"][f"{size}M"] = sha256(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if set(payload) != set(CONDITIONS.values()):
            raise ValueError(f"Unexpected conditions at {size}M: {sorted(payload)}")

        indexed: dict[str, dict[float, dict]] = {}
        checkpoint_counts[f"{size}M"] = {}
        for short, source_key in CONDITIONS.items():
            sequence = payload[source_key]
            checkpoint_counts[f"{size}M"][short] = len(sequence)
            current = {round(float(item["tokens_B"]), 3): item for item in sequence}
            if len(current) != len(sequence):
                raise ValueError(f"Duplicate token checkpoints at {size}M/{short}")
            indexed[short] = current

        shared_tokens = sorted(set(indexed["base"]) & set(indexed["hq"]))
        if set(indexed["base"]) != set(indexed["hq"]):
            raise ValueError(f"Unpaired checkpoints at {size}M")
        if shared_tokens[-1] != 19.999:
            raise ValueError(f"Expected final checkpoint at 19.999B for {size}M")
        pair_tokens[size] = shared_tokens

        for tokens_b in shared_tokens:
            base = indexed["base"][tokens_b]["results"]
            hq = indexed["hq"][tokens_b]["results"]
            for condition_name, result in [("base", base), ("hq", hq)]:
                if not set(TASKS).issubset(result):
                    missing = sorted(set(TASKS) - set(result))
                    missing_records.append(
                        {
                            "parameters_m": size,
                            "condition": condition_name,
                            "tokens_b": tokens_b,
                            "missing_tasks": missing,
                        }
                    )
                for task in set(TASKS) & set(result):
                    for metric in ["acc", "acc_norm", "n"]:
                        if metric not in result[task]:
                            raise ValueError(
                                f"Missing {metric} at {size}M/{condition_name}/{tokens_b}B/{task}"
                            )

            expected_base_average = macro(base, "acc", ORIGINAL_AVERAGE_TASKS)
            expected_hq_average = macro(hq, "acc", ORIGINAL_AVERAGE_TASKS)
            average_errors.extend(
                [
                    abs(expected_base_average - float(base["average"]["acc"])),
                    abs(expected_hq_average - float(hq["average"]["acc"])),
                ]
            )

            for task in [task for task in TASKS if task in base and task in hq]:
                rows.append(
                    {
                        "parameters_m": size,
                        "tokens_b": tokens_b,
                        "task": task,
                        "task_label": TASK_LABELS[task],
                        "n": int(base[task]["n"]),
                        "base_acc": float(base[task]["acc"]),
                        "hq_acc": float(hq[task]["acc"]),
                        "gap_acc_pp": 100 * (float(hq[task]["acc"]) - float(base[task]["acc"])),
                        "base_acc_norm": float(base[task]["acc_norm"]),
                        "hq_acc_norm": float(hq[task]["acc_norm"]),
                        "gap_acc_norm_pp": 100
                        * (float(hq[task]["acc_norm"]) - float(base[task]["acc_norm"])),
                    }
                )

    common_tokens = sorted(set.intersection(*(set(values) for values in pair_tokens.values())))
    audit["checkpoint_counts"] = checkpoint_counts
    audit["paired_checkpoint_count"] = int(sum(len(values) for values in pair_tokens.values()))
    audit["common_checkpoint_count_across_sizes"] = len(common_tokens)
    audit["common_tokens_b"] = common_tokens
    audit["missing_records"] = missing_records
    audit["original_average_definition"] = {
        "tasks": ORIGINAL_AVERAGE_TASKS,
        "metric": "unweighted macro mean of raw acc",
        "max_absolute_reconstruction_error": max(average_errors),
        "note": "The stored field named average is a four-task macro, not a ten-task macro.",
    }
    audit["warnings"] = [
        "One training run per size and data condition; no seed-based uncertainty is available.",
        "Repeated checkpoints from the same run are not independent replicates.",
        "No item-level predictions are included, so paired benchmark confidence intervals cannot be reconstructed.",
        "No ordered data manifest, unique-document count, or train-loss log is included in the current checkout.",
        "README describes the baseline as unfiltered in the design section but as score >= 0.5 in the conclusion; the exact baseline rule requires provenance clarification.",
        "The mechanism claim that filtering removes 67% of data is stated in README but not auditable from included artifacts.",
        "The stored average field covers four tasks, despite ten task records being present.",
    ]
    frame = pd.DataFrame(rows).sort_values(["parameters_m", "tokens_b", "task"]).reset_index(drop=True)
    complete_counts = frame.groupby(["parameters_m", "tokens_b"])["task"].nunique()
    audit["complete_macro10_checkpoint_pairs"] = int((complete_counts == len(TASKS)).sum())
    return frame, audit


def summarize(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    macro_rows: list[dict] = []
    for (size, tokens_b), part in frame.groupby(["parameters_m", "tokens_b"], sort=True):
        if part["task"].nunique() != len(TASKS):
            continue
        row = {
            "parameters_m": size,
            "tokens_b": tokens_b,
            "base_macro10_acc": part["base_acc"].mean(),
            "hq_macro10_acc": part["hq_acc"].mean(),
            "gap_macro10_acc_pp": part["gap_acc_pp"].mean(),
            "base_macro10_acc_norm": part["base_acc_norm"].mean(),
            "hq_macro10_acc_norm": part["hq_acc_norm"].mean(),
            "gap_macro10_acc_norm_pp": part["gap_acc_norm_pp"].mean(),
        }
        original = part[part["task"].isin(ORIGINAL_AVERAGE_TASKS)]
        row.update(
            {
                "base_original_macro4_acc": original["base_acc"].mean(),
                "hq_original_macro4_acc": original["hq_acc"].mean(),
                "gap_original_macro4_acc_pp": original["gap_acc_pp"].mean(),
                "hq_task_wins_raw": int((part["gap_acc_pp"] > 0).sum()),
                "hq_task_ties_raw": int(np.isclose(part["gap_acc_pp"], 0).sum()),
            }
        )
        macro_rows.append(row)
    macro_frame = pd.DataFrame(macro_rows)

    final_tasks = frame[np.isclose(frame["tokens_b"], 19.999)].copy()
    final_macro = macro_frame[np.isclose(macro_frame["tokens_b"], 19.999)].copy()

    family_rows: list[dict] = []
    for size, part in final_tasks.groupby("parameters_m"):
        for family, tasks in FAMILIES.items():
            selected = part[part["task"].isin(tasks)]
            family_rows.append(
                {
                    "parameters_m": size,
                    "family": family,
                    "tasks": ";".join(tasks),
                    "base_macro_acc": selected["base_acc"].mean(),
                    "hq_macro_acc": selected["hq_acc"].mean(),
                    "gap_acc_pp": selected["gap_acc_pp"].mean(),
                }
            )
    family_frame = pd.DataFrame(family_rows)

    stage_rows: list[dict] = []
    for size, part in macro_frame.groupby("parameters_m"):
        for stage, (lower, upper) in STAGES.items():
            selected = part[(part["tokens_b"] > lower) & (part["tokens_b"] <= upper)]
            stage_rows.append(
                {
                    "parameters_m": size,
                    "stage": stage,
                    "n_checkpoints": len(selected),
                    "mean_gap_macro10_acc_pp": selected["gap_macro10_acc_pp"].mean(),
                    "positive_checkpoint_fraction": (selected["gap_macro10_acc_pp"] > 0).mean(),
                }
            )
    stage_frame = pd.DataFrame(stage_rows)

    regression_rows: list[dict] = []
    slope, intercept, r2 = fit_log_size(final_macro, "gap_macro10_acc_pp")
    regression_rows.append(
        {
            "scope": "final_macro10_raw",
            "tokens_b": 19.999,
            "slope_pp_per_ln_parameters": slope,
            "intercept_pp": intercept,
            "r2": r2,
        }
    )
    for task, part in final_tasks.groupby("task"):
        slope, intercept, r2 = fit_log_size(part, "gap_acc_pp")
        regression_rows.append(
            {
                "scope": f"final_task_{task}_raw",
                "tokens_b": 19.999,
                "slope_pp_per_ln_parameters": slope,
                "intercept_pp": intercept,
                "r2": r2,
            }
        )
    for tokens_b, part in macro_frame.groupby("tokens_b"):
        if len(part) != len(SIZES):
            continue
        slope, intercept, r2 = fit_log_size(part, "gap_macro10_acc_pp")
        regression_rows.append(
            {
                "scope": "checkpoint_macro10_raw",
                "tokens_b": tokens_b,
                "slope_pp_per_ln_parameters": slope,
                "intercept_pp": intercept,
                "r2": r2,
            }
        )
    regression_frame = pd.DataFrame(regression_rows)
    return macro_frame, final_tasks, final_macro, family_frame, stage_frame, regression_frame


def make_figure(
    macro_frame: pd.DataFrame, final_tasks: pd.DataFrame, final_macro: pd.DataFrame
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.8,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {56: "#4C78A8", 143: "#59A14F", 263: "#F28E2B", 469: "#B279A2", 803: "#E15759"}
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 6.05), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.05, wspace=0.06, hspace=0.08)

    ax = axes[0, 0]
    final_macro = final_macro.sort_values("parameters_m")
    ax.plot(
        final_macro["parameters_m"],
        100 * final_macro["base_macro10_acc"],
        "o-",
        color="#6B7280",
        linewidth=2,
        label="Full data",
    )
    ax.plot(
        final_macro["parameters_m"],
        100 * final_macro["hq_macro10_acc"],
        "s-",
        color="#2A9D8F",
        linewidth=2,
        label="Score ≥ 0.9",
    )
    ax.set_xscale("log")
    ax.set_xticks(SIZES, [str(size) for size in SIZES])
    ax.set_xlabel("Parameters (M, log scale)")
    ax.set_ylabel("10-task macro accuracy (%) ↑")
    ax.set_title("(a) Final score at 20B tokens")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.plot(
        final_macro["parameters_m"],
        final_macro["gap_macro10_acc_pp"],
        "o-",
        color="#2A9D8F",
        linewidth=2,
        label="Raw accuracy macro",
    )
    ax.plot(
        final_macro["parameters_m"],
        final_macro["gap_macro10_acc_norm_pp"],
        "s--",
        color="#4C78A8",
        linewidth=1.6,
        label="Normalized-accuracy macro",
    )
    ax.set_xscale("log")
    ax.set_xticks(SIZES, [str(size) for size in SIZES])
    ax.set_xlabel("Parameters (M, log scale)")
    ax.set_ylabel("HQ − full-data accuracy (pp)")
    ax.set_title("(b) Final quality-filtering gain")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    heat = (
        final_tasks.pivot(index="task", columns="parameters_m", values="gap_acc_pp")
        .loc[TASKS, SIZES]
    )
    limit = float(np.ceil(np.max(np.abs(heat.to_numpy()))))
    image = ax.imshow(heat.to_numpy(), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(SIZES)), [f"{size}M" for size in SIZES])
    ax.set_yticks(np.arange(len(TASKS)), [TASK_LABELS[task] for task in TASKS])
    for row in range(heat.shape[0]):
        for col in range(heat.shape[1]):
            value = heat.iloc[row, col]
            ax.text(col, row, f"{value:+.1f}", ha="center", va="center", fontsize=6.4)
    ax.set_title("(c) Task-level gap at 20B tokens (pp)")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("HQ − full data (pp)")

    ax = axes[1, 1]
    ax.axhline(0, color="#555555", linewidth=0.8)
    for size in SIZES:
        part = macro_frame[macro_frame["parameters_m"] == size].sort_values("tokens_b")
        ax.plot(
            part["tokens_b"],
            part["gap_macro10_acc_pp"],
            marker="o",
            markersize=2.7,
            linewidth=1.35,
            color=colors[size],
            label=f"{size}M",
        )
    ax.set_xlabel("Training tokens (B)")
    ax.set_ylabel("10-task HQ gain (pp)")
    ax.set_title("(d) Gain along each single training run")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3)

    fig.suptitle(
        "Exploratory data-quality scale sweep: fixed 20B-token budget",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(FIGURES / "fig_quality_scale_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        FIGURES / "fig_quality_scale_summary.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame, audit = load_and_audit()
    macro_frame, final_tasks, final_macro, family_frame, stage_frame, regression_frame = summarize(frame)

    frame.to_csv(TABLES / "paired_checkpoint_task_gaps.csv", index=False)
    macro_frame.to_csv(TABLES / "checkpoint_macro10_summary.csv", index=False)
    final_tasks.to_csv(TABLES / "final_task_scores.csv", index=False)
    final_macro.to_csv(TABLES / "final_macro_summary.csv", index=False)
    family_frame.to_csv(TABLES / "final_task_family_summary.csv", index=False)
    stage_frame.to_csv(TABLES / "training_stage_summary.csv", index=False)
    regression_frame.to_csv(TABLES / "scale_regression_diagnostics.csv", index=False)
    make_figure(macro_frame, final_tasks, final_macro)

    final_regression = regression_frame[regression_frame["scope"] == "final_macro10_raw"].iloc[0]
    checkpoint_regression = regression_frame[regression_frame["scope"] == "checkpoint_macro10_raw"]
    early = checkpoint_regression[checkpoint_regression["tokens_b"] < 10]
    late = checkpoint_regression[checkpoint_regression["tokens_b"] > 15]
    key_numbers = {
        "analysis_type": "exploratory_single_run_five_size_quality_filtering_sweep",
        "source_commit": audit["source_commit"],
        "models": len(SIZES) * len(CONDITIONS),
        "sizes_m": SIZES,
        "training_tokens_b_per_model": 20,
        "paired_size_checkpoint_count": audit["paired_checkpoint_count"],
        "macro10_positive_pairs": int((macro_frame["gap_macro10_acc_pp"] > 0).sum()),
        "macro10_total_pairs": len(macro_frame),
        "final_macro10_raw": final_macro[
            ["parameters_m", "base_macro10_acc", "hq_macro10_acc", "gap_macro10_acc_pp", "hq_task_wins_raw"]
        ].to_dict(orient="records"),
        "final_gap_log_size_regression": {
            "slope_pp_per_ln_parameters": float(final_regression["slope_pp_per_ln_parameters"]),
            "r2": float(final_regression["r2"]),
            "interpretation": "No monotonic final quality-gain trend with size is identified.",
        },
        "checkpoint_log_size_slope": {
            "early_lt_10b_mean": float(early["slope_pp_per_ln_parameters"].mean()),
            "late_gt_15b_mean": float(late["slope_pp_per_ln_parameters"].mean()),
            "note": "Signs vary across checkpoints; this does not support a stable early/late reversal claim.",
        },
        "original_average": audit["original_average_definition"],
        "limitations": audit["warnings"],
    }
    (TABLES / "key_numbers.json").write_text(
        json.dumps(key_numbers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "SOURCE_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    artifacts = {
        str(path.relative_to(OUT)).replace("\\", "/"): sha256(path)
        for path in sorted([*TABLES.glob("*"), *FIGURES.glob("*")])
    }
    (OUT / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "source_script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
                "source_script_sha256": sha256(Path(__file__)),
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
