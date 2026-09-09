"""Recompute the proportional 43M-to-143M scale comparison from public CSVs.

This script deliberately treats the experiment as a two-point proportional
scale-up diagnostic, not as a fitted scaling law.  It requires both seeds and
all six registered conditions before producing any artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "scale_analysis_43m_143m"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

INPUTS = {
    "43M": ROOT
    / "reports"
    / "p0_43m_npu_repro"
    / "tables"
    / "final_metrics_per_seed.csv",
    "143M": ROOT
    / "reports"
    / "p1_143m_proportional_npu"
    / "tables"
    / "final_metrics_per_seed.csv",
}

PARAMETERS = {"43M": 43_461_504, "143M": 143_071_296}
EXPECTED_CONDITIONS = [
    "en_only",
    "zh_only",
    "iid_50",
    "en_first",
    "zh_first",
    "alternating",
]
EXPECTED_SEEDS = {2027, 2028}
LABELS = {
    "en_only": "EN only",
    "zh_only": "ZH only",
    "iid_50": "IID 50:50",
    "en_first": "EN→ZH blocks",
    "zh_first": "ZH→EN blocks",
    "alternating": "Alternating",
}
COLORS = {
    "en_only": "#4C78A8",
    "zh_only": "#E45756",
    "iid_50": "#2A9D8F",
    "en_first": "#F2A541",
    "zh_first": "#8E6C8A",
    "alternating": "#59A14F",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load() -> pd.DataFrame:
    frames = []
    for scale, path in INPUTS.items():
        frame = pd.read_csv(path)
        frame["scale"] = scale
        frame["parameters"] = PARAMETERS[scale]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    expected_rows = len(INPUTS) * len(EXPECTED_CONDITIONS) * len(EXPECTED_SEEDS)
    assert len(data) == expected_rows, (len(data), expected_rows)
    assert set(data["condition"]) == set(EXPECTED_CONDITIONS)
    for scale in INPUTS:
        part = data[data["scale"] == scale]
        for condition in EXPECTED_CONDITIONS:
            seeds = set(part.loc[part["condition"] == condition, "seed"].astype(int))
            assert seeds == EXPECTED_SEEDS, (scale, condition, seeds)
    assert not data[
        [
            "en_test_bpb",
            "zh_test_bpb",
            "mean_test_bpb",
            "en_forgetting_bpb",
            "zh_forgetting_bpb",
        ]
    ].isna().any().any()
    return data


def aggregate(data: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "en_test_bpb",
        "zh_test_bpb",
        "mean_test_bpb",
        "en_forgetting_bpb",
        "zh_forgetting_bpb",
        "median_tokens_per_second",
        "peak_memory_gib",
    ]
    agg = (
        data.groupby(["scale", "parameters", "condition"], sort=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in agg.columns
    ]
    return agg


def schedule_penalties(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale in ["43M", "143M"]:
        part = agg[agg["scale"] == scale].set_index("condition")
        iid = part.loc["iid_50"]
        for condition in ["alternating", "en_first", "zh_first"]:
            current = part.loc[condition]
            en_penalty = 100 * (
                current["en_test_bpb_mean"] / iid["en_test_bpb_mean"] - 1
            )
            zh_penalty = 100 * (
                current["zh_test_bpb_mean"] / iid["zh_test_bpb_mean"] - 1
            )
            rows.append(
                {
                    "scale": scale,
                    "parameters": PARAMETERS[scale],
                    "condition": condition,
                    "mean_penalty_vs_iid_pct": 100
                    * (current["mean_test_bpb_mean"] / iid["mean_test_bpb_mean"] - 1),
                    "en_penalty_vs_iid_pct": en_penalty,
                    "zh_penalty_vs_iid_pct": zh_penalty,
                    "balanced_relative_penalty_pct": (en_penalty + zh_penalty) / 2,
                }
            )
    return pd.DataFrame(rows)


def scale_gains(agg: pd.DataFrame) -> pd.DataFrame:
    wide = agg.pivot(index="condition", columns="scale")
    rows = []
    for condition in EXPECTED_CONDITIONS:
        for language, metric in [
            ("EN", "en_test_bpb_mean"),
            ("ZH", "zh_test_bpb_mean"),
            ("Mean", "mean_test_bpb_mean"),
        ]:
            old = float(wide.loc[condition, (metric, "43M")])
            new = float(wide.loc[condition, (metric, "143M")])
            rows.append(
                {
                    "condition": condition,
                    "language": language,
                    "bpb_43m": old,
                    "bpb_143m": new,
                    "absolute_reduction": old - new,
                    "relative_reduction_pct": 100 * (old - new) / old,
                }
            )
    result = pd.DataFrame(rows)

    iid_reduction = result[result["condition"] == "iid_50"].set_index("language")[
        "absolute_reduction"
    ]
    result["iid_scale_dividend_capture_pct"] = result.apply(
        lambda row: 100 * row["absolute_reduction"] / iid_reduction[row["language"]],
        axis=1,
    )
    return result


def make_figure(
    agg: pd.DataFrame, penalties: pd.DataFrame, gains: pd.DataFrame
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.9), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.06, wspace=0.08, hspace=0.08)

    # (a) Final bilingual mean BPB at each proportional scale.
    ax = axes[0, 0]
    x = np.arange(len(EXPECTED_CONDITIONS))
    width = 0.36
    for offset, scale, hatch in [(-width / 2, "43M", ""), (width / 2, "143M", "//")]:
        part = agg[agg["scale"] == scale].set_index("condition").loc[EXPECTED_CONDITIONS]
        ax.bar(
            x + offset,
            part["mean_test_bpb_mean"],
            width,
            yerr=part["mean_test_bpb_std"],
            capsize=2,
            color="#A7C7E7" if scale == "43M" else "#2A6FBB",
            edgecolor="white",
            linewidth=0.5,
            hatch=hatch,
            label=scale,
        )
    ax.set_xticks(x, [LABELS[c] for c in EXPECTED_CONDITIONS], rotation=28, ha="right")
    ax.set_ylabel("Mean test BPB ↓")
    ax.set_title("(a) Final mean BPB")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)

    # (b) Relative penalty of each bilingual ordering versus IID.
    ax = axes[0, 1]
    schedule_order = ["alternating", "en_first", "zh_first"]
    for condition in schedule_order:
        part = penalties[penalties["condition"] == condition].set_index("scale").loc[["43M", "143M"]]
        ax.plot(
            [0, 1],
            part["balanced_relative_penalty_pct"],
            marker="o",
            linewidth=2,
            color=COLORS[condition],
            label=LABELS[condition],
        )
        for xpos, value in enumerate(part["balanced_relative_penalty_pct"]):
            ax.text(xpos, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks([0, 1], ["43M / 101M tok", "143M / 331M tok"])
    ax.set_ylabel("Balanced relative BPB penalty (%) ↑")
    ax.set_title("(b) Ordering penalty vs. IID")
    ax.set_ylim(0, max(42, penalties["balanced_relative_penalty_pct"].max() + 5))
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    # (c) Forgetting at the end of training.
    ax = axes[1, 0]
    forgetting_specs = [
        ("en_first", "en_forgetting_bpb_mean", "EN→ZH: EN forgotten"),
        ("zh_first", "zh_forgetting_bpb_mean", "ZH→EN: ZH forgotten"),
        ("alternating", "en_forgetting_bpb_mean", "Alternating: EN"),
        ("alternating", "zh_forgetting_bpb_mean", "Alternating: ZH"),
    ]
    xf = np.arange(len(forgetting_specs))
    for offset, scale, color in [(-width / 2, "43M", "#C7C7C7"), (width / 2, "143M", "#E76F51")]:
        part = agg[agg["scale"] == scale].set_index("condition")
        values = [float(part.loc[c, metric]) for c, metric, _ in forgetting_specs]
        ax.bar(xf + offset, values, width, color=color, edgecolor="white", label=scale)
    ax.set_xticks(xf, [label for _, _, label in forgetting_specs], rotation=25, ha="right")
    ax.set_ylabel("Forgetting BPB ↓")
    ax.set_title("(c) End-of-training forgetting")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)

    # (d) Direction-wise relative BPB reduction from 43M to 143M.
    ax = axes[1, 1]
    heat = (
        gains[gains["language"].isin(["EN", "ZH"])]
        .pivot(index="condition", columns="language", values="relative_reduction_pct")
        .loc[EXPECTED_CONDITIONS, ["EN", "ZH"]]
    )
    image = ax.imshow(heat.to_numpy(), cmap="YlGnBu", vmin=0, vmax=20, aspect="auto")
    ax.set_xticks([0, 1], ["English", "Chinese"])
    ax.set_yticks(np.arange(len(EXPECTED_CONDITIONS)), [LABELS[c] for c in EXPECTED_CONDITIONS])
    for row in range(heat.shape[0]):
        for col in range(heat.shape[1]):
            value = heat.iloc[row, col]
            ax.text(col, row, f"{value:.1f}%", ha="center", va="center", color="white" if value > 11 else "black")
    ax.set_title("(d) Direction-wise scale gain")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Reduction (%) ↑")

    fig.suptitle(
        "Two-point proportional scale diagnostic (mean ± sample SD over 2 seeds)",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(FIGURES / "fig_scale_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        FIGURES / "fig_scale_summary.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load()
    agg = aggregate(data)
    penalties = schedule_penalties(agg)
    gains = scale_gains(agg)

    agg.to_csv(TABLES / "scale_metrics_aggregate.csv", index=False)
    penalties.to_csv(TABLES / "schedule_penalties_vs_iid.csv", index=False)
    gains.to_csv(TABLES / "scale_gains_by_language.csv", index=False)
    make_figure(agg, penalties, gains)

    agg_i = agg.set_index(["scale", "condition"])
    pen_i = penalties.set_index(["scale", "condition"])
    gain_i = gains.set_index(["condition", "language"])
    summary = {
        "analysis_type": "two_point_proportional_scale_diagnostic_not_scaling_law",
        "input_sha256": {scale: sha256(path) for scale, path in INPUTS.items()},
        "registered_rows": len(data),
        "conditions": EXPECTED_CONDITIONS,
        "seeds": sorted(EXPECTED_SEEDS),
        "parameter_ratio_143m_over_43m": PARAMETERS["143M"] / PARAMETERS["43M"],
        "token_ratio_143m_over_43m": float(
            data[data["scale"] == "143M"]["tokens"].iloc[0]
            / data[data["scale"] == "43M"]["tokens"].iloc[0]
        ),
        "iid_mean_bpb": {
            scale: float(agg_i.loc[(scale, "iid_50"), "mean_test_bpb_mean"])
            for scale in ["43M", "143M"]
        },
        "iid_mean_relative_reduction_pct": float(
            gain_i.loc[("iid_50", "Mean"), "relative_reduction_pct"]
        ),
        "mean_penalty_vs_iid_pct": {
            scale: {
                condition: float(
                    pen_i.loc[(scale, condition), "mean_penalty_vs_iid_pct"]
                )
                for condition in ["alternating", "en_first", "zh_first"]
            }
            for scale in ["43M", "143M"]
        },
        "preregistered_balanced_relative_penalty_pct": {
            scale: {
                condition: float(
                    pen_i.loc[(scale, condition), "balanced_relative_penalty_pct"]
                )
                for condition in ["alternating", "en_first", "zh_first"]
            }
            for scale in ["43M", "143M"]
        },
        "forgetting": {
            "en_first_english": {
                scale: float(
                    agg_i.loc[(scale, "en_first"), "en_forgetting_bpb_mean"]
                )
                for scale in ["43M", "143M"]
            },
            "zh_first_chinese": {
                scale: float(
                    agg_i.loc[(scale, "zh_first"), "zh_forgetting_bpb_mean"]
                )
                for scale in ["43M", "143M"]
            },
        },
        "scale_dividend_capture_pct": {
            "en_first_english": float(
                gain_i.loc[("en_first", "EN"), "iid_scale_dividend_capture_pct"]
            ),
            "zh_first_chinese": float(
                gain_i.loc[("zh_first", "ZH"), "iid_scale_dividend_capture_pct"]
            ),
            "alternating_english": float(
                gain_i.loc[("alternating", "EN"), "iid_scale_dividend_capture_pct"]
            ),
            "alternating_chinese": float(
                gain_i.loc[("alternating", "ZH"), "iid_scale_dividend_capture_pct"]
            ),
        },
        "limitations": [
            "Only two scale points; no scaling-law exponent is identified.",
            "Parameters, training tokens, updates, and pool size change together.",
            "Two seeds; sample SD is descriptive and not a reliable confidence interval.",
            "Public artifacts do not include per-document NLL, preventing document bootstrap.",
            "Intrinsic BPB only; no downstream-task result is available for these runs.",
        ],
    }
    (TABLES / "key_numbers.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    artifact_hashes = {
        str(path.relative_to(OUT)).replace("\\", "/"): sha256(path)
        for path in sorted(TABLES.glob("*"))
        if path.name != "artifact_manifest.json"
    }
    artifact_hashes.update(
        {
            str(path.relative_to(OUT)).replace("\\", "/"): sha256(path)
            for path in sorted(FIGURES.glob("*"))
        }
    )
    (OUT / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "source_script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
                "source_script_sha256": sha256(Path(__file__)),
                "artifacts": artifact_hashes,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
