"""Generate the figures for the P1 143M beginner report.

Reads dev/test eval JSONs from training/runs (43M NPU) and training/runs_143m
(143M NPU), averages over seeds 2027/2028, and writes PNGs to
training/results/figures/. Labels are English (no CJK font on this machine);
Chinese captions live in the report.

Figures:
  fig_dev_en.png    EN dev BPB curves, 6 conditions, 143M solid vs 43M dashed
  fig_dev_zh.png    same for ZH
  fig_forgetting.png  en_first->EN and zh_first->ZH dev curves with forgetting annotated
  fig_test_bar.png  final mean test BPB per condition (43M vs 143M) + forgetting bars
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mnt/models/CODE/mzy/bilingual_npu_cluster/training")
OUT = ROOT / "results" / "figures"
CONDITIONS = ("en_only", "zh_only", "iid_50", "en_first", "zh_first", "alternating")
SEEDS = (2027, 2028)
SCALES = (("43M", ROOT / "runs", 3072), ("143M", ROOT / "runs_143m", 10112))
COLORS = {
    "en_only": "#d62728", "zh_only": "#ff7f0e", "iid_50": "#1f77b4",
    "en_first": "#2ca02c", "zh_first": "#9467bd", "alternating": "#8c564b",
}


def dev_curve(runs_root: Path, condition: str, seed: int, lang: str):
    evals = runs_root / f"{condition}-s{seed}" / "evals"
    steps, vals = [], []
    for path in sorted(evals.glob("dev_step*.json")):
        steps.append(int(path.stem.replace("dev_step", "")))
        vals.append(json.loads(path.read_text())[lang]["bits_per_byte"])
    return np.array(steps), np.array(vals)


def mean_curve(runs_root: Path, condition: str, lang: str, final: int):
    curves, steps = [], None
    for seed in SEEDS:
        s, v = dev_curve(runs_root, condition, seed, lang)
        steps, curves = s, curves + [v]
    return steps / final * 100.0, np.mean(curves, axis=0), np.std(curves, axis=0, ddof=1)


def plot_lang(lang: str, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for cond in CONDITIONS:
        for name, root, final in SCALES:
            x, m, sd = mean_curve(root, cond, lang, final)
            solid = name == "143M"
            ax.plot(x, m, color=COLORS[cond], ls="-" if solid else "--",
                    lw=1.8 if solid else 1.2, alpha=1.0 if solid else 0.55,
                    label=f"{cond} ({name})" if solid else None)
            if solid:
                ax.fill_between(x, m - sd, m + sd, color=COLORS[cond], alpha=0.12)
    # legend for conditions + one entry explaining dashes
    ax.plot([], [], color="gray", ls="--", lw=1.2, alpha=0.7, label="same condition, 43M (dashed)")
    ax.set_xlabel("training progress (%)")
    ax.set_ylabel(f"{lang.upper()} dev BPB (lower = better)")
    ax.set_title(f"{lang.upper()} dev BPB during training: 143M (solid) vs 43M (dashed)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_forgetting(path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    cases = (("en_first", "en", "EN dev BPB | en_first (EN first half, then ZH)"),
             ("zh_first", "zh", "ZH dev BPB | zh_first (ZH first half, then EN)"))
    for ax, (cond, lang, title) in zip(axes, cases):
        for name, root, final in SCALES:
            x, m, sd = mean_curve(root, cond, lang, final)
            color = "#d62728" if name == "143M" else "#7f7f7f"
            ax.plot(x, m, color=color, lw=1.8, label=f"{name} NPU")
            ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15)
            best_i, final_i = int(np.argmin(m)), len(m) - 1
            best, fin = m[best_i], m[final_i]
            forg = fin - best
            ax.plot([x[best_i]], [best], "o", color=color, ms=4)
            x_arrow = x[final_i] - (7 if name == "43M" else 0)
            ax.annotate("", xy=(x_arrow, fin), xytext=(x_arrow, best),
                        arrowprops=dict(arrowstyle="<->", color=color, lw=1.2))
            ax.text(x_arrow - 3.5, (fin + best) / 2, f"forgetting\n{forg:.3f}",
                    ha="right", va="center", fontsize=9, color=color)
        ax.axvline(50, color="k", ls=":", lw=1, alpha=0.5)
        ax.text(50.5, ax.get_ylim()[0], "language switch", rotation=90, va="bottom", fontsize=8, alpha=0.7)
        ax.set_xlabel("training progress (%)")
        ax.set_ylabel("BPB (lower = better)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_test_bar(path: Path):
    # mean test BPB per condition + forgetting bars
    mean_test, forg = {}, {}
    for name, root, final in SCALES:
        for cond in CONDITIONS:
            vals, f_en, f_zh = [], [], []
            for seed in SEEDS:
                evals = root / f"{cond}-s{seed}" / "evals"
                test = json.loads(sorted(evals.glob("test_step*.json"))[-1].read_text())
                vals.append((test["en"]["bits_per_byte"] + test["zh"]["bits_per_byte"]) / 2)
                for lang, acc in (("en", f_en), ("zh", f_zh)):
                    s, v = dev_curve(root, cond, seed, lang)
                    acc.append(v[-1] - v.min())
            mean_test[(name, cond)] = (np.mean(vals), np.std(vals, ddof=1))
            forg[(name, cond)] = (np.mean(f_en), np.mean(f_zh))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150,
                                   gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(CONDITIONS))
    w = 0.38
    for i, (name, _, _) in enumerate(SCALES):
        m = [mean_test[(name, c)][0] for c in CONDITIONS]
        sd = [mean_test[(name, c)][1] for c in CONDITIONS]
        ax1.bar(x + (i - 0.5) * w, m, w, yerr=sd, capsize=3,
                color="#7f7f7f" if name == "43M" else "#d62728", alpha=0.85, label=f"{name} NPU")
        for xi, mi in zip(x + (i - 0.5) * w, m):
            ax1.text(xi, mi + 0.03, f"{mi:.3f}", ha="center", fontsize=7)
    ax1.set_xticks(x, CONDITIONS, rotation=20, ha="right")
    ax1.set_ylabel("mean of EN/ZH test BPB (lower = better)")
    ax1.set_title("Final test BPB per condition")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend()

    pairs = [("en_first", "EN forgotten"), ("zh_first", "ZH forgotten"),
             ("alternating", "EN (alt)"), ("alternating", "ZH (alt)")]
    xp = np.arange(len(pairs))
    for i, (name, _, _) in enumerate(SCALES):
        vals = []
        for cond, which in pairs:
            en_m, zh_m = forg[(name, cond)]
            vals.append(en_m if which.startswith("EN") else zh_m)
        ax2.bar(xp + (i - 0.5) * w, vals, w, color="#7f7f7f" if name == "43M" else "#d62728",
                alpha=0.85, label=f"{name} NPU")
        for xi, vi in zip(xp + (i - 0.5) * w, vals):
            ax2.text(xi, vi + 0.015, f"{vi:.3f}", ha="center", fontsize=7)
    ax2.set_xticks(xp, [p[1] for p in pairs], rotation=15, ha="right")
    ax2.set_ylabel("forgetting = final dev BPB − best dev BPB")
    ax2.set_title("Forgetting grows with scale")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot_lang("en", OUT / "fig_dev_en.png")
    plot_lang("zh", OUT / "fig_dev_zh.png")
    plot_forgetting(OUT / "fig_forgetting.png")
    plot_test_bar(OUT / "fig_test_bar.png")
    for p in sorted(OUT.glob("*.png")):
        print("wrote", p)


if __name__ == "__main__":
    main()
