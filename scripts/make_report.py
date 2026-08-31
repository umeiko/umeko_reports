from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, load_config, sha256_file, write_json


COLORS = {
    "en_only": "#1f77b4",
    "zh_only": "#d62728",
    "iid_50": "#222222",
    "en_first": "#2ca02c",
    "zh_first": "#9467bd",
    "alternating": "#ff7f0e",
}


def read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluation_at_step(records: list[dict], step: int) -> dict:
    matches = [record for record in records if record["record_type"] == "evaluation" and record["step"] == step]
    if len(matches) != 1:
        raise ValueError(f"expected one evaluation at step {step}, found {len(matches)}")
    return matches[0]


def plot_metric_curves(logs: dict, conditions: list[str], seeds: list[int], figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for condition in conditions:
        condition_logs = [logs[(seed, condition)] for seed in seeds if (seed, condition) in logs]
        if not condition_logs:
            continue
        evaluations = [[record for record in records if record["record_type"] == "evaluation"] for records in condition_logs]
        steps = np.asarray([record["seen_tokens"] / 1e6 for record in evaluations[0]])
        for axis, language in zip(axes, ("en", "zh")):
            values = np.asarray([[record[language]["bits_per_byte"] for record in records] for records in evaluations])
            for seed_values in values:
                axis.plot(steps, seed_values, color=COLORS[condition], alpha=0.18, linewidth=0.9)
            axis.plot(steps, values.mean(axis=0), label=condition, color=COLORS[condition], linewidth=2.0)
            if len(values) > 1:
                axis.fill_between(steps, values.min(axis=0), values.max(axis=0), color=COLORS[condition], alpha=0.08)
            axis.set_title(f"{language.upper()} validation")
            axis.set_xlabel("Seen tokens (millions)")
            axis.set_ylabel("Bits per byte")
            axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "validation_bpb_curves.png", dpi=180)
    fig.savefig(figures / "validation_bpb_curves.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    for condition in conditions:
        trains = [
            [record for record in logs[(seed, condition)] if record["record_type"] == "train"]
            for seed in seeds if (seed, condition) in logs
        ]
        if not trains:
            continue
        steps = np.asarray([record["seen_tokens"] / 1e6 for record in trains[0]])
        values = np.asarray([[record["loss"] for record in records] for records in trains])
        for seed_values in values:
            axis.plot(steps, seed_values, color=COLORS[condition], alpha=0.12, linewidth=0.6)
        axis.plot(steps, values.mean(axis=0), label=condition, color=COLORS[condition], linewidth=1.4)
    axis.set_xlabel("Seen tokens (millions)")
    axis.set_ylabel("Training NLL / token")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "training_loss_curves.png", dpi=180)
    fig.savefig(figures / "training_loss_curves.pdf")
    plt.close(fig)


def plot_order_curves(logs: dict, seeds: list[int], config: dict, figures: Path) -> None:
    ordered = [condition for condition in ("iid_50", "en_first", "zh_first", "alternating") if any((seed, condition) in logs for seed in seeds)]
    if not ordered:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    block_tokens = config["training"]["tokens_per_run"] / 8 / 1e6
    for condition in ordered:
        evaluations = [
            [record for record in logs[(seed, condition)] if record["record_type"] == "evaluation"]
            for seed in seeds if (seed, condition) in logs
        ]
        steps = np.asarray([record["seen_tokens"] / 1e6 for record in evaluations[0]])
        for axis, language in zip(axes, ("en", "zh")):
            values = np.asarray([[record[language]["bits_per_byte"] for record in records] for records in evaluations])
            axis.plot(steps, values.mean(axis=0), label=condition, color=COLORS[condition], linewidth=2.0)
            if len(values) > 1:
                axis.fill_between(steps, values.min(axis=0), values.max(axis=0), color=COLORS[condition], alpha=0.1)
    for axis, language in zip(axes, ("en", "zh")):
        for boundary in range(1, 8):
            axis.axvline(boundary * block_tokens, color="#999999", linestyle=":", linewidth=0.8, alpha=0.6)
        axis.set_title(f"{language.upper()} order effect")
        axis.set_xlabel("Seen tokens (millions)")
        axis.set_ylabel("Bits per byte")
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "order_effect_bpb_curves.png", dpi=180)
    fig.savefig(figures / "order_effect_bpb_curves.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "p0_local.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    training = config["training"]
    seeds = training.get("seeds", [training.get("seed", 2027)])
    report_root = ROOT / "reports" / config["experiment_name"]
    figures = report_root / "figures"
    tables = report_root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    summaries: dict[tuple[int, str], dict] = {}
    logs: dict[tuple[int, str], list[dict]] = {}
    incomplete = []
    for seed in seeds:
        for condition in training["conditions"]:
            run_id = f"{config['experiment_name']}-{condition}-s{seed}"
            run_root = ROOT / "runs" / run_id
            summary_path = run_root / "summary.json"
            metrics_path = run_root / "metrics.jsonl"
            if not summary_path.exists() or not metrics_path.exists():
                incomplete.append(run_id)
                continue
            summaries[(seed, condition)] = json.loads(summary_path.read_text(encoding="utf-8"))
            logs[(seed, condition)] = read_metrics(metrics_path)

    if logs:
        plot_metric_curves(logs, training["conditions"], seeds, figures)
        plot_order_curves(logs, seeds, config, figures)

    rows = []
    for (seed, condition), summary in summaries.items():
        rows.append({
            "condition": condition,
            "seed": seed,
            "tokens": summary["tokens"],
            "en_test_bpb": summary["test"]["en"]["bits_per_byte"],
            "zh_test_bpb": summary["test"]["zh"]["bits_per_byte"],
            "mean_test_bpb": np.mean([summary["test"]["en"]["bits_per_byte"], summary["test"]["zh"]["bits_per_byte"]]),
            "en_forgetting_bpb": summary["forgetting_dev_bpb"]["en"],
            "zh_forgetting_bpb": summary["forgetting_dev_bpb"]["zh"],
            "median_tokens_per_second": summary["median_tokens_per_second_after_warmup"],
            "peak_memory_gib": summary["peak_memory_bytes"] / 2**30,
            # Wall time is retained as an operational diagnostic. It is not a
            # throughput metric because a suspended Windows host can inflate it.
            "elapsed_wall_clock_minutes": summary["elapsed_seconds"] / 60,
            "thermal_pacing_ms_per_update": summary.get("thermal_pacing_ms_per_update", 0.0),
        })
    rows.sort(key=lambda row: (training["conditions"].index(row["condition"]), row["seed"]))
    write_csv(tables / "final_metrics_per_seed.csv", rows)

    aggregate_rows = []
    for condition in training["conditions"]:
        condition_rows = [row for row in rows if row["condition"] == condition]
        if not condition_rows:
            continue
        aggregate = {"condition": condition, "n_seeds": len(condition_rows)}
        for field in ("en_test_bpb", "zh_test_bpb", "mean_test_bpb", "en_forgetting_bpb", "zh_forgetting_bpb", "median_tokens_per_second", "peak_memory_gib"):
            aggregate[f"{field}_mean"], aggregate[f"{field}_sample_std"] = mean_std([row[field] for row in condition_rows])
        aggregate_rows.append(aggregate)
    write_csv(tables / "final_metrics_aggregate.csv", aggregate_rows)

    matched_exposure_rows = []
    midpoint = training["optimizer_updates"] // 2
    for seed in seeds:
        required = ((seed, "en_only"), (seed, "zh_only"), (seed, "iid_50"))
        if not all(key in logs for key in required):
            continue
        iid_final = evaluation_at_step(logs[(seed, "iid_50")], training["optimizer_updates"])
        row = {"seed": seed}
        for language, monolingual_condition in (("en", "en_only"), ("zh", "zh_only")):
            mono_mid = evaluation_at_step(logs[(seed, monolingual_condition)], midpoint)[language]["bits_per_byte"]
            mono_final = evaluation_at_step(logs[(seed, monolingual_condition)], training["optimizer_updates"])[language]["bits_per_byte"]
            iid_value = iid_final[language]["bits_per_byte"]
            row[f"{language}_mono_50m_dev_bpb"] = mono_mid
            row[f"{language}_iid_50m_language_exposure_dev_bpb"] = iid_value
            row[f"{language}_mixing_delta_bpb"] = iid_value - mono_mid
            row[f"{language}_extra_50m_monolingual_gain_bpb"] = mono_mid - mono_final
        matched_exposure_rows.append(row)
    write_csv(tables / "matched_exposure_decomposition.csv", matched_exposure_rows)

    if rows:
        fig, axis = plt.subplots(figsize=(6, 5))
        for row in rows:
            axis.scatter(row["en_test_bpb"], row["zh_test_bpb"], s=32, color=COLORS[row["condition"]], alpha=0.45)
        for aggregate in aggregate_rows:
            axis.scatter(aggregate["en_test_bpb_mean"], aggregate["zh_test_bpb_mean"], s=70, color=COLORS[aggregate["condition"]], edgecolor="white", linewidth=0.8)
            axis.annotate(aggregate["condition"], (aggregate["en_test_bpb_mean"], aggregate["zh_test_bpb_mean"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        axis.set_xlabel("English test bits per byte (lower is better)")
        axis.set_ylabel("Chinese test bits per byte (lower is better)")
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / "final_bilingual_tradeoff.png", dpi=180)
        fig.savefig(figures / "final_bilingual_tradeoff.pdf")
        plt.close(fig)

    materialization_path = ROOT / "data" / config.get("materialized_subdir", "materialized") / "materialization_summary.json"
    materialization = json.loads(materialization_path.read_text(encoding="utf-8")) if materialization_path.exists() else None
    complete = not incomplete
    lines = [
        "# RTX 2080 Ti 全来源中英双语放大实验报告",
        "",
        f"状态：**{'已完成' if complete else '进行中'}**。证据等级：**{len(seeds)} seed、本机 FP16 放大试验；仍不是集群确认性结果**。",
        "",
        "## 实验设置",
        "",
        f"- 每条件 `{training['tokens_per_run']:,}` tokens，`{len(seeds)}` 个 seed，共 `{len(training['conditions']) * len(seeds)}` 个计划运行。",
        f"- 模型参数量 `{config['model']['expected_parameters']:,}`，context `{config['local_data']['context_length']}`，全局 batch `{training['global_batch_sequences']}` 条序列。",
        "- 主双语条件固定 50:50 EN:ZH；各 seed 内四个双语顺序条件使用完全相同的序列多重集，只改变呈现顺序。",
        "- Haidass 模型卡没有披露原始五源配比、训练阶段边界以及 L3/Cosmopedia 具体 config；本实验使用公开、显式、可审计的受控配比，不声称复刻 400B-token 原训练。",
        "",
        "## 本地物化来源",
        "",
        "| Source | Language | Config | Train tokens in pool | Pinned revision |",
        "|---|---|---|---:|---|",
    ]
    if materialization and materialization.get("sources"):
        for source_id, source in materialization["sources"].items():
            lines.append(f"| `{source_id}` | {source['language']} | `{source.get('config')}` | {source['train_tokens']:,} | `{source['revision']}` |")
    lines.extend([
        "",
        "## 结果（mean ± sample SD）",
        "",
        "| Condition | Seeds | EN test BPB | ZH test BPB | Equal-language mean BPB | EN forgetting | ZH forgetting |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for aggregate in aggregate_rows:
        lines.append(
            f"| `{aggregate['condition']}` | {aggregate['n_seeds']} | "
            f"{aggregate['en_test_bpb_mean']:.4f} ± {aggregate['en_test_bpb_sample_std']:.4f} | "
            f"{aggregate['zh_test_bpb_mean']:.4f} ± {aggregate['zh_test_bpb_sample_std']:.4f} | "
            f"{aggregate['mean_test_bpb_mean']:.4f} ± {aggregate['mean_test_bpb_sample_std']:.4f} | "
            f"{aggregate['en_forgetting_bpb_mean']:.4f} ± {aggregate['en_forgetting_bpb_sample_std']:.4f} | "
            f"{aggregate['zh_forgetting_bpb_mean']:.4f} ± {aggregate['zh_forgetting_bpb_sample_std']:.4f} |"
        )
    if matched_exposure_rows:
        lines.extend([
            "",
            "## 等语言曝光分解（validation BPB）",
            "",
            "这里把单语模型在中点（约 50.33M 该语言 tokens）与 IID 模型终点（也约 50.33M/语言）比较。`mixing delta = IID − 单语中点`；正值表示在相同目标语言曝光量下，混合训练的 BPB 更高。该差值是受控的混合效应描述，不直接等同于因果意义上的“参数干扰”。",
            "",
            "| Language | Mono @ 50.33M | IID @ 50.33M/lang | Mixing delta | Extra mono 50.33M gain |",
            "|---|---:|---:|---:|---:|",
        ])
        for language in ("en", "zh"):
            mono_mean, mono_sd = mean_std([row[f"{language}_mono_50m_dev_bpb"] for row in matched_exposure_rows])
            iid_mean, iid_sd = mean_std([row[f"{language}_iid_50m_language_exposure_dev_bpb"] for row in matched_exposure_rows])
            delta_mean, delta_sd = mean_std([row[f"{language}_mixing_delta_bpb"] for row in matched_exposure_rows])
            gain_mean, gain_sd = mean_std([row[f"{language}_extra_50m_monolingual_gain_bpb"] for row in matched_exposure_rows])
            lines.append(
                f"| {language.upper()} | {mono_mean:.4f} ± {mono_sd:.4f} | {iid_mean:.4f} ± {iid_sd:.4f} | "
                f"{delta_mean:+.4f} ± {delta_sd:.4f} | {gain_mean:+.4f} ± {gain_sd:.4f} |"
            )
    paced_runs = [row for row in rows if row["thermal_pacing_ms_per_update"] > 0]
    lines.extend([
        "",
        "## 运行完整性与计时边界",
        "",
        "- 审计要求 12/12 状态完整、训练与模型哈希一致、无非有限 loss/gradient、FP16 scale 不下降、同 seed 共享初始化，以及四个双语条件共享完全相同的数据多重集。",
        "- CSV 中的 `elapsed_wall_clock_minutes` 仅为运维记录：部分运行跨越主机挂起，不能用于性能比较；性能诊断应使用每次更新的 active-compute median tokens/s。",
    ])
    if paced_runs:
        paced = ", ".join(f"`{row['condition']}-s{row['seed']}` ({row['thermal_pacing_ms_per_update']:.0f} ms/update)" for row in paced_runs)
        lines.append(
            f"- 温控重跑：{paced}。首次尝试在 GPU 触及 84°C 并报告软件温控降频后被人工停止并归档；重跑只在每次优化更新后休眠，不改变初始化、数据顺序或优化轨迹。"
        )
    if incomplete:
        lines.extend(["", "## 未完成运行", "", *[f"- `{run_id}`" for run_id in incomplete]])
    else:
        lines.extend([
            "",
            "## 解释边界",
            "",
            "双 seed 只用于检验方向是否在两次初始化下重复，并不足以支撑精确显著性结论。论文中的正式主张仍需预注册的多 seed 集群实验、外部双语任务和层级 bootstrap。",
        ])
    report_path = report_root / "EXPERIMENT_REPORT_CN.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "status": "complete" if complete else "in_progress",
        "complete_run_ids": [summary["run_id"] for summary in summaries.values()],
        "incomplete_run_ids": incomplete,
        "report_sha256": sha256_file(report_path),
        "tables": {path.name: sha256_file(path) for path in sorted(tables.glob("*.csv"))},
        "figures": {path.name: sha256_file(path) for path in sorted(figures.glob("*"))},
    }
    write_json(report_root / "report_manifest.json", manifest)
    print(f"wrote {report_path}; pending={len(incomplete)}")


if __name__ == "__main__":
    main()
