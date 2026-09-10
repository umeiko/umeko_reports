"""Recompute final-checkpoint quality metrics from frozen existing JSONs.

Standard library only. This is a reanalysis, not new training or evaluation.
By default print JSON; --write generates the two current evidence artifacts;
--check verifies those artifacts against a fresh in-memory recomputation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean


SOURCE_COMMIT = "4d45f05583e7370fd452a99635d4c984a9eb9d5e"
SOURCE_REPOSITORY = "https://huggingface.co/DALabCommunity/EXPS"
EXPECTED_SHA256 = {
    56: "1858d4e7b9dae7e699bf69feb92f4c641fe0be485addd4587a873d05155a6763",
    143: "47f35194f60022257f9cf9d0672fa402ec3bbfd498f663b0a7e7dafadabb07fe",
    263: "6d2c39cb58379a008a3fcc19e52fda9278ea42c592be9661639450bc84672f98",
    469: "965433e08f6953bf9d78aac5626fb514fe8691e3d56d70525852c35684205dff",
    803: "850b7b1e6254d98b11c9ab5cc8dc5ee78924848e40fe02fc0c37f60b6e2e77af",
}
TASKS = (
    "mmlu", "arc_easy", "arc_challenge", "sciq", "openbookqa",
    "hellaswag", "piqa", "siqa", "winogrande", "csqa",
)
LABELS = {
    "mmlu": "MMLU", "arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge",
    "sciq": "SciQ", "openbookqa": "OpenBookQA", "hellaswag": "HellaSwag",
    "piqa": "PIQA", "siqa": "SIQA", "winogrande": "WinoGrande", "csqa": "CSQA",
}
WITHOUT_ARC = tuple(task for task in TASKS if task not in {"arc_easy", "arc_challenge"})
METRIC_LABELS = {
    "raw_macro10": "十任务 raw 宏平均（主口径）",
    "source_acc_norm_macro10": "十任务源 acc_norm 宏平均",
    "n_weighted_raw_micro_proxy": "按 n 加权 raw（micro 近似）",
    "raw_macro8_without_arc": "去 ARC-E/C 八任务 raw 宏平均",
}
LIMITATIONS = [
    "All results reuse existing aggregate JSONs; no model was trained or reevaluated.",
    "One run per size-condition; these calculations cannot estimate training-seed uncertainty.",
    "No item-level predictions or original harness configurations are available in these inputs.",
    "Source acc_norm is not chance-normalized accuracy; its exact historical scoring normalization unit is unverified.",
    "The n-weighted result is a summary-derived micro proxy, not independently recovered item-level micro accuracy.",
    "The raw macro10 remains primary; alternative weights and ARC exclusion are post-hoc sensitivity analyses.",
    "Equal task weights give ARC-Easy and ARC-Challenge 20% combined weight; this is not a claim that tasks are independent.",
    "Source accuracy rounding is preserved; no correctness counts are reconstructed by rounding n * acc.",
    "The baseline filtering threshold remains ambiguous in the source README.",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def triplet(base: float, hq: float) -> dict:
    return {"base_fraction": base, "hq_fraction": hq, "gap_pp": 100.0 * (hq - base)}


def sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def final_record(records: list[dict], label: str) -> dict:
    selected = [record for record in records if record.get("iter") == 38146]
    if len(selected) != 1:
        raise ValueError(f"Expected one iter=38146 record at {label}")
    final = selected[0]
    if final.get("tokens_B") != 19.999:
        raise ValueError(f"Expected tokens_B=19.999 at {label}")
    if any(record["iter"] > final["iter"] for record in records):
        raise ValueError(f"Selected record is not final at {label}")
    if not set(TASKS).issubset(final["results"]):
        raise ValueError(f"Missing final tasks at {label}")
    return final


def analyze(workspace: Path) -> dict:
    inputs = []
    scales = []
    flips = []
    expected_n = None
    for size, expected_hash in EXPECTED_SHA256.items():
        relative = f"EXPS/comparison_{size}M-ultra-en-vs-hq/comparison_results.json"
        path = workspace / relative
        observed_hash = digest(path)
        if observed_hash != expected_hash:
            raise ValueError(f"Frozen source hash mismatch: {relative}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if set(data) != {"ultra-en", "ultra-en-hq"}:
            raise ValueError(f"Unexpected conditions: {relative}")
        base_record = final_record(data["ultra-en"], f"{size}M/base")
        hq_record = final_record(data["ultra-en-hq"], f"{size}M/hq")
        base, hq = base_record["results"], hq_record["results"]
        task_rows = []
        sample_counts = {}
        for task in TASKS:
            for condition, result in (("base", base[task]), ("hq", hq[task])):
                if not isinstance(result["n"], int) or result["n"] <= 0:
                    raise ValueError(f"Invalid n: {size}/{condition}/{task}")
                for metric in ("acc", "acc_norm"):
                    value = float(result[metric])
                    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                        raise ValueError(f"Invalid {metric}: {size}/{condition}/{task}")
            if base[task]["n"] != hq[task]["n"]:
                raise ValueError(f"Unmatched n: {size}/{task}")
            sample_counts[task] = base[task]["n"]
            raw = triplet(base[task]["acc"], hq[task]["acc"])
            normalized = triplet(base[task]["acc_norm"], hq[task]["acc_norm"])
            changed = sign(raw["gap_pp"]) != sign(normalized["gap_pp"])
            row = {
                "task": task, "n": sample_counts[task],
                "raw": raw, "source_acc_norm": normalized,
                "gap_sign_changes": changed,
            }
            task_rows.append(row)
            if changed:
                flips.append({
                    "parameters_m": size, "task": task,
                    "raw_gap_pp": raw["gap_pp"],
                    "source_acc_norm_gap_pp": normalized["gap_pp"],
                })
        if expected_n is None:
            expected_n = sample_counts
        elif sample_counts != expected_n:
            raise ValueError(f"Task sample counts vary across final scales: {size}")
        total_n = sum(sample_counts.values())
        aggregates = {}
        for key, tasks, metric, weighted in (
            ("raw_macro10", TASKS, "acc", False),
            ("source_acc_norm_macro10", TASKS, "acc_norm", False),
            ("n_weighted_raw_micro_proxy", TASKS, "acc", True),
            ("raw_macro8_without_arc", WITHOUT_ARC, "acc", False),
        ):
            def aggregate(results: dict) -> float:
                if weighted:
                    return math.fsum(results[t][metric] * sample_counts[t] for t in tasks) / total_n
                return fmean(results[t][metric] for t in tasks)

            aggregates[key] = triplet(aggregate(base), aggregate(hq))
        scales.append({
            "parameters_m": size,
            "checkpoint": {"iter": 38146, "tokens_B": 19.999},
            "total_n": total_n,
            "task_count": len(task_rows),
            "raw_positive_task_count": sum(row["raw"]["gap_pp"] > 0.0 for row in task_rows),
            "source_acc_norm_positive_task_count": sum(row["source_acc_norm"]["gap_pp"] > 0.0 for row in task_rows),
            "aggregates": aggregates,
            "tasks": task_rows,
        })
        inputs.append({"path": relative, "sha256": observed_hash, "frozen_hash_verified": True})
    example = next(row for row in scales[0]["tasks"] if row["task"] == "hellaswag")
    return {
        "schema_version": 1,
        "analysis_kind": "existing_aggregate_results_reanalysis_no_training_or_reevaluation",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "script": "scripts/analysis/audit_quality_metric_sensitivity.py",
        "script_sha256": digest(Path(__file__).resolve()),
        "source_inputs": inputs,
        "primary_metric": "raw_macro10",
        "primary_tasks": list(TASKS),
        "primary_task_weights": {task: 0.1 for task in TASKS},
        "metric_definitions": {
            "raw_macro10": "mean_t(acc_t), equal weights over the ten fixed tasks",
            "source_acc_norm_macro10": "mean_t(source acc_norm_t), with no extra normalization",
            "n_weighted_raw_micro_proxy": "sum_t(n_t * acc_t) / sum_t(n_t); summary-derived micro proxy",
            "raw_macro8_without_arc": "mean_t(acc_t) over the eight tasks excluding ARC-Easy and ARC-Challenge",
            "gap_pp": "100 * (hq_fraction - base_fraction)",
        },
        "acc_norm_semantics": {
            "is_chance_normalized": False,
            "exact_historical_normalization_unit": "unverified; original harness configuration missing",
            "standard_lighteval_reference": "https://github.com/huggingface/lighteval/blob/v0.9.2/src/lighteval/metrics/metrics.py",
            "reference_scope": "Standard acc_norm applies answer-length normalization before correctness aggregation; this does not establish the historical run's exact metric or unit.",
            "counterexample": {
                "parameters_m": 56, "condition": "ultra-en", "task": "hellaswag",
                "raw_accuracy": example["raw"]["base_fraction"],
                "stored_acc_norm": example["source_acc_norm"]["base_fraction"],
                "uniform_four_choice_chance": 0.25,
                "chance_corrected_raw_accuracy": (example["raw"]["base_fraction"] - 0.25) / 0.75,
            },
        },
        "per_scale": scales,
        "sensitivity_summary": {
            "positive_scale_count_by_metric": {
                metric: sum(scale["aggregates"][metric]["gap_pp"] > 0.0 for scale in scales)
                for metric in METRIC_LABELS
            },
            "task_scale_cell_count": len(scales) * len(TASKS),
            "raw_vs_source_acc_norm_sign_change_count": len(flips),
            "sign_change_cells": flips,
        },
        "limitations": LIMITATIONS,
    }


def render_report(data: dict, workspace: Path) -> str:
    scales = data["per_scale"]
    counterexample = data["acc_norm_semantics"]["counterexample"]
    flip_count = data["sensitivity_summary"]["raw_vs_source_acc_norm_sign_change_count"]
    lines = [
        "# 质量指标敏感性：同一批已有结果的复算",
        "",
        "本报告没有训练新模型、没有重新运行评测，也没有增加随机种子。所有数字来自同一份已完成英文质量扫描的最终汇总 JSON。**固定十任务 raw 准确率等权宏平均为主口径**；源 acc_norm、按 n 加权和去除 ARC 的结果均为事后敏感性检查，不替换主结果，不称为新实验。",
        "",
        "## 来源、范围与复现",
        "",
        f"- 来源：[DALabCommunity/EXPS]({SOURCE_REPOSITORY}/tree/{SOURCE_COMMIT})，固定提交 `{SOURCE_COMMIT}`。",
        "- 输入：`EXPS/comparison_{56,143,263,469,803}M-ultra-en-vs-hq/comparison_results.json`；脚本内固定五份 SHA256，任何输入变化都会报错。",
        "- 最终记录固定为 `iter=38146`、`tokens_B=19.999`，所有规模的 full/HQ 十任务齐全、样本数匹配，每个规模总 n 为 35,412。",
        "- 当前重算不使用源 `average` 字段，因为它是四任务均值；不使用中途 checkpoint，也不涉及 263M/6.291B 缺 SciQ 的记录。",
        "- 结构化完整数值与哈希：[quality_metric_sensitivity.json](evidence/quality_metric_sensitivity.json)。",
        "",
        "在 `umeko_reports` 克隆的 `scripts/analysis` 目录执行（Python 3.10+，仅标准库）。将 `WORKSPACE_ROOT` 替换为同时包含 `EXPS`、`.repo-sync` 和 `.reports-sync` 的现有工作区根目录：",
        "",
        "```powershell",
        "python audit_quality_metric_sensitivity.py --workspace WORKSPACE_ROOT --write",
        "python audit_quality_metric_sensitivity.py --workspace WORKSPACE_ROOT --check",
        "```",
        "",
        "不带参数仅向标准输出打印 JSON；`--write` 只生成本报告及 `current/evidence/quality_metric_sensitivity.json`，不会改动历史脚本或原数据；`--check` 在内存中复算并逐字节比较两份产物。工作区可用 `--workspace` 显式指定。",
        "",
        "## 口径定义",
        "",
        "设 a(t,d) 是任务 t 在条件 d 的源 acc，z(t,d) 是源 acc_norm，n(t) 是源任务样本数：",
        "",
        "| 指标 | 计算公式 | 定位 |",
        "|---|---|---|",
        "| 十任务 raw 宏平均 | sum(a(t,d))/10 | 主口径，每任务权重 10% |",
        "| 十任务源 acc_norm 宏平均 | sum(z(t,d))/10 | 原字段敏感性；不添加随机基线校正 |",
        "| 按 n 加权 raw（micro 近似） | sum(n(t)*a(t,d))/sum(n(t)) | 基于汇总值的加权敏感性 |",
        "| 去 ARC 八任务 raw 宏平均 | sum(a(t,d), t 非 ARC-E/C)/8 | 事后检查 ARC 影响 |",
        "",
        "所有分差均为 `100*(HQ−full)`，单位是百分点 pp，不是相对改善百分比。按 n 加权只有在每个源 acc 确为对应样本集的逐题均值时才等于逐题 micro；缺少原 harness 配置和逐题输出，因此这里保留 micro 近似的限定。不会把 n*acc 四舍五入成“答对题数”。MMLU 作为一个已存任务整体参与宏平均；本脚本无法另行审计其内部科目聚合。",
        "",
        "## 五个最终规模的聚合结果",
        "",
        "下表 full/HQ 为准确率百分数，差为 pp；原始高精度值保存在 JSON。",
        "",
        "| 规模 | 指标 | full (%) | HQ (%) | HQ−full (pp) |",
        "|---:|---|---:|---:|---:|",
    ]
    for scale in scales:
        for metric, label in METRIC_LABELS.items():
            item = scale["aggregates"][metric]
            lines.append(f"| {scale['parameters_m']}M | {label} | {100*item['base_fraction']:.4f} | {100*item['hq_fraction']:.4f} | {item['gap_pp']:+.4f} |")
    lines += [
        "",
        "十任务 raw 主宏平均和十任务源 acc_norm 宏平均在五个规模均为正。143M 在按 n 加权后为负，去 ARC 后也为负；469M/803M 去 ARC 后正收益接近零。因此可以陈述“主口径在五个最终模型对上为正”，但不能把它扩大为对任意合理权重或全部任务均稳健的收益。ARC 两任务的联合剔除是诊断，不是预注册检验；按 n 加权也不会自动比等任务宏平均更合理。",
        "",
        "## 完整逐任务分差：raw acc",
        "",
        "每格是 HQ−full（pp），不省略负值。",
        "",
        "| 任务 | n | 56M | 143M | 263M | 469M | 803M |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for field, next_section in (("raw", True), ("source_acc_norm", False)):
        for task in TASKS:
            rows = [next(row for row in scale["tasks"] if row["task"] == task) for scale in scales]
            values = " | ".join(f"{row[field]['gap_pp']:+.4f}" for row in rows)
            lines.append(f"| {LABELS[task]} | {rows[0]['n']} | {values} |")
        if next_section:
            lines += [
                "", "## 完整逐任务分差：源 acc_norm", "",
                "此处原样使用存储的 acc_norm，不把它重新命名为 chance-normalized。", "",
                "| 任务 | n | 56M | 143M | 263M | 469M | 803M |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
    lines += [
        "", "## acc_norm 语义与符号敏感性", "",
        "**源 acc_norm 不是 chance-normalized accuracy。** 若采用标准随机基线校正，应定义 `(accuracy−chance)/(1−chance)`。例如 56M full 的 HellaSwag 是四选一，源字段为：", "",
        f"- raw acc = `{counterexample['raw_accuracy']:.10f}`；源 acc_norm = `{counterexample['stored_acc_norm']:.10f}`。",
        f"- 以 chance=0.25 校正 raw acc 得到 `{counterexample['chance_corrected_raw_accuracy']:.10f}`，与源 acc_norm 不相等。", "",
        "[LightEval v0.9.2 官方实现](https://github.com/huggingface/lighteval/blob/v0.9.2/src/lighteval/metrics/metrics.py) 的 acc_norm 是先对答案 log-likelihood 做长度归一化再计算正确率，且存在不同变体。**这只是标准实现的语义参照，不能认证历史运行采用的确切长度单位或空格处理。** 原 harness commit、配置和任务定义缺失，本报告保留“源 acc_norm”名称，不擅自认定字符、token 或 byte 归一化，也不对原始记录施加新的随机校正。", "",
        f"共 50 个任务×规模单元，其中 **{flip_count} 个**在 raw 与源 acc_norm 下的分差符号不同：", "",
        "| 规模 | 任务 | raw 差 (pp) | 源 acc_norm 差 (pp) |",
        "|---:|---|---:|---:|",
    ]
    for cell in data["sensitivity_summary"]["sign_change_cells"]:
        lines.append(f"| {cell['parameters_m']}M | {LABELS[cell['task']]} | {cell['raw_gap_pp']:+.4f} | {cell['source_acc_norm_gap_pp']:+.4f} |")
    lines += [
        "",
        "例如 HellaSwag 的 56M raw 分差为负，而源 acc_norm 分差为正；因此“HellaSwag 五档均下降”只能附带 raw 口径限定。整体宏平均方向稳定不能证明全部任务模式也稳定。", "",
        "## 可以和不可以推出什么", "",
        "- 可以：在这些已有、单次训练的最终 checkpoint 上，按预先固定在本次重分析中的十任务 raw 等权口径，HQ 均值均较高；大小约 +0.39 至 +1.06 pp，规模序列不单调。",
        "- 可以：任务结果和部分总体方向依赖评分与聚合方式，ARC 对若干规模的 raw 宏平均正收益贡献明显。",
        "- 不可以：由本次重算推断训练种子方差、统计显著性、重训成功概率、任意任务分布上的普遍收益，或已证明的质量—覆盖因果机制。",
        "- 不可以：把新增图表/复算叫作新训练实验；把 checkpoint、任务数量或重采样次数当作新增种子。",
        "- 未解决：原 baseline 阈值冲突、具体 harness 配置、逐题输出、训练清单和唯一曝光量。补 143M/803M 种子只能列为未来研究，不计入当前证据。", "",
        "五份输入哈希已由脚本核对。产物按完整精度计算后仅在展示时四舍五入，细小末位差异不代表新测量。", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    data = analyze(workspace)
    json_text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    report = render_report(data, workspace)
    output = workspace / ".reports-sync" / "reports" / "current"
    artifacts = {
        output / "evidence" / "quality_metric_sensitivity.json": json_text,
        output / "QUALITY_METRIC_SENSITIVITY_CN.md": report,
    }
    if args.check:
        for path, value in artifacts.items():
            if not path.exists() or path.read_bytes() != value.encode("utf-8"):
                raise SystemExit(f"Artifact differs from fresh recomputation: {path}")
        print("Verified frozen inputs and both sensitivity artifacts.")
    elif args.write:
        for path, value in artifacts.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value.encode("utf-8"))
            print(path)
    else:
        print(json_text, end="")


if __name__ == "__main__":
    main()
