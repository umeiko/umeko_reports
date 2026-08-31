from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from common import ROOT, environment_snapshot, load_config, sha256_file, write_json


LANGUAGE_CODE = {"en": 0, "zh": 1}


def build_model(config: dict) -> Qwen3ForCausalLM:
    model_config = config["model"]
    qwen_config = Qwen3Config(
        vocab_size=model_config["vocab_size"],
        hidden_size=model_config["hidden_size"],
        intermediate_size=model_config["intermediate_size"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_attention_heads=model_config["num_attention_heads"],
        num_key_value_heads=model_config["num_key_value_heads"],
        head_dim=model_config["head_dim"],
        max_position_embeddings=config["local_data"]["context_length"],
        rope_theta=model_config["rope_theta"],
        rms_norm_eps=model_config["rms_norm_eps"],
        tie_word_embeddings=model_config["tie_word_embeddings"],
        bos_token_id=3,
        eos_token_id=config["tokenizer"]["eos_token_id"],
        pad_token_id=config["tokenizer"]["eos_token_id"],
        attention_dropout=0.0,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=model_config["num_hidden_layers"],
        use_cache=False,
    )
    qwen_config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(qwen_config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != model_config["expected_parameters"]:
        raise RuntimeError(f"model parameter count changed: {parameter_count}")
    return model


def initialization_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def create_initialization(config: dict, path: Path, seed: int) -> tuple[dict[str, torch.Tensor], str]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_model(config)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    digest = initialization_hash(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        torch.save({"seed": seed, "sha256": digest, "state_dict": state}, path)
    else:
        stored = torch.load(path, map_location="cpu", weights_only=False)
        if stored["sha256"] != digest:
            raise RuntimeError("stored initialization does not match deterministic reconstruction")
        state = stored["state_dict"]
    del model
    return state, digest


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


def sequence_indices(
    labels: np.ndarray,
    available: int,
    seed: int,
    source_arrays: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
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


def learning_rate(step: int, config: dict) -> float:
    training = config["training"]
    peak = training["peak_learning_rate"]
    warmup = training["warmup_updates"]
    total = training["optimizer_updates"]
    if step <= warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.inference_mode()
def evaluate(model: nn.Module, tokens: np.ndarray, bytes_per_sequence: np.ndarray, device: torch.device, micro_batch: int) -> dict:
    model.eval()
    total_nll = 0.0
    total_predicted_tokens = 0
    total_bytes = 0
    for start in range(0, len(tokens), micro_batch):
        batch_array = np.asarray(tokens[start : start + micro_batch], dtype=np.int64)
        batch = torch.from_numpy(batch_array).to(device=device, non_blocking=False)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(input_ids=batch, use_cache=False).logits
        shifted_logits = logits[:, :-1].float().reshape(-1, logits.shape[-1])
        shifted_labels = batch[:, 1:].reshape(-1)
        nll = F.cross_entropy(shifted_logits, shifted_labels, reduction="sum")
        if not torch.isfinite(nll):
            raise FloatingPointError("non-finite held-out NLL")
        total_nll += float(nll.cpu())
        total_predicted_tokens += int(shifted_labels.numel())
        total_bytes += int(np.asarray(bytes_per_sequence[start : start + len(batch_array)]).sum())
        del logits, shifted_logits, shifted_labels, batch
    mean_nll = total_nll / total_predicted_tokens
    return {
        "nll_sum": total_nll,
        "predicted_tokens": total_predicted_tokens,
        "utf8_bytes": total_bytes,
        "nll_per_token": mean_nll,
        "perplexity": math.exp(min(20.0, mean_nll)),
        "bits_per_byte": total_nll / (max(1, total_bytes) * math.log(2.0)),
    }


def save_final_model(model: nn.Module, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    if "lm_head.weight" in state and "model.embed_tokens.weight" in state:
        state.pop("lm_head.weight")
    path = destination / "model.safetensors"
    save_file(state, path, metadata={"format": "pt", "tied_embeddings": "model.embed_tokens.weight=lm_head.weight"})
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def train_condition(condition: str, config: dict, initial_state: dict[str, torch.Tensor], init_hash: str, seed: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda:0")
    training = config["training"]
    local = config["local_data"]
    run_id = f"{config['experiment_name']}-{condition}-s{seed}"
    run_root = ROOT / "runs" / run_id
    summary_path = run_root / "summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"skip completed {run_id}")
            return existing
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "status.json", {"status": "running", "run_id": run_id, "started_unix": time.time()})

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("high")

    data_root = ROOT / "data" / config.get("materialized_subdir", "materialized")
    data = {
        language: {
            split: np.load(data_root / language / f"{split}_tokens.npy", mmap_mode="r")
            for split in ("train", "dev", "test")
        }
        for language in ("en", "zh")
    }
    source_arrays = {
        language: np.load(data_root / language / "train_sources.npy", mmap_mode="r")
        for language in ("en", "zh")
        if (data_root / language / "train_sources.npy").exists()
    }
    byte_counts = {
        language: {
            split: np.load(data_root / language / f"{split}_bytes.npy", mmap_mode="r")
            for split in ("dev", "test")
        }
        for language in ("en", "zh")
    }
    labels = language_schedule(condition, training["sequences_per_run"], seed)
    indices = sequence_indices(labels, local["train_sequences_per_language"], seed, source_arrays or None)
    schedule_path = run_root / "schedule.npy"
    schedule_columns = [labels.astype(np.int32), indices]
    source_sequence_counts = None
    if len(source_arrays) == 2:
        source_codes = np.asarray([
            source_arrays["en" if int(language) == LANGUAGE_CODE["en"] else "zh"][index]
            for language, index in zip(labels.tolist(), indices.tolist())
        ], dtype=np.int32)
        schedule_columns.append(source_codes)
        materialization_summary = json.loads(
            (data_root / "materialization_summary.json").read_text(encoding="utf-8")
        )
        source_vocabulary = materialization_summary["source_vocabulary"]
        source_sequence_counts = {
            source_id: int((source_codes == source_code).sum())
            for source_id, source_code in source_vocabulary.items()
        }
    np.save(schedule_path, np.stack(schedule_columns, axis=1))
    schedule_hash = sha256_file(schedule_path)

    model = build_model(config)
    model.load_state_dict(initial_state, strict=True)
    if initialization_hash({name: value.detach().cpu() for name, value in model.state_dict().items()}) != init_hash:
        raise RuntimeError("condition initialization hash mismatch")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["peak_learning_rate"],
        betas=tuple(training["betas"]),
        eps=training["epsilon"],
        weight_decay=training["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    micro_batch = training["micro_batch_sequences"]
    accumulation = training["gradient_accumulation_steps"]
    global_batch = training["global_batch_sequences"]
    thermal_pacing_ms = max(0.0, float(os.environ.get("HAIDASS_THERMAL_PACING_MS", "0")))
    if micro_batch * accumulation != global_batch:
        raise RuntimeError("microbatch x accumulation does not equal global batch")
    if training["optimizer_updates"] * global_batch != training["sequences_per_run"]:
        raise RuntimeError("update count does not match sequence count")
    if training["tokens_per_run"] != training["sequences_per_run"] * local["context_length"]:
        raise RuntimeError("registered token count mismatch")

    log_path = run_root / "metrics.jsonl"
    temporary_log = log_path.with_suffix(".jsonl.tmp")
    start_time = time.perf_counter()
    peak_memory = 0
    last_train_loss = None
    evaluation_records = []

    def record_evaluation(step: int) -> None:
        nonlocal peak_memory
        record = {"record_type": "evaluation", "step": step, "seen_tokens": step * global_batch * local["context_length"]}
        for language in ("en", "zh"):
            record[language] = evaluate(model, data[language]["dev"], byte_counts[language]["dev"], device, micro_batch)
        record["elapsed_seconds"] = time.perf_counter() - start_time
        record["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        peak_memory = max(peak_memory, record["peak_memory_bytes"])
        evaluation_records.append(record)
        log_handle.write(json.dumps(record) + "\n")
        log_handle.flush()
        model.train()

    try:
        with temporary_log.open("w", encoding="utf-8") as log_handle:
            record_evaluation(0)
            sequence_cursor = 0
            for update in range(1, training["optimizer_updates"] + 1):
                update_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                for _ in range(accumulation):
                    batch_labels = labels[sequence_cursor : sequence_cursor + micro_batch]
                    batch_indices = indices[sequence_cursor : sequence_cursor + micro_batch]
                    sequences = np.empty((micro_batch, local["context_length"]), dtype=np.int64)
                    for row, (language_value, sequence_index) in enumerate(zip(batch_labels.tolist(), batch_indices.tolist())):
                        language = "en" if int(language_value) == LANGUAGE_CODE["en"] else "zh"
                        sequences[row] = data[language]["train"][sequence_index]
                    batch = torch.from_numpy(sequences).to(device=device)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        loss = model(input_ids=batch, labels=batch, use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite training loss at update {update}")
                    accumulated_loss += float(loss.detach().cpu())
                    scaler.scale(loss / accumulation).backward()
                    sequence_cursor += micro_batch
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm at update {update}")
                lr = learning_rate(update, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    raise FloatingPointError(f"FP16 overflow caused a skipped optimizer update at {update}")
                torch.cuda.synchronize(device)
                update_seconds = time.perf_counter() - update_start
                last_train_loss = accumulated_loss / accumulation
                train_record = {
                    "record_type": "train",
                    "step": update,
                    "seen_tokens": update * global_batch * local["context_length"],
                    "language_counts": {
                        "en": int((labels[:sequence_cursor] == 0).sum()) * local["context_length"],
                        "zh": int((labels[:sequence_cursor] == 1).sum()) * local["context_length"],
                    },
                    "loss": last_train_loss,
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "learning_rate": lr,
                    "scale": scaler.get_scale(),
                    "update_seconds": update_seconds,
                    "tokens_per_second": global_batch * local["context_length"] / update_seconds,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                }
                peak_memory = max(peak_memory, train_record["peak_memory_bytes"])
                log_handle.write(json.dumps(train_record) + "\n")
                log_handle.flush()
                if update % training["evaluation_interval_updates"] == 0 or update == training["optimizer_updates"]:
                    record_evaluation(update)
                if update % training["checkpoint_interval_updates"] == 0 and update != training["optimizer_updates"]:
                    write_json(run_root / f"checkpoint-{update:04d}.json", {
                        "step": update,
                        "seen_tokens": train_record["seen_tokens"],
                        "schedule_cursor": sequence_cursor,
                        "note": "Local P0 retains metadata checkpoints; only the final model weights are retained to control disk use.",
                    })
                if thermal_pacing_ms:
                    time.sleep(thermal_pacing_ms / 1000.0)
        os.replace(temporary_log, log_path)
        test_metrics = {
            language: evaluate(model, data[language]["test"], byte_counts[language]["test"], device, micro_batch)
            for language in ("en", "zh")
        }
        model_artifact = save_final_model(model, run_root / "final_checkpoint")
        elapsed = time.perf_counter() - start_time
        train_records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if '"record_type": "train"' in line]
        throughput = [record["tokens_per_second"] for record in train_records[10:]] or [record["tokens_per_second"] for record in train_records]
        summary = {
            "status": "complete",
            "scientific_status": config["scientific_status"],
            "run_id": run_id,
            "condition": condition,
            "seed": seed,
            "model_parameters": config["model"]["expected_parameters"],
            "initialization_sha256": init_hash,
            "schedule_sha256": schedule_hash,
            "config_sha256": sha256_file(Path(config["_config_path"])),
            "tokens": training["tokens_per_run"],
            "source_sequence_counts": source_sequence_counts,
            "source_token_counts": {
                source_id: count * local["context_length"]
                for source_id, count in source_sequence_counts.items()
            } if source_sequence_counts else None,
            "optimizer_updates": training["optimizer_updates"],
            "final_train_loss": last_train_loss,
            "test": test_metrics,
            "forgetting_dev_bpb": {
                language: evaluation_records[-1][language]["bits_per_byte"] - min(record[language]["bits_per_byte"] for record in evaluation_records)
                for language in ("en", "zh")
            },
            "elapsed_seconds": elapsed,
            "median_tokens_per_second_after_warmup": float(np.median(throughput)),
            "thermal_pacing_ms_per_update": thermal_pacing_ms,
            "peak_memory_bytes": peak_memory,
            "model_artifact": model_artifact,
            "metrics_sha256": sha256_file(log_path),
            "environment": environment_snapshot(),
        }
        write_json(summary_path, summary)
        write_json(run_root / "status.json", {"status": "complete", "run_id": run_id, "finished_unix": time.time()})
        print(json.dumps({"run_id": run_id, "elapsed_seconds": elapsed, "test": test_metrics}, indent=2))
        return summary
    except Exception as error:
        write_json(run_root / "status.json", {"status": "failed", "run_id": run_id, "error": repr(error), "failed_unix": time.time()})
        raise
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "p0_local.json"))
    parser.add_argument("--condition", choices=["en_only", "zh_only", "iid_50", "en_first", "zh_first", "alternating"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    materialization = ROOT / "data" / config.get("materialized_subdir", "materialized") / "materialization_summary.json"
    if not materialization.exists():
        raise FileNotFoundError("materialized data missing; run materialize_data.py first")
    conditions = config["training"]["conditions"] if args.all else [args.condition]
    if not args.all and not args.condition:
        parser.error("pass --all or --condition")
    seeds = config["training"].get("seeds", [config["training"].get("seed", 2027)])
    for seed in seeds:
        initial_state, init_hash = create_initialization(config, ROOT / "runs" / f"initialization-s{seed}.pt", seed)
        for condition in conditions:
            train_condition(condition, config, initial_state, init_hash, seed)


if __name__ == "__main__":
    main()
