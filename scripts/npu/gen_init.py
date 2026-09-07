"""Regenerate the pilot P0 deterministic initializations (seed 2027/2028).

Replicates umeko_reports/scripts/train_p0.py create_initialization():
seed python/numpy/torch RNGs, build the HF Qwen3ForCausalLM on CPU, hash the
state dict. Compares against the pilot audit hashes, then writes an HF-format
directory (config.json + model.safetensors) per seed as input for the
hf->mcore checkpoint conversion.

Run inside the mzy-mindspeed container with the ms env python.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
PILOT_CONFIG_PATH = Path("/mnt/models/CODE/mzy/umeko_reports/configs/p0_scaled_100m.json")
AUDIT = Path("/mnt/models/CODE/mzy/umeko_reports/reports/p0_scaled_100m_all_sources_2080ti_run_audit.json")
OUT_ROOT = ROOT / "training" / "init_hf"

PILOT_INIT_SHA256 = {
    2027: "f651fbac80430e78394e449e15f685a46776a454b0e5a913d7703a8256dfd2e8",
    2028: "8bea9f49408806eeae2bc984933628e63173d47d8b4364e1c0b97f67c73eaecb",
}


def build_model(config: dict) -> Qwen3ForCausalLM:
    """Verbatim port of the pilot build_model()."""
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


def main() -> None:
    config = json.loads(PILOT_CONFIG_PATH.read_text(encoding="utf-8"))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    report = {}
    for seed in (2027, 2028):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = build_model(config)
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        digest = initialization_hash(state)
        pilot_digest = PILOT_INIT_SHA256[seed]
        match = digest == pilot_digest
        print(f"seed {seed}: init sha256 {digest[:16]}... {'MATCH' if match else 'MISMATCH'} pilot", flush=True)
        out_dir = OUT_ROOT / f"init-s{seed}"
        model.save_pretrained(out_dir, safe_serialization=True)
        report[seed] = {
            "sha256": digest,
            "pilot_sha256": pilot_digest,
            "match": match,
            "hf_dir": str(out_dir),
            "parameter_count": sum(p.numel() for p in model.parameters()),
        }
        print(f"seed {seed}: init sha256 {digest[:16]}... {'MATCH' if match else 'MISMATCH'} pilot", flush=True)
        del model, state
    (OUT_ROOT / "init_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
