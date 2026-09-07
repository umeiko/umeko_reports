"""Generate deterministic 143M initializations (seed 2027/2028) for the P1 scale-up.

Same procedure as gen_init.py (which replicates the pilot train_p0.py
create_initialization()): seed python/numpy/torch RNGs, build the HF
Qwen3ForCausalLM on CPU with the 143M architecture from
umeko_reports/configs/p1_143m_proportional.json, record the state-dict sha256,
then write an HF-format directory (config.json + model.safetensors) per seed as
input for the hf->mcore checkpoint conversion.

There is no pilot 143M hash to audit against; the recorded sha256 values in
init_report.json are the reference for any future reproduction.

Run inside the mzy-mindspeed container with the ms env python.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
P1_CONFIG_PATH = Path("/mnt/models/CODE/mzy/umeko_reports/configs/p1_143m_proportional.json")
OUT_ROOT = ROOT / "training" / "init_hf_143m"


def build_model(config: dict) -> Qwen3ForCausalLM:
    """Same construction as the pilot build_model() / gen_init.py."""
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
    config = json.loads(P1_CONFIG_PATH.read_text(encoding="utf-8"))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    report = {}
    for seed in (2027, 2028):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = build_model(config)
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        digest = initialization_hash(state)
        out_dir = OUT_ROOT / f"init-s{seed}"
        model.save_pretrained(out_dir, safe_serialization=True)
        report[seed] = {
            "sha256": digest,
            "hf_dir": str(out_dir),
            "parameter_count": sum(p.numel() for p in model.parameters()),
        }
        print(f"seed {seed}: params {report[seed]['parameter_count']:,} init sha256 {digest[:16]}... -> {out_dir}", flush=True)
        del model, state
    (OUT_ROOT / "init_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
