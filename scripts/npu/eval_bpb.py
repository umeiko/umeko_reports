"""Offline BPB evaluation, a verbatim port of the pilot train_p0.evaluate().

Loads an HF-format checkpoint and scores the held-out dev/test split of the
materialized pool with fp16 autocast on NPU; CE summed in fp32;
BPB = nll_sum / (utf8_bytes * ln2).

Usage (inside mzy-mindspeed container, after sourcing container_env.sh):
  $PY eval_bpb.py --hf-model-dir <dir> --pool-root <pool> --split dev \
      --out <out.json> [--micro-batch 4]
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


@torch.inference_mode()
def evaluate(model, tokens, bytes_per_sequence, device, micro_batch):
    model.eval()
    total_nll = 0.0
    total_predicted_tokens = 0
    total_bytes = 0
    for start in range(0, len(tokens), micro_batch):
        batch_array = np.asarray(tokens[start : start + micro_batch], dtype=np.int64)
        batch = torch.from_numpy(batch_array).to(device=device, non_blocking=False)
        with torch.autocast(device_type="npu", dtype=torch.float16):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model-dir", required=True)
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--micro-batch", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("npu:0")
    model = AutoModelForCausalLM.from_pretrained(args.hf_model_dir, torch_dtype=torch.float32)
    model = model.to(device)

    result = {}
    for language in ("en", "zh"):
        tokens = np.load(f"{args.pool_root}/{language}/{args.split}_tokens.npy")
        byte_counts = np.load(f"{args.pool_root}/{language}/{args.split}_bytes.npy")
        result[language] = evaluate(model, tokens, byte_counts, device, args.micro_batch)
    result["hf_model_dir"] = args.hf_model_dir
    result["split"] = args.split
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({lang: result[lang]["bits_per_byte"] for lang in ("en", "zh")}), flush=True)


if __name__ == "__main__":
    main()
