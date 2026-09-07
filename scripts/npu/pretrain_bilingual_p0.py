"""MindSpeed-LLM entry point reproducing the 2080 Ti pilot P0 43M runs.

Reproduces umeko_reports/scripts/train_p0.py semantics on NPU:

- Data: exact pilot sequence order. Pools are the rematerialized
  materialized_100m_all_sources (EN/ZH train_tokens.npy, int32 [102400, 1024]),
  replayed through training/order/order_<condition>_s<seed>.npz
  (labels int8: 0=en 1=zh; indices int32 into that language's train pool).
  The MegatronPretrainingSampler (dataloader_type=single, DP=1) iterates the
  dataset strictly in order, so consumption order == pilot schedule.
- Sequence handling: pilot fed row[:-1] as input and row[1:] as labels to a HF
  model (1023 prediction positions). Here the dataset returns the pre-shifted
  pair directly, so --seq-length must be 1023 and no in-model shift is needed
  beyond Megatron's own label alignment (labels are used as-is by GPTModel).
- Loss: mean CE over all 1023 positions; every micro-batch has the same token
  count, so Megatron's token-count-weighted accumulation equals the pilot's
  plain per-update mean.
- Weight decay: the pilot's HF AdamW applied weight_decay=1e-5 to ALL
  parameters (no no-decay group). Megatron's default excludes 1-D params, so
  _get_param_groups is monkeypatched to force wd_mult=1.0 everywhere.
- Init: weights come from --load (converted from the pilot-parity HF init
  under training/init_hf/init-s<seed>); --finetune resets iteration/optim/rng.

Launch through training/run_one.sh (torchrun, single process per NPU).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

import pretrain_gpt as base  # noqa: F401  (applies mindspeed megatron_adaptor)
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from mindspeed_llm.training.training import pretrain

SEQ_LEN = 1023  # pilot context 1024 with HF shift -> 1023 predicted positions


def force_weight_decay_on_all_params() -> None:
    """Patch megatron.core.optimizer._get_param_groups: no no-decay group."""
    import megatron.core.optimizer as megatron_optimizer

    original = megatron_optimizer._get_param_groups

    def patched(model_chunks, no_weight_decay_cond, scale_lr_cond, lr_mult, *args, **kwargs):
        return original(
            model_chunks, lambda name, param: False, scale_lr_cond, lr_mult, *args, **kwargs
        )

    megatron_optimizer._get_param_groups = patched


class PilotOrderedDataset(Dataset):
    """Replays one pilot run: order file position i -> sequence i consumed."""

    def __init__(self, pool_root: str, order_file: str) -> None:
        order = np.load(order_file)
        self.labels = order["labels"]
        self.indices = order["indices"]
        self.pools = {
            0: np.load(f"{pool_root}/en/train_tokens.npy", mmap_mode="r"),
            1: np.load(f"{pool_root}/zh/train_tokens.npy", mmap_mode="r"),
        }

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int) -> dict:
        lang = int(self.labels[i])
        row = np.asarray(self.pools[lang][int(self.indices[i])], dtype=np.int64)
        return {
            "tokens": torch.from_numpy(row[:-1]),
            "labels": torch.from_numpy(row[1:]),
            "loss_mask": torch.ones(SEQ_LEN, dtype=torch.float32),
            "position_ids": torch.arange(SEQ_LEN, dtype=torch.int64),
        }


def extra_args_provider(parser):
    group = parser.add_argument_group(title="bilingual p0 pilot")
    group.add_argument("--pilot-pool-root", type=str, required=True,
                       help="materialized_100m_all_sources root (en/ zh/ subdirs)")
    group.add_argument("--pilot-order-file", type=str, required=True,
                       help="training/order/order_<condition>_s<seed>.npz")
    return parser


def train_valid_test_datasets_provider(train_val_test_num_samples):
    args = get_args()
    train_ds = PilotOrderedDataset(args.pilot_pool_root, args.pilot_order_file)
    print_rank_0(
        f"> pilot ordered dataset: {len(train_ds)} sequences "
        f"from {args.pilot_order_file}"
    )
    return train_ds, None, None


train_valid_test_datasets_provider.is_distributed = True


def main():
    force_weight_decay_on_all_params()
    pretrain(
        train_valid_test_datasets_provider,
        base.model_provider,
        ModelType.encoder_or_decoder,
        base.forward_step,
        extra_args_provider=extra_args_provider,
    )


if __name__ == "__main__":
    main()
