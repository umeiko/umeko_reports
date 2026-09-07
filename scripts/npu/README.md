# NPU (Ascend + MindSpeed-LLM) reproduction scripts

These scripts ran the P0 43M reproduction and the P1 143M proportional
scale-up on an 8x Ascend NPU host. They are environment-specific (absolute
paths under `/mnt/models/CODE/mzy`, a `mzy-mindspeed` docker container,
MindSpeed-LLM v2.3.0 checked out at `/mnt/models/CODE/MindSpeed-LLM-v2.3.0`)
and are published as a process record, not as a turnkey pipeline.

Pipeline order, per experiment:

1. `materialize_data.py` (repo `scripts/`) builds the token pool from the
   pinned raw shards; P1 uses `configs/p1_143m_proportional.json`.
2. `gen_init.py` / `gen_init_143m.py` build the deterministic HF-format
   initializations (seeds 2027/2028); `convert_ckpt_v2.py` (MindSpeed-LLM)
   converts them to Megatron format.
3. `make_order.py` / `make_order_143m.py` regenerate the per-(condition, seed)
   training orders with the pilot's verbatim schedule logic.
4. `run_one*.sh` trains a single run on one NPU through
   `pretrain_bilingual_p0.py` (custom ordered data provider + all-parameter
   weight decay); `run_wave*.sh` schedules a list of runs per card;
   `run_card.sh` is the per-card driver used for the 43M repro.
5. `eval_run*.sh` converts each checkpoint back to HF and evaluates dev/test
   BPB with `eval_bpb.py`.
6. `compare_results*.py` aggregate the runs; `plot_143m_figures.py` renders
   the P1 figures; `make_repo_tables.py` packages the CSV tables and
   `report_manifest.json` into this repo.

`container_env.sh` is sourced inside the container to set up CANN/MindSpeed
environment variables.
