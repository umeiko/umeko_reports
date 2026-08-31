# Haidass local bilingual experiments

This directory contains isolated RTX 2080 Ti pilots. The scaled pilot covers all
five upstream repositories named by the Haidass1.5 model card, while preserving
the smaller completed pipeline pilot. Results are directional and are not part
of the confirmatory cluster registry.

The scaled run uses the Haidass tokenizer, a 43.46M proxy architecture, the same
language-order definitions, optimizer family, and learning-rate peak as the
cluster plan. It differs deliberately in context length (1,024), precision
(FP16 because Turing has no native BF16 path), two seeds, and 100.66M tokens per
condition. Its controlled mixture includes Ultra-FineWeb EN/ZH, DCLM baseline,
FineMath-4plus, all four Ultra-FineWeb-L3 configs, and all eight Cosmopedia
configs. The model card does not disclose the original source ratios or stage
boundaries, so this run does not claim to reproduce the original 400B-token mix.

The materialized NumPy arrays are sufficient for training. To retain the exact
selected upstream Parquet files as a reproducibility artifact, run:

```powershell
uv run python scripts\download_raw_shards.py --config configs\p0_scaled_100m.json
```

This persists only the 16 frozen experiment shards, not the roughly 19.77 TB
upstream repositories, and records a SHA-256 digest for every local file. Pass
`--raw-shards-root data/raw_shards_100m_all_sources` to `materialize_data.py`
to read the pinned local Parquet files instead of their network URLs.

## Reproduce

```powershell
Set-Location .\local_bilingual_2080ti
uv sync
.\run_local.ps1 -Proxy http://127.0.0.1:8635
```

Run the two-seed, 12-run scaled matrix:

```powershell
.\run_scaled.ps1 -Proxy http://127.0.0.1:8635
```

Stages can be run separately:

```powershell
uv run python .\scripts\materialize_data.py --config .\configs\p0_local.json
uv run python .\scripts\train_p0.py --config .\configs\p0_local.json --all
uv run python .\scripts\make_report.py --config .\configs\p0_local.json
```

The scaled configuration is `configs/p0_scaled_100m.json`. It materializes about
104.86M tokens per language and trains 6 conditions x 2 seeds x 100.66M tokens,
for 1.208B total training tokens. `reports/HAIDASS_DATASET_INVENTORY_CN.md`
records upstream sizes, licenses, pinned revisions, coverage, and the boundary
between disclosed model-card facts and this controlled local design.

Generated data, checkpoints, logs, figures and tables are ignored by Git but
remain under this directory. The completed scaled report is
`reports/p0_scaled_100m_all_sources_2080ti/EXPERIMENT_REPORT_CN.md`;
`runs/*/summary.json` and JSONL logs are the auditable source.

For a Chinese, beginner-oriented explanation of the motivation, all five data
repositories, preprocessing, training, validation, metrics, results, and
scientific interpretation, read
`reports/p0_scaled_100m_all_sources_2080ti/EXPERIMENT_REPORT_BEGINNER_CN.md`.
