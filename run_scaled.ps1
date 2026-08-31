param(
    [string]$Proxy = "http://127.0.0.1:8635",
    [switch]$SkipData,
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
$experimentRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $experimentRoot
$env:HTTP_PROXY = $Proxy
$env:HTTPS_PROXY = $Proxy
$env:HF_HOME = Join-Path $experimentRoot "cache\huggingface"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:TOKENIZERS_PARALLELISM = "false"

uv sync --locked
if (-not $SkipData) {
    uv run python .\scripts\download_raw_shards.py --config .\configs\p0_scaled_100m.json
    uv run python .\scripts\materialize_data.py --config .\configs\p0_scaled_100m.json --raw-shards-root .\data\raw_shards_100m_all_sources
}
uv run python .\scripts\inventory_haidass_data.py --config .\configs\p0_scaled_100m.json
if (-not $SkipTraining) {
    uv run python .\scripts\train_p0.py --config .\configs\p0_scaled_100m.json --all
}
uv run python .\scripts\audit_runs.py --config .\configs\p0_scaled_100m.json
uv run python .\scripts\make_report.py --config .\configs\p0_scaled_100m.json
