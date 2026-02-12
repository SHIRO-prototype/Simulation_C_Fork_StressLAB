# demo_generate.ps1 — Generate all D0-D4 canonical demo runs with seed=42.
#
# Usage:
#   .\scripts\demo_generate.ps1
#
# All outputs go to outputs\. Existing runs with the same config+seed
# will overwrite their previous artifacts.

$ErrorActionPreference = "Stop"

$SEED = 42
$OUTPUT_DIR = "outputs"
$CONFIGS = @(
    "configs/demos/D0_nominal.yaml",
    "configs/demos/D1_short_outage.yaml",
    "configs/demos/D2_long_outage.yaml",
    "configs/demos/D3_sparse_cadence.yaml",
    "configs/demos/D4_high_process_noise.yaml"
)

Write-Host "============================================================"
Write-Host "  StressLAB Demo Generation (seed=$SEED)"
Write-Host "============================================================"
Write-Host ""

foreach ($cfg in $CONFIGS) {
    $label = [System.IO.Path]::GetFileNameWithoutExtension($cfg)
    Write-Host "--- Running $label ---"
    python -m stresslab run `
        --config $cfg `
        --output-dir $OUTPUT_DIR `
        --seed $SEED `
        --progress
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [FAILED] $label (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
    Write-Host "    [OK] $label" -ForegroundColor Green
    Write-Host ""
}

Write-Host "============================================================"
Write-Host "  All 5 demo runs generated in $OUTPUT_DIR\"
Write-Host "============================================================"
Write-Host ""

# Quick sanity check
$pass = 0
$fail = 0
foreach ($cfg in $CONFIGS) {
    $label = [System.IO.Path]::GetFileNameWithoutExtension($cfg)
    $found = Get-ChildItem -Path $OUTPUT_DIR -Filter "summary_*.json" -Recurse |
        Select-String -Pattern "`"run_label`": `"$label`"" -SimpleMatch |
        Select-Object -First 1
    if ($found) {
        Write-Host "  [PASS] $label -> $($found.Filename)" -ForegroundColor Green
        $pass++
    } else {
        Write-Host "  [FAIL] $label - run_label not found in any summary" -ForegroundColor Red
        $fail++
    }
}

Write-Host ""
Write-Host "  Results: $pass passed, $fail failed"

if ($fail -gt 0) {
    exit 1
}
