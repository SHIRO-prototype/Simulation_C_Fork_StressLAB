#!/usr/bin/env bash
# demo_generate.sh — Generate all D0-D4 canonical demo runs with seed=42.
#
# Usage:
#   bash scripts/demo_generate.sh
#
# All outputs go to outputs/. Existing runs with the same config+seed
# will overwrite their previous artifacts.

set -euo pipefail

SEED=42
OUTPUT_DIR="outputs"
CONFIGS=(
  "configs/demos/D0_nominal.yaml"
  "configs/demos/D1_short_outage.yaml"
  "configs/demos/D2_long_outage.yaml"
  "configs/demos/D3_sparse_cadence.yaml"
  "configs/demos/D4_high_process_noise.yaml"
)

echo "============================================================"
echo "  StressLAB Demo Generation (seed=${SEED})"
echo "============================================================"
echo ""

for cfg in "${CONFIGS[@]}"; do
  label=$(basename "$cfg" .yaml)
  echo "--- Running ${label} ---"
  python -m stresslab run \
    --config "$cfg" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --progress
  echo "    [OK] ${label}"
  echo ""
done

echo "============================================================"
echo "  All 5 demo runs generated in ${OUTPUT_DIR}/"
echo ""
echo "  Sanity check: verify each summary JSON contains"
echo "  run_label and scenario_title fields."
echo "============================================================"

# Quick sanity check
PASS=0
FAIL=0
for cfg in "${CONFIGS[@]}"; do
  label=$(basename "$cfg" .yaml)
  found=$(find "$OUTPUT_DIR" -name "summary_*.json" -exec grep -l "\"run_label\": \"${label}\"" {} \; | head -1)
  if [ -n "$found" ]; then
    echo "  [PASS] ${label} -> $(basename "$found")"
    PASS=$((PASS + 1))
  else
    echo "  [FAIL] ${label} — run_label not found in any summary"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "  Results: ${PASS} passed, ${FAIL} failed"

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
