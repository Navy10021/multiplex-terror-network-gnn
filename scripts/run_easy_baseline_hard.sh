#!/usr/bin/env bash
set -euo pipefail

# Run three canonical difficulty presets with one command.
# Usage:
#   bash scripts/run_easy_baseline_hard.sh [SEED] [SIZE] [OUT_ROOT]

SEED="${1:-2025}"
SIZE="${2:-1500}"
OUT_ROOT="${3:-results/repro_runs}"

for difficulty in easy baseline hard; do
  echo "[*] Running difficulty=${difficulty} seed=${SEED} size=${SIZE}"
  python -m src.run_all \
    --config "configs/generator_${difficulty}.json" \
    --size "${SIZE}" \
    --seed "${SEED}" \
    --out_root "${OUT_ROOT}/${difficulty}" \
    --run_reporting_summary

done

echo "[*] Completed runs under: ${OUT_ROOT}"
