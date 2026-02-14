#!/usr/bin/env bash
set -euo pipefail

# Summarize all run directories into CSV + plots.
# Usage:
#   bash scripts/summarize_all.sh [RUNS_ROOT] [OUT_DIR]

RUNS_ROOT="${1:-results/repro_runs}"
OUT_DIR="${2:-results/summary_all}"

mapfile -t RUN_DIRS < <(find "${RUNS_ROOT}" -type f -name "multitask_metrics.json" -print | sed 's#/multitask_metrics.json##' | sort)

if [ "${#RUN_DIRS[@]}" -eq 0 ]; then
  echo "[x] No runs found under ${RUNS_ROOT}. Run scripts/run_easy_baseline_hard.sh first."
  exit 1
fi

python -m src.analysis.plot_multitask_linkpred_summary \
  --run_dirs "${RUN_DIRS[@]}" \
  --out_dir "${OUT_DIR}" \
  --difficulty_mode auto \
  --aggregate \
  --save_runs_csv \
  --write_benchmark_table

echo "[*] Summary written to ${OUT_DIR}"
