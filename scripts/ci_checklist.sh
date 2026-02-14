#!/usr/bin/env bash
set -euo pipefail

echo "[1/5] ruff"
ruff check src tests

echo "[2/5] mypy"
mypy src/ontology src/validation src/cli src/utils

echo "[3/5] pytest"
pytest -q

echo "[4/5] cli health"
python -m src.cli.main --help >/dev/null
python -m src.cli.main run-all --help >/dev/null

echo "[5/5] script syntax"
bash -n scripts/run_easy_baseline_hard.sh scripts/summarize_all.sh

echo "[ok] CI checklist complete"
