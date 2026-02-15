# Reproducibility Guide (STEP 8)

This guide describes the standard one-command workflow for reproducible experiments and summary artifacts.

## 1) Run all three difficulty settings (easy / baseline / hard)

```bash
bash scripts/run_easy_baseline_hard.sh 2025 1500 results/repro_runs
```

Arguments:
- `SEED` (default: `2025`)
- `SIZE` (default: `1500`)
- `OUT_ROOT` (default: `results/repro_runs`)

## 2) Summarize all run folders (CSV + plots)

```bash
bash scripts/summarize_all.sh results/repro_runs results/summary_all
```

Expected outputs (`results/summary_all`):
- `multitask_linkpred_summary_runs.csv`
- `multitask_linkpred_summary_agg.csv`
- `ontology_benchmark_table.csv`
- `ontology_benchmark_table.md`
- `hvt_f1_by_difficulty.png`, `hvt_auc_by_difficulty.png`, ...
- `difficulty_vs_performance_curve.png`

## 3) Recommended output structure

```text
results/
  repro_runs/
    easy/run_.../
      multiplex.json
      ontology_validation_report.json
      pyg_data.pt
      multitask_metrics.json
      linkpred_finance_uniform.json
      linkpred_communication_uniform.json
      run_metadata.json
    baseline/run_.../
    hard/run_.../
  summary_all/
    multitask_linkpred_summary_runs.csv
    multitask_linkpred_summary_agg.csv
    ontology_benchmark_table.csv
    *.png
```

This layout is designed to stay aligned with the README "Typical outputs" section.
