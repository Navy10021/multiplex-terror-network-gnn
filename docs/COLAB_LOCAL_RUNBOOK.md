# Colab/Local Command Runbook

This runbook provides practical commands for running the pipeline in Google Colab or a local shell.

## 1) Data generation and validation

### 1-1. Generate multiplex data
```bash
python -m src.data.multiplex_generator_v3 \
  --config configs/generator_baseline.json \
  --size 1500 \
  --seed 2025 \
  --out_path data/multiplex_baseline/multiplex.json
```

### 1-2. Validate ontology conformance
```bash
python -m src.cli.validate_ontology \
  --manifest data/multiplex_baseline/multiplex.json \
  --ontology ontology/terror.ttl \
  --shapes ontology/constraints.shacl.ttl \
  --json
```

### 1-3. Build PyG dataset
```bash
python -m src.data.build_pyg_dataset_v3 \
  --manifest data/multiplex_baseline/multiplex.json \
  --out_path data/multiplex_baseline/pyg_data.pt
```

### 1-4. Run diagnostics
```bash
python -m src.data.basic_diagnostics_v3 \
  --manifest data/multiplex_baseline/multiplex.json \
  --out_dir data/analysis/multiplex_baseline
```

## 2) End-to-end execution (`run_all`)

### strict mode (default)
```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 1500 \
  --seed 2025 \
  --out_root results
```

### constrained mode (retry-to-conform)
```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 1500 \
  --seed 2025 \
  --out_root results \
  --ontology_mode constrained
```

### report-only mode (continue despite violations)
```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 1500 \
  --seed 2025 \
  --out_root results \
  --ontology_mode report_only
```

## 3) Training commands

### HVT model
```bash
python -m src.models.train_hvt_gnn_v3 \
  --data data/multiplex_baseline/pyg_data.pt \
  --out_dir data/multiplex_baseline
```

### Multitask model
```bash
python -m src.models.train_multitask_gnn_v3 \
  --data data/multiplex_baseline/pyg_data.pt \
  --out_dir data/multiplex_baseline
```

### Layer-wise link prediction
```bash
python -m src.models.train_linkpred_layer_v3 \
  --data data/multiplex_baseline/pyg_data.pt \
  --layer finance \
  --neg_mode uniform \
  --out_dir data/multiplex_baseline
```

## 4) Reporting and summary plots
```bash
python -m src.analysis.plot_multitask_linkpred_summary \
  --run_dirs results/run_* \
  --out_dir results/summary_all
```

## 5) Fast reproducibility sweep
```bash
bash scripts/run_easy_baseline_hard.sh 2025 1500 results/repro_runs
bash scripts/summarize_all.sh results/repro_runs results/summary_all
```

## 6) Lightweight smoke check sequence
```bash
python -m src.run_all --config configs/generator_easy.json --size 200 --seed 7 --out_root results/smoke --ontology_mode report_only
python -m src.data.build_pyg_dataset_v3 --manifest results/smoke/run_generator_easy_s7/multiplex.json --out_path results/smoke/run_generator_easy_s7/pyg_data.pt
python -m src.data.basic_diagnostics_v3 --manifest results/smoke/run_generator_easy_s7/multiplex.json --out_dir results/smoke/run_generator_easy_s7/diagnostics
```

## 7) Practical tips
1. Persist experiment metadata together (`seed`, config path, output path, git commit hash).
2. Always record ontology mode (`strict`, `constrained`, `report_only`) with published results.
3. Prefer `python -m ...` style commands to reduce path differences across Colab/local environments.
