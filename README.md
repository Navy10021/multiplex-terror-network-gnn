# 🔬🕸️ Multiplex Terror Network GNN (Ontology-First)

Synthetic multiplex terror-network generation + GNN learning/research stack with **ontology-driven validation, training regularization, and reporting**.

> **Defensive research only**: this repository is for lawful, defensive CT/criminal-network methodology research on **synthetic data only**.

---

## TL;DR

This repo gives you an end-to-end pipeline:

1. **Generate** synthetic multiplex graphs (`hierarchy`, `finance`, `communication`, `operation`, `ideology`) with configurable structure/noise.
2. **Validate with ontology** (OWL/SHACL contract + runtime checks).
3. **Build PyG dataset** with ontology bridge tensors (`edge_ontology_attr`, `node_ontology_features`, `role_compatibility_mask`).
4. **Train models** including optional ontology-aware losses (Phase D).
5. **Run reporting** with ontology conformance metrics + node-level explanation artifacts (Phase E).

Core one-command runner:

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 1500 \
  --seed 2025 \
  --out_root results
```

---

## Motivation

Real-world hostile/criminal networks are hard to model because they are:

- **Multiplex**: different semantics per layer.
- **Noisy/partial**: missing edges, false edges, copied provenance.
- **Constraint-heavy**: role-relation compatibility, temporal ordering, and consistency rules matter.

This project tackles that by centering ontology across the full lifecycle:

- **Generation quality**: rule-conformant manifests, retry/telemetry in constrained mode.
- **Learning signal**: ontology-aware regularizers in multitask training.
- **Evaluation/explainability**: ontology conformance + violation rates in summary, and node-level explanation JSON with rule evidence.

---

## Highlights

### 1) Ontology assets and validator

- Ontology files:
  - `ontology/terror.ttl`
  - `ontology/constraints.shacl.ttl`
- Python validator + loader:
  - `src/ontology/validator.py`
  - `src/ontology/load.py`
- CLI:

```bash
python -m src.cli.validate_ontology \
  --manifest data/multiplex_baseline/multiplex.json \
  --ontology ontology/terror.ttl \
  --shapes ontology/constraints.shacl.ttl
```

JSON output mode:

```bash
python -m src.cli.validate_ontology --manifest data/multiplex_baseline/multiplex.json --json
```

### 2) Generator v3 + ontology integration

- `src/data/multiplex_generator_v3.py` supports ontology validation and constrained retries.
- `src/run_all.py` writes `ontology_validation_report.json` and telemetry by default.
- Ontology mode presets in runner:
  - `--ontology_mode strict` (default)
  - `--ontology_mode constrained`
  - `--ontology_mode report_only`
- Fine-grained legacy flags (backward compatible):
  - `--ontology_constrained`
  - `--ontology_max_retries`
  - `--ontology_retry_seed_stride`
  - `--no_ontology_strict`

### 3) PyG builder ontology bridge (Phase C)

`src/data/build_pyg_dataset_v3.py` produces:

- `edge_ontology_attr`
- `node_ontology_features`
- `role_compatibility_mask`

plus payload consistency checks against manifest ontology role counts/vocabulary.

### 4) Model Zoo ontology-aware learning (Phase D)

`src/models/train_multitask_gnn_v3.py` supports:

- `--ontology_loss`
- `--ontology_loss_role_weight`
- `--ontology_loss_transitivity_weight`
- `--ontology_loss_temporal_weight`
- `--ontology_max_triplets`

and persists ontology loss settings/final values into `multitask_metrics.json`.

### 5) Reporting + explanations (Phase E)

- `src/analysis/plot_multitask_linkpred_summary.py` now includes ontology columns, e.g.:
  - `ontology_conforms`
  - `ontology_violations_per_1k_edges`
  - ontology-loss diagnostics
- `src/run_all.py` adds:
  - `--run_reporting_summary`
  - `--write_explanations`
  - `--explanations_top_k`
- Generated artifacts:
  - `reporting_summary/multitask_linkpred_summary.csv`
  - `explanations/ontology_explanations.json`
    - includes `rule_chains` (violated/satisfied checks) and `confidence_alignment`

---

## Project Structure

```text
multiplex-terror-network-gnn/
├── README.md
├── PLANS.md
├── PLANS_KOR.md
├── requirements.txt
├── requirements.lock
├── configs/
│   ├── generator_easy.json
│   ├── generator_baseline.json
│   └── generator_hard.json
├── ontology/
│   ├── terror.ttl
│   └── constraints.shacl.ttl
├── src/
│   ├── run_all.py
│   ├── cli/
│   │   └── validate_ontology.py
│   ├── ontology/
│   │   ├── load.py
│   │   └── validator.py
│   ├── data/
│   │   ├── multiplex_generator_v3.py
│   │   ├── build_pyg_dataset_v3.py
│   │   └── basic_diagnostics_v3.py
│   ├── models/
│   │   ├── train_multitask_gnn_v3.py
│   │   ├── train_hvt_gnn_v3.py
│   │   └── train_linkpred_layer_v3.py
│   ├── analysis/
│   │   └── plot_multitask_linkpred_summary.py
│   └── validation/
│       └── schema.py
└── tests/
    └── test_*.py
```

---

## Installation

### 1) Clone

```bash
git clone https://github.com/Navy10021/multiplex-terror-network-gnn.git
cd multiplex-terror-network-gnn
```

### 2) Python env (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

(or use conda if preferred)

### 3) Install dependencies

```bash
pip install -r requirements.lock
```

If you need custom torch/pyg wheels for your CUDA/OS, use `requirements.txt` and install PyTorch/PyG first.

---

## Quick Start

### A) End-to-end run

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 1500 \
  --seed 2025 \
  --out_root results
```

### B) End-to-end with Phase E reporting artifacts

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 800 \
  --seed 2025 \
  --out_root results \
  --run_reporting_summary \
  --write_explanations
```

### C) Ontology mode presets

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 800 \
  --seed 2025 \
  --out_root results \
  --ontology_mode constrained
```

### D) Ontology-constrained generation mode (legacy flags)

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 800 \
  --seed 2025 \
  --out_root results \
  --ontology_constrained \
  --ontology_max_retries 3 \
  --ontology_retry_seed_stride 1
```

### E) Multitask training with ontology loss (Phase D)

```bash
python -m src.models.train_multitask_gnn_v3 \
  --data_path results/<your_run>/pyg_data.pt \
  --encoder transformer \
  --epochs 20 \
  --ontology_loss \
  --ontology_loss_role_weight 0.2 \
  --ontology_loss_transitivity_weight 0.1 \
  --ontology_loss_temporal_weight 0.1
```

---

## Ontology-based Terror Network: Current Status

### ✅ Implemented (current code)

- **Phase A**: validator depth expansion (rule-level violation objects, check histograms, temporal/relation/provenance checks).
- **Phase B**: ontology-constrained retry generation with telemetry.
- **Phase C**: PyG ontology bridge tensors + consistency checks.
- **Phase D**: ontology-aware multitask regularizers + ablation CLI + metrics logging.
- **Phase E**: ontology-aware reporting metrics + node-level explanation artifact generation.

### Key ontology artifacts produced per run

- `ontology_validation_report.json`
- `run_metadata.json` (including ontology paths/status/telemetry and Phase E artifact paths)
- `reporting_summary/multitask_linkpred_summary.csv` *(optional via flag)*
- `explanations/ontology_explanations.json` *(optional via flag)*

### Next-up focus

- richer rule-grounded explanations (model attribution + rule chain alignment),
- better calibration between model confidence and ontology conflict signals,
- broader ontology-aware benchmarking across difficulty presets and seeds.

---

## Plans

- English roadmap: `PLANS.md`
- Korean roadmap: `PLANS_KOR.md`

