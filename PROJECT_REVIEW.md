# PROJECT_REVIEW — Multiplex Terror Network GNN (v3)

## 1. Project Goal

This project provides a **synthetic multiplex network benchmark** designed to evaluate GNN performance under conditions that resemble real operational constraints:
- Partial observability / missing edges
- Noisy observations (false edges)
- Cross-layer copying / provenance effects
- Temporal event streams aggregated into static training graphs
- Multi-task supervision on nodes + link prediction on selected layers

The emphasis is on **reproducible experimentation** and **controlled difficulty scaling**.

---

## 2. Current Pipeline (v3)

### 2.1 Data Generation (multiplex_generator_v3.py)
**Input**
- `--size`, `--seed`, `--out_dir`
- `--config configs/generator_*.json`

**Output**
- `multiplex.json` (single manifest containing nodes, layer edges, events, labels, and metadata)

**Design intent**
- Generator configs provide difficulty knobs (structure strength, randomness, missingness, false/copy edges, observation bias, activity gating).
- `multiplex.json` is treated as the “single source of truth” for all subsequent steps.

---

### 2.2 Dataset Build (build_pyg_dataset_v3.py)
Converts `multiplex.json` → `pyg_data.pt` (PyTorch Geometric `Data`).

**Key contents**
- `x`: node features (categorical one-hot + continuous attributes)
- `y_role`: role class labels
- `y_hvt`: binary HVT labels
- `y_imp`: continuous importance score
- `train_mask`, `val_mask`, `test_mask`
- `edge_index`, `edge_type` (multiplex relation types)
- `edge_attr` (aggregated edge statistics, if present)
- Optional edge provenance flags (if generated), e.g.:
  - `edge_is_false`: injected false edge
  - `edge_is_copied`: copied edge (cross-layer provenance)

**Rationale**
- Keeping provenance flags in `Data` allows:
  - Diagnostics to quantify noise/copy behavior
  - Models to optionally incorporate edge reliability signals

---

### 2.3 Diagnostics (basic_diagnostics_v3.py)
Purpose: confirm that **knobs → measurable statistical changes**.

**Outputs**
- `*.png`: degree distributions per layer, role/region/group counts, role-wise degree boxplots, etc.
- `*.csv`: overlap, noise summaries, burstiness, activity/observability bias, copy provenance breakdowns, etc.

**What to look for**
- Degree distributions shift appropriately as structure/randomness knobs change.
- Layer overlap/Jaccard behaves as expected as cross-layer copying increases.
- False/copy rates match config within tolerance.
- Observability bias summaries reflect intended skew.

---

## 3. Modeling & Training (v3)

### 3.1 Multi-task Node Prediction (train_multitask_gnn_v3.py)
Jointly predicts:
- Role classification
- HVT classification
- Importance regression

**Typical artifacts**
- `multitask_metrics.json`
- `multitask_plots/` (curves & summaries)

**Evaluation**
- Role: macro-F1 / accuracy
- HVT: ROC-AUC / PR-AUC / F1 (threshold strategy supported)
- Importance: R² / RMSE

---

### 3.2 Layer-wise Link Prediction (train_linkpred_layer_v3.py)
Targets: `finance` or `communication`.

**Key design choice: leakage-safe message passing**
For the target layer LP task, validation/test positive edges are removed from the encoder message-passing graph.  
This ensures the model cannot “see” held-out positives during representation learning.

**Edge attributes & flags**
Two-stage handling:
1) `--edge_attr_agg`: aggregates edge attributes into node-level signals (training edges only for leakage safety)
2) `--include_edge_flags`: also aggregate edge provenance flags (`edge_is_false`, `edge_is_copied`) into the node signals

**Negatives**
- `--neg_mode uniform`: standard negative sampling
- `--neg_mode hard_region`: region-aware hard negatives (more realistic confusability)

**Artifacts**
- `linkpred_<layer>_<neg_mode>_v3.json`

---

## 4. Known Gaps / Technical Debt

1) **Config hash standardization**
- Some scripts can pass through generator metadata; ensure consistent, explicit hashing of generator configs for result tracking.

2) **Unified experiment runner**
- A single `run_all_sanity_checks.py` or `run_experiments.py` would reduce friction and improve reproducibility.

3) **Result summarization**
- Provide one canonical summarizer that merges:
  - `multitask_metrics.json`
  - `linkpred_*_v3.json`
  into one CSV + one clean plot (publication-ready).

4) **Testing**
- Add unit tests for:
  - generator output schema validation
  - build_pyg_dataset conversions (masks, edge flags, sizes)
  - leakage-safe edge removal logic
  - deterministic outputs under fixed seeds

---

## 5. Recommended Next Improvements (High ROI)

### A) Difficulty calibration suite
Automate “difficulty curves”:
- Sweep key knobs (missingness/false/copy rates, comm_randomness, structure strength)
- Run diagnostics + models
- Produce a single comparative report for easy/baseline/hard (or custom grids)

### B) Stronger realism for link prediction
- Add time-aware splitting for edges (temporal holdout)
- Evaluate generalization across time, not just random edge splits

### C) Ablations tied to operational constraints
Examples:
- Without edge flags vs with edge flags
- Uniform negatives vs hard-region negatives
- With/without edge_attr_agg
- With/without cross-layer copy mechanisms in the generator

---

## 6. Suggested “What to Claim” (Paper/Report Positioning)

- A controllable synthetic multiplex benchmark with explicit operational constraints
- A leakage-safe link prediction protocol for multiplex layers
- Evidence that noise/observability/copy knobs produce measurable graph shifts (diagnostics)
- Empirical results showing which modeling choices are robust under degraded observability

---

## 7. How to Reproduce (Minimal)

1) Generate:
```bash
python src/multiplex_generator_v3.py --size 1500 --seed 2025 --out_dir data/multiplex_baseline --config configs/generator_baseline.json
```

2) Build:
```bash
python src/build_pyg_dataset_v3.py --manifest data/multiplex_baseline/multiplex.json --out_path data/multiplex_baseline/pyg_data.pt --seed 2025
```

3) Diagnostics:
```bash
python src/basic_diagnostics_v3.py --manifest data/multiplex_baseline/multiplex.json --out_dir data/multiplex_baseline/diagnostics
```

4) Train multi-task:
```bash
python src/models/train_multitask_gnn_v3.py --data_path data/multiplex_baseline/pyg_data.pt
```

5) Train link prediction:
```bash
python src/models/train_linkpred_layer_v3.py --data_path data/multiplex_baseline/pyg_data.pt --layer finance --neg_mode hard_region --edge_attr_agg --include_edge_flags
```

---

## 8. Notes on Responsible Use

This project uses synthetic data and is intended for defensive research and benchmarking.  
Avoid interpreting it as operational intelligence or guidance.
