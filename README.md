# Multiplex Terror Network GNN

Synthetic multiplex terrorist networks and multi-task GNNs for high-value target (HVT) detection and link prediction.

> ⚠️ **Disclaimer**  
> This repository uses **purely synthetic** data to study algorithms for network disruption and risk analysis.  
> No real individuals, organizations, or operational data are used.

---

## 🔍 TL;DR

- We generate **5-layer multiplex terrorist networks** (hierarchy, finance, communication, operations, ideology).
- We train a **multi-task GNN** that jointly learns:
  - High-value target (HVT) classification  
  - Role classification (courier, financier, leader, operative, support)  
  - Node-level importance regression
- We benchmark **layer-wise link prediction** (finance & communication) under multiple difficulty settings:
  - `easy` / `baseline` / `hard` (controlled structural noise & randomness)
- All experiments are fully reproducible via small command-line scripts.

- **One-line summary**: Generate 5-layer multiplex terrorist networks and train multi-task GNN baselines for HVT detection, role classification, and link prediction.

---

## 🧠 Motivation

Real-world terrorist and extremist networks are:

- **Multi-layered**: physical hierarchy, financing, communication, operations, ideology
- **Noisy and incomplete**: only partial observations, missing nodes/edges
- **Risk-sensitive**: analysts care about **who to disrupt** (HVTs), not just who is central

This repository provides a **research sandbox**:

1. A configurable **synthetic data generator** for multiplex terrorist networks.
2. A **multi-task GNN baseline** for *HVT detection + role classification + importance regression*.
3. A set of **layer-wise link prediction baselines** to quantify structural signal quality.

It is intended as a starting point for:

- Network science & security researchers
- GNN / representation learning researchers
- Practitioners exploring risk-aware disruption strategies in complex networks

---

## ✨ Key Features

- **Multiplex Generator (v2)**  
  - 5 layers: `hierarchy`, `finance`, `communication`, `operation`, `ideology`
  - Node attributes: region, group, role, continuous features
  - Difficulty knobs (per config):
    - `finance_structure_strength`
    - `comm_structure_strength`
    - `comm_randomness`
    - `hvt_ratio`, etc.

- **Config-driven Difficulty Levels**
  - `configs/generator_easy.json`
  - `configs/generator_baseline.json`
  - `configs/generator_hard.json`

- **PyG Dataset Builder**
  - Outputs a single `torch_geometric.data.Data` object:
    - `x`, `edge_index`, `edge_type`, `edge_attr`
    - `y_role`, `y_hvt`, `importance_score`
    - `train_mask`, `val_mask`, `test_mask`
    - Meta: `role_mapping`, `region_mapping`, `imp_mean`, `imp_std`, …

- **Models**
  - `MultiTaskRGCN`:
    - Shared R-GCN encoder
    - Heads: role classification, HVT classification, importance regression
  - `HvtRGCN`:
    - Single-task HVT baseline
  - Link prediction:
    - Layer-specific link prediction with R-GCN encoder + simple decoder
    - Uniform vs hard-negative sampling (`uniform`, `hard_region`)

- **Experiment Suite**
  - End-to-end scripts to:
    - Generate data for `easy`, `baseline`, `hard`
    - Build PyG datasets
    - Train multi-task & single-task models
    - Train link-prediction baselines
    - Summarize results (`multitask_linkpred_summary.csv`)

---

## 📁 Project Structure

Recommended layout:

```text
multiplex-terror-network-gnn/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── generator_easy.json
│   ├── generator_baseline.json
│   └── generator_hard.json
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── multiplex_generator_v1.py
│   │   ├── multiplex_generator_v2.py
│   │   ├── build_pyg_dataset.py
│   │   └── basic_diagnostics.py
│   │
│   ├── models/
│   │   ├── train_multitask_gnn.py
│   │   ├── train_hvt_gnn.py
│   │   └── train_linkpred_layer.py
│   │
│   └── analysis/
│       ├── plot_multitask_linkpred_summary.py
│       └── (optional notebooks, plots, etc.)
│
├── data/
│   ├── multiplex_easy/
│   ├── multiplex_baseline/
│   ├── multiplex_hard/
│   └── analysis/
│
├── notebooks/
│   └── multiplex-terror-network-gnn.ipynb
│
└── results/
      └── summary_all
```
You do **not** have to adopt this structure exactly, but the README assumes something close to this layout.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Navy10021/multiplex-terror-network-gnn.git
cd multiplex-terror-network-gnn
```

### 2. Create environment (conda, recommended)

```bash
conda create -n terror-gnn python=3.10 -y
conda activate terror-gnn
```

### 3. Install PyTorch + PyG
Follow the official instructions for your CUDA / OS. Example (CUDA 12.x):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
(Adjust the exact commands to match your CUDA / PyTorch version.)

### 4. Install Python dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start
End-to-end in three steps (works for `easy`, `baseline`, or `hard`).

1) **Generate a multiplex graph**
```bash
python src/data/multiplex_generator_v2.py \
  --config configs/generator_baseline.json \
  --out_dir data/multiplex_baseline
```

2) **Convert to a PyG dataset**
```bash
python src/data/build_pyg_dataset.py \
  --manifest data/multiplex_baseline/multiplex.json \
  --out_path data/multiplex_baseline/pyg_data.pt
```

3) **Train models**
- Multi-task HVT + role + importance
  ```bash
  python src/models/train_multitask_gnn.py \
    --data_path data/multiplex_baseline/pyg_data.pt \
    --hidden_dim 64 --num_layers 3 --epochs 500
  ```
- Layer-wise link prediction (finance or communication) with hard-negative sampling
  ```bash
  python src/models/train_linkpred_layer.py \
    --data_path data/multiplex_baseline/pyg_data.pt \
    --layer finance --neg_mode hard_region --epochs 500
  ```

Artifacts are stored next to the dataset (metrics JSON, training curves, link AUC/AP). Repeat with `configs/generator_baseline.json` or `configs/generator_hard.json` to sweep difficulty.

## 🧪 Smoke Tests & CI
- Run a minimal end-to-end pipeline (generator → PyG dataset) to catch regressions quickly:
  ```bash
  pytest tests/test_smoke_pipeline.py::test_generate_and_build_pyg_roundtrip
  ```
- This uses a small graph (120 nodes) and exercises v2 generator defaults, data conversion, masks, and tensor outputs. It is safe to run locally or in CI as a lightweight sanity check.

## 🗂️ Results & Logging Conventions
- Recommended layout for experiment outputs:
  ```text
  results/
    <difficulty>/          # easy | baseline | hard
      multitask/           # multi-task encoder runs
        <run_name>/
          metrics.json     # loss/accuracy/F1 per split
          curves.png       # optional training curves
      linkpred_finance/
        <run_name>/        # finance layer link prediction
          metrics.json
      linkpred_comm/
        <run_name>/        # communication layer link prediction
          metrics.json
  ```
- When launching training scripts, set `--output_dir` or similar arguments to follow this structure so runs remain comparable across difficulty levels and model types.

## 📊 Example Results (Summary)
Reference runs from `results/summary_all/multitask_linkpred_summary.csv`:

| Difficulty | HVT F1 | HVT AUC | Role F1 (macro) | Importance R² | Finance AUC (uniform) | Communication AUC (uniform) |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 0.25 | 0.893 | 0.575 | 0.551 | 0.981 | 0.864 |
| baseline | 0.50 | 0.918 | 0.280 | 0.331 | 0.980 | 0.864 |
| hard | 0.286 | 0.886 | 0.188 | 0.172 | 0.970 | 0.796 |

- Role F1 and importance R² degrade noticeably under the `hard` setting, while link prediction AUC stays robust—useful for stress-testing models against structural noise.

## 🛠️ Extending the Framework

- **Custom difficulty:** copy a config under `configs/` and tweak `finance_structure_strength`, `comm_structure_strength`, `comm_randomness`, and `hvt_ratio`. Pass it via `--config` to `multiplex_generator_v2.py`.
- **New node/edge features:** extend `generate_multiplex_with_config` in `src/data/multiplex_generator_v2.py` and ensure they are preserved in `build_pyg_dataset.py`.
- **Model variants:**
  - Add heads or encoders in `src/models/train_multitask_gnn.py` for alternative loss balancing or architectures.
  - Swap decoders or negative sampling in `src/models/train_linkpred_layer.py` to test other link-prediction strategies.
- **Reporting:** regenerate summary plots with `src/analysis/plot_multitask_linkpred_summary.py` after adding new runs.

## 📜 License

No explicit license is provided yet. For usage beyond personal or academic experimentation, please contact the maintainer.

## 📚 Citation

If you use this codebase in research, please cite the repository (or open an issue for a formal BibTeX entry):

```
Yoon-seop, Lee. (2025). Multiplex Terror Network GNN (GitHub repository). https://github.com/Navy10021/multiplex-terror-network-gnn
```

If you use this codebase in research, please cite the repository (or open an issue for a formal BibTeX entry):

```
Yoon-seop, Lee. (2025). Multiplex Terror Network GNN (GitHub repository). https://github.com/Navy10021/multiplex-terror-network-gnn
```

## 📬 Contact
For questions, issues, or collaboration:
  - GitHub Issues: please open an issue in this repository.
  - Email: iyunseob4@gmail.com
Contributions, bug reports, and ideas for new experiments are very welcome.




