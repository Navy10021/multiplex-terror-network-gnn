<h1 align="center">Multiplex Terror Network GNN</h1>

<p align="center">
  <b>Synthetic multiplex terrorist networks</b> + <b>multi-task GNNs</b> for
  <b>HVT detection</b>, <b>role inference</b>, and <b>layer-aware link prediction</b>.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-red" />
  <img alt="PyG" src="https://img.shields.io/badge/PyG-2.x-7f3fbf" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

> **Disclaimer**  
> This repository uses **purely synthetic** data to study algorithms for network disruption and risk analysis.  
> **No real individuals, organizations, or operational data** are used.

---

## 🔍 TL;DR

A **purpose-built research sandbox** for disruption and risk analysis on **purely synthetic** multiplex terrorist networks — designed to be **safe, configurable, and reproducible**.

- **Generate** 5-layer multiplex graphs *(hierarchy, finance, communication, operations, ideology)* with controllable structure, noise, and difficulty knobs.
- **Train** a **multi-task R-GCN** for:
  - **HVT detection** (high-value target classification)
  - **Role inference** *(courier, financier, leader, operative, support)*
  - **Node-level importance regression** (continuous criticality score)
- **Benchmark** **layer-aware link prediction** on **finance** & **communication** edges using uniform negatives or **hard-negative sampling**.
- **Reproduce** end-to-end results with **three short CLI commands**.

> **One-line summary:** Generate 5-layer synthetic multiplex terror networks and train multi-task GNN baselines for HVT detection, role inference, importance scoring, and layer-aware link prediction.

---

## 🧠 Motivation

Operational terrorist/extremist networks are difficult to study rigorously because they are:

- **Multiplex**: interactions span hierarchy, financing, communication, operations, and ideology
- **Noisy & incomplete**: partial observability, missing nodes/edges, sampling bias
- **Risk-sensitive**: analysts care about **actionability** (who to disrupt), not only generic centrality

This repository provides a **safe, reproducible sandbox** that captures those realities **without using any real-world data**:

1. **Configurable synthetic generator** for multiplex terrorist networks (with explicit difficulty controls).
2. **Multi-task GNN baselines** for HVT detection, role classification, and importance regression.
3. **Layer-wise link prediction benchmarks** to quantify how much signal each layer provides as difficulty increases.

**Designed for:**
- Network science & security research
- GNN / representation learning experimentation
- Prototyping **risk-aware disruption** strategies on complex graphs

---

## ✨ Highlights

- **Multiplex Generator (v2)**
  - 5 layers: `hierarchy`, `finance`, `communication`, `operation`, `ideology`
  - Node attributes: **region**, **group**, **role**, plus continuous feature vectors
  - Configurable difficulty knobs (examples): `finance_structure_strength`, `comm_structure_strength`, `comm_randomness`, `hvt_ratio`, ...

- **Config-Driven Difficulty**
  - Ready-to-run presets:
    - `configs/generator_easy.json`
    - `configs/generator_baseline.json`
    - `configs/generator_hard.json`
  - Swap configs to **stress-test robustness** under increasing structural noise

- **PyG Dataset Builder**
  - Produces a single `torch_geometric.data.Data` object containing:
    - Graph: `x`, `edge_index`, `edge_type`, `edge_attr`
    - Labels: `y_role`, `y_hvt`, `importance_score`
    - Splits: `train_mask`, `val_mask`, `test_mask`
    - Metadata: `role_mapping`, `region_mapping`, `imp_mean`, `imp_std`, ...

- **Model Zoo**
  - `MultiTaskRGCN`: shared R-GCN encoder + task heads (**role**, **HVT**, **importance**)
  - `HvtRGCN`: single-task HVT baseline
  - Link prediction: R-GCN encoder + decoder with negative sampling modes:
    - `uniform`
    - `hard_region` (hard negatives constrained by region)

- **Experiment Suite**
  - Shell-friendly scripts for end-to-end runs: **generate → build → train → evaluate**
  - Automated result aggregation to `multitask_linkpred_summary.csv`

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
  --config configs/generator_easy.json \
  --out_dir data/multiplex_easy
```

2) **Convert to a PyG dataset**
```bash
python src/data/build_pyg_dataset.py \
  --manifest data/multiplex_easy/multiplex.json \
  --out_path data/multiplex_easy/pyg_data.pt
```

3) **Train models**
- Multi-task HVT + role + importance
  ```bash
  python src/models/train_multitask_gnn.py \
    --data_path data/multiplex_easy/pyg_data.pt \
    --hidden_dim 128 --num_layers 3 --epochs 300
  ```
- Layer-wise link prediction (finance or communication) with hard-negative sampling
  ```bash
  python src/models/train_linkpred_layer.py \
    --data_path data/multiplex_easy/pyg_data.pt \
    --layer finance --neg_mode hard_region --epochs 200
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

A license has **not** been selected yet.

- For **personal or academic experimentation**, you may use the code as-is.
- For any **commercial, operational, or redistributed** use, please contact the maintainer.

If you plan to make this repository broadly reusable, consider adding an OSI-approved license (e.g., **MIT**, **Apache-2.0**) and including a `LICENSE` file at the project root.

---

## 📚 Citation

If you use this repository in academic work, please cite it as:

```text
Lee, Yoon-seop. (2025). Multiplex Terror Network GNN (GitHub repository).
https://github.com/Navy10021/multiplex-terror-network-gnn
```

## 📬 Contact
For questions, issues, or collaboration:
  - GitHub Issues: please open an issue in this repository.
  - Email: iyunseob4@gmail.com
Contributions, bug reports, and ideas for new experiments are very welcome.







