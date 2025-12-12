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

한국어 한 줄 요약:  
5레이어 멀티플렉스 테러 네트워크를 합성하고, HVT 탐지·역할 분류·링크 예측을 위한 멀티태스크 GNN 실험 코드를 제공합니다.

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

## 📊 Example Results (Summary)

## 🛠️ Extending the Framework

## 📜 License

## 📚 Citation

## 📬 Contact
For questions, issues, or collaboration:
  - GitHub Issues: please open an issue in this repository.
  - Email: iyunseob4@gmail.com
Contributions, bug reports, and ideas for new experiments are very welcome.



