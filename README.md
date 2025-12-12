<h1 align="center">Multiplex Terror Network GNN</h1>

<p align="center">
  <b>Synthetic multiplex terrorist networks</b> + <b>multi-task GNN baselines</b> for
  <b>HVT detection</b>, <b>role inference</b>, <b>importance scoring</b>, and <b>layer-aware link prediction</b>.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-red" />
  <img alt="PyG" src="https://img.shields.io/badge/PyG-2.x-7f3fbf" />
  <img alt="Data" src="https://img.shields.io/badge/Data-100%25%20synthetic-lightgrey" />
  <img alt="License" src="https://img.shields.io/badge/License-TBD-lightgrey" />
</p>

---

> **Disclaimer**
>
> This repository uses **purely synthetic** networks to study algorithms for disruption and risk analysis.
> **No real individuals, organizations, or operational data** are used.

## 🔍 TL;DR

A **purpose-built research sandbox** for disruption and risk analysis on **purely synthetic** multiplex terrorist networks — designed to be **configurable and reproducible**.

- **Generate** 5-layer multiplex graphs *(hierarchy, finance, communication, operation, ideology)* with controllable structure, noise, and difficulty knobs.
- **Train** a **multi-task R-GCN** for:
  - **HVT detection** (high-value target classification)
  - **Role inference** *(courier, financier, leader, operative, support)*
  - **Node-level importance regression** (continuous criticality score)
- **Benchmark** **layer-aware link prediction** on **finance** & **communication** edges using:
  - `uniform` negatives
  - `hard_region` hard-negative sampling (negatives constrained by region)
- **Reproduce** end-to-end results via CLI (generator → PyG dataset → train → summarize).

---

## Table of Contents

- [Motivation](#-motivation)
- [Highlights](#-highlights)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Outputs](#-outputs)
- [Example Results](#-example-results-summary)
- [Reproducing Summary Plots](#-reproducing-summary-plots)
- [Configuration](#-configuration-generator-knobs)
- [Smoke Tests](#-smoke-tests--ci)
- [Extending the Framework](#-extending-the-framework)
- [Ethical Considerations](#-ethical-considerations)
- [License](#-license)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 🧠 Motivation

Operational terrorist/extremist networks are difficult to study rigorously because they are often:

- **Multiplex**: interactions span hierarchy, financing, communication, operations, and ideology
- **Noisy & incomplete**: partial observability, missing nodes/edges, sampling bias
- **Risk-sensitive**: analysts care about **actionability** (who to disrupt), not only generic centrality

This repository provides a **safe, reproducible sandbox** that captures these realities **without any real-world data**:

1. **Config-driven multiplex generator** (explicit difficulty controls).
2. **Multi-task GNN baselines** (HVT, role, importance).
3. **Layer-wise link prediction benchmarks** to quantify per-layer signal under increasing noise/difficulty.

---

## ✨ Highlights

### 1) Multiplex Generator (v2)
- 5 layers: `hierarchy`, `finance`, `communication`, `operation`, `ideology`
- Node attributes: `region`, `group`, `role`, plus continuous feature vectors
- Difficulty knobs: `finance_structure_strength`, `comm_structure_strength`, `comm_randomness`, `hvt_ratio`, etc.
- Presets:
  - `configs/generator_easy.json`
  - `configs/generator_baseline.json`
  - `configs/generator_hard.json`

### 2) PyG Dataset Builder
Produces a single `torch_geometric.data.Data` object containing:
- Graph: `x`, `edge_index`, `edge_type`, `edge_attr`
- Labels: `y_role`, `y_hvt`
- Splits: `train_mask`, `val_mask`, `test_mask`
- Metadata: `role_mapping`, `region_mapping`, (and optional importance normalization stats)

### 3) Model Zoo
- `MultiTaskRGCN`: shared R-GCN encoder + task heads (**role**, **HVT**, **importance**)
- `HvtRGCN`: single-task HVT baseline
- Link prediction: R-GCN encoder + decoder with negative sampling modes:
  - `uniform`
  - `hard_region`

### 4) Experiment / Reporting Suite
- Shell-friendly scripts for end-to-end runs: **generate → build → train → evaluate**
- Aggregation + plots from run folders (see `src/analysis/plot_multitask_linkpred_summary.py`)

---

## 📁 Project Structure

Recommended layout:

```text
multiplex-terror-network-gnn/
├── README.md
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
│   ├── data/
│   │   ├── multiplex_generator_v1.py
│   │   ├── multiplex_generator_v2.py
│   │   ├── build_pyg_dataset.py
│   │   └── basic_diagnostics.py
│   ├── models/
│   │   ├── train_multitask_gnn.py
│   │   ├── train_hvt_gnn.py
│   │   └── train_linkpred_layer.py
│   └── analysis/
│       └── plot_multitask_linkpred_summary.py
│
├── data/
│   ├── multiplex_easy/
│   ├── multiplex_baseline/
│   ├── multiplex_hard/
│   └── analysis/
│
└── results/
    └── summary_all/
```

---

## ⚙️ Installation

### 1) Clone the repository
```bash
git clone https://github.com/Navy10021/multiplex-terror-network-gnn.git
cd multiplex-terror-network-gnn
```

### 2) Create environment (conda, recommended)
```bash
conda create -n terror-gnn python=3.10 -y
conda activate terror-gnn
```

### 3) Install PyTorch + PyG
Install PyTorch / PyG based on your OS + CUDA setup. Example (CUDA 12.x):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install PyG following the official instructions for your exact PyTorch/CUDA combination.

### 4) Install remaining dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

End-to-end in **four** steps (works for `easy`, `baseline`, or `hard`).

### 1) Generate a multiplex graph
```bash
python src/data/multiplex_generator_v2.py \
  --size 1500 \
  --seed 2025 \
  --out_dir data/multiplex_baseline \
  --config configs/generator_baseline.json
```

This will create `data/multiplex_baseline/multiplex.json`.

### 2) Convert to a PyG dataset
```bash
python src/data/build_pyg_dataset.py \
  --manifest data/multiplex_baseline/multiplex.json \
  --out_path data/multiplex_baseline/pyg_data.pt
```

### 3) Run diagnostics (optional but recommended)
```bash
python src/data/basic_diagnostics.py \
  --manifest data/multiplex_baseline/multiplex.json \
  --out_dir data/analysis/multiplex_baseline
```

### 4) Train models
Multi-task HVT + role + importance:
```bash
python src/models/train_multitask_gnn.py \
  --data_path data/multiplex_baseline/pyg_data.pt \
  --hidden_dim 64 --num_layers 3 --lr 1e-3 --epochs 500
```

Layer-wise link prediction (finance or communication):
```bash
python src/models/train_linkpred_layer.py \
  --data_path data/multiplex_baseline/pyg_data.pt \
  --layer finance \
  --hidden_dim 64 --num_layers 3 --lr 1e-3 \
  --neg_mode hard_region \
  --epochs 500
```

Repeat with `configs/generator_easy.json` or `configs/generator_hard.json` to sweep difficulty.

---

## 🗂️ Outputs

By default, training artifacts are saved **next to** the dataset (`--data_path` directory).

Example after running the commands above:

```text
data/multiplex_baseline/
├── multiplex.json
├── pyg_data.pt
├── multitask_metrics.json
├── hvt_metrics.json                      # if you run train_hvt_gnn.py
├── linkpred_finance_uniform.json
├── linkpred_finance_hard_region.json
├── linkpred_communication_uniform.json
├── linkpred_communication_hard_region.json
└── multitask_plots/
    ├── loss_curves.png
    ├── hvt_auc_curve.png
    └── ...
```

If you prefer a `results/<difficulty>/<run_name>/...` layout, the simplest option is:
- create the run directory
- place (or copy) `pyg_data.pt` there
- train using `--data_path` pointing at that run directory

This works because the scripts write `*_metrics.json` and plot folders to `os.path.dirname(data_path)`.

---

## 📊 Example Results (Summary)

Reference runs from `results/summary_all/multitask_linkpred_summary.csv`:

| Difficulty | HVT F1 | HVT AUC | Role F1 (macro) | Importance R² | Finance AUC (uniform) | Communication AUC (uniform) |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 0.25 | 0.893 | 0.575 | 0.551 | 0.981 | 0.864 |
| baseline | 0.50 | 0.918 | 0.280 | 0.331 | 0.980 | 0.864 |
| hard | 0.286 | 0.886 | 0.188 | 0.172 | 0.970 | 0.796 |

Notes:
- HVT metrics in the summary use **threshold tuning** on the validation set (see `multitask_metrics.json`).
- Role F1 and importance R² degrade more under `hard`, while link prediction AUC can remain relatively robust—useful for stress-testing representation learning under structural noise.

---

## 📈 Reproducing Summary Plots

After collecting multiple run folders (e.g., `results/run_easy_...`, `results/run_hard_...`), you can aggregate and plot:

```bash
python src/analysis/plot_multitask_linkpred_summary.py \
  --run_dirs results/run_easy results/run_baseline results/run_hard \
  --out_dir results/summary_all
```

The script expects each run directory to contain:
- `multitask_metrics.json`
- `linkpred_<layer>_<neg_mode>.json`
- optionally, `multiplex.json` (for reading generator config)

---

## ⚙️ Configuration (Generator Knobs)

All presets live under `configs/`. The most important parameters:

- `size`: number of nodes
- `hvt_ratio`: fraction of HVT nodes
- `finance_structure_strength`: stronger → more structured/clustered finance edges
- `comm_structure_strength`: stronger → more structured communication edges
- `comm_randomness`: stronger → noisier / less structured communication edges
- (optional) any additional knobs you introduce in `multiplex_generator_v2.py`

Workflow:
1) Copy an existing config (e.g., `generator_baseline.json`)
2) Modify knobs
3) Pass it via `--config` to `multiplex_generator_v2.py`

---

## 🧪 Smoke Tests & CI

Run a minimal end-to-end pipeline (generator → PyG dataset) to catch regressions quickly:

```bash
pytest tests/test_smoke_pipeline.py::test_generate_and_build_pyg_roundtrip
```

This uses a small graph (120 nodes) and exercises:
- v2 generator defaults
- PyG conversion
- masks and tensor outputs

---

## 🛠️ Extending the Framework

- **Custom difficulty:** copy a config under `configs/` and tweak `finance_structure_strength`, `comm_structure_strength`, `comm_randomness`, and `hvt_ratio`. Pass it via `--config` to `multiplex_generator_v2.py`.
- **New node/edge features:** extend `generate_multiplex_with_config` in `src/data/multiplex_generator_v2.py` and ensure they are preserved in `build_pyg_dataset.py`.
- **Model variants:**
  - Add heads or encoders in `src/models/train_multitask_gnn.py` for alternative loss balancing or architectures.
  - Swap decoders or negative sampling in `src/models/train_linkpred_layer.py` to test other link-prediction strategies.
- **Reporting:** regenerate summary plots with `src/analysis/plot_multitask_linkpred_summary.py` after adding new runs.

---

## 🔒 Ethical Considerations

This repository is built with **defensive, academic purposes** in mind:

- ✅ **Synthetic Data Only**  
  All networks are generated synthetically. No real persons, organizations, or operational data are used.
- ✅ **Counter-Terrorism & Lawful Use**  
  Intended for legitimate research in **counter-terrorism**, **criminal network analysis**, and **resilience planning**.
- ✅ **Transparency & Reproducibility**  
  Open code for peer review and academic scrutiny.

**Do NOT use this code to:**
- Target legitimate political groups or civil organizations,
- Conduct unauthorized surveillance,
- Suppress lawful protest, free speech, or assembly,
- Analyze real social networks without proper legal and ethical oversight.

---

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

BibTeX (optional):
```bibtex
@misc{lee2025multiplexterror,
  author       = {Lee, Yoon-seop},
  title        = {Multiplex Terror Network GNN},
  year         = {2025},
  howpublished = {GitHub repository},
  url          = {https://github.com/Navy10021/multiplex-terror-network-gnn}
}
```

---

## 📬 Contact

For questions, issues, or collaboration:
- GitHub Issues: please open an issue in this repository.
- Email: iyunseob4@gmail.com

Contributions, bug reports, and ideas for new experiments are welcome.
