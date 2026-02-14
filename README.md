<h1 align="center">🔬🕸️ Multiplex Terror Network GNN</h1>

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

> **Purpose & Safety Notice (Defensive CT Research Only)**
>
> This repository is intended for **defensive, lawful counter-terrorism (CT)** and **criminal-network analysis** research—e.g., **disruption strategy evaluation**, **risk scoring**, and **resilience planning** on multiplex networks.
> All data are **100% synthetic**. This code is **not** intended for **operational targeting**, **real-world surveillance**, or analysis of real social networks.
> See **[Ethical Considerations](#-ethical-considerations)** for allowed and prohibited uses.


---

## 🔍 TL;DR

A **purpose-built research sandbox** for disruption and risk analysis on **purely synthetic multiplex terrorist networks** — designed to be configurable, leakage-aware, and reproducible.

- **Generate** multi-layer multiplex graphs (e.g., hierarchy, finance, communication, operation, ideology) with explicit knobs for **structure strength**, **randomness**, **missingness/observability**, **false edges**, and **cross-layer copied edges** (provenance-aware noise).
- **Train** a **multi-task GNN (R-GCN / Transformer-style encoder options)** for
  - **HVT detection** *(high-value target classification)*
  - **Role inference** *(courier, financier, leader, operative, support)*
  - **Node-level importance regression** *(continuous criticality score)*
- **Benchmark** **layer-aware link prediction** on **finance** & **communication** edges with:
  - `uniform` negatives
  - `hard_region` hard-negative sampling (negatives constrained by region)
- **Reproduce** end-to-end results via CLI (**generator → PyG dataset → diagnostics → train → summarize**) with seeds + configs tracked for clean comparisons.

---

## 🧠 Motivation

Operational terrorist/extremist networks are difficult to study rigorously because they are often:

- **Multiplex**: interactions span hierarchy, financing, communication, operations, and ideology (with meaningful cross-layer dependencies).
- **Noisy & incomplete**: partial observability, layer-dependent missingness, spurious links (**false edges**), and systematic sampling/visibility bias.
- **Risk-sensitive**: analysts care about **actionability** (who to prioritize/disrupt), not only generic centrality—while models must avoid evaluation artifacts (e.g., edge/label leakage).

This repository provides a **safe, reproducible sandbox** that captures these realities **without any real-world data**:

1. **Config-driven multiplex generator (v3)** with explicit knobs for structure strength, randomness, missingness/observability, false-edge injection, and **cross-layer copied edges** (provenance-aware noise).
2. **Multi-task GNN baselines (v3)** for node-level objectives: **HVT**, **role**, and **importance** (continuous criticality).
3. **Layer-wise link prediction benchmarks (v3)** (e.g., finance/communication) with **uniform vs. hard-region negatives** and a **leakage-safe message-passing protocol** to quantify per-layer signal as difficulty/noise increases.


---

## ✨ Highlights

### 1) Multiplex Generator (v3)
- **5+ layers (relation types)**: `hierarchy`, `finance`, `communication`, `operation`, `ideology` *(extensible)*
- **Node attributes**: `region`, `group`, `role`, plus continuous feature vectors
- **Config-driven difficulty knobs** (examples):
  - Layer structure strength / homophily (community tightness)
  - Layer randomness (esp. communication)
  - **Missingness / observability** (layer-dependent edge drops + visibility bias)
  - **False edges** injection (spurious links)
  - **Cross-layer copied edges** (provenance-aware noise / leakage-like effects)
  - `hvt_ratio`, role priors, burstiness / activity gating (if enabled in config)
- **Presets**:
  - `configs/generator_easy.json`
  - `configs/generator_baseline.json`
  - `configs/generator_hard.json`

> Output is a single manifest: `multiplex.json` (single source of truth for build/diagnostics/training).

### 2) PyG Dataset Builder (v3)
Produces a single `torch_geometric.data.Data` object containing:
- **Graph**
  - `x`, `edge_index`, `edge_type`
  - `edge_attr` *(optional, if built/used)*
  - Optional provenance flags (if present in manifest): `edge_is_false`, `edge_is_copied`
  - Ontology bridge tensors: `edge_ontology_attr`, `node_ontology_features`, `role_compatibility_mask`
- **Labels**
  - `y_role` (multi-class)
  - `y_hvt` (binary)
  - `y_imp` (continuous importance / criticality)
- **Splits**
  - `train_mask`, `val_mask`, `test_mask`
- **Metadata**
  - `role_mapping`, `region_mapping`, `group_mapping` (where applicable)
  - Optional normalization stats for importance/regression targets

### 3) Model Zoo (v3)
- **Multi-task node model**: shared encoder + task heads (**role**, **HVT**, **importance**)
  - Implemented in `train_multitask_gnn_v3.py` (R-GCN / Transformer-style encoder options depending on flags)
  - Phase D: optional ontology-aware regularization (`--ontology_loss`) with weights for role-compatibility/transitivity/temporal constraints
- **Single-task HVT baseline**: `train_hvt_gnn_v3.py`
- **Layer-wise link prediction**: `train_linkpred_layer_v3.py`
  - Negative sampling modes:
    - `uniform`
    - `hard_region` (region-constrained hard negatives)
  - **Leakage-safe message passing**: held-out positives for the target layer are removed from the encoder graph
  - Optional edge-signal usage:
    - `--edge_attr_agg` (aggregate edge attributes into node signals)
    - `--include_edge_flags` (aggregate provenance flags like `edge_is_false/edge_is_copied`)

### 4) Experiment / Reporting Suite
- Shell-friendly scripts for end-to-end runs: **generate → build → diagnostics → train → evaluate**
- Diagnostics for knob validation: `basic_diagnostics_v3.py`
- Aggregation + plots across run folders:
  - `plot_multitask_linkpred_summary.py` (merges multi-task + link-pred outputs into a compact summary)


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
│   │   ├── multiplex_generator_v3.py
│   │   ├── build_pyg_dataset_v3.py
│   │   └── basic_diagnostics_v3.py
│   ├── models/
│   │   ├── train_multitask_gnn_v3.py
│   │   ├── train_hvt_gnn_v3.py
│   │   └── train_linkpred_layer_v3.py
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
pip install -r requirements.lock  # fully pinned
# or
pip install -r requirements.txt   # if you need to adjust torch/pyg for your CUDA
```

---

## 🚀 Quick Start

**One command (auto-managed run folder + metadata)**

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 1500 \
  --seed 2025 \
  --out_root results/
```

This creates a UTC-timestamped run directory (pattern: `run_<date>_<config-hash>_seed<seed>`) containing:
- `multiplex.json` (validated manifest)
- `pyg_data.pt` (PyG dataset)
- `diagnostics/` (plots + CSVs)
- `run_metadata.json` (config hash, git commit, CLI command)
- `DATASET_CARD.md` (layer-wise edge noise/copied rates + artifact paths)

Manifest validation enforces that node IDs are contiguous (`0..N-1`), `meta.num_nodes` matches the node list, and every edge/event endpoint exists—catching malformed inputs early before building datasets or training models.

To validate an existing manifest on its own (e.g., after editing or before sharing), run:

```bash
python -m src.validation.schema data/multiplex_baseline/multiplex.json --summary
```

For ontology-backed rule validation (role/relation and temporal checks), run:

```bash
python -m src.cli.validate_ontology \
  --manifest data/multiplex_baseline/multiplex.json \
  --ontology ontology/terror.ttl \
  --shapes ontology/constraints.shacl.ttl
```

Optional JSON report:

```bash
python -m src.cli.validate_ontology \
  --manifest data/multiplex_baseline/multiplex.json \
  --json
```

`src/data/multiplex_generator_v3.py` and `src/run_all.py` also run ontology checks and write `ontology_validation_report.json` by default. Use `--no_ontology_strict` to keep running while recording violations. For retry-based constrained generation (Phase B), enable `--ontology_constrained` with `--ontology_max_retries` and `--ontology_retry_seed_stride`.

## 🧭 Ontology-based Terror Network: Current Status & Next Focus

### What is already reflected
- OWL/SHACL ontology assets are included (`ontology/terror.ttl`, `ontology/constraints.shacl.ttl`).
- Manifest-level ontology validation is available via `python -m src.cli.validate_ontology`.
- Generator/runner flows write `ontology_validation_report.json` so conformance is tracked as an artifact.

### What to evolve next (research-focused)
- **Generation**: move from post-hoc validation to ontology-constrained sampling with repair telemetry.
- **Builder (PyG)**: expose ontology-derived masks/tensors (role-pair constraint masks, temporal violation candidates).
- **Model Zoo**: add ontology-aware losses (role–relation compatibility, transitivity, temporal consistency).
- **Reporting**: add ontology compliance/violation metrics and rule-grounded explanation outputs per node.

Detailed forward plans are maintained in `PLANS.md` (EN) and `PLANS_KOR.md` (KO), focused on next-stage ontology-driven code evolution.

This exits non-zero on any schema error and prints node/edge/layer/event counts when `--summary` is provided.

---

End-to-end in **four** steps (works for `easy`, `baseline`, or `hard`).

### 1) Generate a multiplex graph
```bash
python src/data/multiplex_generator_v3.py \
  --size 1500 \
  --seed 2025 \
  --out_dir data/multiplex_baseline \
  --config configs/generator_baseline.json
```

This will create `data/multiplex_baseline/multiplex.json`.

### 2) Convert to a PyG dataset
```bash
python src/data/build_pyg_dataset_v3.py \
  --manifest data/multiplex_baseline/multiplex.json \
  --out_path data/multiplex_baseline/pyg_data.pt
```

### 3) Run diagnostics (optional but recommended)
```bash
python src/data/basic_diagnostics_v3.py \
  --manifest data/multiplex_baseline/multiplex.json \
  --out_dir data/analysis/multiplex_baseline
```

### 4) Train models
Multi-task HVT + role + importance:
```bash
python src/models/train_multitask_gnn_v3.py \
  --data_path data/multiplex_baseline/pyg_data.pt \
  --hidden_dim 64 --num_layers 3 --lr 1e-3 --epochs 500
```

Layer-wise link prediction (finance or communication):
```bash
python src/models/train_linkpred_layer_v3.py \
  --data_path data/multiplex_baseline/pyg_data.pt \
  --layer finance \
  --hidden_dim 64 --num_layers 3 --lr 1e-3 \
  --neg_mode hard_region \
  --epochs 500
```

Repeat with `configs/generator_easy.json` or `configs/generator_hard.json` to sweep difficulty.


## 🧪 Jupyter Notebook (Optional)

This repo supports an optional notebook workflow for rapid prototyping, visualization, and debugging.
A Jupyter notebook is provided (e.g., ./notebooks/multiplex-terror-network-gnn.ipynb) that:
- Generates configurable synthetic multiplex network data
- GNN-based model


## 📊 Example Results

Below are representative results from the **current v3 summary export** (`multitask_linkpred_summary.csv`) on the synthetic multiplex benchmarks (**n=1500**, seed=2025, HVT ratio=0.07).

### Multi-task node prediction (v3, Node-level)

| Difficulty | HVT F1 | HVT AUC | Role F1 (macro) | Importance R² |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.619 | 0.977 | 0.568 | 0.732 |
| hard | 0.611 | 0.959 | 0.647 | 0.704 |

### Link Prediction (v3, Edge-level)

> Note: In the current `multitask_linkpred_summary.csv` export, link prediction columns are empty (NaN), which usually means the corresponding `linkpred_*_v3.json` artifacts were not found/merged.  
> Run `train_linkpred_layer_v3.py` for each layer and re-run `plot_multitask_linkpred_summary.py` to populate this table.

| Difficulty | Finance LP AUC/AP | Comm LP AUC/AP |
| --- | --- | --- |
| baseline | — | — |
| hard | — | — |

For link prediction, we evaluate two negative-sampling protocols and (optionally) report the better AUC/AP per difficulty setting:
- `uniform`: negatives sampled uniformly at random
- `hard_region`: negatives sampled from the same region (harder discrimination)

Quick read (from the node tasks above): **baseline** is slightly better on **HVT AUC** and **importance R²**, while **hard** improves **role macro-F1**.


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

## 🛠️ Extending the Framework

- **Custom difficulty:** copy a config under `configs/` and tweak `finance_structure_strength`, `comm_structure_strength`, `comm_randomness`, and `hvt_ratio`. Pass it via `--config` to `multiplex_generator_v2.py`.
- **New node/edge features:** extend `generate_multiplex_with_config` in `src/data/multiplex_generator_v2.py` and ensure they are preserved in `build_pyg_dataset.py`.
- **Model variants:**
  - Add heads or encoders in `src/models/train_multitask_gnn.py` for alternative loss balancing or architectures.
  - Swap decoders or negative sampling in `src/models/train_linkpred_layer.py` to test other link-prediction strategies.
- **Reporting:** regenerate summary plots with `src/analysis/plot_multitask_linkpred_summary.py` after adding new runs.

---

## 🔒 Ethical Considerations

This repository is provided for **defensive and lawful research** only. The following principles apply.

### ✅ Allowed Use
- **Synthetic Data Only**  
  All networks are **100% synthetic** and created solely for experimentation. No real persons, organizations, communications, or operational datasets are included or required.
- **Defensive CT / Criminal Network Research**  
  Appropriate use includes **method development**, **benchmarking**, and **robustness studies** (e.g., disruption simulation, HVT scoring as a research task, and resilience analysis under noise/partial observability).
- **Transparency & Reproducibility**  
  The code is intended to support peer review and methodological comparison.

### 🚫 Prohibited Use
Do **NOT** use this codebase to:
- Conduct or support **operational targeting** of real individuals or groups.
- Perform **unauthorized surveillance**, doxxing, or collection/analysis of real social network data without explicit legal authority and ethical approval.
- **Target or suppress** legitimate political groups, civil society organizations, journalists, activists, or lawful protest.
- Enable discrimination, harassment, intimidation, or violations of privacy, due process, or human rights.

### 🧾 Governance Guidance (If You Use Real Data Elsewhere)
If you adapt ideas from this repo to any real-world context, you are responsible for:
- Obtaining appropriate **legal authorization**, **ethics/IRB review**, and **data governance approvals**.
- Applying **minimization**, **access control**, **audit logging**, and **security controls** to protect sensitive data.
- Ensuring outputs are used for **defensive decision support**, with human oversight and accountability.


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
