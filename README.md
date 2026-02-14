# 🔬🕸️ Multiplex Terror Network GNN

Synthetic multiplex terrorist-network generator + ontology-aware GNN baselines for defensive research on:
- High-Value Target (HVT) detection
- Role inference
- Importance scoring
- Layer-aware link prediction

> **Purpose & Safety (Defensive Use Only)**  
> This repository is for lawful, defensive research only (e.g., robustness testing, disruption simulation, resilience planning).  
> **All data are synthetic.** Do **not** use this project for operational targeting, real-world surveillance, or analysis of real social networks.

---

## 📌 Overview

Real extremist/criminal-style networks differ from generic benchmark graphs:
- **Multiplex semantics:** hierarchy, finance, communication, operation, ideology each carry different meaning.
- **Observational distortion:** false edges, copied edges, and layer-specific missingness coexist.
- **Actionability:** practitioners need interpretable risk signals, not only generic centrality.
- **Evaluation risk:** link prediction can be inflated by leakage if test edges are used during message passing.

This project provides a reproducible sandbox with:
- **Ontology-first data contract** (OWL + SHACL + runtime validator)
- **Config-driven synthetic generation** across easy/baseline/hard difficulty
- **Multi-task node prediction** with shared GNN encoder
- **Leakage-safe layer-wise link prediction** with hard negative sampling options
- **Diagnostics + reporting artifacts** for validation and analysis

---

## 🧭 Why Ontology-First?

This repository originally evolved around ontology validation, and that philosophy is still central:

1. **Generation quality control**: strict/constrained/report-only validation modes
2. **Schema consistency**: role/layer/provenance semantics are enforced as contracts
3. **Training alignment**: ontology-compatible features and optional ontology-aware regularization
4. **Explainability**: rule-violation summaries and explanation artifacts

In short, the ontology is not documentation only—it is an executable guardrail.

---

## 🧱 Multiplex Ontology (v3)

### Layer semantics
- `hierarchy`: command/control structure
- `finance`: monetary/resource flow
- `communication`: contacts/coordination
- `operation`: joint tactical activity
- `ideology`: influence/mentorship/recruitment

### Node schema (conceptual)
- **Roles**: leader, operative, financier, courier, support
- **Context**: region/group attributes
- **Continuous features**: activity/centrality/homophily/consistency-type signals

### Edge schema + provenance
- Core: `layer`, `weight`, `timestamp`, `confidence`
- Provenance tags: `is_structural`, `is_false`, `is_copied`, `source_layer`

### Noise taxonomy
- **False edges**: spurious observed links
- **Copied edges**: correlated cross-layer duplication
- **Missingness**: layer-dependent edge dropout

---

## ✅ Ontology stack in this repo

### 1) OWL model
- `ontology/terror.ttl`
- Core classes include Actor/Role/Relation/Event/Evidence/Provenance/Interaction concepts.

### 2) SHACL constraints
- `ontology/constraints.shacl.ttl`
- Examples: role constraints, event/time constraints, finance value constraints, self-loop constraints, confidence range constraints.

### 3) Runtime validator
- `src/ontology/validator.py`
- Checks that go beyond static SHACL:
  - role whitelist/compatibility
  - hierarchy source-role constraints
  - invalid numeric values (`NaN`, `inf`, malformed)
  - provenance consistency (`is_false`, `copied_from`, `confidence`)
  - layer-interaction temporal checks from manifest rules

### Validation modes
- `strict` (default): fail fast on violations
- `constrained`: retry generation with seed offsets to satisfy constraints
- `report_only`: continue while recording violations

---

## 🧠 Modeling & training

### Multi-task node learning
Shared encoder + task heads:
- HVT binary classification
- Role multi-class classification
- Importance regression

Representative training script:
- `src/models/train_multitask_gnn_v3.py`

### Layer-wise link prediction
- Leakage-safe protocol by removing validation/test edges from encoder graph
- Negative sampling modes (e.g., `uniform`, `hard_region`)

Representative script:
- `src/models/train_linkpred_layer_v3.py`

---

## 🚀 Quick start

### Installation
```bash
git clone https://github.com/Navy10021/multiplex-terror-network-gnn.git
cd multiplex-terror-network-gnn
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
```

> PyTorch/PyG는 환경(CPU/CUDA/Colab)별로 별도 설치가 필요합니다. `docs/INSTALL.md` 설치 레시피를 먼저 확인하세요.

### End-to-end pipeline
```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 1500 \
  --seed 2025 \
  --out_root results
```

### Ontology mode example
```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 800 \
  --seed 2025 \
  --out_root results \
  --ontology_mode constrained
```

### Ontology validation only
```bash
python -m src.cli.validate_ontology \
  --manifest data/multiplex_baseline/multiplex.json \
  --ontology ontology/terror.ttl \
  --shapes ontology/constraints.shacl.ttl \
  --json
```

---

## 📦 Typical outputs

A run directory generally includes:
- `multiplex.json`
- `ontology_validation_report.json`
- `pyg_data.pt`
- `multitask_metrics.json`
- link prediction result JSON files
- `run_metadata.json`
- `explanations/ontology_explanations.json` (when enabled)
- `reporting_summary/multitask_linkpred_summary.csv` (when enabled)
- `DATASET_CARD.md`
- `MODEL_CARD.md`

---

## 🧪 Suggested reproducibility workflow

```bash
# 1) unit tests
pytest -q

# 2) CLI health checks
python -m src.run_all --help
python -m src.cli.validate_ontology --help

# 3) compare difficulty presets
for config in easy baseline hard; do
  python -m src.run_all \
    --config configs/generator_${config}.json \
    --size 1500 --seed 2025 --out_root results/${config}/
done
```

Quick script version:

```bash
bash scripts/run_easy_baseline_hard.sh 2025 1500 results/repro_runs
bash scripts/summarize_all.sh results/repro_runs results/summary_all
```

See `docs/REPRO.md` for full pipeline details.

---

## 📁 Project structure (current)

```text
multiplex-terror-network-gnn/
├── README.md
├── PLANS.md
├── PLANS_KOR.md
├── ontology/
│   ├── terror.ttl
│   └── constraints.shacl.ttl
├── configs/
│   ├── generator_easy.json
│   ├── generator_baseline.json
│   └── generator_hard.json
├── src/
│   ├── run_all.py
│   ├── cli/validate_ontology.py
│   ├── ontology/{load.py,validator.py}
│   ├── data/{multiplex_generator_v3.py,build_pyg_dataset_v3.py,basic_diagnostics_v3.py}
│   ├── models/{train_multitask_gnn_v3.py,train_linkpred_layer_v3.py,...}
│   ├── analysis/plot_multitask_linkpred_summary.py
│   └── validation/schema.py
└── tests/
```

---

## 🔒 Ethical use policy

### Allowed
- Defensive CT/criminal-network research on synthetic or properly governed anonymized data
- Methodological benchmarking, robustness/stress testing, fairness auditing
- Educational use

### Prohibited
- Operational targeting of real persons/groups
- Unauthorized surveillance
- Repression/discrimination/rights-violating use

If adapted to real data: legal authorization, ethics review, governance controls, human oversight, and bias audits are mandatory.

---

## 🤝 Contributing

Contributions are welcome for:
- temporal/dynamic extensions
- robustness under adversarial/noisy settings
- stronger baselines and evaluation protocols
- interpretability and reporting improvements

Please open an issue/PR with clear experiment settings and reproducibility metadata.

---

## 📜 License

Current license status in this repository is **TBD**.
Please contact the maintainer before commercial or redistribution use.

---

## 📬 Contact
- Maintainer: Yoon-seop Lee
- GitHub: `@Navy10021`

---

**Last updated:** 2026-02-14 (v3 README refresh)
