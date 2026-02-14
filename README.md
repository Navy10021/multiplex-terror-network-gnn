# 🔬🕸️ Multiplex Terror Network GNN (Ontology-First)

> **Synthetic multiplex terror-network research stack** for lawful, defensive analysis.
>
> This repository is strictly for **defensive methodology R&D on synthetic data** (no real operational targeting use).

---

## ✨ What this project gives you

An end-to-end pipeline that treats ontology as a first-class engineering contract:

1. **Generate** synthetic multiplex graphs (`hierarchy`, `finance`, `communication`, `operation`, `ideology`).
2. **Validate** graph manifests against ontology + SHACL + runtime rule checks.
3. **Build** PyTorch Geometric artifacts with ontology bridge tensors.
4. **Train** multitask GNNs with optional ontology-aware regularization.
5. **Report & explain** using conformance/violation metrics and node-level explanation JSON.

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 1500 \
  --seed 2025 \
  --out_root results
```

---

## 🧭 Why ontology-first for multiplex threat-network simulation?

Multiplex hostile/criminal network modeling is difficult because:

- **Semantics differ by layer** (money-flow vs command vs communication).
- **Data quality degrades** with missing links, false links, and copied provenance.
- **Reasoning constraints matter** (role compatibility, temporal ordering, confidence bounds).

This project uses ontology to encode these semantics/constraints as a reusable contract across:

- **Data generation quality control** (strict/constrained/report-only modes)
- **Feature engineering** (ontology bridge tensors)
- **Learning objective design** (ontology-aware losses)
- **Evaluation and explainability** (violation histograms + rule chain evidence)

---

## 🧠 Ontology concepts (concrete)

### 1) OWL ontology (`ontology/terror.ttl`)

The ontology defines core classes and properties used as the domain vocabulary:

- **Classes**: `Actor`, `Role`, `Relation`, `Event`, `Evidence`, `EdgeProvenance`, `LayerInteraction`, `InteractionRule`
- **Role subclasses**: `Leader`, `Financier`, `Courier`, `Operative`, `Support`
- **Layer relation subclasses**: `HierarchyRelation`, `FinanceRelation`, `CommunicationRelation`, `OperationRelation`, `IdeologyRelation`
- **Key properties**:
  - structural: `source`, `target`, `hasRole`
  - temporal/event: `timestamp`, `eventType`
  - finance: `txnAmountSum`, `txnCount`
  - provenance/evidence: `isFalseEdge`, `isCopiedEdge`, `copiedFromLayer`, `confidence`
  - interaction rules: `fromLayer`, `toLayer`, `temporalWindowDays`, `ruleType`, `strength`

### 2) SHACL constraints (`ontology/constraints.shacl.ttl`)

SHACL shapes provide declarative guardrails, for example:

- actor-role cardinality (`ActorRoleShape`)
- non-negative event timestamp + allowed event type (`EventTimeShape`, `EventTypeShape`)
- positive transaction amount (`FinanceEdgeAmountShape`)
- no self-loop hierarchy edges (`HierarchyNoSelfLoopShape`)
- provenance confidence range [0,1] (`ProvenanceConfidenceShape`)

### 3) Runtime ontology validator (`src/ontology/validator.py`)

Beyond static SHACL terms, runtime checks evaluate manifest semantics directly:

- role whitelist validation
- hierarchy command-source constraints
- finance value sanity (`txn_amount_sum > 0`, `txn_count >= 0`)
- relation-role compatibility by layer
- provenance validity (`is_false` binary, `confidence` bounds, `copied_from` integrity)
- temporal interaction ordering/lag checks using `ontology.layer_interactions`

Output includes:

- `conforms` flag
- `violations`, `violations_by_check`, `violation_histogram`
- per-check error buckets (`errors_by_check`)
- run-level counts (nodes/layers/events/violation totals)

---

## 🏗️ Pipeline overview

```text
Generator(v3) -> Ontology Validator -> PyG Builder(v3) -> GNN Training(v3) -> Reporting/Explanations
```

### Ontology modes in `src.run_all`

| Mode | Behavior | Typical Use |
|---|---|---|
| `strict` (default) | Validation failure stops run | experiments requiring guaranteed semantic conformance |
| `constrained` | retries generation with shifted seeds | produce conformant manifests under noisy configs |
| `report_only` | record violations but continue | exploratory analysis / ablation |

---

## 📦 Key artifacts per run

- `multiplex.json` (generated manifest)
- `ontology_validation_report.json`
- `run_metadata.json` (ontology settings + stage outputs)
- `pyg_data.pt` (with ontology bridge tensors)
- `multitask_metrics.json`
- `reporting_summary/multitask_linkpred_summary.csv` *(optional)*
- `explanations/ontology_explanations.json` *(optional)*

---

## 🔧 Installation

```bash
git clone https://github.com/Navy10021/multiplex-terror-network-gnn.git
cd multiplex-terror-network-gnn
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
```

> If your CUDA/OS requires custom wheels, install PyTorch/PyG first, then use `requirements.txt`.

---

## 🚀 Quick start recipes

### A) End-to-end run

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 1500 \
  --seed 2025 \
  --out_root results
```

### B) End-to-end + reporting/explanations

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 800 \
  --seed 2025 \
  --out_root results \
  --run_reporting_summary \
  --write_explanations
```

### C) Constrained ontology generation

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 800 \
  --seed 2025 \
  --out_root results \
  --ontology_mode constrained
```

### D) Validate existing manifest only

```bash
python -m src.cli.validate_ontology \
  --manifest data/multiplex_baseline/multiplex.json \
  --ontology ontology/terror.ttl \
  --shapes ontology/constraints.shacl.ttl \
  --json
```

### E) Multitask training with ontology loss

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

## 🧪 Validation checklist (recommended)

```bash
# 1) Unit tests for ontology components
pytest -q tests/test_ontology_validator.py tests/test_ontology_cli.py tests/test_generator_ontology_integration.py

# 2) End-to-end smoke with ontology enabled
python -m src.run_all --config configs/generator_baseline.json --size 120 --seed 2025 --out_root results

# 3) Inspect ontology report
python -m src.cli.validate_ontology --manifest results/<run_dir>/multiplex.json --json
```

---

## 🗂️ Project structure

```text
multiplex-terror-network-gnn/
├── README.md
├── PLANS.md
├── PLANS_KOR.md
├── configs/
├── ontology/
│   ├── terror.ttl
│   └── constraints.shacl.ttl
├── src/
│   ├── run_all.py
│   ├── cli/validate_ontology.py
│   ├── ontology/{load.py,validator.py}
│   ├── data/{multiplex_generator_v3.py,build_pyg_dataset_v3.py,basic_diagnostics_v3.py}
│   ├── models/{train_multitask_gnn_v3.py,train_hvt_gnn_v3.py,train_linkpred_layer_v3.py}
│   └── analysis/plot_multitask_linkpred_summary.py
└── tests/
```

---

## 🛣️ Current status & roadmap pointers

### Implemented phases

- **Phase A**: ontology validator depth expansion (rule-level violations + histograms)
- **Phase B**: constrained generation with retries/telemetry
- **Phase C**: PyG ontology bridge tensors + consistency checks
- **Phase D**: ontology-aware multitask regularizers + metrics logging
- **Phase E**: ontology-aware reporting and explanation artifact generation

### Roadmap docs

- English: `PLANS.md`
- Korean: `PLANS_KOR.md`

---

## 📄 License / usage note

Use this repository only for legal, ethical, and defensive research contexts.
