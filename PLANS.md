# Ontology-Centered Development Plan (PLANS)

This document is a forward engineering plan focused on upgrading this repository into a stronger
**ontology-driven synthetic terror-network research stack**.

---


## 0) Execution status

- [x] **Phase A started**: validator depth expansion implemented (rule-level violations, severity, affected IDs, histogram, temporal/relation-role/provenance checks).
- [ ] Phase B
- [ ] Phase C
- [ ] Phase D
- [ ] Phase E

---

## 1) Current codebase checkpoint (as-is audit)

### 1.1 Generator / Data format / Pipeline map
- Generator v3: `src/data/multiplex_generator_v3.py`
- Manifest schema/validator: `src/validation/schema.py`
- Ontology validator and loader:
  - `src/ontology/load.py`
  - `src/ontology/validator.py`
  - `src/cli/validate_ontology.py`
- Pipeline orchestration: `src/run_all.py`
- PyG builder v3: `src/data/build_pyg_dataset_v3.py`
- Training:
  - `src/models/train_multitask_gnn_v3.py`
  - `src/models/train_hvt_gnn_v3.py`
  - `src/models/train_linkpred_layer_v3.py`
- Diagnostics/reporting:
  - `src/data/basic_diagnostics_v3.py`
  - `src/analysis/plot_multitask_linkpred_summary.py`

### 1.2 Verified baseline health
- Full tests pass.
- `run_all` and ontology CLI entrypoints are healthy (`--help` works).
- Ontology artifacts are integrated in generator + run pipeline.

---

## 2) Strategic objective (next milestone)

Upgrade from “ontology-checked manifests” to “ontology-guided generation/training/reporting”:

1. **Generation validity**: constrain generation with ontology rules, not just post-hoc checks.
2. **Model learning signal**: expose ontology semantics as train-time priors/losses.
3. **Evaluation explainability**: report performance with rule conformance and evidence grounding.

---

## 3) Phase plan (implementation-ready)

## Phase A — Validator depth expansion

### Scope
- Strengthen ontology contract and runtime checks:
  - relation-role compatibility tables
  - temporal window checks by layer interaction
  - provenance consistency (false/copy/confidence)
- Extend report schema:
  - `errors_by_check` (already present) + severity + affected_ids
  - violation histogram by rule key

### Target files
- `src/ontology/validator.py`
- `ontology/constraints.shacl.ttl`
- `tests/test_ontology_validator.py`

### Definition of done
- Validation report includes machine-usable violation breakdown.
- New checks have pass/fail test coverage.

### Validation commands
- `pytest -q tests/test_ontology_validator.py tests/test_ontology_cli.py`

---

## Phase B — Generator v3 constraint-aware generation

### Scope
- Add optional constrained generation mode:
  - rejection/resampling on hard rule violations
  - bounded retries + fallback strategy logging
- Emit generation-time repair telemetry:
  - number of retries
  - repaired rule categories

### Target files
- `src/data/multiplex_generator_v3.py`
- `src/run_all.py`
- `tests/test_generator_ontology_integration.py`

### Definition of done
- Constrained mode yields ontology-conformant manifests under strict mode.
- Telemetry is present in `ontology_validation_report.json` and/or metadata.

### Validation commands
- `pytest -q tests/test_generator_ontology_integration.py tests/test_manifest_validation.py`

---

## Phase C — PyG builder ontology bridge hardening

### Scope
- Expand ontology-derived tensors:
  - edge semantics (logical + temporal + provenance)
  - role-compatibility masks for training constraints
- Add consistency assertions between manifest ontology payload and tensor outputs.

### Target files
- `src/data/build_pyg_dataset_v3.py`
- `tests/test_build_pyg_ontology_bridge.py`

### Definition of done
- Data object stores reproducible ontology tensors/masks.
- Builder tests verify shape/value contracts.

### Validation commands
- `pytest -q tests/test_build_pyg_ontology_bridge.py`

---

## Phase D — Model Zoo ontology-aware learning

### Scope
- Introduce optional ontology losses in multitask model:
  - role-relation compatibility loss
  - hierarchy transitivity consistency penalty
  - temporal ordering penalty
- Add CLI flags for ablation:
  - `--ontology_loss_*` toggles and weights

### Target files
- `src/models/train_multitask_gnn_v3.py`
- (Optional) shared utility under `src/models/` for ontology regularizers
- tests (unit/integration where feasible)

### Definition of done
- Training script supports ontology-loss ablations without breaking legacy runs.
- Metrics JSON captures ontology-loss settings.

### Validation commands
- smoke: `python -m src.models.train_multitask_gnn_v3 --help`
- smoke train: 1-epoch tiny run on generated sample

---

## Phase E — Experiment/reporting integration

### Scope
- Extend summary pipeline with ontology metrics:
  - conformance rate
  - violations per 1k edges/events
  - provenance noise indicators
- Add explanation-ready outputs (JSON) for node-level predictions:
  - model evidence + rule evidence + conflict flags

### Target files
- `src/analysis/plot_multitask_linkpred_summary.py`
- `src/run_all.py`
- `README.md`

### Definition of done
- Summary CSV/plots include ontology-aware columns.
- Example explanation artifact documented in README.

### Validation commands
- summary script smoke with at least 1 run directory

---

## 4) Cross-cutting engineering requirements

1. **Backward compatibility**
   - Existing commands remain valid by default.
   - New ontology modes are opt-in unless explicitly migrated.

2. **Reproducibility**
   - Persist ontology/shapes path and hash in run metadata.
   - Version ontology payload (`ontology.version`) and validator checkset.

3. **Test policy**
   - Every phase adds/updates tests.
   - Keep full suite green before merge.

---

## 5) Suggested near-term execution order

1. Phase A (validator depth)  
2. Phase B (generator constraint mode)  
3. Phase C (builder tensor bridge)  
4. Phase D (ontology-aware losses)  
5. Phase E (reporting + explanation)

This order minimizes regression risk and enables measurable gains at each step.
