# Ontology-Centered Development Plan (PLANS)

This document is a forward engineering plan focused on upgrading this repository into a stronger
**ontology-driven synthetic terror-network research stack**.

---


## 0) Execution status

- [x] **Phase A started**: validator depth expansion implemented (rule-level violations, severity, affected IDs, histogram, temporal/relation-role/provenance checks).
- [x] **Phase B started**: ontology-constrained generation retries + telemetry wired in generator/run pipeline
- [x] **Phase C started**: PyG builder ontology bridge tensors/masks + payload consistency checks
- [x] **Phase D started**: multitask trainer ontology-aware regularizers + ablation flags + metrics wiring
- [x] **Phase E started**: ontology-aware reporting metrics + node explanation artifacts wired into run pipeline

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


## 6) Next evolution proposals (post-Phase E)

- [x] **F1 completed**: ontology_mode presets + strict-failure top-check summaries + actionable guidance

### F1 — Ontology-conformance friendly defaults & UX hardening

**Why now**
- Recent smoke checks show baseline config can fail strict ontology mode in `run_all` unless retries/non-strict mode are enabled.

**Actions**
- Improve strict-mode error guidance and docs for first-run success path.
- Add runbook presets: strict research mode vs non-strict exploratory mode.
- Add conformance summary snippet to terminal output (top failing checks + counts).

**DoD**
- First-time users can complete one end-to-end run without confusion.
- Strict failures produce actionable hints and remediation commands.

**Validation**
- `python -m src.run_all ...` strict + non-strict smoke tests.

- [x] **F2 completed**: explanation JSON now includes rule_chains and confidence_alignment

### F2 — Explanation quality upgrade (model attribution + ontology chain)

**Actions**
- Replace degree-based proxy evidence with model-derived attribution (neighbors/edges contributions).
- Link violations and satisfied rules into per-target rule chains.
- Add `confidence_alignment` between model probability and rule score.

**DoD**
- Explanation JSON contains both model attribution and ontology rule chains.
- README includes one realistic explanation example.

### F3 — Reporting/benchmark standardization

**Actions**
- Add ontology-aware benchmark table templates (base vs +ontology_loss, strict vs non-strict generation).
- Export aggregated ontology metrics (`conformance`, `violations_per_1k_*`) across seeds/difficulties.
- Add lightweight CI smoke job for `run_all` with Phase E flags.

**DoD**
- Reproducible summary table available from one command.
- CI catches regressions in ontology-reporting artifact generation.
