# Ontology-based Validation Layer Plan

## Ask mode: repository tracing

### 1) Data generator location
- Primary generator (v3): `src/data/multiplex_generator_v3.py`
  - Core generation entrypoint: `generate_multiplex_with_config(cfg)`
  - CLI entrypoint: `main()` with `--config/--size/--seed/...`
  - Current validation hook: calls `validate_manifest_dict` before writing manifest.

### 2) Data format / schema location
- Manifest schema + validation CLI: `src/validation/schema.py`
  - Pydantic models for `Manifest`, `Node`, `Layer`, `Edge`, `Event`
  - `validate_manifest_dict`, `validate_manifest_file`, `summarize_manifest`
  - CLI: `python -m src.validation.schema <manifest.json> --summary`
- Existing tests for schema behavior:
  - `tests/test_manifest_validation.py`

### 3) Training / pipeline locations
- End-to-end pipeline entrypoint: `src/run_all.py`
  - Flow: generator v3 -> schema validation -> PyG builder -> diagnostics -> metadata/card.
- Dataset builder (v3): `src/data/build_pyg_dataset_v3.py`
- Main training scripts:
  - Multi-task: `src/models/train_multitask_gnn_v3.py`
  - HVT: `src/models/train_hvt_gnn_v3.py`
  - Layer link prediction: `src/models/train_linkpred_layer_v3.py`

---

## Goal redefinition (fixed, architecture guardrails)
1. **Generation quality/consistency**: enforce domain constraints so synthetic graphs are structurally plausible.
2. **Domain-knowledge injection path**: represent relation semantics in machine-readable form that can later feed features/losses.
3. **Explainability/provenance**: preserve evidence/provenance context so rule-grounded explanations can be attached later.

---

## Implementation roadmap (commit-friendly phases)

## Phase 1 — Ontology schema + CLI validation skeleton

### Scope
- Add ontology artifacts using OWL + SHACL dual-track:
  - `ontology/terror.ttl`
  - `ontology/constraints.shacl.ttl`
- Add CLI entrypoint for ontology checks (manifest -> ontology validation report).

### Files
- New: `ontology/terror.ttl`
- New: `ontology/constraints.shacl.ttl`
- New: `src/cli/validate_ontology.py`
- New/updated package init files if needed (`src/cli/__init__.py`)
- New tests: `tests/test_ontology_cli.py`

### DoD
- CLI runs on a valid manifest and exits `0`.
- CLI reports non-conformance and exits non-zero on invalid manifest.
- Ontology includes Actor/Role/Relation/Event/Evidence(+provenance) concept scaffolding.

### Test commands
- `pytest -q tests/test_ontology_cli.py`

---

## Phase 2 — Validator implementation (manifest -> RDF + SHACL)

### Scope
- Implement ontology validator module:
  - Parse manifest JSON
  - Convert to RDF triples aligned with ontology vocabulary
  - Run SHACL checks via `pyshacl`
  - Return structured validation report

### Files
- New: `src/ontology/load.py` (ontology/shapes loader helpers)
- New: `src/ontology/validator.py` (conversion + SHACL engine)
- New: `src/ontology/__init__.py`
- Update: `src/cli/validate_ontology.py`
- New tests: `tests/test_ontology_validator.py`
- Update dependencies: `requirements.txt` (rdflib/pyshacl)

### DoD
- Programmatic API validates manifest dictionaries/files.
- SHACL violations surfaced with readable messages.
- Unit tests cover pass/fail validation paths.

### Test commands
- `pytest -q tests/test_ontology_validator.py tests/test_ontology_cli.py`

---

## Phase 3 — Generator v3 integration

### Scope
- Integrate ontology validation immediately after generator manifest assembly.
- Add config/CLI flag to control strict enforcement (fail-fast default).
- Emit compact ontology validation report artifact.

### Files
- Update: `src/data/multiplex_generator_v3.py`
- (Optional) Update: `src/run_all.py` for same flag propagation
- New tests: `tests/test_generator_ontology_integration.py`
- Update docs: `README.md` (new validation command + generator behavior)

### DoD
- `multiplex_generator_v3` runs ontology validation by default.
- Invalid manifest in strict mode fails with actionable error.
- Optional bypass flag exists for debugging/backward compatibility.

### Test commands
- `pytest -q tests/test_generator_ontology_integration.py`
- `pytest -q tests/test_manifest_validation.py tests/test_ontology_validator.py tests/test_ontology_cli.py`

---

## Out-of-scope for this implementation batch (future)
- Ontology-aware training losses in `train_multitask_gnn_v3.py`.
- Reasoner-based enrichment and explanation object export.
- Statistical diagnostics layer coupled to ontology constraints.
