# TECH_DEBT Assessment Report (STEP 0)

Date: 2026-02-14  
Scope: repository-wide (`src/`, `tests/`, `configs/`, `ontology/`, docs, dependency files)

## Scan method
A repository snapshot was assessed with lint/testing and structural checks. The report below focuses on actionability over exhaustive detail.

## Completed baseline work (STEP 1~3)
- STEP 1 (tooling standardization)
  - Added `pyproject.toml` with `ruff`/`black`/`isort`/`mypy` settings.
  - Added `.pre-commit-config.yaml`.
  - Cleaned minimal package init files.
- STEP 2 (CI foundation)
  - Added CI workflow and checklist-driven validation.
- STEP 3 (dependency reproducibility)
  - Reworked `requirements.txt` around minimum compatible ranges.
  - Added `requirements.lock` for environment-independent pins.
  - Added `docs/INSTALL.md` for CPU/CUDA/Colab recipes.

## Current debt themes

### A. Style/tooling consistency
1. **Version-suffixed module sprawl (`*_v1.py`, `*_v2.py`, `*_v3.py`)**  
   Risk: style drift and duplicated logic.  
   Action: define an active version policy, move legacy files under a dedicated `legacy/` path, and extract shared utilities.

2. **Warning noise in tests**  
   Risk: real regressions hidden by recurring warnings.  
   Action: define a warning policy and gradually resolve deprecations (especially Pydantic v2 migration warnings).

### B. Import/package boundaries
1. **`src.`-anchored imports are execution-mode sensitive**  
   Risk: behavior differences between module execution and editable install.  
   Action: keep `pip install -e .` as default dev path and tighten internal import conventions.

2. **Public API boundaries are still loose**  
   Risk: accidental reliance on internal modules.  
   Action: declare package-level exports where needed and document stable entry points.

### C. Entrypoint and orchestration
1. **`src/run_all.py` remains large**  
   Risk: high change coupling across generation/validation/reporting.  
   Action: split orchestration into service helpers and keep CLI entrypoint thin.

2. **Ontology mode contract needs stronger operational guidance**  
   Risk: inconsistent strict/constrained/report-only usage across experiments.  
   Action: keep `docs/ONTOLOGY_CONTRACT.md` synchronized with runtime behavior and examples.

### D. Test quality and coverage
1. **Coverage gate not enforced**  
   Risk: blind spots in regression protection.  
   Action: adopt `pytest --cov=src --cov-report=term-missing` and set an incremental threshold.

2. **Contract tests can be further consolidated**  
   Risk: schema behavior spread across multiple files.  
   Action: continue consolidating schema-contract tests and keep success/failure cases explicit.

### E. Dependency and packaging risk
1. **Environment-sensitive dependencies (Torch/PyG)**  
   Risk: install failures in generic CI/dev environments.  
   Action: keep platform-specific install docs explicit and avoid over-pinning GPU-sensitive packages in universal lock files.

2. **Packaging metadata maturity**  
   Risk: friction for reusable package workflows.  
   Action: continue improving project metadata and optional dependency groups.

3. **License finalization**  
   Risk: legal ambiguity for distribution/collaboration.  
   Action: finalize and commit an explicit LICENSE.

## Suggested execution order
1. Reduce warning noise + add coverage gate.
2. Modularize `run_all` orchestration.
3. Rationalize versioned modules and shared utilities.
4. Harden package/API boundaries.
5. Finalize license and packaging polish.
