# CI Checklist (Reusable PR Gate)

This document defines the final quality gate that should be applied to every PR.
CI runs `scripts/ci_checklist.sh` to enforce the checks below.

## 1) Static checks
- [ ] `ruff check src tests`
- [ ] `mypy src/ontology src/validation src/cli src/utils`

## 2) Test suite
- [ ] `pytest -q`

## 3) CLI health checks
- [ ] `python -m src.cli.main --help`
- [ ] `python -m src.cli.main run-all --help`

## 4) Repro script sanity
- [ ] `bash -n scripts/run_easy_baseline_hard.sh scripts/summarize_all.sh`

## 5) Docs alignment review (manual)
- [ ] README Quick Start / CLI / Typical outputs match current code behavior.
- [ ] Any new user-facing feature is documented.

---

## Notes
- Torch/PyG has environment-specific installation behavior. Use `importorskip` for optional tests so CI does not fail due to missing GPU-specific dependencies.
- If this checklist changes, update both `docs/CI_CHECKLIST.md` and `scripts/ci_checklist.sh`.
- CI jobs should stay bounded (`timeout-minutes: 20`, `pip --retries 1 --timeout 30`) to avoid long hangs.
- If mergeability looks stale on a long-lived PR, rebase onto latest `main` and push again.
