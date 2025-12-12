# Project Review: Multiplex Terror Network GNN

## Current Scope
- Generates five-layer synthetic terrorist networks (hierarchy, finance, communication, operations, ideology) with configurable difficulty presets (easy, baseline, hard).
- Provides PyTorch Geometric datasets that include multiplex edge types, node roles, HVT labels, and regression targets.
- Implements multi-task and single-task R-GCN training scripts plus layer-specific link prediction baselines.

## Strengths
- Clear end-to-end workflow from data generation to modeling, making experiments reproducible.
- Multi-task framing (HVT, role, importance) aligns with operational questions while reusing a shared encoder.
- Difficulty knobs in generator configs offer a controlled way to test robustness under noisy observations.

## Risks / Gaps Observed
- No automated tests or smoke checks are present, increasing risk of silent regressions across data and training scripts.
- Documentation stops at installation; runnable commands for each script are implied but not enumerated, which may slow onboarding.
- Result-tracking conventions (output directories, naming) are implicit; standardizing them would help compare runs.

## Suggested Next Steps
1. Add minimal CI or local smoke tests that run the data generator and one epoch of each trainer to catch breakages early.
2. Expand README with command examples for dataset generation, multi-task training, and link prediction, including expected output paths.
3. Define a results/logging structure (e.g., `results/<config>/<model>/`) and document it so new experiments are consistent.
4. Consider packaging the CLI entry points to simplify usage (`python -m src.models.train_multitask_gnn --config ...`).
