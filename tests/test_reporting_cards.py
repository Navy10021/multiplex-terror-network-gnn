import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reporting.cards import write_run_cards
from src.validation.schema import validate_manifest_dict


def _manifest() -> dict:
    return {
        "meta": {
            "num_nodes": 2,
            "seed": 9,
            "generator": "test",
            "config": {
                "missing_edge_rate_finance": 0.1,
                "false_edge_rate_finance": 0.2,
                "cross_layer_copy": [{"from_layer": "finance", "to_layer": "communication", "copy_rate": 0.3}],
            },
        },
        "nodes": [
            {"id": 0, "role": "leader", "region": "x", "group": "g"},
            {"id": 1, "role": "support", "region": "x", "group": "g"},
        ],
        "layers": {
            "finance": {
                "directed": True,
                "edges": [
                    {"source": 0, "target": 1, "is_false": 0, "copied_from": "communication"},
                ],
            }
        },
        "events": [],
    }


def test_write_run_cards_generates_dataset_and_model_cards(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    (run_dir / "multitask_metrics.json").write_text(
        json.dumps(
            {
                "hvt_threshold_tuned": {"test": {"auc": 0.8, "f1": 0.7}},
                "fixed_threshold": {"test": {"role_f1_macro": 0.6, "imp_r2": 0.5}},
                "hvt_calibration": {
                    "enabled": True,
                    "method": "temperature",
                    "temperature": 1.2,
                    "ece_val_before": 0.2,
                    "ece_val_after": 0.1,
                    "brier_val_before": 0.3,
                    "brier_val_after": 0.25,
                    "thr_strategy": "f1",
                    "thr_prevalence": 0.12,
                    "thr_fpr": 0.01,
                    "thr_iqr": {"p25": 0.42, "p75": 0.58},
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = validate_manifest_dict(_manifest())
    paths = write_run_cards(manifest, run_dir=str(run_dir), dataset_path=None, diagnostics_dir=None)

    dataset_card = Path(paths["dataset_card"]).read_text(encoding="utf-8")
    model_card = Path(paths["model_card"]).read_text(encoding="utf-8")

    assert "# DATASET_CARD" in dataset_card
    assert "missing edge rates" in dataset_card
    assert "# MODEL_CARD" in model_card
    assert "HVT auc" in model_card
    assert "calibration enabled" in model_card
    assert "validation ECE (before -> after)" in model_card
    assert "threshold IQR" in model_card
