import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.plot_multitask_linkpred_summary import build_runs_dataframe
from src.run_all import _build_node_explanations


def _write_json(path: Path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_reporting_dataframe_includes_ontology_columns(tmp_path: Path):
    run_dir = tmp_path / "run_baseline_seed7"
    run_dir.mkdir(parents=True)

    _write_json(
        run_dir / "multitask_metrics.json",
        {
            "seed": 7,
            "hvt_threshold_tuned": {"test": {"f1": 0.3, "auc": 0.6}},
            "fixed_threshold": {"test": {"role_f1_macro": 0.4, "imp_r2": 0.2}},
            "ontology_loss": {
                "enabled": True,
                "final_epoch_losses": {
                    "role_relation_compatibility": 0.1,
                    "hierarchy_transitivity": 0.2,
                    "temporal_ordering": 0.3,
                },
            },
        },
    )
    _write_json(
        run_dir / "multiplex.json",
        {
            "meta": {"config": {"finance_structure_strength": 1.0, "comm_structure_strength": 1.0, "comm_randomness": 0.1}},
            "layers": {"hierarchy": {"directed": True, "edges": [{"source": 0, "target": 1}]}},
            "events": [{"event_type": "txn", "u": 0, "v": 1}],
        },
    )
    _write_json(
        run_dir / "ontology_validation_report.json",
        {"conforms": False, "violations_total": 2},
    )

    df = build_runs_dataframe([str(run_dir)], difficulty_mode="auto")
    assert len(df) == 1
    row = df.iloc[0]
    assert "ontology_conforms" in df.columns
    assert "ontology_violations_per_1k_edges" in df.columns
    assert "ontology_loss_enabled" in df.columns
    assert float(row["ontology_conforms"]) == 0.0
    assert float(row["ontology_loss_enabled"]) == 1.0


def test_run_all_help_includes_phase_e_flags():
    proc = subprocess.run(
        [sys.executable, "-m", "src.run_all", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--run_reporting_summary" in proc.stdout
    assert "--write_explanations" in proc.stdout


def test_build_node_explanations_maps_violations_to_targets():
    manifest = {
        "nodes": [{"id": 0}, {"id": 1}],
        "layers": {"hierarchy": {"edges": [{"source": 0, "target": 1}]}},
    }
    report = {
        "conforms": False,
        "violations": [
            {
                "check": "role_compatibility",
                "rule_id": "r1",
                "severity": "error",
                "message": "bad",
                "affected_ids": [0],
            }
        ],
    }
    exps = _build_node_explanations(manifest, report, top_k=2)
    assert len(exps) == 2
    exp0 = [e for e in exps if e["target"] == 0][0]
    assert exp0["ontology_evidence"]["violation_count_for_target"] == 1
    assert exp0["conflict_flags"]["rule_violation_for_target"] is True
    assert "rule_chains" in exp0["ontology_evidence"]
    assert "violated" in exp0["ontology_evidence"]["rule_chains"]
    assert "confidence_alignment" in exp0
    assert 0.0 <= float(exp0["confidence_alignment"]["alignment_score"]) <= 1.0
