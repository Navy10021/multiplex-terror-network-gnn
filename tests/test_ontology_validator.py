import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ontology.validator import OntologyValidationError, validate_manifest_with_ontology


def _write_manifest(tmp_path: Path, overrides=None):
    manifest = {
        "meta": {"num_nodes": 2, "seed": 1, "generator": "test", "config": {}},
        "ontology": {
            "layer_interactions": [
                {
                    "from_layer": "finance",
                    "to_layer": "operation",
                    "rule_type": "correlation",
                    "strength": 0.75,
                    "temporal_window_days": 10,
                }
            ]
        },
        "nodes": [
            {"id": 0, "role": "leader", "region": "x", "group": "g"},
            {"id": 1, "role": "support", "region": "x", "group": "g"},
        ],
        "layers": {
            "hierarchy": {"directed": True, "edges": [{"source": 0, "target": 1, "is_false": 0}]},
            "finance": {
                "directed": True,
                "edges": [
                    {
                        "source": 0,
                        "target": 1,
                        "txn_amount_sum": 2.0,
                        "txn_count": 1.0,
                        "is_false": 0,
                        "copied_from": "communication",
                        "confidence": 0.8,
                    }
                ],
            },
        },
        "events": [
            {"time": 1, "event_type": "txn", "u": 0, "v": 1, "meta": {}},
            {"time": 5, "event_type": "op", "u": 0, "v": 1, "meta": {}},
        ],
    }
    if overrides:
        overrides(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_validate_manifest_with_ontology_success(tmp_path: Path):
    path = _write_manifest(tmp_path)
    report = validate_manifest_with_ontology(
        manifest_path=str(path),
        ontology_path="ontology/terror.ttl",
        shapes_path="ontology/constraints.shacl.ttl",
    )
    assert report["conforms"] is True
    assert report["constraints_checked"] == 8
    assert "violation_histogram" in report
    assert report["counts"]["violations_total"] == 0


def test_validate_manifest_with_ontology_failure(tmp_path: Path):
    def _bad(m):
        m["layers"]["hierarchy"]["edges"][0]["source"] = 1
        m["layers"]["finance"]["edges"][0]["txn_amount_sum"] = -5
        m["events"][0]["event_type"] = "meeting"

    path = _write_manifest(tmp_path, overrides=_bad)

    with pytest.raises(OntologyValidationError) as exc:
        validate_manifest_with_ontology(
            manifest_path=str(path),
            ontology_path="ontology/terror.ttl",
            shapes_path="ontology/constraints.shacl.ttl",
        )

    assert "non-command role" in str(exc.value)
    assert "non-positive" in str(exc.value)
    assert "unsupported event_type" in str(exc.value)


def test_validate_manifest_with_ontology_provenance_failure(tmp_path: Path):
    def _bad(m):
        m["layers"]["finance"]["edges"][0]["is_false"] = 2
        m["layers"]["finance"]["edges"][0]["copied_from"] = "finance"
        m["layers"]["finance"]["edges"][0]["confidence"] = 1.7

    path = _write_manifest(tmp_path, overrides=_bad)

    with pytest.raises(OntologyValidationError) as exc:
        validate_manifest_with_ontology(
            manifest_path=str(path),
            ontology_path="ontology/terror.ttl",
            shapes_path="ontology/constraints.shacl.ttl",
        )

    assert "invalid is_false" in str(exc.value)
    assert "cannot match destination layer" in str(exc.value)
    assert "out-of-range confidence" in str(exc.value)




def test_validate_manifest_with_ontology_non_numeric_values_fail_cleanly(tmp_path: Path):
    def _bad(m):
        edge = m["layers"]["finance"]["edges"][0]
        edge["txn_amount_sum"] = "NaN"
        edge["txn_count"] = "oops"
        edge["is_false"] = 3
        edge["confidence"] = "inf"

    path = _write_manifest(tmp_path, overrides=_bad)

    with pytest.raises(OntologyValidationError) as exc:
        validate_manifest_with_ontology(
            manifest_path=str(path),
            ontology_path="ontology/terror.ttl",
            shapes_path="ontology/constraints.shacl.ttl",
        )

    msg = str(exc.value)
    assert "invalid txn_amount_sum" in msg
    assert "invalid txn_count" in msg
    assert "invalid is_false" in msg
    assert "invalid confidence" in msg


def test_validate_manifest_with_ontology_temporal_interaction_failure(tmp_path: Path):
    def _bad(m):
        m["events"] = [
            {"time": 20, "event_type": "txn", "u": 0, "v": 1, "meta": {}},
            {"time": 3, "event_type": "op", "u": 0, "v": 1, "meta": {}},
        ]

    path = _write_manifest(tmp_path, overrides=_bad)

    with pytest.raises(OntologyValidationError) as exc:
        validate_manifest_with_ontology(
            manifest_path=str(path),
            ontology_path="ontology/terror.ttl",
            shapes_path="ontology/constraints.shacl.ttl",
        )

    assert "temporal interaction rule" in str(exc.value)
