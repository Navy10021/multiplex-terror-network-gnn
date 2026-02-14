import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation.schema import ManifestValidationError, validate_manifest_dict


def _manifest() -> dict:
    return {
        "meta": {"num_nodes": 2, "seed": 7, "generator": "test", "config": {}},
        "nodes": [
            {"id": 0, "role": "leader", "region": "x", "group": "g"},
            {"id": 1, "role": "support", "region": "x", "group": "g"},
        ],
        "layers": {"finance": {"directed": True, "edges": [{"source": 0, "target": 1}]}},
        "events": [{"time": 0, "event_type": "txn", "u": 0, "v": 1, "meta": {}}],
    }


def test_manifest_schema_accepts_valid_sample():
    parsed = validate_manifest_dict(_manifest())
    assert parsed.meta.num_nodes == 2


def test_manifest_schema_rejects_negative_event_time():
    m = _manifest()
    m["events"][0]["time"] = -1
    with pytest.raises(ManifestValidationError) as exc:
        validate_manifest_dict(m)
    assert "event.time must be >= 0" in str(exc.value)


def test_manifest_schema_rejects_non_positive_num_nodes():
    m = _manifest()
    m["meta"]["num_nodes"] = 0
    with pytest.raises(ManifestValidationError) as exc:
        validate_manifest_dict(m)
    assert "meta.num_nodes must be > 0" in str(exc.value)


def test_manifest_schema_rejects_negative_node_id():
    m = _manifest()
    m["nodes"][0]["id"] = -5
    with pytest.raises(ManifestValidationError) as exc:
        validate_manifest_dict(m)
    assert "node.id must be >= 0" in str(exc.value)


def test_manifest_schema_error_includes_actionable_hint_for_edge_reference():
    m = _manifest()
    m["layers"]["finance"]["edges"].append({"source": 3, "target": 1})
    with pytest.raises(ManifestValidationError) as exc:
        validate_manifest_dict(m)
    assert "Check layer edge construction and node ID mapping" in str(exc.value)


def test_manifest_schema_error_includes_actionable_hint_for_num_nodes_mismatch():
    m = _manifest()
    m["meta"]["num_nodes"] = 3
    with pytest.raises(ManifestValidationError) as exc:
        validate_manifest_dict(m)
    msg = str(exc.value)
    assert "does not match nodes length" in msg
    assert "Re-export manifest" in msg
