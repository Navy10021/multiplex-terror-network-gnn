import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.build_pyg_dataset_v3 import build_pyg_data


def _manifest_base():
    return {
        "meta": {"num_nodes": 2, "seed": 1, "generator": "test", "config": {}},
        "ontology": {
            "roles": {
                "allowed_roles": ["leader", "support"],
                "counts": {"leader": 1, "support": 1},
            },
            "relations": {
                "hierarchy": {
                    "domain_roles": ["leader"],
                    "range_roles": ["support"],
                    "logical_props": {"transitive": True, "antisymmetric": True, "symmetric": False},
                    "temporal_props": {"time_ordered": False, "max_lag_days": None},
                },
                "finance": {
                    "domain_roles": ["leader"],
                    "range_roles": ["support"],
                    "logical_props": {"transitive": False, "antisymmetric": False, "symmetric": False},
                    "temporal_props": {"time_ordered": True, "max_lag_days": 30},
                },
            },
        },
        "nodes": [
            {"id": 0, "role": "leader", "region": "x", "group": "g"},
            {"id": 1, "role": "support", "region": "x", "group": "g"},
        ],
        "layers": {
            "hierarchy": {
                "directed": True,
                "edges": [{"source": 0, "target": 1, "is_false": 0, "confidence": 0.9}],
            },
            "finance": {
                "directed": True,
                "edges": [{"source": 0, "target": 1, "txn_amount_sum": 10.0, "is_false": 0, "copied_from": "communication", "confidence": 0.7}],
            },
        },
        "events": [{"time": 1, "event_type": "txn", "u": 0, "v": 1, "meta": {"amount": 10}}],
    }


def test_build_pyg_includes_ontology_bridge_tensors(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest_base()), encoding="utf-8")

    data = build_pyg_data(str(path))

    assert hasattr(data, "edge_ontology_attr")
    assert tuple(data.edge_ontology_attr.shape)[1] == 8
    assert hasattr(data, "node_ontology_features")
    assert tuple(data.node_ontology_features.shape) == (2, 2)
    assert hasattr(data, "role_compatibility_mask")
    # 5 relation types x 2 roles x 2 roles
    assert tuple(data.role_compatibility_mask.shape) == (5, 2, 2)
    assert torch.all(data.node_ontology_features.sum(dim=1) <= 1.0 + 1e-6)


def test_build_pyg_ontology_count_consistency_check(tmp_path: Path):
    m = _manifest_base()
    m["ontology"]["roles"]["counts"]["leader"] = 2
    path = tmp_path / "bad_manifest.json"
    path.write_text(json.dumps(m), encoding="utf-8")

    try:
        build_pyg_data(str(path))
    except ValueError as exc:
        assert "counts mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for inconsistent ontology role counts")
