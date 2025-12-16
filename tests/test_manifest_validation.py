import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation.schema import ManifestValidationError, validate_manifest_dict


def _base_manifest():
    return {
        "meta": {"num_nodes": 3, "seed": 1, "generator": "test", "config": {}},
        "nodes": [
            {"id": 0, "role": "a", "region": "x", "group": "g"},
            {"id": 1, "role": "b", "region": "x", "group": "g"},
            {"id": 2, "role": "c", "region": "x", "group": "g"},
        ],
        "layers": {
            "finance": {"directed": True, "edges": [{"source": 0, "target": 1}]}
        },
        "events": [],
    }


def test_duplicate_nodes_raise():
    manifest = _base_manifest()
    manifest["nodes"][1]["id"] = 0

    with pytest.raises(ManifestValidationError):
        validate_manifest_dict(manifest)


def test_missing_edge_reference_raises():
    manifest = _base_manifest()
    manifest["layers"]["finance"]["edges"].append({"source": 5, "target": 1})

    with pytest.raises(ManifestValidationError):
        validate_manifest_dict(manifest)
