import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.multiplex_generator_v3 import (
    GeneratorConfig,
    load_generator_config,
    validate_generator_config,
)


def test_validate_generator_config_accepts_defaults():
    cfg = GeneratorConfig(size=16, seed=1)
    validate_generator_config(cfg)


def test_validate_generator_config_rejects_rate_out_of_range():
    cfg = GeneratorConfig(size=16, seed=1, missing_edge_rate_finance=1.5)
    with pytest.raises(ValueError, match="missing_edge_rate_finance"):
        validate_generator_config(cfg)


def test_validate_generator_config_rejects_invalid_min_max():
    cfg = GeneratorConfig(size=16, seed=1, txn_events_min=4, txn_events_max=2)
    with pytest.raises(ValueError, match="txn_events_min"):
        validate_generator_config(cfg)


def test_validate_generator_config_rejects_bad_cross_layer_copy_spec():
    cfg = GeneratorConfig(size=16, seed=1, cross_layer_copy=[{"src": "finance", "dst": "communication", "rate": 1.2}])
    with pytest.raises(ValueError, match=r"cross_layer_copy\[0\]\.rate"):
        validate_generator_config(cfg)


def test_load_generator_config_applies_validation(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"missing_event_rate_txn": -0.1}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing_event_rate_txn"):
        load_generator_config(str(path), size=32, seed=3)
