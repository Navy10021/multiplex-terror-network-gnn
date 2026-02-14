import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.multiplex_generator_v3 import (
    GeneratorConfig,
    generate_multiplex_with_config,
    generate_with_ontology_constraints,
)
from src.ontology.validator import validate_manifest_dict_with_ontology


def test_generator_v3_manifest_passes_ontology_validation():
    cfg = GeneratorConfig(size=64, seed=7)
    manifest = generate_multiplex_with_config(cfg)

    report = validate_manifest_dict_with_ontology(
        manifest_dict=manifest,
        ontology_path="ontology/terror.ttl",
        shapes_path="ontology/constraints.shacl.ttl",
    )

    assert report["conforms"] is True
    assert report["constraints_checked"] == 8


def test_ontology_constrained_generation_reports_telemetry(monkeypatch):
    cfg = GeneratorConfig(size=32, seed=11)
    state = {"count": 0}

    def _fake_validate(*, manifest_dict, ontology_path, shapes_path, strict=True):
        state["count"] += 1
        if state["count"] < 3:
            return {
                "conforms": False,
                "errors": ["synthetic failure"],
                "violation_histogram": {"rule.synthetic": 1},
                "counts": {"violations_error": 1},
            }
        return {
            "conforms": True,
            "errors": [],
            "violation_histogram": {},
            "counts": {"violations_error": 0},
        }

    monkeypatch.setattr(
        "src.data.multiplex_generator_v3.validate_manifest_dict_with_ontology",
        _fake_validate,
    )

    manifest, report, telemetry = generate_with_ontology_constraints(
        cfg=cfg,
        ontology_path="ontology/terror.ttl",
        shapes_path="ontology/constraints.shacl.ttl",
        max_retries=4,
        retry_seed_stride=5,
    )

    assert isinstance(manifest, dict)
    assert report["conforms"] is True
    assert telemetry["attempts"] == 3
    assert telemetry["failed_attempts"] == 2
    assert telemetry["successful_attempt"] == 3
    assert telemetry["seed_attempts"] == [11, 16, 21]
    assert telemetry["violation_histogram"]["rule.synthetic"] == 2


def test_ontology_constrained_retry_policy_filters_rules(monkeypatch):
    cfg = GeneratorConfig(size=16, seed=3)

    def _fake_validate(*, manifest_dict, ontology_path, shapes_path, strict=True):
        return {
            "conforms": False,
            "errors": ["non target violation"],
            "violations": [
                {
                    "check": "roles",
                    "rule_id": "roles.allowed",
                    "message": "bad role",
                    "severity": "error",
                    "affected_ids": [1],
                }
            ],
            "violation_histogram": {"roles.allowed": 1},
            "counts": {"violations_error": 1},
        }

    monkeypatch.setattr(
        "src.data.multiplex_generator_v3.validate_manifest_dict_with_ontology",
        _fake_validate,
    )

    _, report, telemetry = generate_with_ontology_constraints(
        cfg=cfg,
        ontology_path="ontology/terror.ttl",
        shapes_path="ontology/constraints.shacl.ttl",
        max_retries=5,
        retry_seed_stride=2,
        retry_on_rule_ids=["finance.amount_positive"],
        retry_on_severities=["error"],
    )

    assert report["conforms"] is False
    # since violated rule is not in retry_on_rule_ids, should stop after first attempt
    assert telemetry["attempts"] == 1
    assert telemetry["retry_log"][0]["retry_triggered"] is False
