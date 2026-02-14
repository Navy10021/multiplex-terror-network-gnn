import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ontology.report_schema import ONTOLOGY_REPORT_SCHEMA_VERSION, validate_ontology_report_schema


def test_ontology_report_schema_accepts_valid_report():
    report = {
        "schema_version": ONTOLOGY_REPORT_SCHEMA_VERSION,
        "conforms": True,
        "constraints_checked": 8,
        "errors": [],
        "errors_by_check": {"roles": []},
        "violations": [],
        "violations_by_check": {"roles": []},
        "violation_histogram": {},
        "assets": {"ontology": "ontology/terror.ttl", "shapes": "ontology/constraints.shacl.ttl"},
        "counts": {"nodes": 2, "layers": 1, "events": 0, "violations_total": 0, "violations_error": 0},
    }
    out = validate_ontology_report_schema(report)
    assert out["conforms"] is True


def test_ontology_report_schema_rejects_invalid_severity():
    report = {
        "schema_version": ONTOLOGY_REPORT_SCHEMA_VERSION,
        "conforms": False,
        "constraints_checked": 8,
        "errors": ["x"],
        "errors_by_check": {"roles": ["x"]},
        "violations": [
            {"check": "roles", "rule_id": "roles.allowed", "message": "bad", "severity": "fatal", "affected_ids": [1]}
        ],
        "violations_by_check": {
            "roles": [
                {"check": "roles", "rule_id": "roles.allowed", "message": "bad", "severity": "fatal", "affected_ids": [1]}
            ]
        },
        "violation_histogram": {"roles.allowed": 1},
        "assets": {"ontology": "ontology/terror.ttl", "shapes": "ontology/constraints.shacl.ttl"},
        "counts": {"nodes": 2, "layers": 1, "events": 0, "violations_total": 1, "violations_error": 1},
    }
    with pytest.raises(Exception):
        validate_ontology_report_schema(report)


def test_ontology_report_schema_rejects_invalid_schema_version():
    report = {
        "schema_version": "0.0.1",
        "conforms": True,
        "constraints_checked": 8,
        "errors": [],
        "errors_by_check": {"roles": []},
        "violations": [],
        "violations_by_check": {"roles": []},
        "violation_histogram": {},
        "assets": {"ontology": "ontology/terror.ttl", "shapes": "ontology/constraints.shacl.ttl"},
        "counts": {"nodes": 2, "layers": 1, "events": 0, "violations_total": 0, "violations_error": 0},
    }
    with pytest.raises(Exception):
        validate_ontology_report_schema(report)
