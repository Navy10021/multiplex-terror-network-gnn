from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field, validator

_ALLOWED_SEVERITIES = {"error", "warn", "warning", "info", "critical"}


class OntologyViolation(BaseModel):
    check: str
    rule_id: str
    message: str
    severity: str = "error"
    affected_ids: List[Any] = Field(default_factory=list)

    @validator("severity")
    def _severity_allowed(cls, v: str) -> str:
        if str(v) not in _ALLOWED_SEVERITIES:
            raise ValueError(f"unsupported severity '{v}'")
        return str(v)


class OntologyReportCounts(BaseModel):
    nodes: int
    layers: int
    events: int
    violations_total: int
    violations_error: int


class OntologyReport(BaseModel):
    conforms: bool
    constraints_checked: int
    errors: List[str] = Field(default_factory=list)
    errors_by_check: Dict[str, List[str]] = Field(default_factory=dict)
    violations: List[OntologyViolation] = Field(default_factory=list)
    violations_by_check: Dict[str, List[OntologyViolation]] = Field(default_factory=dict)
    violation_histogram: Dict[str, int] = Field(default_factory=dict)
    assets: Dict[str, str] = Field(default_factory=dict)
    counts: OntologyReportCounts


def validate_ontology_report_schema(report: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize ontology report structure.

    Returns a JSON-serializable normalized dictionary.
    """

    parsed = OntologyReport.parse_obj(report)
    return parsed.dict()
