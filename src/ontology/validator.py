from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.ontology.load import ensure_ontology_assets
from src.validation.schema import (
    Manifest,
    ManifestValidationError,
    validate_manifest_dict,
    validate_manifest_file,
)


class OntologyValidationError(Exception):
    """Raised when ontology-backed manifest validation fails."""


_ALLOWED_ROLES = {"leader", "financier", "courier", "operative", "support"}
_COMMAND_ROLES = {"leader", "financier", "operative"}


def _validate_constraints(manifest: Manifest) -> List[str]:
    errors: List[str] = []

    for node in manifest.nodes:
        if node.role not in _ALLOWED_ROLES:
            errors.append(f"node {node.id} has unknown role '{node.role}'")

    hierarchy = manifest.layers.get("hierarchy")
    if hierarchy:
        role_by_id = {n.id: n.role for n in manifest.nodes}
        for idx, edge in enumerate(hierarchy.edges):
            if edge.source == edge.target:
                errors.append(f"hierarchy edge {idx} is self-loop ({edge.source}->{edge.target})")
            source_role = role_by_id.get(edge.source)
            if source_role not in _COMMAND_ROLES:
                errors.append(
                    f"hierarchy edge {idx} source node {edge.source} has non-command role '{source_role}'"
                )

    finance = manifest.layers.get("finance")
    if finance:
        for idx, edge in enumerate(finance.edges):
            amount = getattr(edge, "txn_amount_sum", None)
            if amount is None:
                continue
            if float(amount) <= 0:
                errors.append(f"finance edge {idx} has non-positive txn_amount_sum={amount}")

    for idx, event in enumerate(manifest.events or []):
        if int(event.time) < 0:
            errors.append(f"event {idx} has negative timestamp={event.time}")

    return errors


def _build_report(manifest: Manifest, assets: Dict[str, str]) -> Dict[str, Any]:
    errors = _validate_constraints(manifest)
    return {
        "conforms": len(errors) == 0,
        "constraints_checked": 4,
        "errors": errors,
        "assets": assets,
        "counts": {
            "nodes": len(manifest.nodes),
            "layers": len(manifest.layers),
            "events": len(manifest.events or []),
        },
    }


def validate_manifest_dict_with_ontology(
    manifest_dict: Dict[str, Any],
    ontology_path: str,
    shapes_path: str,
) -> Dict[str, Any]:
    try:
        assets = ensure_ontology_assets(ontology_path=ontology_path, shapes_path=shapes_path)
    except FileNotFoundError as exc:
        raise OntologyValidationError(str(exc)) from exc

    try:
        manifest = validate_manifest_dict(manifest_dict)
    except ManifestValidationError as exc:
        raise OntologyValidationError(f"manifest schema validation failed: {exc}") from exc

    report = _build_report(manifest, assets)
    if not report["conforms"]:
        raise OntologyValidationError("ontology validation failed: " + "; ".join(report["errors"]))
    return report


def validate_manifest_with_ontology(
    manifest_path: str,
    ontology_path: str,
    shapes_path: str,
) -> Dict[str, Any]:
    try:
        assets = ensure_ontology_assets(ontology_path=ontology_path, shapes_path=shapes_path)
    except FileNotFoundError as exc:
        raise OntologyValidationError(str(exc)) from exc

    try:
        manifest = validate_manifest_file(manifest_path)
    except ManifestValidationError as exc:
        raise OntologyValidationError(f"manifest schema validation failed: {exc}") from exc

    report = _build_report(manifest, assets)
    if not report["conforms"]:
        raise OntologyValidationError("ontology validation failed: " + "; ".join(report["errors"]))

    return report


def write_ontology_report(report: Dict[str, Any], path: str) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
