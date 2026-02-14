from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
_ALLOWED_EVENT_TYPES = {"comm", "txn", "op"}
_KNOWN_LAYERS = {"hierarchy", "finance", "communication", "operation", "ideology"}
_REQUIRED_ONTOLOGY_TERMS = (
    ":Actor a owl:Class .",
    ":Role a owl:Class .",
    ":Evidence a owl:Class .",
    ":EdgeProvenance a owl:Class",
    ":hasRole a owl:ObjectProperty",
    ":copiedFromLayer a owl:DatatypeProperty",
)
_REQUIRED_SHAPES_TERMS = (
    ":ActorRoleShape a sh:NodeShape",
    ":HierarchyNoSelfLoopShape a sh:NodeShape",
    ":FinanceEdgeAmountShape a sh:NodeShape",
    ":EventTypeShape a sh:NodeShape",
)


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _missing_terms(text: str, terms: Iterable[str]) -> List[str]:
    return [term for term in terms if term not in text]


def _validate_ontology_contract(ontology_text: str, shapes_text: str) -> List[str]:
    errors: List[str] = []

    missing_ontology = _missing_terms(ontology_text, _REQUIRED_ONTOLOGY_TERMS)
    if missing_ontology:
        errors.append("ontology is missing required terms: " + ", ".join(missing_ontology))

    missing_shapes = _missing_terms(shapes_text, _REQUIRED_SHAPES_TERMS)
    if missing_shapes:
        errors.append("shapes are missing required terms: " + ", ".join(missing_shapes))

    return errors


def _validate_roles(manifest: Manifest) -> List[str]:
    errors: List[str] = []
    for node in manifest.nodes:
        if node.role not in _ALLOWED_ROLES:
            errors.append(f"node {node.id} has unknown role '{node.role}'")
    return errors


def _validate_hierarchy(manifest: Manifest) -> List[str]:
    errors: List[str] = []
    hierarchy = manifest.layers.get("hierarchy")
    if not hierarchy:
        return errors

    role_by_id = {n.id: n.role for n in manifest.nodes}
    for idx, edge in enumerate(hierarchy.edges):
        if edge.source == edge.target:
            errors.append(f"hierarchy edge {idx} is self-loop ({edge.source}->{edge.target})")
        source_role = role_by_id.get(edge.source)
        if source_role not in _COMMAND_ROLES:
            errors.append(
                f"hierarchy edge {idx} source node {edge.source} has non-command role '{source_role}'"
            )
    return errors


def _validate_finance(manifest: Manifest) -> List[str]:
    errors: List[str] = []
    finance = manifest.layers.get("finance")
    if not finance:
        return errors

    for idx, edge in enumerate(finance.edges):
        amount = getattr(edge, "txn_amount_sum", None)
        if amount is not None and float(amount) <= 0:
            errors.append(f"finance edge {idx} has non-positive txn_amount_sum={amount}")

        txn_count = getattr(edge, "txn_count", None)
        if txn_count is not None and float(txn_count) < 0:
            errors.append(f"finance edge {idx} has negative txn_count={txn_count}")

    return errors


def _validate_events(manifest: Manifest) -> List[str]:
    errors: List[str] = []
    for idx, event in enumerate(manifest.events or []):
        if int(event.time) < 0:
            errors.append(f"event {idx} has negative timestamp={event.time}")
        if event.event_type not in _ALLOWED_EVENT_TYPES:
            errors.append(f"event {idx} has unsupported event_type='{event.event_type}'")
    return errors


def _iter_layer_edges(manifest: Manifest) -> Iterable[Tuple[str, Any]]:
    for layer_name, layer in manifest.layers.items():
        for edge in layer.edges:
            yield layer_name, edge


def _validate_provenance(manifest: Manifest) -> List[str]:
    errors: List[str] = []
    for layer_name, edge in _iter_layer_edges(manifest):
        is_false = getattr(edge, "is_false", None)
        if is_false is not None and int(is_false) not in (0, 1):
            errors.append(f"{layer_name} edge ({edge.source}->{edge.target}) has invalid is_false={is_false}")

        copied_from = getattr(edge, "copied_from", None)
        if copied_from is None:
            continue

        if copied_from not in _KNOWN_LAYERS:
            errors.append(
                f"{layer_name} edge ({edge.source}->{edge.target}) copied_from unknown layer '{copied_from}'"
            )
        if copied_from == layer_name:
            errors.append(
                f"{layer_name} edge ({edge.source}->{edge.target}) copied_from cannot match destination layer"
            )

    return errors


def _build_report(manifest: Manifest, assets: Dict[str, str]) -> Dict[str, Any]:
    ontology_text = _read_text(assets["ontology"])
    shapes_text = _read_text(assets["shapes"])

    checks = {
        "ontology_contract": _validate_ontology_contract(ontology_text, shapes_text),
        "roles": _validate_roles(manifest),
        "hierarchy": _validate_hierarchy(manifest),
        "finance": _validate_finance(manifest),
        "events": _validate_events(manifest),
        "provenance": _validate_provenance(manifest),
    }

    all_errors: List[str] = []
    for errs in checks.values():
        all_errors.extend(errs)

    return {
        "conforms": len(all_errors) == 0,
        "constraints_checked": len(checks),
        "errors": all_errors,
        "errors_by_check": checks,
        "assets": assets,
        "counts": {
            "nodes": len(manifest.nodes),
            "layers": len(manifest.layers),
            "events": len(manifest.events or []),
        },
    }


def _raise_if_invalid(report: Dict[str, Any]) -> None:
    if report["conforms"]:
        return
    raise OntologyValidationError("ontology validation failed: " + "; ".join(report["errors"]))


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
    _raise_if_invalid(report)
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
    _raise_if_invalid(report)
    return report


def write_ontology_report(report: Dict[str, Any], path: str) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
