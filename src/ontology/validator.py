from __future__ import annotations

import json
import math
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
    ":RelationRoleCompatibilityShape a sh:NodeShape",
    ":TemporalInteractionShape a sh:NodeShape",
)


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _missing_terms(text: str, terms: Iterable[str]) -> List[str]:
    return [term for term in terms if term not in text]


def _violation(
    check: str,
    rule_id: str,
    message: str,
    severity: str = "error",
    affected_ids: List[Any] | None = None,
) -> Dict[str, Any]:
    return {
        "check": check,
        "rule_id": rule_id,
        "message": message,
        "severity": severity,
        "affected_ids": affected_ids or [],
    }


def _to_float(value: Any) -> Tuple[float | None, str | None]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None, f"non-numeric value={value!r}"
    if not math.isfinite(out):
        return None, f"non-finite value={value!r}"
    return out, None


def _to_int(value: Any) -> Tuple[int | None, str | None]:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None, f"non-integer value={value!r}"
    return out, None


def _validate_ontology_contract(ontology_text: str, shapes_text: str) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []

    missing_ontology = _missing_terms(ontology_text, _REQUIRED_ONTOLOGY_TERMS)
    if missing_ontology:
        violations.append(
            _violation(
                check="ontology_contract",
                rule_id="ontology.required_terms",
                message="ontology is missing required terms: " + ", ".join(missing_ontology),
            )
        )

    missing_shapes = _missing_terms(shapes_text, _REQUIRED_SHAPES_TERMS)
    if missing_shapes:
        violations.append(
            _violation(
                check="ontology_contract",
                rule_id="shapes.required_terms",
                message="shapes are missing required terms: " + ", ".join(missing_shapes),
            )
        )

    return violations


def _validate_roles(manifest: Manifest) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    for node in manifest.nodes:
        if node.role not in _ALLOWED_ROLES:
            violations.append(
                _violation(
                    check="roles",
                    rule_id="roles.allowed",
                    message=f"node {node.id} has unknown role '{node.role}'",
                    affected_ids=[node.id],
                )
            )
    return violations


def _validate_hierarchy(manifest: Manifest) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    hierarchy = manifest.layers.get("hierarchy")
    if not hierarchy:
        return violations

    role_by_id = {n.id: n.role for n in manifest.nodes}
    for idx, edge in enumerate(hierarchy.edges):
        if edge.source == edge.target:
            violations.append(
                _violation(
                    check="hierarchy",
                    rule_id="hierarchy.no_self_loop",
                    message=f"hierarchy edge {idx} is self-loop ({edge.source}->{edge.target})",
                    affected_ids=[edge.source, edge.target],
                )
            )
        source_role = role_by_id.get(edge.source)
        if source_role not in _COMMAND_ROLES:
            violations.append(
                _violation(
                    check="hierarchy",
                    rule_id="hierarchy.command_source_role",
                    message=f"hierarchy edge {idx} source node {edge.source} has non-command role '{source_role}'",
                    affected_ids=[edge.source],
                )
            )
    return violations


def _validate_finance(manifest: Manifest) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    finance = manifest.layers.get("finance")
    if not finance:
        return violations

    for idx, edge in enumerate(finance.edges):
        amount = getattr(edge, "txn_amount_sum", None)
        if amount is not None:
            amount_val, amount_err = _to_float(amount)
            if amount_err:
                violations.append(
                    _violation(
                        check="finance",
                        rule_id="finance.amount_numeric",
                        message=f"finance edge {idx} has invalid txn_amount_sum ({amount_err})",
                        affected_ids=[edge.source, edge.target],
                    )
                )
            elif amount_val <= 0:
                violations.append(
                    _violation(
                        check="finance",
                        rule_id="finance.amount_positive",
                        message=f"finance edge {idx} has non-positive txn_amount_sum={amount}",
                        affected_ids=[edge.source, edge.target],
                    )
                )

        txn_count = getattr(edge, "txn_count", None)
        if txn_count is not None:
            count_val, count_err = _to_float(txn_count)
            if count_err:
                violations.append(
                    _violation(
                        check="finance",
                        rule_id="finance.txn_count_numeric",
                        message=f"finance edge {idx} has invalid txn_count ({count_err})",
                        affected_ids=[edge.source, edge.target],
                    )
                )
            elif count_val < 0:
                violations.append(
                    _violation(
                        check="finance",
                        rule_id="finance.txn_count_non_negative",
                        message=f"finance edge {idx} has negative txn_count={txn_count}",
                        affected_ids=[edge.source, edge.target],
                    )
                )

    return violations


def _validate_events(manifest: Manifest) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    for idx, event in enumerate(manifest.events or []):
        time_val, time_err = _to_int(event.time)
        if time_err:
            violations.append(
                _violation(
                    check="events",
                    rule_id="events.time_integer",
                    message=f"event {idx} has invalid timestamp ({time_err})",
                    affected_ids=[event.u, event.v],
                )
            )
        elif time_val < 0:
            violations.append(
                _violation(
                    check="events",
                    rule_id="events.non_negative_time",
                    message=f"event {idx} has negative timestamp={event.time}",
                    affected_ids=[event.u, event.v],
                )
            )
        if event.event_type not in _ALLOWED_EVENT_TYPES:
            violations.append(
                _violation(
                    check="events",
                    rule_id="events.allowed_type",
                    message=f"event {idx} has unsupported event_type='{event.event_type}'",
                    affected_ids=[event.u, event.v],
                )
            )
    return violations


def _iter_layer_edges(manifest: Manifest) -> Iterable[Tuple[str, Any]]:
    for layer_name, layer in manifest.layers.items():
        for edge in layer.edges:
            yield layer_name, edge


def _validate_relation_role_compatibility(manifest: Manifest) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    role_by_id = {n.id: n.role for n in manifest.nodes}

    # Conservative defaults to avoid over-constraining the synthetic generator.
    relation_rules: Dict[str, Dict[str, set[str]]] = {
        "hierarchy": {
            "source": {"leader", "financier", "operative"},
            "target": {"leader", "financier", "courier", "operative", "support"},
        },
        "finance": {
            "source": {"leader", "financier"},
            "target": {"leader", "financier", "courier", "operative", "support"},
        },
    }

    for layer_name, layer in manifest.layers.items():
        spec = relation_rules.get(layer_name)
        if not spec:
            continue
        allowed_src = spec["source"]
        allowed_dst = spec["target"]

        for idx, edge in enumerate(layer.edges):
            src_role = role_by_id.get(edge.source)
            dst_role = role_by_id.get(edge.target)
            if src_role not in allowed_src:
                violations.append(
                    _violation(
                        check="relation_role_compatibility",
                        rule_id=f"{layer_name}.source_role_allowed",
                        message=f"{layer_name} edge {idx} source node {edge.source} role '{src_role}' is not in allowed source roles",
                        affected_ids=[edge.source],
                    )
                )
            if dst_role not in allowed_dst:
                violations.append(
                    _violation(
                        check="relation_role_compatibility",
                        rule_id=f"{layer_name}.target_role_allowed",
                        message=f"{layer_name} edge {idx} target node {edge.target} role '{dst_role}' is not in allowed target roles",
                        affected_ids=[edge.target],
                    )
                )

    return violations


def _validate_provenance(manifest: Manifest) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    for layer_name, edge in _iter_layer_edges(manifest):
        is_false = getattr(edge, "is_false", None)
        if is_false is not None:
            is_false_val, is_false_err = _to_int(is_false)
            if is_false_err or is_false_val not in (0, 1):
                details = is_false_err or f"expected 0/1, got {is_false!r}"
                violations.append(
                    _violation(
                        check="provenance",
                        rule_id="provenance.is_false_binary",
                        message=f"{layer_name} edge ({edge.source}->{edge.target}) has invalid is_false ({details})",
                        affected_ids=[edge.source, edge.target],
                    )
                )

        confidence = getattr(edge, "confidence", None)
        if confidence is not None:
            conf_val, conf_err = _to_float(confidence)
            if conf_err:
                violations.append(
                    _violation(
                        check="provenance",
                        rule_id="provenance.confidence_numeric",
                        message=f"{layer_name} edge ({edge.source}->{edge.target}) has invalid confidence ({conf_err})",
                        affected_ids=[edge.source, edge.target],
                    )
                )
            elif not (0.0 <= conf_val <= 1.0):
                violations.append(
                    _violation(
                        check="provenance",
                        rule_id="provenance.confidence_range",
                        message=f"{layer_name} edge ({edge.source}->{edge.target}) has out-of-range confidence={confidence}",
                        affected_ids=[edge.source, edge.target],
                    )
                )

        copied_from = getattr(edge, "copied_from", None)
        if copied_from is None:
            continue

        if copied_from not in _KNOWN_LAYERS:
            violations.append(
                _violation(
                    check="provenance",
                    rule_id="provenance.copied_from_known_layer",
                    message=f"{layer_name} edge ({edge.source}->{edge.target}) copied_from unknown layer '{copied_from}'",
                    affected_ids=[edge.source, edge.target],
                )
            )
        if copied_from == layer_name:
            violations.append(
                _violation(
                    check="provenance",
                    rule_id="provenance.copied_from_not_same_layer",
                    message=f"{layer_name} edge ({edge.source}->{edge.target}) copied_from cannot match destination layer",
                    affected_ids=[edge.source, edge.target],
                )
            )

    return violations


def _event_times_by_type(manifest: Manifest) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {"comm": [], "txn": [], "op": []}
    for ev in manifest.events or []:
        if ev.event_type in out:
            out[ev.event_type].append(int(ev.time))
    return out


def _validate_temporal_interactions(manifest: Manifest, manifest_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    ontology = manifest_dict.get("ontology") if isinstance(manifest_dict.get("ontology"), dict) else {}
    interactions = ontology.get("layer_interactions") if isinstance(ontology.get("layer_interactions"), list) else []
    if not interactions:
        return violations

    event_map = {"communication": "comm", "finance": "txn", "operation": "op"}
    times_by_type = _event_times_by_type(manifest)

    for idx, rule in enumerate(interactions):
        if not isinstance(rule, dict):
            continue
        from_layer = str(rule.get("from_layer", "")).strip()
        to_layer = str(rule.get("to_layer", "")).strip()
        max_lag = rule.get("temporal_window_days", None)
        if to_layer != "operation" or from_layer not in event_map:
            continue
        if max_lag is None:
            continue

        from_type = event_map[from_layer]
        to_type = event_map["operation"]
        if not times_by_type[from_type] or not times_by_type[to_type]:
            continue

        latest_from = max(times_by_type[from_type])
        earliest_to = min(times_by_type[to_type])
        lag = earliest_to - latest_from

        if lag < 0:
            violations.append(
                _violation(
                    check="temporal_interactions",
                    rule_id=f"interaction[{idx}].ordering",
                    message=(
                        f"temporal interaction rule {idx} violated: operation event occurs before {from_layer} "
                        f"(earliest_op={earliest_to}, latest_{from_layer}={latest_from})"
                    ),
                    affected_ids=[from_layer, to_layer],
                )
            )
            continue

        if int(lag) > int(max_lag):
            violations.append(
                _violation(
                    check="temporal_interactions",
                    rule_id=f"interaction[{idx}].max_lag",
                    message=(
                        f"temporal interaction rule {idx} violated: lag={lag} exceeds temporal_window_days={max_lag} "
                        f"for {from_layer}->{to_layer}"
                    ),
                    affected_ids=[from_layer, to_layer],
                )
            )

    return violations


def _build_report(manifest: Manifest, assets: Dict[str, str], manifest_dict: Dict[str, Any]) -> Dict[str, Any]:
    ontology_text = _read_text(assets["ontology"])
    shapes_text = _read_text(assets["shapes"])

    checks: Dict[str, List[Dict[str, Any]]] = {
        "ontology_contract": _validate_ontology_contract(ontology_text, shapes_text),
        "roles": _validate_roles(manifest),
        "hierarchy": _validate_hierarchy(manifest),
        "finance": _validate_finance(manifest),
        "events": _validate_events(manifest),
        "relation_role_compatibility": _validate_relation_role_compatibility(manifest),
        "provenance": _validate_provenance(manifest),
        "temporal_interactions": _validate_temporal_interactions(manifest, manifest_dict),
    }

    all_violations: List[Dict[str, Any]] = []
    for vlist in checks.values():
        all_violations.extend(vlist)

    failure_violations = [v for v in all_violations if str(v.get("severity", "error")) in {"error", "critical"}]
    histogram: Dict[str, int] = {}
    for v in all_violations:
        rid = str(v.get("rule_id", "unknown"))
        histogram[rid] = int(histogram.get(rid, 0) + 1)

    errors_by_check = {k: [v["message"] for v in vlist] for k, vlist in checks.items()}

    return {
        "conforms": len(failure_violations) == 0,
        "constraints_checked": len(checks),
        "errors": [v["message"] for v in failure_violations],
        "errors_by_check": errors_by_check,
        "violations": all_violations,
        "violations_by_check": checks,
        "violation_histogram": histogram,
        "assets": assets,
        "counts": {
            "nodes": len(manifest.nodes),
            "layers": len(manifest.layers),
            "events": len(manifest.events or []),
            "violations_total": len(all_violations),
            "violations_error": len(failure_violations),
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
    strict: bool = True,
) -> Dict[str, Any]:
    try:
        assets = ensure_ontology_assets(ontology_path=ontology_path, shapes_path=shapes_path)
    except FileNotFoundError as exc:
        raise OntologyValidationError(str(exc)) from exc

    try:
        manifest = validate_manifest_dict(manifest_dict)
    except ManifestValidationError as exc:
        raise OntologyValidationError(f"manifest schema validation failed: {exc}") from exc

    report = _build_report(manifest, assets, manifest_dict)
    if strict:
        _raise_if_invalid(report)
    return report


def validate_manifest_with_ontology(
    manifest_path: str,
    ontology_path: str,
    shapes_path: str,
    strict: bool = True,
) -> Dict[str, Any]:
    try:
        assets = ensure_ontology_assets(ontology_path=ontology_path, shapes_path=shapes_path)
    except FileNotFoundError as exc:
        raise OntologyValidationError(str(exc)) from exc

    try:
        manifest = validate_manifest_file(manifest_path)
    except ManifestValidationError as exc:
        raise OntologyValidationError(f"manifest schema validation failed: {exc}") from exc

    manifest_dict = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    report = _build_report(manifest, assets, manifest_dict)
    if strict:
        _raise_if_invalid(report)
    return report


def write_ontology_report(report: Dict[str, Any], path: str) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
