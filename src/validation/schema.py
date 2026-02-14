from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, ValidationError, validator


class Meta(BaseModel):
    num_nodes: int
    seed: int
    generator: str
    config: Dict[str, Any] = Field(default_factory=dict)
    copy_summary: Dict[str, Dict[str, int]] = Field(default_factory=dict)


class Node(BaseModel):
    id: int
    role: str
    region: str
    group: str
    ideology: float = 0.0
    skill_level: float = 0.0
    radicalization: float = 0.0
    past_incidents: float = 0.0
    activity_rate: float = 1.0
    observability: float = 1.0
    importance_score: float = 0.0
    high_value_target: int = 0
    op_cell_id: Optional[int] = None


class Edge(BaseModel):
    source: int
    target: int
    is_false: Optional[int] = None
    copied_from: Optional[str] = None

    class Config:
        extra = "allow"


class Layer(BaseModel):
    directed: bool
    edges: List[Edge]

    @validator("edges")
    def _ensure_edges_have_nodes(cls, v: List[Edge]):
        if v is None:
            raise ValueError("edges list is required")
        return v


class Event(BaseModel):
    time: int
    event_type: str
    u: int
    v: int
    meta: Any


class Manifest(BaseModel):
    meta: Meta
    nodes: List[Node]
    layers: Dict[str, Layer]
    events: List[Event]

    class Config:
        extra = "allow"

    @validator("nodes")
    def _nodes_not_empty(cls, v: List[Node]):
        if v is None or len(v) == 0:
            raise ValueError("nodes list cannot be empty")
        return v

    @validator("layers")
    def _layers_not_empty(cls, v: Dict[str, Layer]):
        if v is None or len(v) == 0:
            raise ValueError("at least one layer is required")
        return v

    @validator("events")
    def _events_not_none(cls, v: List[Event]):
        return v or []

    @validator("layers", pre=True)
    def _layers_required(cls, v: Dict[str, Layer]):
        if v is None:
            raise ValueError("layers section is required")
        return v

    @validator("events", pre=True)
    def _events_required(cls, v: List[Event]):
        if v is None:
            return []
        return v

    @validator("nodes", pre=True)
    def _nodes_required(cls, v: List[Node]):
        if v is None:
            raise ValueError("nodes list is required")
        return v

    @validator("meta")
    def _meta_required(cls, v: Meta):
        if v is None:
            raise ValueError("meta section is required")
        return v

    @validator("layers")
    def _validate_node_references(
        cls, layers: Dict[str, Layer], values: Dict[str, Any]
    ) -> Dict[str, Layer]:
        nodes: List[Node] = values.get("nodes", [])
        meta: Optional[Meta] = values.get("meta")

        node_ids = [n.id for n in nodes]
        node_set: Set[int] = set(node_ids)

        errors = []

        counts = Counter(node_ids)
        duplicates = {nid for nid, freq in counts.items() if freq > 1}
        if duplicates:
            errors.append(f"duplicate node ids detected: {sorted(duplicates)}")

        expected_count = meta.num_nodes if meta else len(nodes)
        expected_ids = set(range(expected_count))
        missing = expected_ids - node_set
        unexpected = node_set - expected_ids
        if missing:
            errors.append(
                f"missing node ids for range 0..{expected_count - 1}: {sorted(missing)}"
            )
        if unexpected:
            errors.append(
                f"unexpected node ids (outside 0..{expected_count - 1}): {sorted(unexpected)}"
            )

        if meta and meta.num_nodes != len(nodes):
            errors.append(
                f"meta.num_nodes={meta.num_nodes} does not match nodes length={len(nodes)}"
            )

        for lname, layer in layers.items():
            for idx, edge in enumerate(layer.edges):
                for endpoint in (edge.source, edge.target):
                    if endpoint not in node_set:
                        errors.append(
                            f"layer '{lname}' edge {idx} references missing node {endpoint}"
                        )

        for idx, event in enumerate(values.get("events", [])):
            for endpoint in (event.u, event.v):
                if endpoint not in node_set:
                    errors.append(
                        f"event {idx} references missing node {endpoint}"
                    )

        if errors:
            raise ValueError("; ".join(errors))

        return layers


class ManifestValidationError(Exception):
    pass


def validate_manifest_dict(manifest: Dict[str, Any]) -> Manifest:
    try:
        return Manifest.parse_obj(manifest)
    except ValidationError as e:
        raise ManifestValidationError(str(e)) from e


def validate_manifest_file(path: str) -> Manifest:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return validate_manifest_dict(manifest)


def summarize_manifest(manifest: Manifest) -> Dict[str, int]:
    """Return basic counts for a validated manifest.

    Counts include nodes, total edges across layers, number of layers,
    and events. This is useful for quick smoke checks and reporting.
    """

    edge_total = sum(len(layer.edges) for layer in manifest.layers.values())

    return {
        "nodes": len(manifest.nodes),
        "edges": edge_total,
        "layers": len(manifest.layers),
        "events": len(manifest.events or []),
    }


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate a manifest JSON file and optionally summarize it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", help="Path to manifest JSON file")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print node/edge/layer/event counts after validation",
    )

    args = parser.parse_args()

    try:
        manifest = validate_manifest_file(args.path)
    except ManifestValidationError as exc:  # pragma: no cover - exercised in CLI usage
        parser.error(str(exc))
        return 1

    print(f"Validation succeeded for {args.path}")

    if args.summary:
        summary = summarize_manifest(manifest)
        print(
            "Summary: nodes={nodes}, edges={edges}, layers={layers}, events={events}".format(
                **summary
            )
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - entrypoint wrapper
    raise SystemExit(_cli())
