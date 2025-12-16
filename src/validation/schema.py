from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

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
