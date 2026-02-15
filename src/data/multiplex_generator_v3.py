import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.ontology.validator import (
    OntologyValidationError,
    validate_manifest_dict_with_ontology,
    write_ontology_report,
)
from src.utils.exp_logging import build_artifact_dir, collect_run_metadata, write_run_metadata
from src.validation.schema import validate_manifest_dict

# -----------------------------
# Data class definitions
# -----------------------------


@dataclass
class Node:
    id: int
    role: str
    region: str
    group: str
    ideology: float

    # continuous attributes (used as node features)
    skill_level: float = 0.0
    radicalization: float = 0.0
    past_incidents: float = 0.0
    # dynamics / observation
    activity_rate: float = 1.0  # fraction of days active (used by on/off activity)
    observability: float = 1.0  # surveillance/measurement likelihood (used by biased observation)


    # labels / targets
    importance_score: float = 0.0
    high_value_target: int = 0  # 0/1

    # optional metadata
    op_cell_id: int = -1


@dataclass
class Edge:
    source: int
    target: int


@dataclass
class Event:
    time: int
    event_type: str  # "comm", "txn", "op"
    u: int
    v: int
    meta: Any


# -----------------------------
# Generator configuration
# -----------------------------


@dataclass
class GeneratorConfig:
    size: int = 1500
    seed: int = 2025

    # optional: externalize categorical distributions
    role_probs: Optional[Dict[str, float]] = None
    regions: Optional[List[str]] = None
    region_probs: Optional[List[float]] = None
    groups: Optional[List[str]] = None
    group_probs: Optional[List[float]] = None

    # finance layer
    finance_avg_out_degree: float = 18.0
    finance_w_group: float = 2.0
    finance_w_region: float = 1.5
    finance_w_ideo: float = 1.0
    finance_w_tier_dist: float = 1.0
    finance_base_bias: float = 0.1
    finance_structure_strength: float = 1.0

    # communication layer
    comm_avg_degree: float = 3.5
    comm_alpha0: float = 0.1
    comm_alpha_group: float = 1.5
    comm_alpha_region: float = 1.0
    comm_alpha_hier: float = 1.5
    comm_alpha_fin: float = 1.5
    comm_structure_strength: float = 1.0
    comm_randomness: float = 0.0  # 0~1 mixture with uniform

    # ideology layer
    ideo_threshold: float = 0.2

    # operation layer (cells)
    op_num_cells: int = 20
    op_cell_size: int = 4
    op_cell_homophily_strength: float = 2.0
    op_cell_w_group: float = 1.0
    op_cell_w_region: float = 0.6
    op_cell_w_ideo: float = 0.8
    op_inter_cell_bridge_rate: float = 0.05
    op_allow_overlap: bool = False
    op_role_template: Optional[Dict[str, float]] = None
    op_role_template_strength: float = 1.0

    # observation noise (edges)
    missing_edge_rate_hierarchy: float = 0.0
    missing_edge_rate_finance: float = 0.0
    missing_edge_rate_communication: float = 0.0
    missing_edge_rate_operation: float = 0.0
    missing_edge_rate_ideology: float = 0.0

    false_edge_rate_hierarchy: float = 0.0
    false_edge_rate_finance: float = 0.0
    false_edge_rate_communication: float = 0.0
    false_edge_rate_operation: float = 0.0
    false_edge_rate_ideology: float = 0.0

    false_edge_event_scale: float = 0.25

    # events
    num_days: int = 300
    txn_events_min: int = 1
    txn_events_max: int = 3
    comm_events_min: int = 1
    comm_events_max: int = 5
    op_events_min: int = 1
    op_events_max: int = 3

    missing_event_rate_txn: float = 0.0
    missing_event_rate_comm: float = 0.0
    missing_event_rate_op: float = 0.0

    # temporal burstiness
    event_burstiness: float = 0.0  # 0=uniform, 1=always from campaigns
    campaign_count: int = 3
    campaign_length: int = 20

    # activity on/off (node-level; impacts event generation)
    activity_onoff: bool = False
    activity_p_off_to_on: float = 0.03  # OFF->ON probability per day
    activity_p_on_to_off: float = 0.05  # ON->OFF probability per day
    activity_role_multiplier: Optional[Dict[str, float]] = None

    # biased observation (node-level observability; impacts edge missingness / false edges)
    observation_bias: bool = False
    observability_role_weight: Optional[Dict[str, float]] = None
    observability_region_weight: Optional[Dict[str, float]] = None
    obs_missing_bias_strength: float = 0.0  # >0 => low-observability nodes lose more edges
    obs_false_edge_bias_gamma: float = 0.0  # >0 => false edges prefer high-observability nodes

    # cross-layer edge copy (increase overlap/correlation across layers)
    cross_layer_copy: Optional[List[Dict[str, Any]]] = None

    # HVT
    hvt_ratio: float = 0.05


def _filter_cfg_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(GeneratorConfig)}
    return {k: v for k, v in d.items() if k in allowed}




def validate_generator_config(cfg: GeneratorConfig) -> None:
    """Validate generator config values before generation.

    Raises:
        ValueError: when a config invariant is violated.
    """

    def _check_unit_interval(name: str, value: float) -> None:
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{name} must be in [0,1], got {value}")

    if int(cfg.size) <= 0:
        raise ValueError(f"size must be > 0, got {cfg.size}")
    if int(cfg.num_days) <= 0:
        raise ValueError(f"num_days must be > 0, got {cfg.num_days}")
    if int(cfg.campaign_count) <= 0:
        raise ValueError(f"campaign_count must be > 0, got {cfg.campaign_count}")
    if int(cfg.campaign_length) <= 0:
        raise ValueError(f"campaign_length must be > 0, got {cfg.campaign_length}")

    unit_interval_fields = {
        "hvt_ratio": cfg.hvt_ratio,
        "event_burstiness": cfg.event_burstiness,
        "activity_p_off_to_on": cfg.activity_p_off_to_on,
        "activity_p_on_to_off": cfg.activity_p_on_to_off,
        "missing_edge_rate_hierarchy": cfg.missing_edge_rate_hierarchy,
        "missing_edge_rate_finance": cfg.missing_edge_rate_finance,
        "missing_edge_rate_communication": cfg.missing_edge_rate_communication,
        "missing_edge_rate_operation": cfg.missing_edge_rate_operation,
        "missing_edge_rate_ideology": cfg.missing_edge_rate_ideology,
        "false_edge_rate_hierarchy": cfg.false_edge_rate_hierarchy,
        "false_edge_rate_finance": cfg.false_edge_rate_finance,
        "false_edge_rate_communication": cfg.false_edge_rate_communication,
        "false_edge_rate_operation": cfg.false_edge_rate_operation,
        "false_edge_rate_ideology": cfg.false_edge_rate_ideology,
        "false_edge_event_scale": cfg.false_edge_event_scale,
        "missing_event_rate_txn": cfg.missing_event_rate_txn,
        "missing_event_rate_comm": cfg.missing_event_rate_comm,
        "missing_event_rate_op": cfg.missing_event_rate_op,
    }
    for name, value in unit_interval_fields.items():
        _check_unit_interval(name, float(value))

    min_max_pairs = [
        ("txn_events", int(cfg.txn_events_min), int(cfg.txn_events_max)),
        ("comm_events", int(cfg.comm_events_min), int(cfg.comm_events_max)),
        ("op_events", int(cfg.op_events_min), int(cfg.op_events_max)),
    ]
    for label, lo, hi in min_max_pairs:
        if lo <= 0 or hi <= 0:
            raise ValueError(f"{label}_min/max must be > 0, got ({lo}, {hi})")
        if lo > hi:
            raise ValueError(f"{label}_min must be <= {label}_max, got ({lo}, {hi})")

    if cfg.cross_layer_copy is not None:
        if not isinstance(cfg.cross_layer_copy, list):
            raise ValueError("cross_layer_copy must be a list of specs")
        for i, spec in enumerate(cfg.cross_layer_copy):
            if not isinstance(spec, dict):
                raise ValueError(f"cross_layer_copy[{i}] must be an object")
            src = spec.get("src")
            dst = spec.get("dst")
            rate = spec.get("rate", None)
            if not isinstance(src, str) or not src.strip():
                raise ValueError(f"cross_layer_copy[{i}].src must be a non-empty string")
            if not isinstance(dst, str) or not dst.strip():
                raise ValueError(f"cross_layer_copy[{i}].dst must be a non-empty string")
            if rate is None:
                raise ValueError(f"cross_layer_copy[{i}].rate is required")
            _check_unit_interval(f"cross_layer_copy[{i}].rate", float(rate))

def load_generator_config(config_path: Optional[str], size: int, seed: int) -> GeneratorConfig:
    """Load GeneratorConfig from JSON and override size/seed from CLI."""
    if config_path is None:
        cfg = GeneratorConfig(size=size, seed=seed)
        validate_generator_config(cfg)
        print("[*] No config file provided. Using default GeneratorConfig.")
        return cfg

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    cfg = GeneratorConfig(**_filter_cfg_dict(cfg_dict))
    cfg.size = size
    cfg.seed = seed
    validate_generator_config(cfg)

    print(f"[*] Loaded GeneratorConfig from {config_path}")
    return cfg


# -----------------------------
# Defaults
# -----------------------------


DEFAULT_ROLE_PROBS: Dict[str, float] = {
    "leader": 0.024,
    "financier": 0.044,
    "courier": 0.073,
    "operative": 0.256,
    "support": 0.603,
}

DEFAULT_REGIONS = ["Africa", "Asia", "Europe", "MiddleEast"]
DEFAULT_GROUPS = ["GroupA", "GroupB", "GroupC"]

ROLE_TIER = {
    "leader": 0,
    "financier": 1,
    "operative": 2,
    "courier": 2,
    "support": 3,
}

DEFAULT_ACTIVITY_ROLE_MULTIPLIER: Dict[str, float] = {
    "leader": 0.9,
    "financier": 1.0,
    "operative": 1.2,
    "courier": 1.3,
    "support": 0.8,
}

DEFAULT_OBSERVABILITY_ROLE_WEIGHT: Dict[str, float] = {
    "leader": 1.3,
    "financier": 1.1,
    "operative": 1.0,
    "courier": 0.9,
    "support": 0.8,
}

# Regions are configurable; these weights are only used when the region string matches.
DEFAULT_OBSERVABILITY_REGION_WEIGHT: Dict[str, float] = {
    "MiddleEast": 1.2,
    "Europe": 1.0,
    "Asia": 0.95,
    "Africa": 0.85,
}



# -----------------------------
# Utilities
# -----------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def _normalize_probs(probs: Optional[List[float]], n: int) -> np.ndarray:
    if probs is None:
        p = np.ones(n, dtype=float)
    else:
        p = np.array(probs, dtype=float)
        if p.size != n:
            raise ValueError(f"Probability length mismatch: expected {n}, got {p.size}")
    s = float(p.sum())
    if s <= 0:
        raise ValueError("Probabilities must sum to a positive value")
    return p / s


def sample_categorical(items: List[str], probs: Optional[List[float]], size: int) -> List[str]:
    p = _normalize_probs(probs, len(items))
    return np.random.choice(items, size=size, p=p).tolist()


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


# -----------------------------
# Node generation
# -----------------------------


def generate_nodes(cfg: GeneratorConfig) -> List[Node]:
    num_nodes = cfg.size

    role_probs = cfg.role_probs or DEFAULT_ROLE_PROBS
    roles = list(role_probs.keys())
    role_p = _normalize_probs(list(role_probs.values()), len(roles))
    sampled_roles = np.random.choice(roles, size=num_nodes, p=role_p).tolist()

    regions = cfg.regions or DEFAULT_REGIONS
    sampled_regions = sample_categorical(regions, cfg.region_probs, num_nodes)

    groups = cfg.groups or DEFAULT_GROUPS
    sampled_groups = sample_categorical(groups, cfg.group_probs, num_nodes)

    ideology = np.random.rand(num_nodes)

    # Continuous attributes: intentionally noisy (no hard leakage)
    role_skill_base = {
        "leader": 0.80,
        "financier": 0.72,
        "operative": 0.60,
        "courier": 0.45,
        "support": 0.30,
    }
    base_skill = np.array([role_skill_base.get(r, 0.4) for r in sampled_roles], dtype=float)
    skill = _clamp01(base_skill + np.random.normal(0.0, 0.08, size=num_nodes))

    # radicalization correlated with ideological extremity (0.0 at center, 1.0 at extremes)
    extremity = 2.0 * np.abs(ideology - 0.5)
    radical = _clamp01(extremity + np.random.normal(0.0, 0.10, size=num_nodes))

    # past_incidents: Poisson-like count (stored as float)
    role_inc_lambda = {
        "leader": 1.2,
        "financier": 1.1,
        "operative": 1.6,
        "courier": 1.3,
        "support": 0.6,
    }
    lam = np.array([role_inc_lambda.get(r, 1.0) for r in sampled_roles], dtype=float)
    past = np.random.poisson(lam=lam).astype(float)

    nodes: List[Node] = []
    for i in range(num_nodes):
        nodes.append(
            Node(
                id=i,
                role=sampled_roles[i],
                region=sampled_regions[i],
                group=sampled_groups[i],
                ideology=float(ideology[i]),
                skill_level=float(skill[i]),
                radicalization=float(radical[i]),
                past_incidents=float(past[i]),
            )
        )
    return nodes


# -----------------------------
# Hierarchy layer
# -----------------------------


# -----------------------------
# Activity on/off + biased observation helpers
# -----------------------------


def compute_observability_scores(cfg: GeneratorConfig, nodes: List[Node]) -> np.ndarray:
    """Return per-node observability scores in [0,1]."""
    n = len(nodes)
    if not bool(getattr(cfg, "observation_bias", False)) or n == 0:
        return np.ones(n, dtype=np.float32)

    role_w = cfg.observability_role_weight or DEFAULT_OBSERVABILITY_ROLE_WEIGHT
    region_w = cfg.observability_region_weight or DEFAULT_OBSERVABILITY_REGION_WEIGHT

    raw = np.array(
        [float(role_w.get(nd.role, 1.0)) * float(region_w.get(nd.region, 1.0)) for nd in nodes],
        dtype=np.float32,
    )
    raw = raw / (float(raw.max()) + 1e-12)
    raw = raw + np.random.normal(loc=0.0, scale=0.03, size=n).astype(np.float32)
    raw = np.clip(raw, 0.05, 1.0)
    return raw.astype(np.float32)


def generate_activity_matrix(cfg: GeneratorConfig, nodes: List[Node]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (active[N,T] bool, activity_rate[N] float)."""
    num_nodes = len(nodes)
    T = int(getattr(cfg, "num_days", 1))
    if T <= 0:
        T = 1

    if not bool(getattr(cfg, "activity_onoff", False)) or num_nodes == 0:
        active = np.ones((num_nodes, T), dtype=bool)
        rates = np.ones(num_nodes, dtype=np.float32)
        return active, rates

    role_mult = cfg.activity_role_multiplier or DEFAULT_ACTIVITY_ROLE_MULTIPLIER
    p01_base = float(np.clip(cfg.activity_p_off_to_on, 0.0, 1.0))
    p10_base = float(np.clip(cfg.activity_p_on_to_off, 0.0, 1.0))

    active = np.zeros((num_nodes, T), dtype=bool)
    rates = np.zeros(num_nodes, dtype=np.float32)

    for i, nd in enumerate(nodes):
        m = float(role_mult.get(nd.role, 1.0))
        # Higher m => more active on average: increase OFF->ON, decrease ON->OFF
        p01 = float(np.clip(p01_base * m, 0.0, 1.0))
        p10 = float(np.clip(p10_base / max(m, 1e-6), 0.0, 1.0))

        denom = p01 + p10
        p_on = (p01 / denom) if denom > 0 else 0.5
        state = bool(np.random.rand() < p_on)

        for t in range(T):
            active[i, t] = state
            if state:
                if np.random.rand() < p10:
                    state = False
            else:
                if np.random.rand() < p01:
                    state = True

        rates[i] = float(active[i].mean())

    return active, rates


def _edge_key(u: int, v: int, directed: bool) -> Tuple[int, int]:
    if directed:
        return (int(u), int(v))
    a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
    return (a, b)


def apply_cross_layer_edge_copy(
    cfg: GeneratorConfig,
    layer_edges: Dict[str, List[Edge]],
    layer_directed: Dict[str, bool],
) -> Tuple[Dict[str, List[Edge]], Dict[str, Dict[Tuple[int, int], str]]]:
    """Copy a fraction of edges from one layer to another.

    cfg.cross_layer_copy: list of dicts, each with:
      - src: str layer name
      - dst: str layer name
      - rate: float in [0,1]
      - preserve_direction: bool (optional; only relevant when src and dst are directed)
      - direction_mode: str (optional; when src is undirected and dst is directed: 'random'|'both'|'min_to_max'|'max_to_min')
    """
    specs = cfg.cross_layer_copy or []
    if not specs:
        provenance = {k: {} for k in layer_edges.keys()}
        return layer_edges, provenance

    # current key sets
    key_sets: Dict[str, set] = {}
    for lname, edges in layer_edges.items():
        directed = bool(layer_directed.get(lname, False))
        s = set()
        for e in edges:
            k = _edge_key(e.source, e.target, directed)
            if k[0] != k[1]:
                s.add(k)
        key_sets[lname] = s

    provenance: Dict[str, Dict[Tuple[int, int], str]] = {k: {} for k in layer_edges.keys()}

    for spec in specs:
        if not isinstance(spec, dict):
            continue
        src = spec.get("src")
        dst = spec.get("dst")
        rate = float(spec.get("rate", 0.0))
        if src not in key_sets or dst not in key_sets:
            continue
        if rate <= 0.0:
            continue

        src_directed = bool(layer_directed.get(src, False))
        dst_directed = bool(layer_directed.get(dst, False))

        src_keys = list(key_sets[src])
        if not src_keys:
            continue

        n_pick = int(round(rate * len(src_keys)))
        n_pick = max(0, min(n_pick, len(src_keys)))
        if n_pick <= 0:
            continue

        pick_idx = np.random.choice(len(src_keys), size=n_pick, replace=False)
        preserve_direction = bool(spec.get("preserve_direction", True))
        direction_mode = str(spec.get("direction_mode", "random"))

        for j in pick_idx:
            k = src_keys[int(j)]

            # translate into dst key(s)
            dst_keys: List[Tuple[int, int]] = []
            if src_directed:
                u, v = int(k[0]), int(k[1])
                if dst_directed:
                    if preserve_direction:
                        dst_keys = [(u, v)]
                    else:
                        dst_keys = [(u, v)] if np.random.rand() < 0.5 else [(v, u)]
                else:
                    dst_keys = [_edge_key(u, v, directed=False)]
            else:
                a, b = int(k[0]), int(k[1])
                if dst_directed:
                    if direction_mode == "both":
                        dst_keys = [(a, b), (b, a)]
                    elif direction_mode == "min_to_max":
                        dst_keys = [(a, b)]
                    elif direction_mode == "max_to_min":
                        dst_keys = [(b, a)]
                    else:  # random
                        dst_keys = [(a, b)] if np.random.rand() < 0.5 else [(b, a)]
                else:
                    dst_keys = [(a, b)]

            for dk in dst_keys:
                dk = _edge_key(dk[0], dk[1], directed=dst_directed)
                if dk[0] == dk[1]:
                    continue
                if dk in key_sets[dst]:
                    continue
                key_sets[dst].add(dk)
                provenance[dst][dk] = str(src)

    # rebuild lists (preserve original order, append newly copied edges)
    rebuilt: Dict[str, List[Edge]] = {}
    for lname, edges in layer_edges.items():
        directed = bool(layer_directed.get(lname, False))
        seen = set()
        ordered: List[Tuple[int, int]] = []
        for e in edges:
            k = _edge_key(e.source, e.target, directed)
            if k[0] == k[1] or k in seen:
                continue
            seen.add(k)
            ordered.append(k)

        new_keys = [k for k in key_sets[lname] if k not in seen]
        new_keys = sorted(new_keys)
        all_keys = ordered + new_keys
        rebuilt[lname] = [Edge(source=int(k[0]), target=int(k[1])) for k in all_keys]

    return rebuilt, provenance

def build_hierarchy_edges(nodes: List[Node]) -> List[Edge]:
    leaders = [n.id for n in nodes if n.role == "leader"]
    financiers = [n.id for n in nodes if n.role == "financier"]
    operatives = [n.id for n in nodes if n.role == "operative"]
    couriers = [n.id for n in nodes if n.role == "courier"]
    supports = [n.id for n in nodes if n.role == "support"]

    edges: List[Edge] = []
    if not leaders:
        return edges

    # leader -> financier/operative
    for idx, nid in enumerate(financiers + operatives):
        leader_id = leaders[idx % len(leaders)]
        edges.append(Edge(source=leader_id, target=nid))

    # mid -> courier/support
    mid = financiers + operatives
    if mid:
        for idx, nid in enumerate(couriers + supports):
            parent = mid[idx % len(mid)]
            edges.append(Edge(source=parent, target=nid))

    return edges


# -----------------------------
# Finance layer (structured)
# -----------------------------


def build_finance_edges(
    nodes: List[Node],
    avg_out_degree: float,
    w_group: float,
    w_region: float,
    w_ideo: float,
    w_tier_dist: float,
    base_bias: float,
) -> List[Edge]:
    id_to_node: Dict[int, Node] = {n.id: n for n in nodes}
    num_nodes = len(nodes)
    ideology = np.array([n.ideology for n in nodes], dtype=float)
    tiers = np.array([ROLE_TIER.get(n.role, 3) for n in nodes], dtype=float)

    financiers = [n.id for n in nodes if n.role == "financier"]
    if not financiers:
        return []

    edges: List[Edge] = []

    for u in financiers:
        target_k = max(5, int(np.random.normal(loc=avg_out_degree, scale=4.0)))

        u_node = id_to_node[u]
        u_group = u_node.group
        u_region = u_node.region
        u_ideo = u_node.ideology
        u_tier = ROLE_TIER.get(u_node.role, 3)

        candidates = [v for v in range(num_nodes) if v != u]
        cand_groups = np.array([id_to_node[v].group for v in candidates])
        cand_regions = np.array([id_to_node[v].region for v in candidates])
        cand_ideo = ideology[candidates]
        cand_tiers = tiers[candidates]

        same_group = (cand_groups == u_group).astype(float)
        same_region = (cand_regions == u_region).astype(float)
        ideo_sim = 1.0 - np.abs(cand_ideo - u_ideo)
        tier_dist = np.abs(cand_tiers - u_tier)

        score = (
            base_bias
            + w_group * same_group
            + w_region * same_region
            + w_ideo * ideo_sim
            - w_tier_dist * tier_dist
        )
        score = np.maximum(score, 1e-6)
        weights = score / score.sum()

        k = min(target_k, len(candidates))
        chosen = np.random.choice(len(candidates), size=k, replace=False, p=weights)
        for idx in chosen:
            v = int(candidates[int(idx)])
            edges.append(Edge(source=u, target=v))

    return edges


# -----------------------------
# Communication layer (structured + randomness)
# -----------------------------


def build_communication_edges(
    nodes: List[Node],
    hierarchy_edges: List[Edge],
    finance_edges: List[Edge],
    avg_degree: float,
    alpha0: float,
    alpha_group: float,
    alpha_region: float,
    alpha_hier: float,
    alpha_fin: float,
    randomness: float,
) -> List[Edge]:
    id_to_node: Dict[int, Node] = {n.id: n for n in nodes}
    num_nodes = len(nodes)

    hier_set = set((e.source, e.target) for e in hierarchy_edges) | set((e.target, e.source) for e in hierarchy_edges)
    fin_set = set((e.source, e.target) for e in finance_edges) | set((e.target, e.source) for e in finance_edges)

    edges_set = set()

    groups = np.array([n.group for n in nodes])
    regions = np.array([n.region for n in nodes])

    mix = float(np.clip(randomness, 0.0, 1.0))

    for u in range(num_nodes):
        target_k = max(1, int(np.random.normal(loc=avg_degree, scale=1.0)))

        u_group = groups[u]
        u_region = regions[u]

        candidates = np.array([v for v in range(num_nodes) if v != u], dtype=int)
        cand_groups = groups[candidates]
        cand_regions = regions[candidates]

        same_group = (cand_groups == u_group).astype(float)
        same_region = (cand_regions == u_region).astype(float)

        hier_link = np.array([1.0 if (u, int(v)) in hier_set else 0.0 for v in candidates], dtype=float)
        fin_link = np.array([1.0 if (u, int(v)) in fin_set else 0.0 for v in candidates], dtype=float)

        score = (
            alpha0
            + alpha_group * same_group
            + alpha_region * same_region
            + alpha_hier * hier_link
            + alpha_fin * fin_link
        )
        score = np.maximum(score, 1e-6)
        w = score / score.sum()

        # mix with uniform for controllable randomness
        if mix > 0:
            uni = np.full_like(w, 1.0 / len(w))
            w = (1.0 - mix) * w + mix * uni
            w = w / w.sum()

        k = min(target_k, len(candidates))
        chosen = np.random.choice(len(candidates), size=k, replace=False, p=w)
        for idx in chosen:
            v = int(candidates[int(idx)])
            a, b = (u, v) if u < v else (v, u)
            edges_set.add((a, b))

    return [Edge(source=a, target=b) for (a, b) in edges_set]


# -----------------------------
# Ideology layer
# -----------------------------


def build_ideology_edges(nodes: List[Node], threshold: float) -> List[Edge]:
    num_nodes = len(nodes)
    ideology = np.array([n.ideology for n in nodes], dtype=float)
    edges: List[Edge] = []
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if abs(float(ideology[u]) - float(ideology[v])) < threshold:
                edges.append(Edge(source=u, target=v))
    return edges


# -----------------------------
# Operation layer (homophilous cells + bridges)
# -----------------------------


def _role_quota_from_template(template: Dict[str, float], cell_size: int) -> Dict[str, int]:
    """Accept either fractions summing to ~1, or integer-like counts."""
    vals = list(template.values())
    if not vals:
        return {}

    # heuristic: if sums close to 1, treat as fractions
    s = float(sum(float(x) for x in vals))
    if 0.9 <= s <= 1.1:
        raw = {k: int(round(float(v) * cell_size)) for k, v in template.items()}
        # adjust to exact cell_size
        total = sum(raw.values())
        if total == 0:
            return {}
        # fix rounding drift
        while total < cell_size:
            # add one to the largest fractional weight role
            k = max(template.keys(), key=lambda kk: float(template[kk]))
            raw[k] += 1
            total += 1
        while total > cell_size:
            k = max(raw.keys(), key=lambda kk: raw[kk])
            if raw[k] > 0:
                raw[k] -= 1
                total -= 1
            else:
                break
        return {k: int(v) for k, v in raw.items() if int(v) > 0}

    # otherwise treat as counts
    raw2 = {k: int(round(float(v))) for k, v in template.items()}
    # cap total
    total2 = sum(raw2.values())
    if total2 <= 0:
        return {}
    if total2 > cell_size:
        # scale down proportionally
        scale = cell_size / total2
        raw2 = {k: int(np.floor(v * scale)) for k, v in raw2.items()}
        # fill remainder
        rem = cell_size - sum(raw2.values())
        for k in sorted(template.keys(), key=lambda kk: float(template[kk]), reverse=True):
            if rem <= 0:
                break
            raw2[k] += 1
            rem -= 1
    return {k: int(v) for k, v in raw2.items() if int(v) > 0}


def build_operation_edges(nodes: List[Node], cfg: GeneratorConfig) -> Tuple[List[Edge], Dict[Tuple[int, int], int]]:
    """Return (edges, false_edge_flags empty here; noise applied later)."""
    num_nodes = len(nodes)

    op_edges_set = set()
    cell_members: List[List[int]] = []

    all_ids = list(range(num_nodes))
    available = set(all_ids)

    group_arr = np.array([n.group for n in nodes])
    region_arr = np.array([n.region for n in nodes])
    ideo_arr = np.array([n.ideology for n in nodes], dtype=float)
    role_arr = np.array([n.role for n in nodes])

    template_quota: Optional[Dict[str, int]] = None
    if cfg.op_role_template:
        template_quota = _role_quota_from_template(cfg.op_role_template, cfg.op_cell_size)

    def _pick_node(anchor_id: int, pool: List[int], desired_role: Optional[str]) -> Optional[int]:
        if not pool:
            return None

        anchor_group = group_arr[anchor_id]
        anchor_region = region_arr[anchor_id]
        anchor_ideo = float(ideo_arr[anchor_id])

        pool_np = np.array(pool, dtype=int)

        same_group = (group_arr[pool_np] == anchor_group).astype(float)
        same_region = (region_arr[pool_np] == anchor_region).astype(float)
        ideo_sim = 1.0 - np.abs(ideo_arr[pool_np] - anchor_ideo)

        sim = (
            cfg.op_cell_w_group * same_group
            + cfg.op_cell_w_region * same_region
            + cfg.op_cell_w_ideo * ideo_sim
        )
        sim = np.maximum(sim, 0.0)

        # convert similarity to weights
        w = np.exp(float(cfg.op_cell_homophily_strength) * sim)

        if desired_role is not None:
            desired = (role_arr[pool_np] == desired_role).astype(float)
            # role template strength acts as a multiplicative boost
            boost = 1.0 + float(cfg.op_role_template_strength) * desired
            w = w * boost

        w = np.maximum(w, 1e-12)
        w = w / w.sum()

        chosen_idx = int(np.random.choice(len(pool_np), size=1, replace=False, p=w)[0])
        return int(pool_np[chosen_idx])

    for cid in range(int(cfg.op_num_cells)):
        if not cfg.op_allow_overlap and len(available) < int(cfg.op_cell_size):
            break

        pool_for_anchor = list(available) if not cfg.op_allow_overlap else all_ids
        anchor = int(random.choice(pool_for_anchor))

        chosen: List[int] = [anchor]

        # build desired roles list
        desired_roles: List[Optional[str]] = []
        if template_quota:
            for r, c in template_quota.items():
                desired_roles.extend([r] * int(c))
            # ensure length = cell_size-1 (anchor already fixed)
            # if template wants more than remaining slots, truncate
            desired_roles = desired_roles[: max(0, int(cfg.op_cell_size) - 1)]

        while len(chosen) < int(cfg.op_cell_size):
            desired_role = None
            if desired_roles:
                desired_role = desired_roles.pop(0)

            if cfg.op_allow_overlap:
                candidate_pool = [i for i in all_ids if i not in chosen]
            else:
                candidate_pool = [i for i in available if i not in chosen]

            picked = _pick_node(anchor, candidate_pool, desired_role)
            if picked is None:
                break
            chosen.append(picked)

        if len(chosen) < 2:
            continue

        # assign cell id
        for nid in chosen:
            nodes[nid].op_cell_id = cid

        cell_members.append(chosen)

        # fully connect within the cell (undirected)
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                a, b = chosen[i], chosen[j]
                x, y = (a, b) if a < b else (b, a)
                op_edges_set.add((x, y))

        if not cfg.op_allow_overlap:
            for nid in chosen:
                if nid in available:
                    available.remove(nid)

    # add inter-cell bridges
    if cfg.op_inter_cell_bridge_rate > 0 and len(cell_members) >= 2:
        attempts = max(1, int(round(cfg.op_inter_cell_bridge_rate * len(cell_members) * cfg.op_cell_size)))
        for _ in range(attempts):
            c1, c2 = random.sample(range(len(cell_members)), 2)
            m1 = cell_members[c1]
            m2 = cell_members[c2]

            # prefer courier nodes as bridges when available
            cands1 = [n for n in m1 if nodes[n].role == "courier"] or m1
            cands2 = [n for n in m2 if nodes[n].role == "courier"] or m2

            u = int(random.choice(cands1))
            v = int(random.choice(cands2))
            a, b = (u, v) if u < v else (v, u)
            op_edges_set.add((a, b))

    edges = [Edge(source=a, target=b) for (a, b) in op_edges_set]
    return edges, {}


# -----------------------------
# Observation noise
# -----------------------------


def apply_edge_observation_noise(
    edges: List[Edge],
    num_nodes: int,
    directed: bool,
    missing_rate: float,
    false_rate: float,
    obs_scores: Optional[np.ndarray] = None,
    missing_bias_strength: float = 0.0,
    false_bias_gamma: float = 0.0,
    allowed_false_sources: Optional[set[int]] = None,
    allowed_false_targets: Optional[set[int]] = None,
) -> Tuple[List[Edge], Dict[Tuple[int, int], int]]:
    """Return (noised_edges, false_edge_flags).

    - missing_rate: fraction of *true* edges that are dropped.
    - false_rate: fraction of *kept* edges to add as false positives.
    - obs_scores: per-node observability in [0,1]. If provided:
        * missing_bias_strength > 0 biases missingness toward low-observability nodes
        * false_bias_gamma > 0 biases false edges toward high-observability nodes
    """
    if edges is None:
        edges = []

    miss = float(np.clip(missing_rate, 0.0, 1.0))
    false = float(np.clip(false_rate, 0.0, 1.0))

    def _key(u: int, v: int) -> Tuple[int, int]:
        return _edge_key(u, v, directed=directed)

    # de-duplicate first
    existing_keys: List[Tuple[int, int]] = []
    seen = set()
    for e in edges:
        k = _key(e.source, e.target)
        if k[0] == k[1] or k in seen:
            continue
        seen.add(k)
        existing_keys.append(k)

    # -------------------------
    # missing edges
    # -------------------------
    n_drop = int(round(len(existing_keys) * miss))
    keep_mask = np.ones(len(existing_keys), dtype=bool)

    if n_drop > 0 and len(existing_keys) > 0:
        if obs_scores is not None and missing_bias_strength > 0.0:
            obs = np.asarray(obs_scores, dtype=np.float32)
            w = []
            for (u, v) in existing_keys:
                mu = float(obs[int(u)]) if 0 <= int(u) < len(obs) else 1.0
                mv = float(obs[int(v)]) if 0 <= int(v) < len(obs) else 1.0
                mean_obs = 0.5 * (mu + mv)
                # low observability => higher drop weight
                ww = 1.0 + float(missing_bias_strength) * (1.0 - mean_obs)
                w.append(max(1e-6, ww))
            w = np.asarray(w, dtype=np.float64)
            p = w / (w.sum() + 1e-12)
            drop_idx = np.random.choice(len(existing_keys), size=min(n_drop, len(existing_keys)), replace=False, p=p)
        else:
            drop_idx = np.random.choice(len(existing_keys), size=min(n_drop, len(existing_keys)), replace=False)

        keep_mask[drop_idx] = False

    kept_keys = [k for i, k in enumerate(existing_keys) if bool(keep_mask[i])]

    # -------------------------
    # false edges
    # -------------------------
    n_add_target = int(round(len(kept_keys) * false))
    false_flags: Dict[Tuple[int, int], int] = {}

    kept_set = set(kept_keys)

    if n_add_target > 0 and num_nodes > 1:
        if obs_scores is not None and false_bias_gamma > 0.0:
            obs = np.asarray(obs_scores, dtype=np.float64)
            obs = np.clip(obs, 1e-6, 1.0)
            w_nodes = np.power(obs, float(false_bias_gamma))
            w_nodes = w_nodes / (w_nodes.sum() + 1e-12)
        else:
            w_nodes = None

        tries = 0
        max_tries = max(2000, n_add_target * 100)
        while len(false_flags) < n_add_target and tries < max_tries:
            tries += 1
            if w_nodes is None:
                u = int(np.random.randint(0, num_nodes))
                v = int(np.random.randint(0, num_nodes - 1))
                if v >= u:
                    v += 1
            else:
                u = int(np.random.choice(num_nodes, p=w_nodes))
                v = int(np.random.choice(num_nodes, p=w_nodes))
                if v == u:
                    continue

            k = _key(u, v)
            if k[0] == k[1]:
                continue
            if allowed_false_sources is not None and int(k[0]) not in allowed_false_sources:
                continue
            if allowed_false_targets is not None and int(k[1]) not in allowed_false_targets:
                continue
            if k in kept_set or k in false_flags:
                continue
            false_flags[k] = 1

    # rebuild Edge list
    out_edges: List[Edge] = [Edge(source=int(k[0]), target=int(k[1])) for k in kept_keys]
    out_edges.extend([Edge(source=int(k[0]), target=int(k[1])) for k in false_flags.keys()])

    return out_edges, false_flags


def filter_edges_by_role_constraints(
    edges: List[Edge],
    role_by_id: Dict[int, str],
    allowed_source_roles: set[str],
    allowed_target_roles: Optional[set[str]] = None,
    directed: bool = True,
    false_flags: Optional[Dict[Tuple[int, int], int]] = None,
) -> Tuple[List[Edge], Optional[Dict[Tuple[int, int], int]]]:
    filtered: List[Edge] = []
    filtered_false: Optional[Dict[Tuple[int, int], int]] = {} if false_flags is not None else None

    for e in edges:
        k = _edge_key(int(e.source), int(e.target), directed=directed)
        src_role = role_by_id.get(int(k[0]))
        dst_role = role_by_id.get(int(k[1]))

        if src_role not in allowed_source_roles:
            continue
        if allowed_target_roles is not None and dst_role not in allowed_target_roles:
            continue

        filtered.append(Edge(source=int(k[0]), target=int(k[1])))
        if filtered_false is not None and false_flags is not None and k in false_flags:
            filtered_false[k] = 1

    return filtered, filtered_false

def _make_campaign_windows(cfg: GeneratorConfig) -> List[Tuple[int, int]]:
    # windows represented as (center, half_length)
    if cfg.campaign_count <= 0 or cfg.campaign_length <= 0:
        return []
    centers = np.random.randint(0, max(1, cfg.num_days), size=int(cfg.campaign_count)).tolist()
    half = max(1, int(cfg.campaign_length // 2))
    return [(int(c), half) for c in centers]


def _sample_time(cfg: GeneratorConfig, windows: List[Tuple[int, int]]) -> int:
    if cfg.num_days <= 1:
        return 0

    b = float(np.clip(cfg.event_burstiness, 0.0, 1.0))
    if b <= 0.0 or not windows:
        return int(np.random.randint(0, cfg.num_days))

    if np.random.rand() > b:
        return int(np.random.randint(0, cfg.num_days))

    center, half = windows[int(np.random.randint(0, len(windows)))]
    # truncated normal around center
    sd = max(1.0, float(half) / 3.0)
    t = int(round(np.random.normal(loc=float(center), scale=sd)))
    t = int(np.clip(t, 0, cfg.num_days - 1))
    return t


def build_events(
    cfg: GeneratorConfig,
    finance_edges: List[Edge],
    comm_edges: List[Edge],
    op_edges: List[Edge],
    false_fin: Dict[Tuple[int, int], int],
    false_comm: Dict[Tuple[int, int], int],
    false_op: Dict[Tuple[int, int], int],
    active: Optional[np.ndarray] = None,
    max_resample: int = 12,
) -> List[Event]:
    windows = _make_campaign_windows(cfg)
    events: List[Event] = []

    def _sample_time_active(u: int, v: int) -> Optional[int]:
        if active is None:
            return _sample_time(cfg, windows)
        T = int(cfg.num_days)
        if T <= 0:
            return 0
        for _ in range(max(1, int(max_resample))):
            t = _sample_time(cfg, windows)
            if 0 <= u < active.shape[0] and 0 <= v < active.shape[0] and 0 <= t < active.shape[1]:
                if bool(active[u, t]) and bool(active[v, t]):
                    return int(t)
        return None

    # finance -> txn (directed)
    for e in finance_edges:
        k = int(np.random.randint(cfg.txn_events_min, cfg.txn_events_max + 1))
        if (e.source, e.target) in false_fin:
            k = max(1, int(round(k * cfg.false_edge_event_scale)))
        for _ in range(k):
            t = _sample_time_active(int(e.source), int(e.target))
            if t is None:
                continue
            amount = float(np.random.lognormal(mean=8.0, sigma=1.0))
            events.append(Event(time=int(t), event_type="txn", u=int(e.source), v=int(e.target), meta={"amount": amount}))

    # communication -> comm (undirected)
    for e in comm_edges:
        a, b = (int(e.source), int(e.target)) if int(e.source) < int(e.target) else (int(e.target), int(e.source))
        k = int(np.random.randint(cfg.comm_events_min, cfg.comm_events_max + 1))
        if (a, b) in false_comm:
            k = max(1, int(round(k * cfg.false_edge_event_scale)))
        for _ in range(k):
            t = _sample_time_active(a, b)
            if t is None:
                continue
            duration = int(np.random.exponential(scale=5.0))
            events.append(Event(time=int(t), event_type="comm", u=a, v=b, meta={"duration": duration}))

    # operation -> op (undirected)
    for e in op_edges:
        a, b = (int(e.source), int(e.target)) if int(e.source) < int(e.target) else (int(e.target), int(e.source))
        k = int(np.random.randint(cfg.op_events_min, cfg.op_events_max + 1))
        if (a, b) in false_op:
            k = max(1, int(round(k * cfg.false_edge_event_scale)))
        for _ in range(k):
            t = _sample_time_active(a, b)
            if t is None:
                continue
            events.append(Event(time=int(t), event_type="op", u=a, v=b, meta={}))

    # event missingness
    if cfg.missing_event_rate_txn > 0:
        events = [ev for ev in events if ev.event_type != "txn" or np.random.rand() > cfg.missing_event_rate_txn]
    if cfg.missing_event_rate_comm > 0:
        events = [ev for ev in events if ev.event_type != "comm" or np.random.rand() > cfg.missing_event_rate_comm]
    if cfg.missing_event_rate_op > 0:
        events = [ev for ev in events if ev.event_type != "op" or np.random.rand() > cfg.missing_event_rate_op]

    events.sort(key=lambda ev: (int(ev.time), str(ev.event_type)))
    return events


def compute_importance_and_hvt(
    nodes: List[Node],
    hierarchy_edges: List[Edge],
    finance_edges: List[Edge],
    comm_edges: List[Edge],
    events: List[Event],
    hvt_ratio: float,
) -> None:
    num_nodes = len(nodes)

    hier_out = np.zeros(num_nodes, dtype=int)
    fin_out = np.zeros(num_nodes, dtype=int)
    comm_deg = np.zeros(num_nodes, dtype=int)
    op_count = np.zeros(num_nodes, dtype=int)

    for e in hierarchy_edges:
        hier_out[e.source] += 1
    for e in finance_edges:
        fin_out[e.source] += 1
    for e in comm_edges:
        comm_deg[e.source] += 1
        comm_deg[e.target] += 1
    for ev in events:
        if ev.event_type == "op":
            op_count[ev.u] += 1
            op_count[ev.v] += 1

    base_role_score = {
        "leader": 80.0,
        "financier": 70.0,
        "operative": 50.0,
        "courier": 40.0,
        "support": 20.0,
    }

    scores = np.zeros(num_nodes, dtype=float)
    for n in nodes:
        role_base = base_role_score.get(n.role, 30.0)
        s = (
            role_base
            + 5.0 * float(hier_out[n.id])
            + 3.0 * float(fin_out[n.id])
            + 1.5 * float(comm_deg[n.id])
            + 4.0 * float(op_count[n.id])
            + 2.5 * float(n.skill_level)
            + 1.5 * float(n.past_incidents)
        )
        scores[n.id] = float(s)
        n.importance_score = float(s)

    thr = float(np.quantile(scores, 1.0 - float(hvt_ratio)))
    for n in nodes:
        n.high_value_target = int(n.importance_score >= thr)


# -----------------------------
# Aggregate events into per-edge stats
# -----------------------------


def aggregate_event_stats(events: List[Event]) -> Tuple[Dict[Tuple[int, int], Dict[str, float]], Dict[Tuple[int, int], Dict[str, float]], Dict[Tuple[int, int], Dict[str, float]]]:
    txn: Dict[Tuple[int, int], Dict[str, float]] = {}
    comm: Dict[Tuple[int, int], Dict[str, float]] = {}
    op: Dict[Tuple[int, int], Dict[str, float]] = {}

    for ev in events:
        if ev.event_type == "txn":
            k = (int(ev.u), int(ev.v))
            d = txn.setdefault(k, {"txn_count": 0.0, "txn_amount_sum": 0.0, "txn_amount_max": 0.0})
            amt = float(ev.meta.get("amount", 0.0)) if isinstance(ev.meta, dict) else 0.0
            d["txn_count"] += 1.0
            d["txn_amount_sum"] += amt
            d["txn_amount_max"] = max(float(d["txn_amount_max"]), amt)
        elif ev.event_type == "comm":
            a, b = (int(ev.u), int(ev.v)) if int(ev.u) < int(ev.v) else (int(ev.v), int(ev.u))
            k = (a, b)
            d = comm.setdefault(k, {"comm_count": 0.0, "comm_duration_sum": 0.0})
            dur = float(ev.meta.get("duration", 0.0)) if isinstance(ev.meta, dict) else 0.0
            d["comm_count"] += 1.0
            d["comm_duration_sum"] += dur
        elif ev.event_type == "op":
            a, b = (int(ev.u), int(ev.v)) if int(ev.u) < int(ev.v) else (int(ev.v), int(ev.u))
            k = (a, b)
            d = op.setdefault(k, {"op_count": 0.0})
            d["op_count"] += 1.0

    # add means
    for d in txn.values():
        c = max(1.0, float(d.get("txn_count", 0.0)))
        d["txn_amount_mean"] = float(d.get("txn_amount_sum", 0.0)) / c
    for d in comm.values():
        c = max(1.0, float(d.get("comm_count", 0.0)))
        d["comm_duration_mean"] = float(d.get("comm_duration_sum", 0.0)) / c

    return txn, comm, op


# -----------------------------
# Main generator
# -----------------------------


def generate_multiplex_with_config(cfg: GeneratorConfig) -> Dict[str, Any]:
    """Generate a multiplex manifest (v3) with optional:
    - node activity on/off (affects event generation)
    - biased observation (affects edge missingness / false edges)
    - cross-layer edge copy (increases overlap across layers)

    Output schema is compatible with build_pyg_dataset_v3.py.
    """
    set_seed(cfg.seed)

    nodes = generate_nodes(cfg)
    num_nodes = int(cfg.size)

    # --------------------------------------------------
    # 0) node-level activity + observability
    # --------------------------------------------------
    obs_scores = compute_observability_scores(cfg, nodes)
    active_mat, activity_rates = generate_activity_matrix(cfg, nodes)

    for i, nd in enumerate(nodes):
        nd.observability = float(obs_scores[i]) if i < len(obs_scores) else 1.0
        nd.activity_rate = float(activity_rates[i]) if i < len(activity_rates) else 1.0

    role_by_id: Dict[int, str] = {n.id: n.role for n in nodes}
    hierarchy_allowed_src = {"leader", "financier", "operative"}
    hierarchy_allowed_dst = {"leader", "financier", "courier", "operative", "support"}
    finance_allowed_src = {"leader", "financier"}
    finance_allowed_dst = {"leader", "financier", "courier", "operative", "support"}
    hierarchy_allowed_src_ids = {nid for nid, role in role_by_id.items() if role in hierarchy_allowed_src}
    hierarchy_allowed_dst_ids = {nid for nid, role in role_by_id.items() if role in hierarchy_allowed_dst}
    finance_allowed_src_ids = {nid for nid, role in role_by_id.items() if role in finance_allowed_src}
    finance_allowed_dst_ids = {nid for nid, role in role_by_id.items() if role in finance_allowed_dst}

    # --------------------------------------------------
    # 1) base layers
    # --------------------------------------------------
    hierarchy_edges = build_hierarchy_edges(nodes)

    fs = float(cfg.finance_structure_strength)
    finance_edges = build_finance_edges(
        nodes,
        avg_out_degree=cfg.finance_avg_out_degree,
        w_group=cfg.finance_w_group * fs,
        w_region=cfg.finance_w_region * fs,
        w_ideo=cfg.finance_w_ideo * fs,
        w_tier_dist=cfg.finance_w_tier_dist * fs,
        base_bias=cfg.finance_base_bias,
    )

    # ideology and operation first (communication depends on finance/hierarchy)
    ideology_edges = build_ideology_edges(nodes, threshold=cfg.ideo_threshold)
    operation_edges, _ = build_operation_edges(nodes, cfg)

    cs = float(cfg.comm_structure_strength)
    cr = float(cfg.comm_randomness)

    communication_edges = build_communication_edges(
        nodes,
        hierarchy_edges,
        finance_edges,
        avg_degree=cfg.comm_avg_degree,
        alpha0=cfg.comm_alpha0 * (1.0 + cr),
        alpha_group=cfg.comm_alpha_group * cs,
        alpha_region=cfg.comm_alpha_region * cs,
        alpha_hier=cfg.comm_alpha_hier * cs,
        alpha_fin=cfg.comm_alpha_fin * cs,
        randomness=cr,
    )

    # --------------------------------------------------
    # 1b) cross-layer edge copy (optional)
    # --------------------------------------------------
    layer_edges = {
        "hierarchy": hierarchy_edges,
        "finance": finance_edges,
        "communication": communication_edges,
        "operation": operation_edges,
        "ideology": ideology_edges,
    }
    layer_directed = {
        "hierarchy": True,
        "finance": True,
        "communication": False,
        "operation": False,
        "ideology": False,
    }

    layer_edges, copy_provenance = apply_cross_layer_edge_copy(cfg, layer_edges, layer_directed)
    hierarchy_edges = layer_edges["hierarchy"]
    finance_edges = layer_edges["finance"]
    communication_edges = layer_edges["communication"]
    operation_edges = layer_edges["operation"]
    ideology_edges = layer_edges["ideology"]

    # Enforce ontology role compatibility after optional cross-layer copy.
    hierarchy_edges, _ = filter_edges_by_role_constraints(
        hierarchy_edges,
        role_by_id=role_by_id,
        allowed_source_roles=hierarchy_allowed_src,
        allowed_target_roles=hierarchy_allowed_dst,
        directed=True,
    )
    finance_edges, _ = filter_edges_by_role_constraints(
        finance_edges,
        role_by_id=role_by_id,
        allowed_source_roles=finance_allowed_src,
        allowed_target_roles=finance_allowed_dst,
        directed=True,
    )

    # --------------------------------------------------
    # 2) observation noise (optional; per-layer rates)
    # --------------------------------------------------
    obs_for_noise = obs_scores if bool(cfg.observation_bias) else None
    miss_bias = float(cfg.obs_missing_bias_strength) if bool(cfg.observation_bias) else 0.0
    false_gamma = float(cfg.obs_false_edge_bias_gamma) if bool(cfg.observation_bias) else 0.0

    hierarchy_edges, false_hier = apply_edge_observation_noise(
        hierarchy_edges,
        num_nodes=num_nodes,
        directed=True,
        missing_rate=float(cfg.missing_edge_rate_hierarchy),
        false_rate=float(cfg.false_edge_rate_hierarchy),
        obs_scores=obs_for_noise,
        missing_bias_strength=miss_bias,
        false_bias_gamma=false_gamma,
        allowed_false_sources=hierarchy_allowed_src_ids,
        allowed_false_targets=hierarchy_allowed_dst_ids,
    )

    finance_edges, false_fin = apply_edge_observation_noise(
        finance_edges,
        num_nodes=num_nodes,
        directed=True,
        missing_rate=float(cfg.missing_edge_rate_finance),
        false_rate=float(cfg.false_edge_rate_finance),
        obs_scores=obs_for_noise,
        missing_bias_strength=miss_bias,
        false_bias_gamma=false_gamma,
        allowed_false_sources=finance_allowed_src_ids,
        allowed_false_targets=finance_allowed_dst_ids,
    )

    communication_edges, false_comm = apply_edge_observation_noise(
        communication_edges,
        num_nodes=num_nodes,
        directed=False,
        missing_rate=float(cfg.missing_edge_rate_communication),
        false_rate=float(cfg.false_edge_rate_communication),
        obs_scores=obs_for_noise,
        missing_bias_strength=miss_bias,
        false_bias_gamma=false_gamma,
    )

    operation_edges, false_op = apply_edge_observation_noise(
        operation_edges,
        num_nodes=num_nodes,
        directed=False,
        missing_rate=float(cfg.missing_edge_rate_operation),
        false_rate=float(cfg.false_edge_rate_operation),
        obs_scores=obs_for_noise,
        missing_bias_strength=miss_bias,
        false_bias_gamma=false_gamma,
    )

    ideology_edges, false_ideo = apply_edge_observation_noise(
        ideology_edges,
        num_nodes=num_nodes,
        directed=False,
        missing_rate=float(cfg.missing_edge_rate_ideology),
        false_rate=float(cfg.false_edge_rate_ideology),
        obs_scores=obs_for_noise,
        missing_bias_strength=miss_bias,
        false_bias_gamma=false_gamma,
    )

    # Re-apply constraints because false-edge injection can introduce invalid role pairings.
    hierarchy_edges, false_hier = filter_edges_by_role_constraints(
        hierarchy_edges,
        role_by_id=role_by_id,
        allowed_source_roles=hierarchy_allowed_src,
        allowed_target_roles=hierarchy_allowed_dst,
        directed=True,
        false_flags=false_hier,
    )
    finance_edges, false_fin = filter_edges_by_role_constraints(
        finance_edges,
        role_by_id=role_by_id,
        allowed_source_roles=finance_allowed_src,
        allowed_target_roles=finance_allowed_dst,
        directed=True,
        false_flags=false_fin,
    )

    # --------------------------------------------------
    # 3) events (affected by activity on/off + edge false-positives)
    # --------------------------------------------------
    events = build_events(
        cfg,
        finance_edges=finance_edges,
        comm_edges=communication_edges,
        op_edges=operation_edges,
        false_fin=false_fin,
        false_comm=false_comm,
        false_op=false_op,
        active=active_mat,
    )

    # --------------------------------------------------
    # 4) node targets (importance/HVT)
    # --------------------------------------------------
    compute_importance_and_hvt(
        nodes,
        hierarchy_edges=hierarchy_edges,
        finance_edges=finance_edges,
        comm_edges=communication_edges,
        events=events,
        hvt_ratio=float(cfg.hvt_ratio),
    )

    # --------------------------------------------------
    # 5) aggregate per-edge stats from events
    # --------------------------------------------------
    txn_stats, comm_stats, op_stats = aggregate_event_stats(events)
    id_to_node: Dict[int, Node] = {n.id: n for n in nodes}

    # --------------------------------------------------
    # 6) assemble manifest
    # --------------------------------------------------
    def _k(u: int, v: int, directed: bool) -> Tuple[int, int]:
        return _edge_key(int(u), int(v), directed=directed)

    # hierarchy (directed)
    hier_edges_out: List[Dict[str, Any]] = []
    for e in hierarchy_edges:
        k = _k(e.source, e.target, directed=True)
        d: Dict[str, Any] = {"source": int(k[0]), "target": int(k[1])}
        d["is_false"] = int(k in false_hier)
        d["copied_from"] = copy_provenance.get("hierarchy", {}).get(k)
        hier_edges_out.append(d)

    # finance (directed)
    fin_edges_out: List[Dict[str, Any]] = []
    for e in finance_edges:
        k = _k(e.source, e.target, directed=True)
        st = txn_stats.get(k, {"txn_count": 0.0, "txn_amount_sum": 0.0, "txn_amount_max": 0.0, "txn_amount_mean": 0.0})
        if float(st.get("txn_amount_sum", 0.0)) <= 0.0:
            # Keep ontology-consistent finance attributes even when all events were dropped
            # (e.g., due to activity gating or missing-event noise).
            st = {
                "txn_count": max(1.0, float(st.get("txn_count", 0.0))),
                "txn_amount_sum": 1.0,
                "txn_amount_max": 1.0,
                "txn_amount_mean": 1.0,
            }
        d = {"source": int(k[0]), "target": int(k[1]), **{kk: float(vv) for kk, vv in st.items()}}
        d["is_false"] = int(k in false_fin)
        d["copied_from"] = copy_provenance.get("finance", {}).get(k)
        fin_edges_out.append(d)

    # communication (undirected)
    comm_edges_out: List[Dict[str, Any]] = []
    for e in communication_edges:
        k = _k(e.source, e.target, directed=False)
        st = comm_stats.get(k, {"comm_count": 0.0, "comm_duration_sum": 0.0, "comm_duration_mean": 0.0})
        d = {"source": int(k[0]), "target": int(k[1]), **{kk: float(vv) for kk, vv in st.items()}}
        d["is_false"] = int(k in false_comm)
        d["copied_from"] = copy_provenance.get("communication", {}).get(k)
        comm_edges_out.append(d)

    # operation (undirected)
    op_edges_out: List[Dict[str, Any]] = []
    for e in operation_edges:
        k = _k(e.source, e.target, directed=False)
        st = op_stats.get(k, {"op_count": 0.0})
        a, b = int(k[0]), int(k[1])
        same_cell = int(id_to_node.get(a).op_cell_id == id_to_node.get(b).op_cell_id) if (a in id_to_node and b in id_to_node) else 0
        d = {"source": a, "target": b, **{kk: float(vv) for kk, vv in st.items()}, "same_cell": int(same_cell)}
        d["is_false"] = int(k in false_op)
        d["copied_from"] = copy_provenance.get("operation", {}).get(k)
        op_edges_out.append(d)

    # ideology (undirected)
    ideo_edges_out: List[Dict[str, Any]] = []
    for e in ideology_edges:
        k = _k(e.source, e.target, directed=False)
        a, b = int(k[0]), int(k[1])
        u_ideo = float(id_to_node[a].ideology) if a in id_to_node else 0.5
        v_ideo = float(id_to_node[b].ideology) if b in id_to_node else 0.5
        sim = 1.0 - abs(u_ideo - v_ideo)
        d = {"source": a, "target": b, "similarity": float(sim)}
        d["is_false"] = int(k in false_ideo)
        d["copied_from"] = copy_provenance.get("ideology", {}).get(k)
        ideo_edges_out.append(d)

    # summary of copy provenance (for quick diagnostics)
    copy_summary: Dict[str, Dict[str, int]] = {}
    for lname, mp in (copy_provenance or {}).items():
        cnt: Dict[str, int] = {}
        for src in mp.values():
            cnt[str(src)] = int(cnt.get(str(src), 0) + 1)
        if cnt:
            copy_summary[str(lname)] = cnt

    manifest = {
        "meta": {
            "num_nodes": int(cfg.size),
            "seed": int(cfg.seed),
            "generator": "multiplex_generator_v3",
            "config": asdict(cfg),
            "copy_summary": copy_summary,
        },
        "nodes": [asdict(n) for n in nodes],
        "layers": {
            "hierarchy": {"directed": True, "edges": hier_edges_out},
            "finance": {"directed": True, "edges": fin_edges_out},
            "communication": {"directed": False, "edges": comm_edges_out},
            "operation": {"directed": False, "edges": op_edges_out},
            "ideology": {"directed": False, "edges": ideo_edges_out},
        },
        "events": [asdict(ev) for ev in events],
    }

    return manifest



def generate_with_ontology_constraints(
    cfg: GeneratorConfig,
    ontology_path: str,
    shapes_path: str,
    max_retries: int = 3,
    retry_seed_stride: int = 1,
    retry_on_rule_ids: Optional[List[str]] = None,
    retry_on_severities: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Generate manifests with ontology-aware retries.

    Returns (manifest, ontology_report, telemetry).
    Retries are governed by:
    - max_retries
    - retry_seed_stride
    - retry_on_rule_ids (empty => any rule can trigger retry)
    - retry_on_severities (default: ["error", "critical"]) 
    """
    attempts = max(1, int(max_retries))
    stride = max(1, int(retry_seed_stride))
    target_rules = {str(x).strip() for x in (retry_on_rule_ids or []) if str(x).strip()}
    target_sev = {str(x).strip() for x in (retry_on_severities or ["error", "critical"]) if str(x).strip()}

    telemetry: Dict[str, Any] = {
        "mode": "ontology_constrained",
        "max_retries": attempts,
        "attempts": 0,
        "failed_attempts": 0,
        "successful_attempt": None,
        "seed_attempts": [],
        "violation_histogram": {},
        "attempt_summaries": [],
        "retry_policy": {
            "retry_seed_stride": stride,
            "retry_on_rule_ids": sorted(list(target_rules)),
            "retry_on_severities": sorted(list(target_sev)),
        },
        "retry_log": [],
    }

    last_manifest: Dict[str, Any] | None = None
    last_report: Dict[str, Any] | None = None

    for attempt in range(attempts):
        seed_i = int(cfg.seed) + attempt * stride
        cfg_i = replace(cfg, seed=seed_i)
        manifest = generate_multiplex_with_config(cfg_i)
        report = validate_manifest_dict_with_ontology(
            manifest_dict=manifest,
            ontology_path=ontology_path,
            shapes_path=shapes_path,
            strict=False,
        )

        violations = report.get("violations", []) if isinstance(report.get("violations"), list) else []
        matched = []
        for v in violations:
            if not isinstance(v, dict):
                continue
            sev = str(v.get("severity", "error"))
            rid = str(v.get("rule_id", ""))
            if target_sev and sev not in target_sev:
                continue
            if target_rules and rid not in target_rules:
                continue
            matched.append(rid)

        should_retry = (not bool(report.get("conforms", False))) and (attempt < attempts - 1) and (len(matched) > 0 or len(target_rules) == 0)

        telemetry["attempts"] = int(telemetry["attempts"] + 1)
        telemetry["seed_attempts"].append(seed_i)
        telemetry["attempt_summaries"].append(
            {
                "attempt": attempt + 1,
                "seed": seed_i,
                "conforms": bool(report.get("conforms", False)),
                "violations_error": int((report.get("counts") or {}).get("violations_error", len(report.get("errors", [])))),
                "matched_retry_rules": sorted(list(set(matched))),
                "retry_triggered": bool(should_retry),
            }
        )

        telemetry["retry_log"].append(
            {
                "attempt": attempt + 1,
                "seed": seed_i,
                "conforms": bool(report.get("conforms", False)),
                "matched_retry_rules": sorted(list(set(matched))),
                "retry_triggered": bool(should_retry),
            }
        )

        for k, v in (report.get("violation_histogram") or {}).items():
            telemetry["violation_histogram"][str(k)] = int(telemetry["violation_histogram"].get(str(k), 0) + int(v))

        last_manifest = manifest
        last_report = report

        if bool(report.get("conforms", False)):
            telemetry["successful_attempt"] = attempt + 1
            return manifest, report, telemetry

        telemetry["failed_attempts"] = int(telemetry["failed_attempts"] + 1)

        if not should_retry:
            break

    assert last_manifest is not None and last_report is not None
    return last_manifest, last_report, telemetry




def main() -> None:
    parser = argparse.ArgumentParser(description="Structured Multiplex Terrorist Network Generator v3")
    parser.add_argument("--size", type=int, default=1500, help="number of nodes")
    parser.add_argument("--out_dir", type=str, default=None, help="output directory (legacy/manual)")
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="If provided, create an auto-named run folder under this root (UTC+hash+seed)",
    )
    parser.add_argument("--run_prefix", type=str, default="run", help="Prefix for auto-named run directories")
    parser.add_argument("--seed", type=int, default=2025, help="random seed")
    parser.add_argument("--config", type=str, default=None, help="JSON config file for generator (optional)")
    parser.add_argument("--ontology", type=str, default="ontology/terror.ttl", help="Ontology TTL path")
    parser.add_argument("--shapes", type=str, default="ontology/constraints.shacl.ttl", help="SHACL constraints path")
    parser.add_argument("--no_ontology_strict", action="store_true", help="Do not fail generation on ontology-rule violations")
    parser.add_argument("--ontology_constrained", action="store_true", help="Retry generation with shifted seeds until ontology validation conforms")
    parser.add_argument("--ontology_max_retries", type=int, default=3, help="Max generation attempts in ontology_constrained mode")
    parser.add_argument("--ontology_retry_seed_stride", type=int, default=1, help="Seed increment per attempt in ontology_constrained mode")

    args = parser.parse_args()

    if args.out_root:
        out_dir = build_artifact_dir(args.out_root, args.config, args.seed, prefix=args.run_prefix)
    elif args.out_dir:
        out_dir = args.out_dir
    else:
        parser.error("Either --out_root or --out_dir must be provided.")

    os.makedirs(out_dir, exist_ok=True)

    cfg = load_generator_config(args.config, size=args.size, seed=args.seed)

    ontology_telemetry = {"mode": "single_pass", "attempts": 1, "failed_attempts": 0}
    if args.ontology_constrained:
        manifest, ontology_report, ontology_telemetry = generate_with_ontology_constraints(
            cfg=cfg,
            ontology_path=args.ontology,
            shapes_path=args.shapes,
            max_retries=args.ontology_max_retries,
            retry_seed_stride=args.ontology_retry_seed_stride,
        )
        if ontology_report.get("conforms", False):
            print(f"[*] Ontology-constrained generation passed at attempt {ontology_telemetry.get('successful_attempt')}")
        else:
            msg = "ontology-constrained generation exhausted retries without conformance"
            if not args.no_ontology_strict:
                raise OntologyValidationError(msg)
            print(f"[!] Ontology validation warning (strict disabled): {msg}")
    else:
        manifest = generate_multiplex_with_config(cfg)
        ontology_report = validate_manifest_dict_with_ontology(
            manifest_dict=manifest,
            ontology_path=args.ontology,
            shapes_path=args.shapes,
            strict=False,
        )
        if ontology_report.get("conforms", False):
            print("[*] Ontology validation passed")
        elif not args.no_ontology_strict:
            raise OntologyValidationError("ontology validation failed: " + "; ".join(ontology_report.get("errors", [])))
        else:
            print("[!] Ontology validation warning (strict disabled)")

    ontology_report["generation_telemetry"] = ontology_telemetry
    manifest_model = validate_manifest_dict(manifest)

    ontology_report = {"conforms": True, "constraints_checked": 0, "errors": []}
    try:
        ontology_report = validate_manifest_dict_with_ontology(
            manifest_dict=manifest,
            ontology_path=args.ontology,
            shapes_path=args.shapes,
        )
        print("[*] Ontology validation passed")
    except OntologyValidationError as exc:
        ontology_report = {"conforms": False, "constraints_checked": 4, "errors": [str(exc)]}
        if not args.no_ontology_strict:
            raise
        print(f"[!] Ontology validation warning (strict disabled): {exc}")

    out_path = os.path.join(out_dir, "multiplex.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[*] Saved multiplex manifest to: {out_path}")

    ontology_report_path = os.path.join(out_dir, "ontology_validation_report.json")
    write_ontology_report(ontology_report, ontology_report_path)
    print(f"[*] Saved ontology report to: {ontology_report_path}")

    metadata = collect_run_metadata(
        out_dir=out_dir,
        config_path=args.config,
        seed=args.seed,
        extra={
            "manifest_path": os.path.abspath(out_path),
            "validated": True,
            "generator": manifest_model.meta.generator,
            "ontology_validation": ontology_report,
            "ontology_generation_telemetry": ontology_telemetry,
        },
    )
    meta_path = write_run_metadata(out_dir, metadata)
    print(f"[*] Saved run metadata to: {meta_path}")


if __name__ == "__main__":
    main()
