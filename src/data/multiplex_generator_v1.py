"""
multiplex_generator.py

Multiplex synthetic terror network generator.

Outputs:
  - nodes.csv                : node metadata (role, region, group, skill, radicalization, etc.)
  - layers/hierarchy.csv     : hierarchy layer (directed)
  - layers/finance.csv       : finance layer (directed, weighted, temporal)
  - layers/communication.csv : communication layer (undirected edge + multiple timestamps)
  - layers/operation.csv     : operation layer (sparse undirected)
  - layers/ideology.csv      : ideology similarity layer (undirected, similarity)
  - events.jsonl             : unified temporal log of all layer events
  - labels.csv               : per-node role + importance labels (high_value_target, etc.)
  - multiplex.json           : manifest of all file paths

Example:
  python multiplex_generator.py --size 1000 --out_dir ./out --seed 2025 --scenario active_comm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# 0. Global configuration
# -----------------------------

# Role distribution (approximate fractions)
ROLE_DIST: Dict[str, float] = {
    "leader": 0.02,
    "financier": 0.05,
    "courier": 0.08,
    "operative": 0.25,
    "support": 0.60,
}

REGIONS: List[str] = [
    "MiddleEast",
    "Africa",
    "Europe",
    "Asia",
]

GROUPS: List[str] = [
    "GroupA",
    "GroupB",
    "GroupC",
]

# Region-level connection preference (for geo-clustered networks)
# Higher values increase connection probability
REGION_AFFINITY: Dict[Tuple[str, str], float] = {}
for r1 in REGIONS:
    for r2 in REGIONS:
        if r1 == r2:
            REGION_AFFINITY[(r1, r2)] = 3.0
        else:
            # Make MiddleEast <-> Europe relatively high
            if {r1, r2} == {"MiddleEast", "Europe"}:
                REGION_AFFINITY[(r1, r2)] = 2.0
            # MiddleEast <-> Africa is moderate
            elif {r1, r2} == {"MiddleEast", "Africa"}:
                REGION_AFFINITY[(r1, r2)] = 1.2
            # Others use the default
            else:
                REGION_AFFINITY[(r1, r2)] = 0.8

# Layer-specific role-pair edge propensity (relative weight)
ROLE_PAIR_PROPENSITY = {
    "hierarchy": {
        ("leader", "operative"): 5.0,
        ("leader", "support"): 3.0,
        ("leader", "financier"): 2.5,
        ("operative", "operative"): 1.0,
        ("support", "operative"): 0.8,
    },
    "finance": {
        ("financier", "leader"): 3.0,
        ("financier", "operative"): 2.5,
        ("financier", "support"): 2.0,
        ("support", "financier"): 0.6,
    },
    "communication": {
        ("courier", "courier"): 4.0,
        ("courier", "operative"): 3.5,
        ("operative", "courier"): 3.5,
        ("operative", "operative"): 1.8,
        ("leader", "operative"): 1.2,
        ("operative", "leader"): 1.2,
        ("leader", "leader"): 0.1,
    },
    "operation": {
        ("operative", "operative"): 2.5,
        ("leader", "operative"): 1.5,
        ("operative", "leader"): 1.5,
        ("courier", "operative"): 1.5,
        ("operative", "courier"): 1.5,
    },
}

# Layer weights for cross-layer importance score calculation
LAYER_IMPORTANCE_WEIGHT = {
    "hierarchy": 1.5,
    "finance": 1.3,
    "communication": 1.0,
    "operation": 1.4,
    "ideology": 0.8,
}


# -----------------------------
# 1. Utilities
# -----------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_probs(values: List[float]) -> np.ndarray:
    arr = np.array(values, dtype=float)
    s = arr.sum()
    if s <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / s


def iso_now_minus_days(rng: np.random.Generator, max_days: int = 365 * 5) -> str:
    days = rng.integers(0, max_days + 1)
    seconds = rng.integers(0, 86400)
    ts = datetime.utcnow() - timedelta(days=int(days), seconds=int(seconds))
    return ts.isoformat()


def power_law_degrees(
    n: int,
    exponent: float = 2.5,
    avg_target: float = 3.0,
    rng: np.random.Generator | None = None,
) -> List[int]:
    """Sample from continuous Pareto (power-law) and scale the mean degree to avg_target."""
    if rng is None:
        rng = np.random.default_rng()
    min_deg = 1
    max_deg = max(3, int(avg_target * 10))
    alpha = exponent

    r = rng.random(n)
    xs = (((max_deg ** (1 - alpha) - min_deg ** (1 - alpha)) * r + min_deg ** (1 - alpha))) ** (
        1 / (1 - alpha)
    )
    degs = np.round(xs).astype(int)
    cur_avg = float(degs.mean())
    if cur_avg > 0:
        scale = avg_target / cur_avg
        degs = np.clip(np.round(degs * scale), min_deg, max_deg).astype(int)
    # Add one if the sum is odd
    if degs.sum() % 2 == 1:
        idx = rng.integers(0, n)
        degs[idx] += 1
    return degs.tolist()


# -----------------------------
# 2. Node generation
# -----------------------------


def generate_nodes(
    num_nodes: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a node table containing role / region / group / skill / radicalization.
    """
    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    roles = list(ROLE_DIST.keys())
    role_probs = normalize_probs(list(ROLE_DIST.values()))

    rows = []
    for i in range(num_nodes):
        node_id = f"N{i:06d}"
        role = np_rng.choice(roles, p=role_probs)
        region = np_rng.choice(REGIONS)
        group = np_rng.choice(GROUPS)

        # Slightly boost skill_level for leaders / operatives
        base_skill_mu = 0.55 if role in {"leader", "operative"} else 0.45
        skill = float(
            np.clip(np_rng.normal(loc=base_skill_mu, scale=0.15), 0.0, 1.0)
        )

        # radicalization: leader/operative > courier > support
        if role in {"leader", "operative"}:
            rad = np_rng.beta(3.0, 2.0)  # mean ~0.6
        elif role == "courier":
            rad = np_rng.beta(2.5, 2.5)  # mean ~0.5
        else:
            rad = np_rng.beta(2.0, 4.0)  # mean ~0.33
        radicalization = float(np.clip(rad, 0.0, 1.0))

        past_incidents = int(
            np.clip(
                np_rng.poisson(0.1 if role != "leader" else 0.6),
                0,
                None,
            )
        )

        created_at = iso_now_minus_days(np_rng, max_days=365 * 5)

        rows.append(
            {
                "node_id": node_id,
                "role": role,
                "region": region,
                "group": group,
                "skill_level": round(skill, 3),
                "radicalization": round(radicalization, 3),
                "past_incidents": past_incidents,
                "created_at": created_at,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# 3. Layer-wise edge generation
# -----------------------------


def _region_factor(region_a: str, region_b: str) -> float:
    return REGION_AFFINITY.get((region_a, region_b), 1.0)


def _role_pair_weight(layer: str, role_a: str, role_b: str) -> float:
    base = ROLE_PAIR_PROPENSITY.get(layer, {})
    return base.get((role_a, role_b), 0.3)


def _sample_edge_pairs_weighted(
    candidates: List[Tuple[str, str, float]],
    target_m: int,
    rng: np.random.Generator,
    directed: bool = True,
) -> List[Tuple[str, str]]:
    """
    Sample target_m edges from candidate (u, v, weight) tuples using the provided weights.
    """
    if not candidates:
        return []

    weights = np.array([w for (_, _, w) in candidates], dtype=float)
    probs = normalize_probs(weights)
    k = min(target_m, len(candidates))
    idx = rng.choice(len(candidates), size=k, replace=False, p=probs)

    seen = set()
    edges: List[Tuple[str, str]] = []
    for i in idx:
        u, v, _ = candidates[i]
        key = (u, v) if directed else tuple(sorted((u, v)))
        if key in seen:
            continue
        seen.add(key)
        edges.append((u, v))
    return edges


def generate_hierarchy_layer(
    nodes: pd.DataFrame,
    avg_children_per_leader: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Hierarchy layer:
      - Forest rooted at leaders
      - Includes some informal cross-links
    """
    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    leaders = nodes.loc[nodes["role"] == "leader", "node_id"].tolist()
    if not leaders:
        # If no leaders exist, treat top 2% by skill as leaders
        top_k = max(1, int(0.02 * len(nodes)))
        leaders = (
            nodes.sort_values("skill_level", ascending=False)
            .head(top_k)["node_id"]
            .tolist()
        )

    non_leaders = nodes.loc[~nodes["node_id"].isin(leaders), "node_id"].tolist()
    edges = []

    # Connect each non-leader to a senior leader or operative in the same region
    for nid in non_leaders:
        node_row = nodes[nodes["node_id"] == nid].iloc[0]
        region = node_row["region"]

        # Candidate superiors (leaders/operatives)
        sup_candidates = nodes[
            (nodes["role"].isin(["leader", "operative"]))
            & (nodes["region"] == region)
        ]["node_id"].tolist()
        if not sup_candidates:
            sup_candidates = leaders

        superior = py_rng.choice(sup_candidates)
        edges.append(
            {
                "source": superior,
                "target": nid,
                "relation": "superior",
            }
        )

    # Add a few informal links
    extra_links = max(1, int(0.02 * len(non_leaders)))
    for _ in range(extra_links):
        a, b = py_rng.sample(non_leaders, 2)
        edges.append(
            {
                "source": a,
                "target": b,
                "relation": "informal",
            }
        )

    return pd.DataFrame(edges)


def generate_finance_layer(
    nodes: pd.DataFrame,
    base_tx_per_financier: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Finance layer:
      - directed, weighted
      - financier -> others
      - transaction amount: lognormal (heavy tail)
      - includes timestamp
    """
    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    financiers = nodes.loc[nodes["role"] == "financier", "node_id"].tolist()
    if not financiers:
        # Treat top 3% by skill as financiers if none exist
        top_k = max(1, int(0.03 * len(nodes)))
        financiers = (
            nodes.sort_values("skill_level", ascending=False)
            .head(top_k)["node_id"]
            .tolist()
        )

    all_ids = nodes["node_id"].tolist()
    rows = []

    for fid in financiers:
        # Transaction count per financier (Poisson)
        tx_count = max(1, int(np_rng.poisson(lam=base_tx_per_financier)))
        src_row = nodes[nodes["node_id"] == fid].iloc[0]
        src_region = src_row["region"]

        for _ in range(tx_count):
            # Choose recipients using region affinity + role weighting
            weights = []
            for _, r in nodes.iterrows():
                if r["node_id"] == fid:
                    weights.append(0.0)
                    continue
                w = _region_factor(src_region, r["region"])
                w *= _role_pair_weight("finance", "financier", r["role"])
                weights.append(w)
            probs = normalize_probs(weights)
            dst_idx = np_rng.choice(len(nodes), p=probs)
            dst_id = nodes.iloc[dst_idx]["node_id"]

            amount = float(
                np.clip(
                    np_rng.lognormal(mean=8.0, sigma=1.5),  # roughly hundreds to tens of thousands
                    50.0,
                    1e7,
                )
            )
            ts = iso_now_minus_days(np_rng, max_days=365 * 3)

            rows.append(
                {
                    "source": fid,
                    "target": dst_id,
                    "amount": round(amount, 2),
                    "timestamp": ts,
                    "channel": py_rng.choice(
                        ["hawala", "cash", "crypto", "bank"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def generate_communication_layer(
    nodes: pd.DataFrame,
    avg_degree: float,
    days_span: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Communication layer:
      - undirected logical edges
      - multiple events (timestamps) per edge
      - degree follows a power-law
    """
    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    n = len(nodes)
    node_ids = nodes["node_id"].tolist()
    roles = nodes["role"].tolist()
    regions = nodes["region"].tolist()

    degs = power_law_degrees(
        n=n,
        exponent=2.4,
        avg_target=avg_degree,
        rng=np_rng,
    )

    # Build stub list then randomly match (configuration model style)
    stubs = []
    for nid, d in zip(node_ids, degs):
        stubs.extend([nid] * max(0, int(d)))
    py_rng.shuffle(stubs)

    pairs = []
    seen_pairs = set()
    while len(stubs) >= 2:
        a = stubs.pop()
        b = stubs.pop()
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        pairs.append(key)

    rows = []
    base_date = datetime.utcnow() - timedelta(days=days_span)

    for (a, b) in pairs:
        # Weight contact frequency by role/region
        row_a = nodes[nodes["node_id"] == a].iloc[0]
        row_b = nodes[nodes["node_id"] == b].iloc[0]

        role_factor = _role_pair_weight(
            "communication",
            row_a["role"],
            row_b["role"],
        )
        region_factor = _region_factor(row_a["region"], row_b["region"])

        # Average number of events (communication frequency)
        lam = 0.5 + role_factor + 0.3 * region_factor
        num_events = max(1, int(np_rng.poisson(lam=lam)))

        timestamps = []
        for _ in range(num_events):
            # Combine exponential + uniform for burstiness
            offset_days = float(
                np_rng.exponential(scale=days_span / 6.0)
            )
            offset_days = min(offset_days, days_span)
            seconds = int(np_rng.integers(0, 86400))
            ts = base_date + timedelta(days=offset_days, seconds=seconds)
            timestamps.append(ts.isoformat())

        rows.append(
            {
                "source": a,
                "target": b,
                "num_events": num_events,
                "timestamps": json.dumps(sorted(timestamps)),
                "channel": py_rng.choice(
                    ["phone", "email", "in_person", "encrypted_chat"]
                ),
            }
        )

    return pd.DataFrame(rows)


def generate_operation_layer(
    nodes: pd.DataFrame,
    ops_edge_frac: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Operation layer:
      - very sparse
      - represents joint operation counts (joint_ops)
    """
    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    node_ids = nodes["node_id"].tolist()
    n = len(node_ids)
    target_edges = max(1, int(n * ops_edge_frac))

    rows = []
    seen = set()
    for _ in range(target_edges * 3):  # over-sample to allow filtering
        if len(rows) >= target_edges:
            break
        a, b = py_rng.sample(node_ids, 2)
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)

        row_a = nodes[nodes["node_id"] == a].iloc[0]
        row_b = nodes[nodes["node_id"] == b].iloc[0]

        # Increase probability of joint ops if same region / same group
        w = 1.0
        if row_a["region"] == row_b["region"]:
            w *= 1.5
        if row_a["group"] == row_b["group"]:
            w *= 1.5
        # Slightly higher probability for courier + operative pairs
        if {"courier", "operative"} <= {row_a["role"], row_b["role"]}:
            w *= 1.3

        # Number of joint_ops
        mean_ops = 1.0 * w
        joint_ops = max(1, int(np_rng.poisson(lam=mean_ops)))

        rows.append(
            {
                "source": a,
                "target": b,
                "joint_ops": joint_ops,
            }
        )

    return pd.DataFrame(rows)


def generate_ideology_layer(
    nodes: pd.DataFrame,
    sim_threshold: float,
    emb_dim: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Ideology layer:
      - generate node-level ideology embeddings (shifted by role)
      - create edge when cos similarity > threshold
      - since this is O(n^2), increase threshold or sample when size is large
    """
    np_rng = np.random.default_rng(seed)

    ids = nodes["node_id"].tolist()
    roles = nodes["role"].tolist()

    # Generate embeddings
    emb: Dict[str, np.ndarray] = {}
    for nid, role in zip(ids, roles):
        base = np_rng.normal(loc=0.0, scale=1.0, size=emb_dim)
        if role in {"leader", "operative"}:
            base += 0.7
        elif role == "courier":
            base += 0.3
        v = base / (np.linalg.norm(base) + 1e-9)
        emb[nid] = v

    rows = []
    n = len(ids)
    for i in range(n):
        a = ids[i]
        va = emb[a]
        for j in range(i + 1, n):
            b = ids[j]
            vb = emb[b]
            cos = float(
                np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
            )
            if cos >= sim_threshold:
                rows.append(
                    {
                        "source": a,
                        "target": b,
                        "similarity": round(cos, 3),
                    }
                )

    return pd.DataFrame(rows)


# -----------------------------
# 4. Event flattening + label generation
# -----------------------------


def flatten_events(
    finance: pd.DataFrame,
    comm: pd.DataFrame,
    ops: pd.DataFrame,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate event-level logs for finance / communication / operation layers.
    """
    np_rng = np.random.default_rng(seed)
    events = []
    eid = 0

    # communication: expand edge timestamps into individual events
    for _, row in comm.iterrows():
        ts_list = json.loads(row["timestamps"])
        for ts in ts_list:
            events.append(
                {
                    "event_id": f"E{eid:09d}",
                    "timestamp": ts,
                    "event_type": "comm",
                    "layer": "communication",
                    "nodes": [row["source"], row["target"]],
                    "meta": {
                        "channel": row["channel"],
                    },
                }
            )
            eid += 1

    # finance: treat each transaction as an individual event
    for _, row in finance.iterrows():
        events.append(
            {
                "event_id": f"E{eid:09d}",
                "timestamp": row["timestamp"],
                "event_type": "txn",
                "layer": "finance",
                "nodes": [row["source"], row["target"]],
                "meta": {
                    "amount": row["amount"],
                    "channel": row["channel"],
                },
            }
        )
        eid += 1

    # operation: create events equal to joint_ops
    for _, row in ops.iterrows():
        base_ts = datetime.utcnow() - timedelta(days=30)
        for k in range(int(row["joint_ops"])):
            ts = base_ts + timedelta(days=int(np_rng.integers(0, 30)))
            events.append(
                {
                    "event_id": f"E{eid:09d}",
                    "timestamp": ts.isoformat(),
                    "event_type": "op",
                    "layer": "operation",
                    "nodes": [row["source"], row["target"]],
                    "meta": {
                        "seq": k + 1,
                        "total_joint_ops": int(row["joint_ops"]),
                    },
                }
            )
            eid += 1

    return events

def compute_importance_and_labels(
    nodes: pd.DataFrame,
    hierarchy: pd.DataFrame,
    finance: pd.DataFrame,
    comm: pd.DataFrame,
    ops: pd.DataFrame,
    ideology: pd.DataFrame,
    top_k_frac: float = 0.05,
) -> pd.DataFrame:
    """
    Compute cross-layer importance scores from layer degrees + node attributes,
    then label the top_k_frac nodes as high_value_target.

    Potential improvements:
      - z-score layer degrees to avoid domination by specific layers (e.g., ideology)
      - jointly consider skill_level / radicalization / past_incidents
      - add role-based priors (leader / financier)
      - min-max scale final scores to the 10~100 importance_score range
    """

    eps = 1e-8
    node_ids = nodes["node_id"].tolist()
    n = len(node_ids)
    id2idx = {nid: i for i, nid in enumerate(node_ids)}

    # ---------- 1) Degree vectors per layer (aligned to node order) ----------
    def deg_vector_from_edges(df: pd.DataFrame, undirected: bool) -> np.ndarray:
        vec = np.zeros(n, dtype=float)
        if df is None or df.empty:
            return vec
        for _, r in df.iterrows():
            u = r["source"]
            v = r["target"]
            if u in id2idx:
                vec[id2idx[u]] += 1.0
            if v in id2idx:
                vec[id2idx[v]] += 1.0
            if not undirected:
                # Count both inbound and outbound as involvement even if directed
                # v already incremented above, so no additional action
                pass
        return vec

    deg_h = deg_vector_from_edges(hierarchy, undirected=False)
    deg_f = deg_vector_from_edges(finance, undirected=False)
    deg_c = deg_vector_from_edges(comm, undirected=True)
    deg_o = deg_vector_from_edges(ops, undirected=True)
    deg_i = deg_vector_from_edges(ideology, undirected=True)

    # ---------- 2) Z-score per layer ----------
    def zscore(x: np.ndarray) -> np.ndarray:
        mu = x.mean()
        sd = x.std()
        if sd < eps:
            return np.zeros_like(x)
        z = (x - mu) / (sd + eps)
        # Clip extreme outliers for stability
        return np.clip(z, -3.0, 3.0)

    deg_h_z = zscore(deg_h)
    deg_f_z = zscore(deg_f)
    deg_c_z = zscore(deg_c)
    deg_o_z = zscore(deg_o)
    deg_i_z = zscore(deg_i)

    # ---------- 3) Node attributes (skill, radicalization, incidents) ----------
    skill = nodes["skill_level"].to_numpy(dtype=float)          # already near 0~1
    rad = nodes["radicalization"].to_numpy(dtype=float)         # 0~1
    inc = nodes["past_incidents"].to_numpy(dtype=float)         # non-negative integer

    # z-score incidents after log1p
    inc_log = np.log1p(inc)
    inc_z = zscore(inc_log)

    # Also z-score skill and radicalization
    skill_z = zscore(skill)
    rad_z = zscore(rad)

    # ---------- 4) Combine layers/features: weighted sum ----------
    w_layer = LAYER_IMPORTANCE_WEIGHT  # reuse global weights

    # Layer contribution
    layer_score = (
        w_layer["hierarchy"] * deg_h_z
        + w_layer["finance"] * deg_f_z
        + w_layer["communication"] * deg_c_z
        + w_layer["operation"] * deg_o_z
        + w_layer["ideology"] * deg_i_z
    )

    # Internal feature contribution (weights tunable)
    w_skill = 0.8
    w_rad = 1.0
    w_inc = 0.6

    intrinsic_score = w_skill * skill_z + w_rad * rad_z + w_inc * inc_z

    # ---------- 5) Role-based additive prior ----------
    role = nodes["role"].tolist()
    role_bonus = np.zeros(n, dtype=float)
    for i, r in enumerate(role):
        if r == "leader":
            role_bonus[i] += 1.0
        elif r == "financier":
            role_bonus[i] += 0.5
        # courier / operative / support remain 0

    # ---------- 6) Final raw score ----------
    raw_score = layer_score + intrinsic_score + role_bonus

    # ---------- 7) Min-max scaling → range 10 ~ 100 ----------
    s_min = raw_score.min()
    s_max = raw_score.max()
    if s_max - s_min < eps:
        norm_score = np.zeros_like(raw_score)
    else:
        norm_score = (raw_score - s_min) / (s_max - s_min + eps)

    # Final importance_score: 10 ~ 100 (roughly similar to previous scale)
    importance_score = 10.0 + 90.0 * norm_score

    # ---------- 8) Top_k_frac → high_value_target ----------
    k = max(1, int(len(nodes) * top_k_frac))
    top_indices = np.argsort(-importance_score)[:k]
    is_top = np.zeros(n, dtype=int)
    is_top[top_indices] = 1

    labels = []
    for i, row in nodes.reset_index(drop=True).iterrows():
        labels.append(
            {
                "node_id": row["node_id"],
                "role": row["role"],
                "region": row["region"],
                "group": row["group"],
                "importance_score": round(float(importance_score[i]), 3),
                "high_value_target": int(is_top[i]),
            }
        )

    return pd.DataFrame(labels)


# -----------------------------
# 5. Full pipeline
# -----------------------------


def generate_multiplex_dataset(
    num_nodes: int,
    out_dir: str,
    seed: int,
    scenario: str,
) -> None:
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "layers"))

    print(f"[*] Generating nodes (n={num_nodes})...")
    nodes = generate_nodes(num_nodes=num_nodes, seed=seed)
    nodes.to_csv(os.path.join(out_dir, "nodes.csv"), index=False)

    # Scenario-specific parameters
    if scenario == "balanced":
        avg_comm_degree = 3.0
        base_tx_per_financier = 20.0
        ops_edge_frac = 0.01
        ideology_sim_threshold = 0.75
    elif scenario == "active_comm":
        avg_comm_degree = 4.0
        base_tx_per_financier = 18.0
        ops_edge_frac = 0.03
        ideology_sim_threshold = 0.78
    elif scenario == "high_ops":
        avg_comm_degree = 3.2
        base_tx_per_financier = 15.0
        ops_edge_frac = 0.05
        ideology_sim_threshold = 0.75
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    print("[*] Generating hierarchy layer...")
    hierarchy = generate_hierarchy_layer(nodes, seed=seed)
    hierarchy_path = os.path.join(out_dir, "layers", "hierarchy.csv")
    hierarchy.to_csv(hierarchy_path, index=False)

    print("[*] Generating finance layer...")
    finance = generate_finance_layer(
        nodes,
        base_tx_per_financier=base_tx_per_financier,
        seed=seed,
    )
    finance_path = os.path.join(out_dir, "layers", "finance.csv")
    finance.to_csv(finance_path, index=False)

    print("[*] Generating communication layer...")
    communication = generate_communication_layer(
        nodes,
        avg_degree=avg_comm_degree,
        days_span=365,
        seed=seed,
    )
    comm_path = os.path.join(out_dir, "layers", "communication.csv")
    communication.to_csv(comm_path, index=False)

    print("[*] Generating operation layer...")
    operation = generate_operation_layer(
        nodes,
        ops_edge_frac=ops_edge_frac,
        seed=seed,
    )
    op_path = os.path.join(out_dir, "layers", "operation.csv")
    operation.to_csv(op_path, index=False)

    print("[*] Generating ideology layer...")
    ideology = generate_ideology_layer(
        nodes,
        sim_threshold=ideology_sim_threshold,
        emb_dim=6,
        seed=seed,
    )
    ideology_path = os.path.join(out_dir, "layers", "ideology.csv")
    ideology.to_csv(ideology_path, index=False)

    print("[*] Flattening events...")
    events = flatten_events(finance, communication, operation, seed=seed)
    events_path = os.path.join(out_dir, "events.jsonl")
    with open(events_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    print("[*] Computing labels (importance + high_value_target)...")
    labels = compute_importance_and_labels(
        nodes,
        hierarchy,
        finance,
        communication,
        operation,
        ideology,
        top_k_frac=0.05,
    )
    labels_path = os.path.join(out_dir, "labels.csv")
    labels.to_csv(labels_path, index=False)

    print("[*] Writing manifest (multiplex.json)...")
    manifest = {
        "nodes": os.path.abspath(os.path.join(out_dir, "nodes.csv")),
        "layers": {
            "hierarchy": os.path.abspath(hierarchy_path),
            "finance": os.path.abspath(finance_path),
            "communication": os.path.abspath(comm_path),
            "operation": os.path.abspath(op_path),
            "ideology": os.path.abspath(ideology_path),
        },
        "events": os.path.abspath(events_path),
        "labels": os.path.abspath(labels_path),
        "meta": {
            "num_nodes": int(num_nodes),
            "seed": int(seed),
            "scenario": scenario,
            "generated_at": datetime.utcnow().isoformat(),
        },
    }
    with open(os.path.join(out_dir, "multiplex.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[*] Done. Dataset saved to:", os.path.abspath(out_dir))


# -----------------------------
# 6. CLI entrypoint
# -----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Multiplex Terror Network Synthetic Data Generator"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2000,
        help="number of nodes",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./out",
        help="output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="balanced",
        choices=["balanced", "active_comm", "high_ops"],
        help="generation scenario",
    )

    args = parser.parse_args()
    generate_multiplex_dataset(
        num_nodes=args.size,
        out_dir=args.out_dir,
        seed=args.seed,
        scenario=args.scenario,
    )


if __name__ == "__main__":
    main()
