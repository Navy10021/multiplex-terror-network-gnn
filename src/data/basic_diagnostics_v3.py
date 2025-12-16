"""
basic_diagnostics_v3.py

Basic diagnostics + extended checks for multiplex synthetic terror networks (v1/v2/v3 compatible).

Adds v3-focused diagnostics:
  - Edge noise diagnostics via 'is_false' flags (per layer)
  - Layer overlap (Jaccard) across layers
  - Edge attribute summaries (txn/comm/op aggregated stats)
  - Operation cell homophily/purity diagnostics (if nodes contain op_cell_id)
  - Event burstiness metrics (Fano factor, burstiness coefficient)

Usage:
  python basic_diagnostics_v3.py --manifest path/to/multiplex.json --out_dir ./analysis_output_v3
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple, Optional, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.validation.schema import validate_manifest_file


# -------------------------------------
# Utilities
# -------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series([default] * (0 if df is None else len(df)))
    return df[col].fillna(default)


def load_multiplex(manifest_path: str):
    with open(manifest_path, "r", encoding="utf-8") as f:
        mani = json.load(f)

    # --------------------------------------------------
    # 1) nodes / labels
    # --------------------------------------------------
    nodes_raw = mani.get("nodes")

    if isinstance(nodes_raw, str):
        # v1 style: nodes.csv path
        nodes = pd.read_csv(nodes_raw)
        labels_path = mani.get("labels")
        if labels_path is None:
            raise ValueError("v1 format requires 'labels' (CSV path).")
        labels = pd.read_csv(labels_path)

    elif isinstance(nodes_raw, list):
        df_nodes = pd.DataFrame(nodes_raw)

        if "node_id" in df_nodes.columns:
            pass
        elif "id" in df_nodes.columns:
            df_nodes = df_nodes.rename(columns={"id": "node_id"})
        else:
            raise ValueError("'nodes' must include an 'id' or 'node_id' column.")

        for col in ["node_id", "role", "region", "group"]:
            if col not in df_nodes.columns:
                raise ValueError(f"Required column '{col}' missing in nodes.")

        # optional columns (v2/v3)
        for col, default in [
            ("ideology", 0.5),
            ("skill_level", 0.0),
            ("radicalization", 0.0),
            ("past_incidents", 0.0),
            ("activity_rate", 1.0),
            ("observability", 1.0),
            ("importance_score", 0.0),
            ("high_value_target", 0),
            ("op_cell_id", -1),
        ]:
            if col not in df_nodes.columns:
                df_nodes[col] = default

        nodes = df_nodes[["node_id", "role", "region", "group"]].copy()
        labels = df_nodes[
            [
                "node_id",
                "role",
                "region",
                "group",
                "ideology",
                "skill_level",
                "radicalization",
                "past_incidents",
                "activity_rate",
                "observability",
                "importance_score",
                "high_value_target",
                "op_cell_id",
            ]
        ].copy()

    else:
        raise ValueError(f"Unrecognized type for 'nodes': {type(nodes_raw)}")

    # --------------------------------------------------
    # 2) layers
    # --------------------------------------------------
    layers: Dict[str, pd.DataFrame] = {}
    raw_layers = mani.get("layers", {})

    for layer_name, layer_obj in raw_layers.items():
        if isinstance(layer_obj, str):
            layers[layer_name] = pd.read_csv(layer_obj)
        elif isinstance(layer_obj, dict) and "edges" in layer_obj:
            df_layer = pd.DataFrame(layer_obj["edges"])
            if "source" not in df_layer.columns or "target" not in df_layer.columns:
                raise ValueError(f"Layer '{layer_name}' requires 'source' and 'target'.")
            layers[layer_name] = df_layer
        else:
            raise ValueError(f"Unrecognized layer format for '{layer_name}': {type(layer_obj)}")

    # --------------------------------------------------
    # 3) events (optional)
    # --------------------------------------------------
    events_raw = mani.get("events", None)
    if isinstance(events_raw, str):
        df_events = pd.read_csv(events_raw)
    elif isinstance(events_raw, list):
        df_events = pd.DataFrame(events_raw)
    elif events_raw is None:
        df_events = pd.DataFrame()
    else:
        raise ValueError(f"Unrecognized type for 'events': {type(events_raw)}")

    return mani, nodes, labels, layers, df_events


def _undirected_edge_keys(df: pd.DataFrame) -> set:
    """Return undirected unique edge keys as (min(u,v), max(u,v))."""
    if df is None or df.empty:
        return set()
    u = df["source"].astype(int).to_numpy()
    v = df["target"].astype(int).to_numpy()
    a = np.minimum(u, v)
    b = np.maximum(u, v)
    return set(zip(a.tolist(), b.tolist()))


def _directed_edge_keys(df: pd.DataFrame) -> set:
    if df is None or df.empty:
        return set()
    u = df["source"].astype(int).to_numpy()
    v = df["target"].astype(int).to_numpy()
    return set(zip(u.tolist(), v.tolist()))


def degree_dict_from_edges(df: pd.DataFrame, directed: bool) -> dict[int, int]:
    if df is None or df.empty:
        return {}
    G = nx.from_pandas_edgelist(
        df, "source", "target", create_using=nx.DiGraph() if directed else nx.Graph()
    )
    return dict(G.degree())


# -------------------------------------
# 0. Manifest meta (v3 config)
# -------------------------------------

def print_meta(mani: Dict[str, Any]) -> None:
    meta = mani.get("meta", {})
    print("=" * 80)
    print("[0] Manifest meta")
    print("=" * 80)
    if not meta:
        print("(no meta found)")
        return
    gen = meta.get("generator", "unknown")
    seed = meta.get("seed", None)
    n = meta.get("num_nodes", None)
    print(f"generator: {gen}")
    if n is not None:
        print(f"num_nodes: {n}")
    if seed is not None:
        print(f"seed: {seed}")

    cfg = meta.get("config", None)
    if isinstance(cfg, dict):
        # print a compact subset
        keys = [
            "finance_structure_strength", "comm_structure_strength", "comm_randomness",
            "op_cell_homophily_strength", "op_inter_cell_bridge_rate",
            "missing_edge_rate_finance", "false_edge_rate_finance",
            "missing_edge_rate_communication", "false_edge_rate_communication",
            "event_burstiness", "missing_event_rate_txn", "missing_event_rate_comm", "missing_event_rate_op",
            "hvt_ratio",
        ]
        present = {k: cfg.get(k) for k in keys if k in cfg}
        if present:
            print("config (selected):")
            for k, v in present.items():
                print(f"  - {k}: {v}")


# -------------------------------------
# 1-1. Basic structure checks
# -------------------------------------

def basic_stats(nodes: pd.DataFrame, layers: dict[str, pd.DataFrame], out_dir: str):
    print("=" * 80)
    print("[1-1] Basic structure")
    print("=" * 80)

    print(f"#nodes: {len(nodes)}")
    for lname, df in layers.items():
        print(f"[{lname}] #edges = {len(df)}")

    print("\nRole distribution (ratio):")
    print(nodes["role"].value_counts(normalize=True).round(3))

    print("\nRegion distribution (ratio):")
    print(nodes["region"].value_counts(normalize=True).round(3))

    print("\nGroup distribution (ratio):")
    print(nodes["group"].value_counts(normalize=True).round(3))

    ensure_dir(out_dir)

    for col, fname, title in [
        ("role", "role_counts.png", "Role counts"),
        ("region", "region_counts.png", "Region counts"),
        ("group", "group_counts.png", "Group counts"),
    ]:
        plt.figure()
        nodes[col].value_counts().plot(kind="bar")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()


# -------------------------------------
# 1-2. Degree distributions per layer
# -------------------------------------

def plot_degree_hist(layer_df: pd.DataFrame, directed: bool, title: str, out_path: str):
    if layer_df is None or layer_df.empty:
        print(f"[WARN] {title} : empty layer, skip.")
        return

    if directed:
        G = nx.from_pandas_edgelist(layer_df, "source", "target", create_using=nx.DiGraph())
        out_deg = [d for _, d in G.out_degree()]
        in_deg = [d for _, d in G.in_degree()]

        for degs, suffix, xlabel in [(out_deg, "_out", "out-degree"), (in_deg, "_in", "in-degree")]:
            plt.figure()
            plt.hist(degs, bins=30)
            plt.yscale("log")
            plt.xlabel(xlabel)
            plt.ylabel("count (log)")
            plt.title(f"{title} - {xlabel}")
            plt.tight_layout()
            plt.savefig(out_path.replace(".png", f"{suffix}.png"))
            plt.close()
    else:
        G = nx.from_pandas_edgelist(layer_df, "source", "target", create_using=nx.Graph())
        deg = [d for _, d in G.degree()]
        plt.figure()
        plt.hist(deg, bins=30)
        plt.yscale("log")
        plt.xlabel("degree")
        plt.ylabel("count (log)")
        plt.title(f"{title} - degree")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


def degree_distributions(layers: dict[str, pd.DataFrame], out_dir: str):
    print("=" * 80)
    print("[1-2] Layer-wise degree distributions")
    print("=" * 80)

    ensure_dir(out_dir)

    specs = [
        ("hierarchy", True, "Hierarchy layer", "deg_hierarchy.png"),
        ("finance", True, "Finance layer", "deg_finance.png"),
        ("communication", False, "Communication layer", "deg_communication.png"),
        ("operation", False, "Operation layer", "deg_operation.png"),
        ("ideology", False, "Ideology layer", "deg_ideology.png"),
    ]

    for name, directed, title, fn in specs:
        if name in layers:
            plot_degree_hist(layers[name], directed, title, os.path.join(out_dir, fn))


# -------------------------------------
# 1-3. Role-wise degree stats
# -------------------------------------

def rolewise_degree_stats(nodes: pd.DataFrame, layers: dict[str, pd.DataFrame], out_dir: str):
    print("=" * 80)
    print("[1-3] Role-wise degree stats")
    print("=" * 80)

    ensure_dir(out_dir)
    df = nodes.copy()

    if "hierarchy" in layers and not layers["hierarchy"].empty:
        G_h = nx.from_pandas_edgelist(layers["hierarchy"], "source", "target", create_using=nx.DiGraph())
        df["hier_out_deg"] = df["node_id"].map(dict(G_h.out_degree())).fillna(0)
        print("\nHierarchy out-degree by role:")
        print(df.groupby("role")["hier_out_deg"].describe().round(3))

    if "finance" in layers and not layers["finance"].empty:
        G_f = nx.from_pandas_edgelist(layers["finance"], "source", "target", create_using=nx.DiGraph())
        df["fin_out_deg"] = df["node_id"].map(dict(G_f.out_degree())).fillna(0)
        print("\nFinance out-degree by role:")
        print(df.groupby("role")["fin_out_deg"].describe().round(3))

    if "communication" in layers and not layers["communication"].empty:
        G_c = nx.from_pandas_edgelist(layers["communication"], "source", "target", create_using=nx.Graph())
        df["comm_deg"] = df["node_id"].map(dict(G_c.degree())).fillna(0)
        print("\nCommunication degree by role:")
        print(df.groupby("role")["comm_deg"].describe().round(3))

    # boxplots (when columns exist)
    cols = [c for c in ["hier_out_deg", "fin_out_deg", "comm_deg"] if c in df.columns]
    if cols:
        plt.figure(figsize=(4 * len(cols), 4))
        for i, col in enumerate(cols):
            plt.subplot(1, len(cols), i + 1)
            df.boxplot(column=col, by="role", rot=45)
            plt.title(col)
            plt.suptitle("")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rolewise_degrees_boxplot.png"))
        plt.close()


# -------------------------------------
# 1-4. Cross-layer degree correlations + overlap
# -------------------------------------

def cross_layer_correlations(nodes: pd.DataFrame, layers: dict[str, pd.DataFrame], out_dir: str):
    print("=" * 80)
    print("[1-4] Cross-layer degree correlations")
    print("=" * 80)

    ensure_dir(out_dir)

    deg_fin = degree_dict_from_edges(layers.get("finance", pd.DataFrame()), directed=True)
    deg_comm = degree_dict_from_edges(layers.get("communication", pd.DataFrame()), directed=False)
    deg_ops = degree_dict_from_edges(layers.get("operation", pd.DataFrame()), directed=False)
    deg_hier = degree_dict_from_edges(layers.get("hierarchy", pd.DataFrame()), directed=True)
    deg_ideo = degree_dict_from_edges(layers.get("ideology", pd.DataFrame()), directed=False)

    df_deg = nodes[["node_id", "role"]].copy()
    df_deg["deg_fin"] = df_deg["node_id"].map(deg_fin).fillna(0)
    df_deg["deg_comm"] = df_deg["node_id"].map(deg_comm).fillna(0)
    df_deg["deg_ops"] = df_deg["node_id"].map(deg_ops).fillna(0)
    df_deg["deg_hier"] = df_deg["node_id"].map(deg_hier).fillna(0)
    df_deg["deg_ideo"] = df_deg["node_id"].map(deg_ideo).fillna(0)

    print("\nCorrelation matrix (deg_fin, deg_comm, deg_ops, deg_hier, deg_ideo):")
    print(df_deg[["deg_fin", "deg_comm", "deg_ops", "deg_hier", "deg_ideo"]].corr().round(3))

    # scatter examples
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(df_deg["deg_fin"], df_deg["deg_comm"], s=5, alpha=0.5)
    plt.xlabel("deg_fin")
    plt.ylabel("deg_comm")
    plt.title("Finance vs Communication degree")
    plt.subplot(1, 2, 2)
    plt.scatter(df_deg["deg_hier"], df_deg["deg_ops"], s=5, alpha=0.5)
    plt.xlabel("deg_hier")
    plt.ylabel("deg_ops")
    plt.title("Hierarchy vs Operation degree")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "degree_scatter_examples.png"))
    plt.close()


def layer_overlap_diagnostics(layers: dict[str, pd.DataFrame], out_dir: str):
    print("=" * 80)
    print("[1-4b] Layer overlap (Jaccard)")
    print("=" * 80)
    ensure_dir(out_dir)

    names = ["hierarchy", "finance", "communication", "operation", "ideology"]
    keys = {}
    for n in names:
        df = layers.get(n, pd.DataFrame())
        # use undirected keys for overlap comparisons (more interpretable)
        keys[n] = _undirected_edge_keys(df)

    # compute pairwise jaccard for a few important pairs
    pairs = [
        ("finance", "communication"),
        ("communication", "operation"),
        ("finance", "operation"),
        ("ideology", "communication"),
    ]

    rows = []
    for a, b in pairs:
        A, B = keys.get(a, set()), keys.get(b, set())
        inter = len(A & B)
        union = len(A | B) if (A or B) else 1
        j = inter / union
        rows.append({"layer_a": a, "layer_b": b, "jaccard": j, "intersection": inter, "union": union})

    dfj = pd.DataFrame(rows)
    print(dfj.round(4).to_string(index=False))

    dfj.to_csv(os.path.join(out_dir, "layer_overlap_jaccard.csv"), index=False)


# -------------------------------------
# 1-5. Label checks (importance_score / HVT)
# -------------------------------------

def label_diagnostics(labels: pd.DataFrame, out_dir: str):
    print("=" * 80)
    print("[1-5] Label diagnostics (importance_score, high_value_target)")
    print("=" * 80)

    ensure_dir(out_dir)

    if "importance_score" in labels.columns:
        print("\nimportance_score stats:")
        print(labels["importance_score"].describe().round(3))

        plt.figure()
        plt.hist(labels["importance_score"], bins=30)
        plt.yscale("log")
        plt.xlabel("importance_score")
        plt.ylabel("count (log)")
        plt.title("Importance score distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "importance_score_hist.png"))
        plt.close()

    if "high_value_target" in labels.columns:
        print("\nHVT ratio:")
        print(labels["high_value_target"].value_counts(normalize=True).round(3))

        if "role" in labels.columns:
            print("\nHVT ratio by role:")
            print(labels.groupby("role")["high_value_target"].mean().round(3))


# -------------------------------------
# 1-6. Edge noise + attributes
# -------------------------------------

def edge_noise_diagnostics(layers: dict[str, pd.DataFrame], out_dir: str):
    print("=" * 80)
    print("[1-6] Edge noise diagnostics (is_false)")
    print("=" * 80)
    ensure_dir(out_dir)

    rows = []
    for lname, df in layers.items():
        if df is None or df.empty:
            rows.append({"layer": lname, "edges": 0, "false_edges": 0, "false_ratio": 0.0})
            continue
        if "is_false" in df.columns:
            false_cnt = int((df["is_false"].fillna(0).astype(int) == 1).sum())
        else:
            false_cnt = 0
        rows.append({"layer": lname, "edges": int(len(df)), "false_edges": false_cnt, "false_ratio": (false_cnt / max(1, len(df)))})

    out = pd.DataFrame(rows).sort_values("layer")
    print(out.round(4).to_string(index=False))
    out.to_csv(os.path.join(out_dir, "edge_noise_summary.csv"), index=False)


def edge_attribute_diagnostics(layers: dict[str, pd.DataFrame], out_dir: str):
    print("=" * 80)
    print("[1-7] Edge attribute diagnostics (txn/comm/op stats)")
    print("=" * 80)
    ensure_dir(out_dir)

    # Finance
    fin = layers.get("finance", pd.DataFrame())
    if fin is not None and not fin.empty:
        cols = [c for c in ["txn_count", "txn_amount_sum", "txn_amount_mean", "txn_amount_max"] if c in fin.columns]
        if cols:
            print("\n[finance] attribute summary:")
            print(fin[cols].describe().round(3))
            for c in cols:
                plt.figure()
                x = fin[c].fillna(0.0).astype(float).to_numpy()
                if c in ("txn_amount_sum", "txn_amount_mean", "txn_amount_max"):
                    x = np.log1p(x)
                    plt.xlabel(f"log1p({c})")
                else:
                    plt.xlabel(c)
                plt.hist(x, bins=30)
                plt.yscale("log")
                plt.title(f"Finance {c} distribution")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"finance_{c}.png"))
                plt.close()

    # Communication
    comm = layers.get("communication", pd.DataFrame())
    if comm is not None and not comm.empty:
        cols = [c for c in ["comm_count", "comm_duration_sum", "comm_duration_mean"] if c in comm.columns]
        if cols:
            print("\n[communication] attribute summary:")
            print(comm[cols].describe().round(3))
            for c in cols:
                plt.figure()
                x = comm[c].fillna(0.0).astype(float).to_numpy()
                if c in ("comm_count", "comm_duration_sum"):
                    x = np.log1p(x)
                    plt.xlabel(f"log1p({c})")
                else:
                    plt.xlabel(c)
                plt.hist(x, bins=30)
                plt.yscale("log")
                plt.title(f"Communication {c} distribution")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"communication_{c}.png"))
                plt.close()

    # Operation
    op = layers.get("operation", pd.DataFrame())
    if op is not None and not op.empty:
        if "op_count" in op.columns:
            print("\n[operation] attribute summary:")
            print(op[["op_count"]].describe().round(3))
            plt.figure()
            x = np.log1p(op["op_count"].fillna(0.0).astype(float).to_numpy())
            plt.hist(x, bins=30)
            plt.yscale("log")
            plt.xlabel("log1p(op_count)")
            plt.title("Operation op_count distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "operation_op_count.png"))
            plt.close()


# -------------------------------------
# 1-8. Operation cell purity (if available)
# -------------------------------------

def operation_cell_purity(labels: pd.DataFrame, out_dir: str):
    print("=" * 80)
    print("[1-8] Operation cell purity (homophily)")
    print("=" * 80)
    ensure_dir(out_dir)

    if "op_cell_id" not in labels.columns:
        print("No op_cell_id in nodes/labels; skipping.")
        return
    df = labels.copy()
    # treat -1 as unassigned
    df = df[df["op_cell_id"].astype(int) >= 0].copy()
    if df.empty:
        print("No assigned cells; skipping.")
        return

    def _purity(series: pd.Series) -> float:
        vc = series.value_counts()
        if vc.empty:
            return 0.0
        return float(vc.max()) / float(vc.sum())

    rows = []
    for cid, g in df.groupby("op_cell_id"):
        rows.append({
            "cell_id": int(cid),
            "size": int(len(g)),
            "group_purity": _purity(g["group"]),
            "region_purity": _purity(g["region"]),
            "ideology_std": float(g["ideology"].astype(float).std(ddof=0)) if "ideology" in g.columns else float("nan"),
        })

    out = pd.DataFrame(rows).sort_values("cell_id")
    print(out.round(4).to_string(index=False))
    out.to_csv(os.path.join(out_dir, "op_cell_purity.csv"), index=False)

    # visualize purity distribution
    for col in ["group_purity", "region_purity"]:
        plt.figure()
        plt.hist(out[col].fillna(0.0), bins=10)
        plt.xlabel(col)
        plt.ylabel("count")
        plt.title(f"{col} distribution across cells")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{col}_hist.png"))
        plt.close()


# -------------------------------------
# 1-9. Event burstiness
# -------------------------------------

def event_burstiness_diagnostics(df_events: pd.DataFrame, out_dir: str):
    print("=" * 80)
    print("[1-9] Event temporal burstiness")
    print("=" * 80)
    ensure_dir(out_dir)

    if df_events is None or df_events.empty:
        print("No events provided; skipping.")
        return
    if "event_type" not in df_events.columns:
        print("No event_type in events; skipping.")
        return
    if "time" not in df_events.columns:
        # v1 fallback
        print("No 'time' column found; trying timestamp parsing via original diagnostics pattern.")
        return

    df = df_events.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(0).astype(int)

    rows = []
    for et, g in df.groupby("event_type"):
        counts = g.groupby("time").size().sort_index()
        mu = float(counts.mean()) if len(counts) else 0.0
        var = float(counts.var(ddof=0)) if len(counts) else 0.0
        fano = (var / mu) if mu > 0 else float("nan")

        # inter-event time burstiness
        times = g["time"].sort_values().to_numpy()
        if len(times) >= 3:
            iet = np.diff(times)
            m = float(np.mean(iet))
            s = float(np.std(iet, ddof=0))
            B = (s - m) / (s + m) if (s + m) > 0 else float("nan")
        else:
            B = float("nan")

        rows.append({"event_type": et, "days_with_events": int(len(counts)), "mean_daily": mu, "var_daily": var, "fano": fano, "iet_burstiness": B})

        # plot time series
        plt.figure()
        counts.plot()
        plt.title(f"{et} daily counts")
        plt.xlabel("day")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{et}_daily_counts_ts.png"))
        plt.close()

    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    out.to_csv(os.path.join(out_dir, "event_burstiness_summary.csv"), index=False)


# -------------------------------------
# v3: Activity / Observability / Copy diagnostics
# -------------------------------------

def activity_observability_diagnostics(labels: pd.DataFrame, out_dir: str) -> None:
    """Summarize node-level activity_rate and observability (if present)."""
    ensure_dir(out_dir)
    if labels is None or labels.empty:
        print("No labels; skipping activity/observability diagnostics.")
        return

    if "activity_rate" not in labels.columns and "observability" not in labels.columns:
        print("No activity_rate/observability in labels; skipping.")
        return

    df = labels.copy()
    for col, default in [("activity_rate", 1.0), ("observability", 1.0)]:
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    # summary table
    rows = []
    for col in ["activity_rate", "observability"]:
        s = df[col]
        rows.append({
            "metric": col,
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "p05": float(s.quantile(0.05)),
            "median": float(s.median()),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "activity_observability_summary.csv"), index=False)
    print("[activity/observability] summary saved:", os.path.join(out_dir, "activity_observability_summary.csv"))

    # histograms
    for col in ["activity_rate", "observability"]:
        plt.figure()
        df[col].hist(bins=30)
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{col}_hist.png"))
        plt.close()

    # role-wise boxplots (if role exists)
    if "role" in df.columns:
        for col in ["activity_rate", "observability"]:
            plt.figure(figsize=(10, 4))
            df.boxplot(column=col, by="role", grid=False)
            plt.suptitle("")
            plt.title(f"{col} by role")
            plt.xlabel("role")
            plt.ylabel(col)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{col}_by_role_box.png"))
            plt.close()


def copy_provenance_diagnostics(mani: Dict[str, Any], layers: Dict[str, pd.DataFrame], out_dir: str) -> None:
    """Check cross-layer copy behavior using per-edge 'copied_from' and meta.copy_summary."""
    ensure_dir(out_dir)

    # meta summary (if present)
    meta = mani.get("meta", {}) if isinstance(mani, dict) else {}
    copy_summary = meta.get("copy_summary") if isinstance(meta, dict) else None
    if isinstance(copy_summary, dict) and copy_summary:
        # flatten
        rows = []
        for dst, mp in copy_summary.items():
            if isinstance(mp, dict):
                for src, cnt in mp.items():
                    rows.append({"dst_layer": str(dst), "src_layer": str(src), "copied_edges": int(cnt)})
        if rows:
            df_cs = pd.DataFrame(rows)
            df_cs.to_csv(os.path.join(out_dir, "copy_summary_meta.csv"), index=False)

    # edge-level provenance
    rows2 = []
    for lname, df in layers.items():
        if df is None or df.empty or "copied_from" not in df.columns:
            rows2.append({"layer": lname, "has_copied_from": False, "copied_edges": 0, "total_edges": int(0 if df is None else len(df)), "copied_frac": 0.0})
            continue
        s = df["copied_from"].astype(str)
        copied_mask = (~df["copied_from"].isna()) & (s != "None") & (s != "nan") & (s != "")
        copied_edges = int(copied_mask.sum())
        total_edges = int(len(df))
        rows2.append({"layer": lname, "has_copied_from": True, "copied_edges": copied_edges, "total_edges": total_edges, "copied_frac": float(copied_edges / max(1, total_edges))})

        # breakdown by source
        if copied_edges > 0:
            vc = df.loc[copied_mask, "copied_from"].value_counts()
            df_vc = vc.reset_index()
            df_vc.columns = ["copied_from", "count"]
            df_vc.to_csv(os.path.join(out_dir, f"{lname}_copied_from_breakdown.csv"), index=False)

    df2 = pd.DataFrame(rows2)
    df2.to_csv(os.path.join(out_dir, "copy_provenance_summary.csv"), index=False)
    print("[copy] summary saved:", os.path.join(out_dir, "copy_provenance_summary.csv"))


def observability_false_edge_bias(labels: pd.DataFrame, layers: Dict[str, pd.DataFrame], out_dir: str) -> None:
    """If observation bias is enabled, false edges should skew toward high-observability endpoints.

    This checks whether edges with is_false=1 have higher mean endpoint observability than is_false=0.
    """
    ensure_dir(out_dir)
    if labels is None or labels.empty:
        print("No labels; skipping observability false-edge bias diagnostics.")
        return
    if "observability" not in labels.columns:
        print("No 'observability' in labels; skipping.")
        return

    obs = labels[["node_id", "observability"]].copy()
    obs["observability"] = pd.to_numeric(obs["observability"], errors="coerce").fillna(1.0)

    rows = []
    for lname, df in layers.items():
        if df is None or df.empty:
            continue
        if "is_false" not in df.columns:
            continue
        if "source" not in df.columns or "target" not in df.columns:
            continue

        tmp = df[["source", "target", "is_false"]].copy()
        tmp["is_false"] = pd.to_numeric(tmp["is_false"], errors="coerce").fillna(0).astype(int)

        tmp = tmp.merge(obs.rename(columns={"node_id": "source", "observability": "obs_u"}), on="source", how="left")
        tmp = tmp.merge(obs.rename(columns={"node_id": "target", "observability": "obs_v"}), on="target", how="left")
        tmp[["obs_u", "obs_v"]] = tmp[["obs_u", "obs_v"]].fillna(1.0)
        tmp["obs_mean"] = 0.5 * (tmp["obs_u"] + tmp["obs_v"])

        g = tmp.groupby("is_false")["obs_mean"].agg(["count", "mean", "std"]).reset_index()
        g["layer"] = lname
        rows.append(g)

        # quick plot
        if tmp["is_false"].nunique() >= 2:
            plt.figure()
            tmp.boxplot(column="obs_mean", by="is_false", grid=False)
            plt.suptitle("")
            plt.title(f"{lname}: mean endpoint observability by is_false")
            plt.xlabel("is_false")
            plt.ylabel("obs_mean")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{lname}_obs_by_is_false_box.png"))
            plt.close()

    if rows:
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(os.path.join(out_dir, "observability_false_edge_bias.csv"), index=False)
        print("[obs_bias] summary saved:", os.path.join(out_dir, "observability_false_edge_bias.csv"))
    else:
        print("No layers with is_false found; skipping observability false-edge bias diagnostics.")


# -------------------------------------
# Main
# -------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Basic diagnostics v3 for multiplex synthetic terror network")
    parser.add_argument("--manifest", type=str, required=True, help="Path to multiplex.json")
    parser.add_argument("--out_dir", type=str, default="./analysis_output_v3", help="Directory to save plots / reports")
    args = parser.parse_args()

    validate_manifest_file(args.manifest)

    print("[*] Loading multiplex dataset from:", args.manifest)
    mani, nodes, labels, layers, df_events = load_multiplex(args.manifest)

    print_meta(mani)

    basic_stats(nodes, layers, out_dir=os.path.join(args.out_dir, "1_basic_stats"))
    degree_distributions(layers, out_dir=os.path.join(args.out_dir, "2_degree_dists"))
    rolewise_degree_stats(nodes, layers, out_dir=os.path.join(args.out_dir, "3_rolewise"))
    cross_layer_correlations(nodes, layers, out_dir=os.path.join(args.out_dir, "4_cross_layer"))
    layer_overlap_diagnostics(layers, out_dir=os.path.join(args.out_dir, "4b_overlap"))
    label_diagnostics(labels, out_dir=os.path.join(args.out_dir, "5_labels"))
    activity_observability_diagnostics(labels, out_dir=os.path.join(args.out_dir, "5b_activity_obs"))
    edge_noise_diagnostics(layers, out_dir=os.path.join(args.out_dir, "6_edge_noise"))
    false_edge_observability_diagnostics(labels, layers, out_dir=os.path.join(args.out_dir, "6b_false_edge_obs"))
    copy_provenance_diagnostics(mani, layers, out_dir=os.path.join(args.out_dir, "6c_copy"))
    edge_attribute_diagnostics(layers, out_dir=os.path.join(args.out_dir, "7_edge_attr"))
    operation_cell_purity(labels, out_dir=os.path.join(args.out_dir, "8_op_cells"))
    event_burstiness_diagnostics(df_events, out_dir=os.path.join(args.out_dir, "9_events_burst"))

    print("\n[*] Diagnostics completed.")
    print("    Output directory:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
