"""
data/analysis/basic_diagnostics.py

Basic diagnostics and visualization script for multiplex synthetic terror networks.

Features:
  - 1-1. Node/edge counts and role/region/group distributions
  - 1-2. Degree distributions per layer (histograms)
  - 1-3. Role-wise degree statistics per layer
  - 1-4. Cross-layer degree correlations
  - 1-5. importance_score / high_value_target label checks
  - 1-6. Event (communication/finance/operation) timing and scale distributions
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# -------------------------------------
# Utilities
# -------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_multiplex(manifest_path: str):

    with open(manifest_path, "r", encoding="utf-8") as f:
        mani = json.load(f)

    # --------------------------------------------------
    # 1) handle nodes / labels
    # --------------------------------------------------
    nodes_raw = mani.get("nodes")

    # (A) v1 style: CSV path string
    if isinstance(nodes_raw, str):
        nodes = pd.read_csv(nodes_raw)
        labels_path = mani.get("labels")
        if labels_path is None:
            raise ValueError("v1 format requires 'labels' to provide a CSV path.")
        labels = pd.read_csv(labels_path)

    # (B) v2 style: node information inline as a JSON list
    elif isinstance(nodes_raw, list):
        df_nodes = pd.DataFrame(nodes_raw)

        # normalize id / node_id
        if "node_id" in df_nodes.columns:
            pass
        elif "id" in df_nodes.columns:
            df_nodes = df_nodes.rename(columns={"id": "node_id"})
        else:
            raise ValueError("'nodes' must include an 'id' or 'node_id' column.")

        # minimum columns needed for analysis/diagnostics (and v1 compatibility)
        for col in ["node_id", "role", "region", "group"]:
            if col not in df_nodes.columns:
                raise ValueError(f"Required column '{col}' missing in nodes.")

        # fill optional columns with defaults if missing
        if "skill_level" not in df_nodes.columns:
            df_nodes["skill_level"] = 0.0
        if "radicalization" not in df_nodes.columns:
            df_nodes["radicalization"] = 0.0
        if "past_incidents" not in df_nodes.columns:
            df_nodes["past_incidents"] = 0.0
        if "importance_score" not in df_nodes.columns:
            df_nodes["importance_score"] = 0.0
        if "high_value_target" not in df_nodes.columns:
            df_nodes["high_value_target"] = 0

        # mimic the v1 nodes.csv / labels.csv split
        nodes = df_nodes[["node_id", "role", "region", "group"]].copy()
        labels = df_nodes[
            [
                "node_id",
                "role",
                "region",
                "group",
                "skill_level",
                "radicalization",
                "past_incidents",
                "importance_score",
                "high_value_target",
            ]
        ].copy()

    else:
        raise ValueError(f"Unrecognized type for 'nodes' field: {type(nodes_raw)}")

    # --------------------------------------------------
    # 2) handle layers
    # --------------------------------------------------
    layers: Dict[str, pd.DataFrame] = {}
    raw_layers = mani.get("layers", {})

    for layer_name, layer_obj in raw_layers.items():
        # (A) v1 style: CSV path string
        if isinstance(layer_obj, str):
            layers[layer_name] = pd.read_csv(layer_obj)

        # (B) v2 style: {"directed": bool, "edges": [ {...}, ... ]}
        elif isinstance(layer_obj, dict) and "edges" in layer_obj:
            df_layer = pd.DataFrame(layer_obj["edges"])

            # minimum expected columns for basic_diagnostics: source / target
            if "source" not in df_layer.columns or "target" not in df_layer.columns:
                raise ValueError(
                    f"Layer '{layer_name}' requires 'source' and 'target' columns."
                )

            # Other fields (amount / num_events / joint_ops / similarity, etc.)
            # can be handled via .get() or fillna downstream.
            layers[layer_name] = df_layer

        else:
            raise ValueError(
                f"Unrecognized layer format for '{layer_name}': {type(layer_obj)}"
            )

    # --------------------------------------------------
    # 3) handle events (optional)
    # --------------------------------------------------
    events_raw = mani.get("events", None)

    if isinstance(events_raw, str):
        # v1 style: CSV path
        df_events = pd.read_csv(events_raw)
    elif isinstance(events_raw, list):
        # v2 style: inline list
        df_events = pd.DataFrame(events_raw)
    elif events_raw is None:
        # events may be missing entirely
        df_events = pd.DataFrame()
    else:
        raise ValueError(f"Unrecognized type for 'events' field: {type(events_raw)}")

    return mani, nodes, labels, layers, df_events


def degree_dict_from_edges(df: pd.DataFrame, directed: bool) -> dict[str, int]:
    if df is None or df.empty:
        return {}
    G = nx.from_pandas_edgelist(
        df, "source", "target", create_using=nx.DiGraph() if directed else nx.Graph()
    )
    return dict(G.degree())


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

    print("\nRole distribution:")
    print(nodes["role"].value_counts())
    print("\nRole distribution (ratio):")
    print(nodes["role"].value_counts(normalize=True).round(3))

    print("\nRegion distribution:")
    print(nodes["region"].value_counts())
    print("\nRegion distribution (ratio):")
    print(nodes["region"].value_counts(normalize=True).round(3))

    print("\nGroup distribution:")
    print(nodes["group"].value_counts())
    print("\nGroup distribution (ratio):")
    print(nodes["group"].value_counts(normalize=True).round(3))

    # Save simple role/region/group bar charts
    ensure_dir(out_dir)

    plt.figure()
    nodes["role"].value_counts().plot(kind="bar")
    plt.title("Role counts")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "role_counts.png"))
    plt.close()

    plt.figure()
    nodes["region"].value_counts().plot(kind="bar")
    plt.title("Region counts")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "region_counts.png"))
    plt.close()

    plt.figure()
    nodes["group"].value_counts().plot(kind="bar")
    plt.title("Group counts")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "group_counts.png"))
    plt.close()


# -------------------------------------
# 1-2. Degree distributions per layer
# -------------------------------------


def plot_degree_hist(layer_df: pd.DataFrame, directed: bool, title: str, out_path: str):
    if layer_df is None or layer_df.empty:
        print(f"[WARN] {title} : empty layer, skip.")
        return

    if directed:
        G = nx.from_pandas_edgelist(
            layer_df, "source", "target", create_using=nx.DiGraph()
        )
        out_deg = [d for _, d in G.out_degree()]
        in_deg = [d for _, d in G.in_degree()]

        plt.figure()
        plt.hist(out_deg, bins=30)
        plt.yscale("log")
        plt.xlabel("out-degree")
        plt.ylabel("count (log)")
        plt.title(f"{title} - out-degree")
        plt.tight_layout()
        plt.savefig(out_path.replace(".png", "_out.png"))
        plt.close()

        plt.figure()
        plt.hist(in_deg, bins=30)
        plt.yscale("log")
        plt.xlabel("in-degree")
        plt.ylabel("count (log)")
        plt.title(f"{title} - in-degree")
        plt.tight_layout()
        plt.savefig(out_path.replace(".png", "_in.png"))
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

    # hierarchy: directed
    plot_degree_hist(
        layers["hierarchy"],
        directed=True,
        title="Hierarchy layer",
        out_path=os.path.join(out_dir, "deg_hierarchy.png"),
    )

    # finance: directed
    plot_degree_hist(
        layers["finance"],
        directed=True,
        title="Finance layer",
        out_path=os.path.join(out_dir, "deg_finance.png"),
    )

    # communication: undirected
    plot_degree_hist(
        layers["communication"],
        directed=False,
        title="Communication layer",
        out_path=os.path.join(out_dir, "deg_communication.png"),
    )

    # operation: undirected
    plot_degree_hist(
        layers["operation"],
        directed=False,
        title="Operation layer",
        out_path=os.path.join(out_dir, "deg_operation.png"),
    )

    # ideology: undirected
    plot_degree_hist(
        layers["ideology"],
        directed=False,
        title="Ideology layer",
        out_path=os.path.join(out_dir, "deg_ideology.png"),
    )


# -------------------------------------
# 1-3. Role-wise structural characteristics
# -------------------------------------


def rolewise_degree_stats(nodes: pd.DataFrame, layers: dict[str, pd.DataFrame], out_dir: str):
    print("=" * 80)
    print("[1-3] Role-wise degree stats")
    print("=" * 80)

    ensure_dir(out_dir)

    # hierarchy out-degree
    G_h = nx.from_pandas_edgelist(
        layers["hierarchy"], "source", "target", create_using=nx.DiGraph()
    )
    h_out = dict(G_h.out_degree())
    df = nodes.copy()
    df["hier_out_deg"] = df["node_id"].map(h_out).fillna(0)

    print("\nHierarchy out-degree by role:")
    print(df.groupby("role")["hier_out_deg"].describe().round(3))

    # finance out-degree
    G_f = nx.from_pandas_edgelist(
        layers["finance"], "source", "target", create_using=nx.DiGraph()
    )
    f_out = dict(G_f.out_degree())
    df["fin_out_deg"] = df["node_id"].map(f_out).fillna(0)

    print("\nFinance out-degree by role:")
    print(df.groupby("role")["fin_out_deg"].describe().round(3))

    # communication degree
    G_c = nx.from_pandas_edgelist(
        layers["communication"], "source", "target", create_using=nx.Graph()
    )
    c_deg = dict(G_c.degree())
    df["comm_deg"] = df["node_id"].map(c_deg).fillna(0)

    print("\nCommunication degree by role:")
    print(df.groupby("role")["comm_deg"].describe().round(3))

    # Simple boxplot examples (hierarchy / finance / communication)
    plt.figure(figsize=(10, 4))
    for i, col in enumerate(["hier_out_deg", "fin_out_deg", "comm_deg"]):
        plt.subplot(1, 3, i + 1)
        df.boxplot(column=col, by="role", rot=45)
        plt.title(col)
        plt.suptitle("")  # drop top spacing
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rolewise_degrees_boxplot.png"))
    plt.close()


# -------------------------------------
# 1-4. Cross-layer degree correlations
# -------------------------------------


def cross_layer_correlations(nodes: pd.DataFrame, layers: dict[str, pd.DataFrame], out_dir: str):
    print("=" * 80)
    print("[1-4] Cross-layer degree correlations")
    print("=" * 80)

    ensure_dir(out_dir)

    deg_fin = degree_dict_from_edges(layers["finance"], directed=True)
    deg_comm = degree_dict_from_edges(layers["communication"], directed=False)
    deg_ops = degree_dict_from_edges(layers["operation"], directed=False)
    deg_hier = degree_dict_from_edges(layers["hierarchy"], directed=True)
    deg_ideo = degree_dict_from_edges(layers["ideology"], directed=False)

    df_deg = nodes[["node_id", "role"]].copy()
    df_deg["deg_fin"] = df_deg["node_id"].map(deg_fin).fillna(0)
    df_deg["deg_comm"] = df_deg["node_id"].map(deg_comm).fillna(0)
    df_deg["deg_ops"] = df_deg["node_id"].map(deg_ops).fillna(0)
    df_deg["deg_hier"] = df_deg["node_id"].map(deg_hier).fillna(0)
    df_deg["deg_ideo"] = df_deg["node_id"].map(deg_ideo).fillna(0)

    print("\nCorrelation matrix (deg_fin, deg_comm, deg_ops, deg_hier, deg_ideo):")
    print(df_deg[["deg_fin", "deg_comm", "deg_ops", "deg_hier", "deg_ideo"]].corr().round(3))

    # Example scatter plots
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


# -------------------------------------
# 1-5. Label checks (importance_score / HVT)
# -------------------------------------


def label_diagnostics(labels: pd.DataFrame, out_dir: str):
    print("=" * 80)
    print("[1-5] Label diagnostics (importance_score, high_value_target)")
    print("=" * 80)

    ensure_dir(out_dir)

    print("\nimportance_score stats:")
    print(labels["importance_score"].describe().round(3))

    print("\nHVT ratio:")
    print(labels["high_value_target"].value_counts(normalize=True).round(3))

    print("\nHVT ratio by role:")
    print(labels.groupby("role")["high_value_target"].mean().round(3))

    # importance_score histogram
    plt.figure()
    plt.hist(labels["importance_score"], bins=30)
    plt.yscale("log")
    plt.xlabel("importance_score")
    plt.ylabel("count (log)")
    plt.title("Importance score distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "importance_score_hist.png"))
    plt.close()

    # importance_score boxplot by role
    plt.figure(figsize=(6, 4))
    labels.boxplot(column="importance_score", by="role", rot=45)
    plt.title("importance_score by role")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "importance_score_by_role.png"))
    plt.close()


# -------------------------------------
# 1-6. Event/time characteristics
# -------------------------------------
def event_time_diagnostics(df_events: pd.DataFrame, out_dir: str):
    """
    Event/time diagnostics:
      - v1: uses a 'timestamp' (datetime string) column
      - v2: also supports numeric time indices such as 'time' or 'step'
    """
    os.makedirs(out_dir, exist_ok=True)

    if df_events is None or df_events.empty:
        print("\n[1-6] No events provided; skipping temporal diagnostics.")
        return

    print("\nEvent type counts:")
    if "event_type" in df_events.columns:
        print(df_events["event_type"].value_counts())
    else:
        print("  [WARN] 'event_type' column missing; skip per-type counts.")
        print("  Available columns:", list(df_events.columns))

    # --------------------------------------------------
    # 1) auto-detect time column
    # --------------------------------------------------
    time_candidates = ["timestamp", "time", "date", "t", "step", "day"]
    time_col = None
    for cand in time_candidates:
        if cand in df_events.columns:
            time_col = cand
            break

    if time_col is None:
        print(
            "\n[1-6] Could not find a time column (timestamp/time/date/step/etc.)."
        )
        print("       → skipping temporal diagnostics.")
        print("       Available columns:", list(df_events.columns))
        return

    print(f"\n[1-6] Using '{time_col}' as time column for temporal diagnostics.")

    # --------------------------------------------------
    # 2) preprocess according to time column type
    #    - numeric: use as step/day index
    #    - string/datetime: parse to datetime
    # --------------------------------------------------
    col = df_events[time_col]

    if pd.api.types.is_numeric_dtype(col):
        # numeric: use as-is for step/day index
        df_events = df_events.copy()
        df_events["time_bin"] = col.astype(int)
        # assume daily bins and reuse as 'date'
        df_events["date"] = df_events["time_bin"]
    else:
        # string/datetime: parse then convert to date
        df_events = df_events.copy()
        df_events["timestamp_dt"] = pd.to_datetime(col, errors="coerce")
        df_events["date"] = df_events["timestamp_dt"].dt.date

    # --------------------------------------------------
    # 3) daily counts/statistics by event type
    # --------------------------------------------------
    def _describe_event_type(event_name: str):
        df_sub = df_events[df_events.get("event_type", "") == event_name].copy()
        if df_sub.empty:
            print(f"\n[{event_name}] no events found.")
            return

        daily_counts = df_sub.groupby("date").size()

        print(f"\n[{event_name}] daily event count stats:")
        print(daily_counts.describe())

        # optionally save simple histograms or time series plots here
        # e.g., histogram
        import matplotlib.pyplot as plt

        plt.figure()
        daily_counts.hist(bins=30)
        plt.title(f"{event_name} daily counts")
        plt.xlabel("count")
        plt.ylabel("frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{event_name}_daily_count_hist.png"))
        plt.close()

    # diagnose in comm / txn / op order
    _describe_event_type("comm")
    _describe_event_type("txn")
    _describe_event_type("op")


# -------------------------------------
# Main
# -------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Basic diagnostics for multiplex synthetic terror network"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to multiplex.json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./analysis_output",
        help="Directory to save plots / reports",
    )

    args = parser.parse_args()

    manifest_path = args.manifest
    out_dir = args.out_dir

    print("[*] Loading multiplex dataset from:", manifest_path)
    mani, nodes, labels, layers, df_events = load_multiplex(manifest_path)

    # 1-1
    basic_stats(nodes, layers, out_dir=os.path.join(out_dir, "1_basic_stats"))

    # 1-2
    degree_distributions(layers, out_dir=os.path.join(out_dir, "2_degree_dists"))

    # 1-3
    rolewise_degree_stats(nodes, layers, out_dir=os.path.join(out_dir, "3_rolewise"))

    # 1-4
    cross_layer_correlations(
        nodes,
        layers,
        out_dir=os.path.join(out_dir, "4_cross_layer"),
    )

    # 1-5
    label_diagnostics(labels, out_dir=os.path.join(out_dir, "5_labels"))

    # 1-6
    event_time_diagnostics(df_events, out_dir=os.path.join(out_dir, "6_events"))

    print("\n[*] Diagnostics completed.")
    print("    Output directory:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
