"""
src/data/build_pyg_dataset.py

Convert synthetic multiplex terror network data (v1 or v2) into a PyTorch Geometric `Data` object.

Outputs (single homogeneous graph):
  - x           : [N, F] node features
  - edge_index  : [2, E] edges (concatenated from all layers)
  - edge_type   : [E] relation type index (0=hier, 1=finance, 2=comm, 3=ops, 4=ideo)
  - edge_attr   : [E, 1] scalar edge weight per edge (amount / num_events / joint_ops / similarity / 1.0)
  - y_role      : [N] role class index
  - y_hvt       : [N] high_value_target (0/1)
  - train_mask / val_mask / test_mask : [N] boolean masks (node-level split)
  - metadata fields attached directly to `Data`:
      node_id, role_mapping, region_mapping, group_mapping, layer_type_mapping,
      importance_score, imp_mean, imp_std, generator_meta, generator_config

Key reproducibility improvement:
  - node split is now deterministic using `--split_seed` (default: manifest meta.seed if present, else 42).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# -----------------------------
# Utility functions
# -----------------------------

def load_multiplex(manifest_path: str):
    """
    Loader that supports both v1 and v2 multiplex.json formats.

    v1 format (legacy):
      {
        "nodes": "nodes.csv",
        "labels": "labels.csv",
        "layers": {
          "hierarchy": "hierarchy_edges.csv",
          "finance": "finance_edges.csv",
          ...
        }
      }

    v2 format (example produced by multiplex_generator_v2.py):
      {
        "meta": { ... },
        "nodes": [ { ... }, { ... }, ... ],      # node info inline as a JSON list
        "layers": {
          "hierarchy": { "directed": true,  "edges": [ {...}, ... ] },
          "finance":   { "directed": true,  "edges": [ {...}, ... ] },
          "communication": { "directed": false, "edges": [ {...}, ... ] },
          ...
        },
        "events": [ ... ],   # optional (not used in PyG conversion)
        ...
      }
    """
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

        # minimum required columns (used for v1 merge compatibility)
        for col in ["node_id", "role", "region", "group"]:
            if col not in df_nodes.columns:
                raise ValueError(f"Required column '{col}' missing in nodes.")

        # Fill with defaults if any are missing so conversion still works.
        for col, default in [
            ("skill_level", 0.0),
            ("radicalization", 0.0),
            ("past_incidents", 0.0),
            ("importance_score", 0.0),
            ("high_value_target", 0),
        ]:
            if col not in df_nodes.columns:
                df_nodes[col] = default

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
    raw_layers = mani.get("layers", {}) or {}

    for layer_name, layer_obj in raw_layers.items():
        # (A) v1 style: CSV path string
        if isinstance(layer_obj, str):
            layers[layer_name] = pd.read_csv(layer_obj)

        # (B) v2 style: {"directed": bool, "edges": [ {...}, ... ]}
        elif isinstance(layer_obj, dict) and "edges" in layer_obj:
            df_layer = pd.DataFrame(layer_obj["edges"])
            if df_layer.empty:
                layers[layer_name] = df_layer
                continue

            # build_pyg_dataset expects at least source/target for each layer
            if "source" not in df_layer.columns or "target" not in df_layer.columns:
                raise ValueError(
                    f"Layer '{layer_name}' requires 'source' and 'target' columns."
                )
            layers[layer_name] = df_layer

        else:
            raise ValueError(
                f"Unrecognized layer format for '{layer_name}': {type(layer_obj)}"
            )

    return mani, nodes, labels, layers


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float, tol: float = 1e-6) -> None:
    """Validate that the provided split ratios are non-negative and sum to 1.0."""
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(r < 0 for r in ratios):
        raise ValueError("train/val/test ratios must be non-negative.")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > tol:
        raise ValueError("train/val/test ratios must sum to 1.0.")


def encode_categorical(series: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    """Encode a categorical series into 0..C-1 indices and return (indices, mapping)."""
    uniques = sorted(series.unique().tolist())
    mapping = {v: i for i, v in enumerate(uniques)}
    idx = series.map(mapping).astype(int).to_numpy()
    return idx, mapping


def one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(indices), num_classes), dtype=np.float32)
    out[np.arange(len(indices)), indices] = 1.0
    return out


# -----------------------------
# Main conversion logic
# -----------------------------

def _default_split_seed_from_manifest(mani: dict) -> int:
    """
    Priority:
      1) meta.seed (preferred, since it reflects the generator CLI seed)
      2) meta.config.seed (if present)
      3) fallback = 42
    """
    meta = mani.get("meta", {}) or {}
    if isinstance(meta, dict):
        if "seed" in meta and meta["seed"] is not None:
            try:
                return int(meta["seed"])
            except Exception:
                pass
        cfg = meta.get("config", {}) or {}
        if isinstance(cfg, dict) and "seed" in cfg and cfg["seed"] is not None:
            try:
                return int(cfg["seed"])
            except Exception:
                pass
    return 42


def build_pyg_data(
    manifest_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_seed: Optional[int] = None,
) -> Data:
    """
    Read multiplex.json and construct a PyG `Data`.

    split_seed:
      - if provided, used for deterministic node split.
      - if None, uses manifest meta.seed (or meta.config.seed) when available.
    """
    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    mani, nodes, labels, layers = load_multiplex(manifest_path)

    # -------------------------
    # 1) node index mapping
    # -------------------------
    node_ids = nodes["node_id"].tolist()
    num_nodes = len(node_ids)
    node2idx = {nid: i for i, nid in enumerate(node_ids)}

    # merge labels.csv with nodes to consolidate role/region/group/importance_score/hvt
    df = nodes.merge(labels, on=["node_id", "role", "region", "group"], how="left")

    # -------------------------
    # 2) build node features
    # -------------------------
    role_idx, role_map = encode_categorical(df["role"])
    region_idx, region_map = encode_categorical(df["region"])
    group_idx, group_map = encode_categorical(df["group"])

    region_oh = one_hot(region_idx, len(region_map))
    group_oh = one_hot(group_idx, len(group_map))

    cont_cols = ["skill_level", "radicalization", "past_incidents"]
    for col in cont_cols:
        if col not in df.columns:
            raise ValueError(f"Continuous feature column {col} is missing in the DataFrame.")

    cont_feats = df[cont_cols].to_numpy(dtype=np.float32)
    cont_mean = cont_feats.mean(axis=0, keepdims=True)
    cont_std = cont_feats.std(axis=0, keepdims=True) + 1e-8
    cont_feats = (cont_feats - cont_mean) / cont_std

    x_np = np.concatenate([region_oh, group_oh, cont_feats], axis=1)
    x = torch.from_numpy(x_np)

    # -------------------------
    # 3) build labels
    # -------------------------
    y_role = torch.from_numpy(role_idx.astype(np.int64))
    y_hvt = torch.from_numpy(df["high_value_target"].astype(np.int64).to_numpy())

    # -------------------------
    # 4) edges
    # -------------------------
    layer_type_map = {
        "hierarchy": 0,
        "finance": 1,
        "communication": 2,
        "operation": 3,
        "ideology": 4,
    }

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_type_list: List[int] = []
    edge_attr_vals: List[float] = []

    def _add_edges_from_layer(layer_name: str, df_layer: pd.DataFrame):
        nonlocal edge_src, edge_dst, edge_type_list, edge_attr_vals

        if df_layer is None or df_layer.empty:
            return
        if layer_name not in layer_type_map:
            return

        ltype = layer_type_map[layer_name]

        for _, row in df_layer.iterrows():
            u = row["source"]
            v = row["target"]

            if u not in node2idx or v not in node2idx:
                continue

            ui = node2idx[u]
            vi = node2idx[v]

            edge_src.append(ui)
            edge_dst.append(vi)
            edge_type_list.append(ltype)

            if layer_name == "hierarchy":
                w = 1.0
            elif layer_name == "finance":
                w = float(row.get("amount", 1.0))
            elif layer_name == "communication":
                w = float(row.get("num_events", 1.0))
            elif layer_name == "operation":
                w = float(row.get("joint_ops", 1.0))
            elif layer_name == "ideology":
                w = float(row.get("similarity", 1.0))
            else:
                w = 1.0

            edge_attr_vals.append(w)

    for lname in ["hierarchy", "finance", "communication", "operation", "ideology"]:
        if lname in layers:
            _add_edges_from_layer(lname, layers[lname])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_type_list, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float32).view(-1, 1)

    # -------------------------
    # 5) deterministic node split
    # -------------------------
    if split_seed is None:
        split_seed = _default_split_seed_from_manifest(mani)

    gen = torch.Generator().manual_seed(int(split_seed))
    perm = torch.randperm(num_nodes, generator=gen)

    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # importance_score train statistics (mean/std)
    imp_np = df["importance_score"].to_numpy(dtype=np.float32)
    train_idx_np = train_idx.cpu().numpy() if hasattr(train_idx, "cpu") else train_idx.numpy()
    imp_train = imp_np[train_idx_np]
    if imp_train.size > 0:
        imp_mean = float(imp_train.mean())
        imp_std = float(imp_train.std())
    else:
        imp_mean = float(imp_np.mean())
        imp_std = float(imp_np.std())

    print(f"[*] split_seed={split_seed} | importance_score train mean={imp_mean:.3f}, std={imp_std:.3f}")

    # -------------------------
    # 6) assemble Data
    # -------------------------
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_attr=edge_attr,
        y_role=y_role,
        y_hvt=y_hvt,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    data.node_id = node_ids
    data.role_mapping = role_map
    data.region_mapping = region_map
    data.group_mapping = group_map
    data.layer_type_mapping = layer_type_map

    data.importance_score = torch.from_numpy(df["importance_score"].to_numpy(dtype=np.float32))
    data.imp_mean = float(imp_mean)
    data.imp_std = float(imp_std)

    data.generator_meta = mani.get("meta", {}) or {}
    meta = data.generator_meta if isinstance(data.generator_meta, dict) else {}
    data.generator_config = meta.get("config", {}) if isinstance(meta, dict) else {}

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Convert multiplex synthetic dataset to a PyTorch Geometric Data object."
    )
    parser.add_argument("--manifest", type=str, required=True, help="Path to multiplex.json from the generator")
    parser.add_argument("--out_path", type=str, required=True, help="Output .pt file path to save PyG Data")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training node ratio (default: 0.7)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation node ratio (default: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test node ratio (default: 0.15)")
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Seed for deterministic node split. Default: manifest meta.seed (or 42 if missing).",
    )

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    print("[*] Building PyG Data from:", args.manifest)
    data = build_pyg_data(
        manifest_path=args.manifest,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )

    torch.save(data, args.out_path)
    print("[*] Saved PyG Data to:", os.path.abspath(args.out_path))
    print("    #nodes   :", data.num_nodes)
    print("    #edges   :", data.edge_index.size(1))
    print("    x.shape  :", tuple(data.x.shape))
    print("    edge_attr.shape:", tuple(data.edge_attr.shape))


if __name__ == "__main__":
    main()
