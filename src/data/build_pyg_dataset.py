"""
./data/build_pyg_dataset.py

multiplex_generator.py가 만든 synthetic terror multiplex 데이터를
PyTorch Geometric Data 객체로 변환하는 스크립트.

생성되는 것:
  - 단일 homogeneous 그래프 Data:
      * x           : [num_nodes, num_features] 노드 피처
      * edge_index  : [2, num_edges] (모든 레이어 edge를 concat)
      * edge_type   : [num_edges] (0=hier, 1=finance, 2=comm, 3=ops, 4=ideo)
      * edge_attr   : [num_edges, 1] (레이어별 의미 다른 weight: amount, similarity 등)
      * y_role      : [num_nodes] (int, 역할 클래스 인덱스)
      * y_hvt       : [num_nodes] (int, 0/1 high_value_target)
      * train_mask / val_mask / test_mask : [num_nodes] boolean mask
      * node_id     : list[str], 원본 노드 ID (Data에 그대로 넣어둠)

사용 예 (Colab):
  !python data/build_pyg_dataset.py \
      --manifest ./data/multiplex_v1_1/multiplex.json \
      --out_path ./data/multiplex_v1_1/pyg_data.pt
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# -----------------------------
# 유틸 함수
# -----------------------------

def load_multiplex(manifest_path: str):
    """
    multiplex.json 포맷을 v1 / v2 둘 다 지원하는 로더.

    v1 포맷 (기존):
      {
        "nodes": "nodes.csv",
        "labels": "labels.csv",
        "layers": {
          "hierarchy": "hierarchy_edges.csv",
          "finance": "finance_edges.csv",
          ...
        }
      }

    v2 포맷 (multiplex_generator_v2.py 가 생성하는 형태 예시):
      {
        "nodes": [ { ... }, { ... }, ... ],      # 노드 정보가 JSON 리스트로 인라인 저장
        "layers": {
          "hierarchy": { "directed": true,  "edges": [ {...}, ... ] },
          "finance":   { "directed": false, "edges": [ {...}, ... ] },
          ...
        },
        "events": [ ... ],   # (있을 수도 있음, PyG 변환에서는 사용 X)
        ...
      }
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        mani = json.load(f)

    # --------------------------------------------------
    # 1) nodes / labels 처리
    # --------------------------------------------------
    nodes_raw = mani.get("nodes")

    # (A) v1 스타일: CSV 경로 문자열
    if isinstance(nodes_raw, str):
        # 기존 코드와 동일하게 처리
        nodes = pd.read_csv(nodes_raw)
        labels_path = mani.get("labels")
        if labels_path is None:
            raise ValueError("v1 포맷에서는 'labels'에 CSV 경로가 있어야 합니다.")
        labels = pd.read_csv(labels_path)

    # (B) v2 스타일: 노드 정보가 리스트(JSON 인라인)
    elif isinstance(nodes_raw, list):
        df_nodes = pd.DataFrame(nodes_raw)

        # id / node_id 정규화
        if "node_id" in df_nodes.columns:
            pass
        elif "id" in df_nodes.columns:
            df_nodes = df_nodes.rename(columns={"id": "node_id"})
        else:
            raise ValueError("nodes 객체에 'id' 또는 'node_id' 컬럼이 필요합니다.")

        # 최소 필수 컬럼 체크 (v1에서 merge에 사용) :contentReference[oaicite:2]{index=2}
        for col in ["node_id", "role", "region", "group"]:
            if col not in df_nodes.columns:
                raise ValueError(f"nodes에 필수 컬럼 '{col}' 이(가) 없습니다.")

        # v2에서는 skill_level, radicalization, past_incidents, importance_score,
        # high_value_target 이 이미 nodes 안에 들어있을 가능성이 높음.
        # 없을 경우 기본값으로 채워서라도 빌드 가능하게 함.
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

        # v1의 nodes.csv / labels.csv 구조를 흉내 내서 분리
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
        raise ValueError(f"'nodes' 필드 타입을 알 수 없습니다: {type(nodes_raw)}")

    # --------------------------------------------------
    # 2) layers 처리
    # --------------------------------------------------
    layers: Dict[str, pd.DataFrame] = {}
    raw_layers = mani.get("layers", {})

    for layer_name, layer_obj in raw_layers.items():
        # (A) v1 스타일: CSV 경로 문자열
        if isinstance(layer_obj, str):
            layers[layer_name] = pd.read_csv(layer_obj)

        # (B) v2 스타일: {"directed": bool, "edges": [ {...}, ... ]}
        elif isinstance(layer_obj, dict) and "edges" in layer_obj:
            df_layer = pd.DataFrame(layer_obj["edges"])

            # build_pyg_dataset 은 각 레이어별로 최소한 source / target 을 기대 :contentReference[oaicite:3]{index=3}
            if "source" not in df_layer.columns or "target" not in df_layer.columns:
                raise ValueError(
                    f"레이어 '{layer_name}' 에 'source', 'target' 컬럼이 필요합니다."
                )

            # amount / num_events / joint_ops / similarity 는 있으면 그대로 사용,
            # 없으면 나중에 build_pyg_data() 에서 get("...", 1.0) 으로 처리됨.
            layers[layer_name] = df_layer

        else:
            raise ValueError(
                f"레이어 '{layer_name}' 형식을 인식할 수 없습니다: {type(layer_obj)}"
            )

    return mani, nodes, labels, layers



def encode_categorical(
    series: pd.Series,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    범주형 시리즈를 0..C-1 인덱스로 인코딩하고,
    (index_array, mapping) 반환.
    """
    uniques = sorted(series.unique().tolist())
    mapping = {v: i for i, v in enumerate(uniques)}
    idx = series.map(mapping).astype(int).to_numpy()
    return idx, mapping


def one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(indices), num_classes), dtype=np.float32)
    out[np.arange(len(indices)), indices] = 1.0
    return out


# -----------------------------
# 메인 변환 로직
# -----------------------------


def build_pyg_data(
    manifest_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Data:
    """
    multiplex.json을 읽어서 PyG Data를 구성.
    """
    assert abs(train_ratio + val_ratio - 1.0) > 1e-6 or test_ratio >= 0, \
        "train/val/test 비율을 명시적으로 넣어주세요."

    mani, nodes, labels, layers = load_multiplex(manifest_path)

    # -------------------------
    # 1. 노드 인덱스 매핑
    # -------------------------
    node_ids = nodes["node_id"].tolist()
    num_nodes = len(node_ids)
    node2idx = {nid: i for i, nid in enumerate(node_ids)}

    # labels.csv를 nodes와 merge해서 role/region/group/importance_score/hvt 정리
    df = nodes.merge(labels, on=["node_id", "role", "region", "group"], how="left")


    # -------------------------
    # 2. 노드 피처 구성
    #    - 범주형: region, group → one-hot  (role은 label로만 사용)
    #    - 연속형: skill_level, radicalization, past_incidents
    # -------------------------
    role_idx, role_map = encode_categorical(df["role"])
    region_idx, region_map = encode_categorical(df["region"])
    group_idx, group_map = encode_categorical(df["group"])

    # role_oh = one_hot(role_idx, len(role_map))  # <- 더 이상 x에 포함하지 않음
    region_oh = one_hot(region_idx, len(region_map))
    group_oh = one_hot(group_idx, len(group_map))

    cont_cols = ["skill_level", "radicalization", "past_incidents"]
    for col in cont_cols:
        if col not in df.columns:
            raise ValueError(f"연속형 피처 컬럼 {col} 이(가) DataFrame에 없습니다.")

    
    cont_feats = df[cont_cols].to_numpy(dtype=np.float32)
    cont_mean = cont_feats.mean(axis=0, keepdims=True)
    cont_std = cont_feats.std(axis=0, keepdims=True) + 1e-8
    cont_feats = (cont_feats - cont_mean) / cont_std
    
    x_np = np.concatenate([region_oh, group_oh, cont_feats], axis=1)
    x = torch.from_numpy(x_np) 


    # -------------------------
    # 3. 라벨 구성
    #    - y_role: 역할 인덱스
    #    - y_hvt: high_value_target (0/1)
    # -------------------------
    y_role = torch.from_numpy(role_idx.astype(np.int64))          # [num_nodes]
    y_hvt = torch.from_numpy(df["high_value_target"].astype(np.int64).to_numpy())

    # -------------------------
    # 4. edge_index / edge_type / edge_attr 구성
    # -------------------------
    # layer 이름 → 타입 인덱스 매핑
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

            # edge_attr: 레이어별로 의미 있는 scalar weight 하나를 선택
            if layer_name == "hierarchy":
                # relation(superior/informal)를 굳이 숫자로 encode하지 않고 1.0으로 둠
                w = 1.0
            elif layer_name == "finance":
                # amount 사용
                w = float(row.get("amount", 1.0))
            elif layer_name == "communication":
                # num_events 사용
                w = float(row.get("num_events", 1.0))
            elif layer_name == "operation":
                # joint_ops 사용
                w = float(row.get("joint_ops", 1.0))
            elif layer_name == "ideology":
                # similarity 사용
                w = float(row.get("similarity", 1.0))
            else:
                w = 1.0

            edge_attr_vals.append(w)

    for lname in ["hierarchy", "finance", "communication", "operation", "ideology"]:
        if lname not in layers:
            continue
        _add_edges_from_layer(lname, layers[lname])

    edge_index = torch.tensor(
        [edge_src, edge_dst], dtype=torch.long
    )  # [2, num_edges]
    edge_type = torch.tensor(edge_type_list, dtype=torch.long)   # [num_edges]
    edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float32).view(-1, 1)  # [num_edges, 1]

    # -------------------------
    # 5. train/val/test mask 생성 (노드 단위)
    # -------------------------
    num_nodes = x.size(0)
    perm = torch.randperm(num_nodes)

    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)
    n_test = num_nodes - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # importance_score train 통계 (mean/std)
    imp_np = df["importance_score"].to_numpy(dtype=np.float32)
    # train_idx는 torch.Tensor이므로 numpy 인덱싱용으로 변환
    train_idx_np = train_idx.cpu().numpy() if hasattr(train_idx, "cpu") else train_idx.numpy()
    imp_train = imp_np[train_idx_np]
    if imp_train.size > 0:
        imp_mean = float(imp_train.mean())
        imp_std = float(imp_train.std())
    else:
        # 혹시라도 train이 비어 있는 극단 상황에서는 전체 통계 사용
        imp_mean = float(imp_np.mean())
        imp_std = float(imp_np.std())
    print(f"[*] importance_score train mean={imp_mean:.3f}, std={imp_std:.3f}")

    # -------------------------
    # 6. Data 객체 구성
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


    # 원본 node_id / mapping / importance_score 등 메타 정보 추가
    data.node_id = node_ids
    data.role_mapping = role_map
    data.region_mapping = region_map
    data.group_mapping = group_map
    data.layer_type_mapping = layer_type_map

    # importance_score 텐서 추가 (float32)
    data.importance_score = torch.from_numpy(
        df["importance_score"].to_numpy(dtype=np.float32)
    )
    # importance_score train 통계 (mean/std) 메타 정보 저장
    data.imp_mean = float(imp_mean)
    data.imp_std = float(imp_std)

    return data


# -----------------------------
# CLI
# -----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert multiplex synthetic terror dataset to PyTorch Geometric Data."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to multiplex.json from multiplex_generator.py",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output .pt file path to save PyG Data",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training node ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation node ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test node ratio (default: 0.15)",
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    print("[*] Building PyG Data from:", args.manifest)
    data = build_pyg_data(
        manifest_path=args.manifest,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    torch.save(data, args.out_path)
    print("[*] Saved PyG Data to:", os.path.abspath(args.out_path))
    print("    #nodes   :", data.num_nodes)
    print("    #edges   :", data.edge_index.size(1))
    print("    x.shape  :", tuple(data.x.shape))
    print("    edge_attr.shape:", tuple(data.edge_attr.shape))


if __name__ == "__main__":
    main()
