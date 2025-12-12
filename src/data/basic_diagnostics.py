"""
data/analysis/basic_diagnostics.py

Multiplex synthetic terror network 기본 진단 / 시각화 스크립트.

기능:
  - 1-1. 노드/엣지 수, 역할/지역/그룹 분포
  - 1-2. 레이어별 degree 분포 (히스토그램)
  - 1-3. 역할별 레이어 degree 통계
  - 1-4. 크로스 레이어 degree 상관관계
  - 1-5. importance_score / high_value_target 라벨 검증
  - 1-6. 이벤트(communication / finance / operation) 시간·규모 분포
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
# 유틸
# -------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_multiplex(manifest_path: str):

    with open(manifest_path, "r", encoding="utf-8") as f:
        mani = json.load(f)

    # --------------------------------------------------
    # 1) nodes / labels 처리
    # --------------------------------------------------
    nodes_raw = mani.get("nodes")

    # (A) v1 스타일: CSV 경로 문자열
    if isinstance(nodes_raw, str):
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

        # 분석/진단에서 최소로 필요할 컬럼들 (v1과 호환 위해)
        for col in ["node_id", "role", "region", "group"]:
            if col not in df_nodes.columns:
                raise ValueError(f"nodes에 필수 컬럼 '{col}' 이(가) 없습니다.")

        # 아래 컬럼들은 없으면 기본값으로 채워서라도 사용 가능하게 처리
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

        # v1의 nodes.csv / labels.csv 구조를 흉내 낸 분리
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

            # basic_diagnostics에서 최소로 기대하는 컬럼: source / target
            if "source" not in df_layer.columns or "target" not in df_layer.columns:
                raise ValueError(
                    f"레이어 '{layer_name}' 에 'source', 'target' 컬럼이 필요합니다."
                )

            # 그 외 amount / num_events / joint_ops / similarity 등은
            # 코드 안에서 .get() 또는 fillna 로 처리되게 두면 됨.
            layers[layer_name] = df_layer

        else:
            raise ValueError(
                f"레이어 '{layer_name}' 형식을 인식할 수 없습니다: {type(layer_obj)}"
            )

    # --------------------------------------------------
    # 3) events 처리 (optional)
    # --------------------------------------------------
    events_raw = mani.get("events", None)

    if isinstance(events_raw, str):
        # v1 스타일: CSV 경로
        df_events = pd.read_csv(events_raw)
    elif isinstance(events_raw, list):
        # v2 스타일: 리스트 인라인
        df_events = pd.DataFrame(events_raw)
    elif events_raw is None:
        # 이벤트가 없을 수도 있음
        df_events = pd.DataFrame()
    else:
        raise ValueError(f"'events' 필드 타입을 알 수 없습니다: {type(events_raw)}")

    return mani, nodes, labels, layers, df_events


def degree_dict_from_edges(df: pd.DataFrame, directed: bool) -> dict[str, int]:
    if df is None or df.empty:
        return {}
    G = nx.from_pandas_edgelist(
        df, "source", "target", create_using=nx.DiGraph() if directed else nx.Graph()
    )
    return dict(G.degree())


# -------------------------------------
# 1-1. 기본 구조 점검
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

    # 역할/지역/그룹 바 차트 저장 (간단 버전)
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
# 1-2. 레이어별 degree 분포
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
# 1-3. 역할별 구조적 특징
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

    # 간단 boxplot 예시 (hierarchy / finance / communication)
    plt.figure(figsize=(10, 4))
    for i, col in enumerate(["hier_out_deg", "fin_out_deg", "comm_deg"]):
        plt.subplot(1, 3, i + 1)
        df.boxplot(column=col, by="role", rot=45)
        plt.title(col)
        plt.suptitle("")  # 상단 공백 제거
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rolewise_degrees_boxplot.png"))
    plt.close()


# -------------------------------------
# 1-4. 크로스 레이어 degree 상관관계
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

    # 산점도 몇 개만 예시
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
# 1-5. 라벨 (importance_score / HVT) 검증
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

    # importance_score 히스토그램
    plt.figure()
    plt.hist(labels["importance_score"], bins=30)
    plt.yscale("log")
    plt.xlabel("importance_score")
    plt.ylabel("count (log)")
    plt.title("Importance score distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "importance_score_hist.png"))
    plt.close()

    # role별 importance_boxplot
    plt.figure(figsize=(6, 4))
    labels.boxplot(column="importance_score", by="role", rot=45)
    plt.title("importance_score by role")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "importance_score_by_role.png"))
    plt.close()


# -------------------------------------
# 1-6. 이벤트/시간 특성
# -------------------------------------
def event_time_diagnostics(df_events: pd.DataFrame, out_dir: str):
    """
    이벤트/시간 진단:
      - v1: 'timestamp' (datetime string) 컬럼을 사용
      - v2: 'time', 'step' 등 숫자형 타임 인덱스를 사용했어도 동작하도록 확장
    """
    os.makedirs(out_dir, exist_ok=True)

    if df_events is None or df_events.empty:
        print("\n[1-6] No events provided; skipping temporal diagnostics.")
        return

    print("\nEvent type counts:")
    if "event_type" in df_events.columns:
        print(df_events["event_type"].value_counts())
    else:
        print("  [WARN] 'event_type' 컬럼이 없어 타입별 카운트는 생략합니다.")
        print("  Available columns:", list(df_events.columns))

    # --------------------------------------------------
    # 1) 시간 컬럼 자동 탐색
    # --------------------------------------------------
    time_candidates = ["timestamp", "time", "date", "t", "step", "day"]
    time_col = None
    for cand in time_candidates:
        if cand in df_events.columns:
            time_col = cand
            break

    if time_col is None:
        print(
            "\n[1-6] 시간 정보 컬럼(timestamp/time/date/step 등)을 찾지 못했습니다."
        )
        print("       → temporal diagnostics는 스킵합니다.")
        print("       Available columns:", list(df_events.columns))
        return

    print(f"\n[1-6] Using '{time_col}' as time column for temporal diagnostics.")

    # --------------------------------------------------
    # 2) 시간 컬럼 타입에 따라 전처리
    #    - 숫자형이면 step/day 인덱스로 사용
    #    - 문자열/타임스탬프면 datetime 으로 파싱
    # --------------------------------------------------
    col = df_events[time_col]

    if pd.api.types.is_numeric_dtype(col):
        # 숫자형: step/day 인덱스로 그대로 사용
        df_events = df_events.copy()
        df_events["time_bin"] = col.astype(int)
        # 일 단위로 본다고 가정하고 'date' = time_bin 으로 사용
        df_events["date"] = df_events["time_bin"]
    else:
        # 문자열 / datetime: to_datetime 후 날짜로 변환
        df_events = df_events.copy()
        df_events["timestamp_dt"] = pd.to_datetime(col, errors="coerce")
        df_events["date"] = df_events["timestamp_dt"].dt.date

    # --------------------------------------------------
    # 3) 이벤트 타입별 일일 카운트 / 통계
    # --------------------------------------------------
    def _describe_event_type(event_name: str):
        df_sub = df_events[df_events.get("event_type", "") == event_name].copy()
        if df_sub.empty:
            print(f"\n[{event_name}] no events found.")
            return

        daily_counts = df_sub.groupby("date").size()

        print(f"\n[{event_name}] daily event count stats:")
        print(daily_counts.describe())

        # 간단히 히스토그램이나 타임 시리즈 플롯을 저장하고 싶으면 여기서 추가
        # 예: 히스토그램
        import matplotlib.pyplot as plt

        plt.figure()
        daily_counts.hist(bins=30)
        plt.title(f"{event_name} daily counts")
        plt.xlabel("count")
        plt.ylabel("frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{event_name}_daily_count_hist.png"))
        plt.close()

    # comm / txn / op 순서로 진단
    _describe_event_type("comm")
    _describe_event_type("txn")
    _describe_event_type("op")


# -------------------------------------
# 메인
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
