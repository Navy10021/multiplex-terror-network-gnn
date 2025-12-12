import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import random


# -----------------------------
# 데이터 클래스 정의
# -----------------------------
@dataclass
class Node:
    id: int
    role: str
    region: str
    group: str
    ideology: float
    importance_score: float = 0.0
    high_value_target: int = 0  # 0/1


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
# Generator 설정 (config용)
# -----------------------------
@dataclass
class GeneratorConfig:
    size: int = 1500
    seed: int = 2025

    # finance layer
    finance_avg_out_degree: float = 18.0
    finance_w_group: float = 2.0
    finance_w_region: float = 1.5
    finance_w_ideo: float = 1.0
    finance_w_tier_dist: float = 1.0
    finance_base_bias: float = 0.1
    finance_structure_strength: float = 1.0  # 구조 강도 knob

    # communication layer
    comm_avg_degree: float = 3.5
    comm_alpha0: float = 0.1
    comm_alpha_group: float = 1.5
    comm_alpha_region: float = 1.0
    comm_alpha_hier: float = 1.5
    comm_alpha_fin: float = 1.5
    comm_structure_strength: float = 1.0
    comm_randomness: float = 0.0          # 0~1

    # ideology / operation
    ideo_threshold: float = 0.2
    op_num_cells: int = 20
    op_cell_size: int = 4

    # importance / HVT
    hvt_ratio: float = 0.05


def load_generator_config(
    config_path: Optional[str],
    size: int,
    seed: int,
) -> GeneratorConfig:
    """
    JSON 파일에서 GeneratorConfig를 불러오고,
    CLI에서 넘긴 size/seed로 덮어쓴다.
    """
    if config_path is None:
        cfg = GeneratorConfig(size=size, seed=seed)
        print("[*] No config file provided. Using default GeneratorConfig.")
        return cfg

    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    cfg = GeneratorConfig(**cfg_dict)
    cfg.size = size
    cfg.seed = seed

    print(f"[*] Loaded GeneratorConfig from {config_path}")
    print(cfg)
    return cfg

# -----------------------------
# 설정값 (분포/가중치 등)
# -----------------------------
ROLE_PROBS = {
    "leader": 0.024,      # ~ 3%
    "financier": 0.044,   # ~ 4%
    "courier": 0.073,     # ~ 7%
    "operative": 0.256,   # ~ 26%
    "support": 0.603,     # ~ 60%
}

REGIONS = ["Africa", "Asia", "Europe", "MiddleEast"]
GROUPS = ["GroupA", "GroupB", "GroupC"]

# tier: 계층적 거리 계산용
ROLE_TIER = {
    "leader": 0,
    "financier": 1,
    "operative": 2,
    "courier": 2,
    "support": 3,
}


# -----------------------------
# 유틸 함수들
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def sample_roles(num_nodes: int) -> List[str]:
    roles = list(ROLE_PROBS.keys())
    probs = np.array(list(ROLE_PROBS.values()))
    probs = probs / probs.sum()
    sampled = np.random.choice(roles, size=num_nodes, p=probs)
    return sampled.tolist()


def sample_regions(num_nodes: int) -> List[str]:
    return np.random.choice(REGIONS, size=num_nodes).tolist()


def sample_groups(num_nodes: int) -> List[str]:
    return np.random.choice(GROUPS, size=num_nodes).tolist()


def sample_ideologies(num_nodes: int) -> np.ndarray:
    # [0,1] 균등 분포 이념 좌표
    return np.random.rand(num_nodes)

# -----------------------------
# Hierarchy 레이어 생성
# -----------------------------
def build_hierarchy_edges(nodes: List[Node]) -> List[Edge]:
    """
    매우 단순한 트리형 계층 구조 생성.
    - leaders 최상단
    - 그 아래에 financiers/operatives
    - 그 아래에 couriers/support
    """
    leaders = [n.id for n in nodes if n.role == "leader"]
    financiers = [n.id for n in nodes if n.role == "financier"]
    operatives = [n.id for n in nodes if n.role == "operative"]
    couriers = [n.id for n in nodes if n.role == "courier"]
    supports = [n.id for n in nodes if n.role == "support"]

    edges: List[Edge] = []

    if not leaders:
        return edges

    # leader -> financier/operative
    for idx, f_id in enumerate(financiers + operatives):
        leader_id = leaders[idx % len(leaders)]
        edges.append(Edge(source=leader_id, target=f_id))

    # 중간 계층 -> couriers/support
    mid_level = financiers + operatives
    if mid_level:
        for idx, nid in enumerate(couriers + supports):
            parent = mid_level[idx % len(mid_level)]
            edges.append(Edge(source=parent, target=nid))

    return edges


# -----------------------------
# Finance 레이어 (structured v2)
# -----------------------------
def build_finance_edges(
    nodes: List[Node],
    hierarchy_edges: List[Edge],
    avg_out_degree: float = 18.0,
    w_group: float = 2.0,
    w_region: float = 1.5,
    w_ideo: float = 1.0,
    w_tier_dist: float = 1.0,
    base_bias: float = 0.1,
) -> List[Edge]:
    """
    Financier -> 다른 노드로 가는 구조적 자금 흐름 엣지 생성.
    score(u,v) = base_bias +
                 w_group * 1(same_group) +
                 w_region * 1(same_region) +
                 w_ideo * ideology_sim(u,v) -
                 w_tier_dist * tier_distance(u,v)
    을 weight로 사용해, financier 당 target_out_degree 만큼 샘플링.
    """
    id_to_node: Dict[int, Node] = {n.id: n for n in nodes}
    num_nodes = len(nodes)
    ideology = np.array([n.ideology for n in nodes])
    roles = [n.role for n in nodes]

    # tier 정보
    tiers = np.array([ROLE_TIER[role] for role in roles])

    financiers = [n.id for n in nodes if n.role == "financier"]
    if not financiers:
        return []

    edges: List[Edge] = []

    # 각 financier별 target out-degree (약간의 랜덤성 포함)
    for fid in financiers:
        # 평균 18, 표준편차 4 정도로 설정
        target_k = max(5, int(np.random.normal(loc=avg_out_degree, scale=4.0)))
        u = fid

        u_node = id_to_node[u]
        u_group = u_node.group
        u_region = u_node.region
        u_ideo = u_node.ideology
        u_tier = ROLE_TIER[u_node.role]

        # 후보 v: 자기 자신 제외
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

        # 음수/zero 방지
        score = np.maximum(score, 1e-6)
        weights = score / score.sum()

        # 후보 수보다 target_k가 클 수 있으니 min
        k = min(target_k, len(candidates))

        # 중복 없이 샘플링
        chosen_idx = np.random.choice(len(candidates), size=k, replace=False, p=weights)
        for idx in chosen_idx:
            v = candidates[idx]
            edges.append(Edge(source=u, target=int(v)))

    return edges


# -----------------------------
# Communication 레이어 (structured v2)
# -----------------------------
def build_communication_edges(
    nodes: List[Node],
    hierarchy_edges: List[Edge],
    finance_edges: List[Edge],
    avg_degree: float = 3.5,
    alpha0: float = 0.1,
    alpha_group: float = 1.5,
    alpha_region: float = 1.0,
    alpha_hier: float = 1.5,
    alpha_fin: float = 1.5,
) -> List[Edge]:
    """
    hierarchy/finance/group/region 구조를 반영하는 연락망 생성.
    - 노드별 target degree를 평균 avg_degree 주변에서 설정.
    - weight(i,j) = alpha0 + alpha_group * 1(same_group) + ... 를 기반으로 샘플링.
    - undirected에 가깝게, (i,j) 한 번만 생성 (source < target 규칙).
    """
    id_to_node: Dict[int, Node] = {n.id: n for n in nodes}
    num_nodes = len(nodes)

    # hierarchy / finance adjacency set
    hier_set = set((e.source, e.target) for e in hierarchy_edges) | set(
        (e.target, e.source) for e in hierarchy_edges
    )
    fin_set = set((e.source, e.target) for e in finance_edges) | set(
        (e.target, e.source) for e in finance_edges
    )

    edges_set = set()

    # 미리 group/region 배열 준비
    groups = np.array([n.group for n in nodes])
    regions = np.array([n.region for n in nodes])

    for u in range(num_nodes):
        target_k = max(1, int(np.random.normal(loc=avg_degree, scale=1.0)))

        u_group = groups[u]
        u_region = regions[u]

        # 후보: 자기 자신 제외
        candidates = np.array([v for v in range(num_nodes) if v != u])
        cand_groups = groups[candidates]
        cand_regions = regions[candidates]

        same_group = (cand_groups == u_group).astype(float)
        same_region = (cand_regions == u_region).astype(float)

        # hierarchy/finance 연결 여부
        hier_link = np.array(
            [1.0 if (u, int(v)) in hier_set or (int(v), u) in hier_set else 0.0 for v in candidates]
        )
        fin_link = np.array(
            [1.0 if (u, int(v)) in fin_set or (int(v), u) in fin_set else 0.0 for v in candidates]
        )

        score = (
            alpha0
            + alpha_group * same_group
            + alpha_region * same_region
            + alpha_hier * hier_link
            + alpha_fin * fin_link
        )
        score = np.maximum(score, 1e-6)
        weights = score / score.sum()

        k = min(target_k, len(candidates))
        chosen_idx = np.random.choice(len(candidates), size=k, replace=False, p=weights)
        for idx in chosen_idx:
            v = int(candidates[idx])
            a, b = (u, v) if u < v else (v, u)
            edges_set.add((a, b))

    # set -> Edge 리스트
    edges = [Edge(source=a, target=b) for (a, b) in edges_set]
    return edges


# -----------------------------
# Ideology 레이어 (간단한 유사도 기반)
# -----------------------------
def build_ideology_edges(nodes: List[Node], threshold: float = 0.2) -> List[Edge]:
    """
    ideology 값이 가까운 노드들 사이에 edge 생성.
    threshold: |ideo_u - ideo_v| < threshold 면 연결.
    """
    num_nodes = len(nodes)
    ideology = np.array([n.ideology for n in nodes])
    edges: List[Edge] = []

    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if abs(ideology[u] - ideology[v]) < threshold:
                edges.append(Edge(source=u, target=v))
    return edges


# -----------------------------
# Operation 레이어 (간단한 소수의 작전 셀)
# -----------------------------
def build_operation_edges(nodes: List[Node], num_cells: int = 20, cell_size: int = 4) -> List[Edge]:
    """
    작은 작전 셀(operation cell)을 몇 개 만든다.
    - 각 셀은 무작위 노드 중 몇 명씩 묶고, 완전 그래프 형태로 연결.
    """
    num_nodes = len(nodes)
    all_ids = list(range(num_nodes))
    random.shuffle(all_ids)

    edges: List[Edge] = []

    ptr = 0
    for _ in range(num_cells):
        if ptr + cell_size > num_nodes:
            break
        cell = all_ids[ptr : ptr + cell_size]
        ptr += cell_size
        # 완전 그래프
        for i in range(len(cell)):
            for j in range(i + 1, len(cell)):
                u, v = cell[i], cell[j]
                edges.append(Edge(source=u, target=v))
    return edges


# -----------------------------
# Event 생성 (간단 버전)
# -----------------------------
def build_events(
    finance_edges: List[Edge],
    comm_edges: List[Edge],
    op_edges: List[Edge],
    num_days: int = 300,
) -> List[Event]:
    """
    매우 단순한 이벤트 생성:
    - finance edge 당 1~3개 txn 이벤트
    - comm edge 당 1~5개 comm 이벤트
    - op edge 당 1~3개 op 이벤트
    time: [0, num_days) 정수 일 단위
    """
    events: List[Event] = []

    # finance -> txn
    for e in finance_edges:
        k = np.random.randint(1, 4)
        for _ in range(k):
            t = int(np.random.randint(0, num_days))
            amount = float(np.random.lognormal(mean=8.0, sigma=1.0))  # log-normal 자금 규모
            events.append(
                Event(
                    time=t,
                    event_type="txn",
                    u=e.source,
                    v=e.target,
                    meta={"amount": amount},
                )
            )

    # communication -> comm
    for e in comm_edges:
        k = np.random.randint(1, 6)
        for _ in range(k):
            t = int(np.random.randint(0, num_days))
            duration = int(np.random.exponential(scale=5.0))  # 통화 길이 같은 것
            events.append(
                Event(
                    time=t,
                    event_type="comm",
                    u=e.source,
                    v=e.target,
                    meta={"duration": duration},
                )
            )

    # operation -> op
    for e in op_edges:
        k = np.random.randint(1, 4)
        for _ in range(k):
            t = int(np.random.randint(0, num_days))
            events.append(
                Event(
                    time=t,
                    event_type="op",
                    u=e.source,
                    v=e.target,
                    meta={},
                )
            )

    return events


# -----------------------------
# importance_score & HVT 계산
# -----------------------------
def compute_importance_and_hvt(
    nodes: List[Node],
    hierarchy_edges: List[Edge],
    finance_edges: List[Edge],
    comm_edges: List[Edge],
    op_events: List[Event],
    hvt_ratio: float = 0.05,
) -> None:
    num_nodes = len(nodes)
    id_to_idx = {n.id: n.id for n in nodes}

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
    for ev in op_events:
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
        i = id_to_idx[n.id]
        role_base = base_role_score.get(n.role, 30.0)
        s = (
            role_base
            + 5.0 * hier_out[i]
            + 3.0 * fin_out[i]
            + 1.5 * comm_deg[i]
            + 4.0 * op_count[i]
        )
        scores[i] = s
        n.importance_score = float(s)

    # HVT: 상위 hvt_ratio 퍼센트
    threshold = np.quantile(scores, 1.0 - hvt_ratio)
    for n in nodes:
        n.high_value_target = int(n.importance_score >= threshold)


# -----------------------------
# 메인 제너레이터
# -----------------------------
def generate_multiplex_with_config(cfg: GeneratorConfig) -> Dict[str, Any]:
    """
    GeneratorConfig 기반으로 multiplex manifest를 생성.
    """
    set_seed(cfg.seed)

    num_nodes = cfg.size

    # 1) 노드 생성
    roles = sample_roles(num_nodes)
    regions = sample_regions(num_nodes)
    groups = sample_groups(num_nodes)
    ideologies = sample_ideologies(num_nodes)

    nodes: List[Node] = []
    for i in range(num_nodes):
        nodes.append(
            Node(
                id=i,
                role=roles[i],
                region=regions[i],
                group=groups[i],
                ideology=float(ideologies[i]),
            )
        )

    # 2) 계층 레이어
    hierarchy_edges = build_hierarchy_edges(nodes)

    # 3) finance 레이어 (config + 구조강도 knob)
    fs = cfg.finance_structure_strength
    finance_edges = build_finance_edges(
        nodes,
        hierarchy_edges,
        avg_out_degree=cfg.finance_avg_out_degree,
        w_group=cfg.finance_w_group * fs,
        w_region=cfg.finance_w_region * fs,
        w_ideo=cfg.finance_w_ideo * fs,
        w_tier_dist=cfg.finance_w_tier_dist * fs,
        base_bias=cfg.finance_base_bias,
    )

    # 4) ideology 레이어
    ideology_edges = build_ideology_edges(nodes, threshold=cfg.ideo_threshold)

    # 5) operation 레이어
    operation_edges = build_operation_edges(
        nodes,
        num_cells=cfg.op_num_cells,
        cell_size=cfg.op_cell_size,
    )

    # 6) communication 레이어 (config + 구조강도/랜덤성 knob)
    cs = cfg.comm_structure_strength
    cr = cfg.comm_randomness

    alpha_group = cfg.comm_alpha_group * cs
    alpha_region = cfg.comm_alpha_region * cs
    alpha_hier = cfg.comm_alpha_hier * cs
    alpha_fin = cfg.comm_alpha_fin * cs
    alpha0 = cfg.comm_alpha0 * (1.0 + cr)

    communication_edges = build_communication_edges(
        nodes,
        hierarchy_edges,
        finance_edges,
        avg_degree=cfg.comm_avg_degree,
        alpha0=alpha0,
        alpha_group=alpha_group,
        alpha_region=alpha_region,
        alpha_hier=alpha_hier,
        alpha_fin=alpha_fin,
    )

    # 7) 이벤트 생성
    events = build_events(finance_edges, communication_edges, operation_edges, num_days=300)

    # 8) importance_score / HVT 계산
    op_events = [ev for ev in events if ev.event_type == "op"]
    compute_importance_and_hvt(
        nodes,
        hierarchy_edges,
        finance_edges,
        communication_edges,
        op_events,
        hvt_ratio=cfg.hvt_ratio,
    )

    # 9) manifest JSON 구성
    manifest = {
        "meta": {
            "num_nodes": num_nodes,
            "seed": cfg.seed,
            "generator": "multiplex_generator_v2",
            "config": asdict(cfg),
        },
        "nodes": [asdict(n) for n in nodes],
        "layers": {
            "hierarchy": {
                "directed": True,
                "edges": [asdict(e) for e in hierarchy_edges],
            },
            "finance": {
                "directed": True,
                "edges": [asdict(e) for e in finance_edges],
            },
            "communication": {
                "directed": False,
                "edges": [asdict(e) for e in communication_edges],
            },
            "operation": {
                "directed": False,
                "edges": [asdict(e) for e in operation_edges],
            },
            "ideology": {
                "directed": False,
                "edges": [asdict(e) for e in ideology_edges],
            },
        },
        "events": [asdict(ev) for ev in events],
    }

    return manifest

def generate_multiplex(
    num_nodes: int,
    seed: int,
) -> Dict[str, Any]:

    cfg = GeneratorConfig(size=num_nodes, seed=seed)
    return generate_multiplex_with_config(cfg)


def main():
    parser = argparse.ArgumentParser(description="Structured Multiplex Terrorist Network Generator v2")
    parser.add_argument("--size", type=int, default=1500, help="number of nodes")
    parser.add_argument("--out_dir", type=str, required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=2025, help="random seed")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON config file for generator (optional)",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_generator_config(args.config, size=args.size, seed=args.seed)
    manifest = generate_multiplex_with_config(cfg)

    out_path = os.path.join(args.out_dir, "multiplex.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[*] Saved multiplex manifest to: {out_path}")


if __name__ == "__main__":
    main()
