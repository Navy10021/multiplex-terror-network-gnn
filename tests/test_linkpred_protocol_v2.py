import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.train_linkpred_layer_v3 import (
    sample_negative_edges_degree,
    split_edges,
    unique_edges_with_optional_time,
)


def test_temporal_split_respects_time_order():
    # 6 unique edges with increasing times
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [1, 2, 2, 3, 3, 4]], dtype=torch.long)
    edge_time = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)

    train, val, test = split_edges(
        edge_index,
        train_ratio=0.5,
        val_ratio=1 / 3,
        seed=7,
        split_mode="temporal",
        edge_time=edge_time,
    )

    # train gets earliest 3, val gets next 2, test gets last 1
    assert train.size(1) == 3
    assert val.size(1) == 2
    assert test.size(1) == 1
    assert torch.equal(train, edge_index[:, :3])
    assert torch.equal(val, edge_index[:, 3:5])
    assert torch.equal(test, edge_index[:, 5:])


def test_unique_edges_with_optional_time_uses_earliest_timestamp_for_duplicates():
    # duplicate undirected edge (0,1) appears twice with times 8 and 3
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 2]], dtype=torch.long)
    edge_time = torch.tensor([8.0, 3.0, 5.0], dtype=torch.float32)

    uniq_e, uniq_t = unique_edges_with_optional_time(edge_index, directed=False, edge_time=edge_time)

    assert uniq_e.size(1) == 2
    pairs = {(int(uniq_e[0, i]), int(uniq_e[1, i])): float(uniq_t[i]) for i in range(uniq_e.size(1))}
    assert pairs[(0, 1)] == 3.0
    assert pairs[(0, 2)] == 5.0


def test_degree_negative_sampling_avoids_existing_edges_and_self_loops():
    num_nodes = 5
    existing = {(0, 1), (1, 2), (2, 3)}
    degrees = torch.tensor([10, 10, 2, 1, 1], dtype=torch.float32).numpy()

    neg = sample_negative_edges_degree(
        num_nodes=num_nodes,
        num_samples=20,
        existing=existing,
        directed=False,
        degree_weights=degrees,
        seed=11,
    )

    assert neg.shape == (2, 20)
    for u, v in zip(neg[0].tolist(), neg[1].tolist()):
        assert u != v
        key = (min(int(u), int(v)), max(int(u), int(v)))
        assert key not in existing
