import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pyg = pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.train_linkpred_layer_v3 import (
    assert_hard_region_negatives,
    assert_no_target_edge_leakage,
    build_train_graph_without_leakage,
)


def test_encoder_graph_excludes_heldout_target_edges():
    # edges: first 3 are target relation(=1), last is non-target relation(=0)
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 3]], dtype=torch.long)
    edge_type = torch.tensor([1, 1, 1, 0], dtype=torch.long)
    data = Data(x=torch.randn(4, 3), edge_index=edge_index, edge_type=edge_type)

    heldout = torch.tensor([[1], [2]], dtype=torch.long)
    train_graph = build_train_graph_without_leakage(
        data,
        target_rel=1,
        heldout_pos=heldout,
        directed=True,
    )

    # should not raise
    assert_no_target_edge_leakage(
        train_graph,
        target_rel=1,
        heldout_pos=heldout,
        directed=True,
    )


def test_hard_region_negative_constraint_assertion():
    regions = torch.tensor([0, 0, 1, 1], dtype=torch.long).numpy()

    valid_neg = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    assert_hard_region_negatives(valid_neg, regions)

    invalid_neg = torch.tensor([[0], [2]], dtype=torch.long)
    with pytest.raises(AssertionError):
        assert_hard_region_negatives(invalid_neg, regions)
