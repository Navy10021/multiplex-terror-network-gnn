import subprocess
import sys

import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.train_multitask_gnn_v3 import compute_ontology_regularization_losses


def test_train_multitask_help_includes_ontology_flags():
    proc = subprocess.run(
        [sys.executable, "-m", "src.models.train_multitask_gnn_v3", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--ontology_loss" in proc.stdout
    assert "--ontology_loss_role_weight" in proc.stdout
    assert "--ontology_loss_transitivity_weight" in proc.stdout
    assert "--ontology_loss_temporal_weight" in proc.stdout


def test_compute_ontology_regularization_losses_returns_expected_keys():
    role_logits = torch.tensor([
        [4.0, -1.0],
        [-1.0, 4.0],
        [4.0, -1.0],
    ])
    edge_index = torch.tensor([[0, 1, 1], [1, 2, 0]], dtype=torch.long)
    edge_type = torch.tensor([0, 0, 0], dtype=torch.long)

    # all transitive + time_ordered for this synthetic relation
    edge_ontology_attr = torch.tensor([
        [1, 1, 0, 1, 1, 0.1, 1.0, 0.0],
        [1, 1, 0, 1, 1, 0.1, 1.0, 0.0],
        [1, 1, 0, 1, 1, 0.1, 1.0, 0.0],
    ], dtype=torch.float32)

    # relation 0 allows role0 -> role1 only
    role_mask = torch.zeros((1, 2, 2), dtype=torch.float32)
    role_mask[0, 0, 1] = 1.0

    losses = compute_ontology_regularization_losses(
        role_logits=role_logits,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_ontology_attr=edge_ontology_attr,
        role_compatibility_mask=role_mask,
        max_triplets=32,
    )

    assert set(losses.keys()) == {
        "role_relation_compatibility",
        "hierarchy_transitivity",
        "temporal_ordering",
    }
    for value in losses.values():
        assert torch.is_tensor(value)
        assert value.dim() == 0
        assert torch.isfinite(value).item()
