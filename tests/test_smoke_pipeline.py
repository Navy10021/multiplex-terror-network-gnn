import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Ensure repository root is on the path for src imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.build_pyg_dataset import build_pyg_data
from src.data.multiplex_generator_v2 import GeneratorConfig, generate_multiplex_with_config


def _write_manifest(tmp_path: Path, cfg: GeneratorConfig) -> Path:
    manifest = generate_multiplex_with_config(cfg)
    manifest_path = tmp_path / "multiplex.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f)
    return manifest_path


def test_generate_and_build_pyg_roundtrip(tmp_path: Path):
    cfg = GeneratorConfig(
        size=120,
        seed=42,
        finance_structure_strength=0.8,
        comm_structure_strength=0.8,
        comm_randomness=0.1,
        hvt_ratio=0.1,
        op_num_cells=10,
        op_cell_size=3,
    )

    manifest_path = _write_manifest(tmp_path, cfg)

    data = build_pyg_data(
        manifest_path=str(manifest_path),
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )

    num_nodes = cfg.size

    assert data.x.size(0) == num_nodes
    assert data.edge_index.size(1) > 0
    assert data.y_role.numel() == num_nodes
    assert data.y_hvt.shape[0] == num_nodes
    assert data.importance_score.shape[0] == num_nodes

    # Ensure HVT labels exist and train/val/test masks cover all nodes
    assert data.y_hvt.sum() > 0
    mask_total = (
        data.train_mask.sum().item()
        + data.val_mask.sum().item()
        + data.test_mask.sum().item()
    )
    assert mask_total == num_nodes

    # Train/val/test splits should roughly follow the provided ratios
    train_ratio = data.train_mask.sum().item() / num_nodes
    val_ratio = data.val_mask.sum().item() / num_nodes
    test_ratio = data.test_mask.sum().item() / num_nodes

    assert abs(train_ratio - 0.6) < 0.1
    assert abs(val_ratio - 0.2) < 0.1
    assert abs(test_ratio - 0.2) < 0.1

    # Sanity check basic tensor types
    assert isinstance(data.edge_index, torch.Tensor)
    assert isinstance(data.edge_type, torch.Tensor)
    assert isinstance(data.edge_attr, torch.Tensor)
    assert isinstance(data.importance_score, torch.Tensor)


def test_invalid_split_ratios_raise(tmp_path: Path):
    cfg = GeneratorConfig(
        size=60,
        seed=7,
        finance_structure_strength=0.7,
        comm_structure_strength=0.7,
        comm_randomness=0.2,
        hvt_ratio=0.05,
        op_num_cells=5,
        op_cell_size=3,
    )

    manifest_path = _write_manifest(tmp_path, cfg)

    with pytest.raises(ValueError):
        build_pyg_data(
            manifest_path=str(manifest_path),
            train_ratio=0.5,
            val_ratio=0.6,
            test_ratio=0.1,
        )

    with pytest.raises(ValueError):
        build_pyg_data(
            manifest_path=str(manifest_path),
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=-0.1,
        )
