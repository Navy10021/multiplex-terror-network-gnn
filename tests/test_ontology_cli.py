import json
import subprocess
import sys
from pathlib import Path


def _manifest(tmp_path: Path) -> Path:
    data = {
        "meta": {"num_nodes": 2, "seed": 1, "generator": "test", "config": {}},
        "nodes": [
            {"id": 0, "role": "leader", "region": "x", "group": "g"},
            {"id": 1, "role": "support", "region": "x", "group": "g"},
        ],
        "layers": {"hierarchy": {"directed": True, "edges": [{"source": 0, "target": 1}]}},
        "events": [],
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_validate_ontology_cli_success(tmp_path: Path):
    manifest = _manifest(tmp_path)
    cmd = [
        sys.executable,
        "-m",
        "src.cli.validate_ontology",
        "--manifest",
        str(manifest),
        "--ontology",
        "ontology/terror.ttl",
        "--shapes",
        "ontology/constraints.shacl.ttl",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "Ontology validation succeeded" in result.stdout


def test_validate_ontology_cli_missing_file_fails(tmp_path: Path):
    manifest = _manifest(tmp_path)
    cmd = [
        sys.executable,
        "-m",
        "src.cli.validate_ontology",
        "--manifest",
        str(manifest),
        "--ontology",
        "ontology/not-found.ttl",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "ontology file not found" in result.stderr
