from __future__ import annotations

from pathlib import Path
from typing import Dict


def ensure_ontology_assets(ontology_path: str, shapes_path: str) -> Dict[str, str]:
    paths = {
        "ontology": Path(ontology_path),
        "shapes": Path(shapes_path),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        missing_desc = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise FileNotFoundError(f"Missing ontology assets: {missing_desc}")

    return {name: str(path.resolve()) for name, path in paths.items()}
