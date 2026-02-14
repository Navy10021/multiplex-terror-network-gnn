import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.multiplex_generator_v3 import GeneratorConfig, generate_multiplex_with_config
from src.ontology.validator import validate_manifest_dict_with_ontology


def test_generator_v3_manifest_passes_ontology_validation():
    cfg = GeneratorConfig(size=64, seed=7)
    manifest = generate_multiplex_with_config(cfg)

    report = validate_manifest_dict_with_ontology(
        manifest_dict=manifest,
        ontology_path="ontology/terror.ttl",
        shapes_path="ontology/constraints.shacl.ttl",
    )

    assert report["conforms"] is True
    assert report["constraints_checked"] == 8
