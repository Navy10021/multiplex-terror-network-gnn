from __future__ import annotations

import argparse
from pathlib import Path

from src.ontology.validator import OntologyValidationError, validate_manifest_with_ontology


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate manifest against ontology-backed constraints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--ontology", default="ontology/terror.ttl", help="Path to ontology TTL")
    parser.add_argument("--shapes", default="ontology/constraints.shacl.ttl", help="Path to SHACL constraints TTL")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    for label, path in (("ontology", args.ontology), ("shapes", args.shapes)):
        if not Path(path).exists():
            parser.error(f"{label} file not found: {path}")

    try:
        report = validate_manifest_with_ontology(
            manifest_path=args.manifest,
            ontology_path=args.ontology,
            shapes_path=args.shapes,
        )
    except OntologyValidationError as exc:
        parser.error(str(exc))

    print(f"Ontology validation succeeded: {args.manifest}")
    print(f"Checked constraints: {report['constraints_checked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
