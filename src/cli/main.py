from __future__ import annotations

"""Unified CLI entrypoint for multiplex-terror-network-gnn."""

import subprocess
import sys

COMMANDS = {
    "run-all": "src.run_all",
    "validate-ontology": "src.cli.validate_ontology",
}


def main() -> int:
    """Dispatch subcommands to module-level CLIs."""
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: multiplex-gnn {run-all,validate-ontology} [args...]")
        print("\nCommands:")
        print("  run-all            Run full pipeline (src.run_all)")
        print("  validate-ontology  Validate ontology (src.cli.validate_ontology)")
        return 0

    cmd = argv[0]
    module = COMMANDS.get(cmd)
    if module is None:
        print(f"[x] Unknown command: {cmd}")
        print("Use --help to see available commands.")
        return 2

    rest = argv[1:]
    return subprocess.call([sys.executable, "-m", module, *rest])


if __name__ == "__main__":
    raise SystemExit(main())
