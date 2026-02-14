import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.run_all import _resolve_ontology_mode


def test_run_all_help_executes():
    proc = subprocess.run(
        [sys.executable, "-m", "src.run_all", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "generate" in proc.stdout
    assert "--ontology" in proc.stdout
    assert "--ontology_constrained" in proc.stdout


def test_run_all_help_includes_ontology_mode():
    proc = subprocess.run(
        [sys.executable, "-m", "src.run_all", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--ontology_mode" in proc.stdout


def test_resolve_ontology_mode_presets():
    import argparse

    ns = argparse.Namespace(ontology_mode="strict", no_ontology_strict=False, ontology_constrained=False)
    assert _resolve_ontology_mode(ns) == (True, False)

    ns = argparse.Namespace(ontology_mode="constrained", no_ontology_strict=False, ontology_constrained=False)
    assert _resolve_ontology_mode(ns) == (True, True)

    ns = argparse.Namespace(ontology_mode="report_only", no_ontology_strict=False, ontology_constrained=False)
    assert _resolve_ontology_mode(ns) == (False, False)


def test_resolve_ontology_mode_legacy_flags_override():
    import argparse

    ns = argparse.Namespace(ontology_mode="strict", no_ontology_strict=True, ontology_constrained=False)
    assert _resolve_ontology_mode(ns) == (False, False)

    ns = argparse.Namespace(ontology_mode="report_only", no_ontology_strict=False, ontology_constrained=True)
    assert _resolve_ontology_mode(ns) == (True, True)
