import subprocess
import sys


def test_cli_main_help():
    proc = subprocess.run([sys.executable, "-m", "src.cli.main", "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "run-all" in proc.stdout
    assert "validate-ontology" in proc.stdout


def test_cli_run_all_help_passthrough():
    proc = subprocess.run([sys.executable, "-m", "src.cli.main", "run-all", "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "End-to-end runner" in proc.stdout
