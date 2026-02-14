import subprocess
import sys


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

