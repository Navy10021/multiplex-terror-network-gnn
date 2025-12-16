from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional


def _sha256_file(path: str) -> Optional[str]:
    if path is None or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(config_path: Optional[str]) -> Optional[str]:
    return _sha256_file(config_path)


def git_commit_hash(short: bool = False) -> Optional[str]:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        return commit[:7] if short else commit
    except Exception:
        return None


def command_line() -> str:
    return " ".join(sys.argv)


def build_run_id(cfg_hash: Optional[str], seed: int, prefix: str = "run", timestamp: Optional[str] = None) -> str:
    ts = timestamp or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    suffix = (cfg_hash or "nocfg")[:8]
    return f"{prefix}_{ts}_{suffix}_seed{seed}"


def build_artifact_dir(out_root: str, config_path: Optional[str], seed: int, prefix: str = "run") -> str:
    cfg_hash = config_hash(config_path)
    run_id = build_run_id(cfg_hash, seed=seed, prefix=prefix)
    return os.path.join(out_root, run_id)


def collect_run_metadata(
    *,
    out_dir: str,
    config_path: Optional[str],
    seed: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg_hash = config_hash(config_path)
    meta: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "config_path": os.path.abspath(config_path) if config_path else None,
        "config_sha256": cfg_hash,
        "git_commit": git_commit_hash(),
        "command": command_line(),
        "out_dir": os.path.abspath(out_dir),
    }
    if extra:
        meta.update(extra)
    return meta


def write_run_metadata(out_dir: str, metadata: Dict[str, Any], filename: str = "run_metadata.json") -> str:
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return path
