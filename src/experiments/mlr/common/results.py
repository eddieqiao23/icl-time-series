"""Reproducibility metadata and stable result serialization."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=False,
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def build_manifest(*, repo_root: Path, experiment: str, command: list[str],
                   parameters: dict[str, Any], checkpoints: list[dict[str, Any]],
                   seeds: dict[str, Any], outputs: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": experiment,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repo_root),
        "command": command,
        "parameters": parameters,
        "seeds": seeds,
        "checkpoints": checkpoints,
        "outputs": outputs,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            ),
        },
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

