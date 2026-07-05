"""Shared bootstrap helpers for command-line scripts."""

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def configure_stdio() -> None:
    """Configure terminal output encoding when the runtime supports it."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def bootstrap_project(reexec_venv: bool = False) -> Path:
    """Prepare stdio, optional venv re-exec, and project import path."""
    configure_stdio()
    if reexec_venv:
        ensure_project_python()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    return PROJECT_ROOT


def ensure_project_python() -> None:
    """Restart the current script with the project virtual environment if available."""
    if not PROJECT_VENV_PYTHON.exists():
        return

    current_python = Path(sys.executable).resolve()
    project_python = PROJECT_VENV_PYTHON.resolve()
    if current_python == project_python:
        return
    if os.environ.get("PAPERVLM_SKIP_PYTHON_REEXEC") == "1":
        return

    env = os.environ.copy()
    env["PAPERVLM_SKIP_PYTHON_REEXEC"] = "1"
    command = [str(project_python), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command, env=env))


def default_bge_model(default_model: str) -> str:
    """Prefer the local mirrored BGE model if available."""
    candidates = [
        PROJECT_ROOT / "models" / "bge-small-en-v1.5-hf-mirror",
        PROJECT_ROOT / "models" / "bge-small-en-v1.5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.relative_to(PROJECT_ROOT))
    return default_model
