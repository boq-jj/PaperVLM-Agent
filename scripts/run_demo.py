"""Command-line entrypoint for launching the Gradio demo."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def _ensure_project_python() -> None:
    """Restart with the project virtual environment when available."""
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
    command = [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command, env=env))


_ensure_project_python()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.gradio_app import APP_CSS, build_demo  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Launch the PaperVLM-Agent Gradio demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser.parse_args()


def main() -> None:
    """Launch the local Gradio application."""
    args = parse_args()
    demo = build_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
