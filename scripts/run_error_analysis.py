"""Command-line entrypoint for text RAG error analysis."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


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

from src.eval.error_analysis import run_error_analysis  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate error analysis reports from text RAG evaluation results."
    )
    parser.add_argument(
        "--eval-result",
        required=True,
        help="Path to run_eval_text_rag.py result JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eval/analysis",
        help="Directory for generated analysis artifacts.",
    )
    return parser.parse_args()


def print_summary(output: dict[str, Any]) -> None:
    """Print concise error analysis summary."""
    summary = output["summary"]
    print("Error analysis summary:")
    print(f"total: {summary['total']}")
    print(f"page_hit_rate: {summary['page_hit_rate']:.4f}")
    print(f"average_top_score: {summary['average_top_score']:.4f}")
    print("error_label_counts:")
    for label, count in summary["error_label_counts"].items():
        print(f"  {label}: {count}")

    print("Output files:")
    for name, path in output["output_paths"].items():
        print(f"  {name}: {Path(path).as_posix()}")


def main() -> None:
    """Run error analysis without re-running evaluation."""
    args = parse_args()
    try:
        output = run_error_analysis(
            eval_result_path=args.eval_result,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error analysis failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print_summary(output)


if __name__ == "__main__":
    main()
