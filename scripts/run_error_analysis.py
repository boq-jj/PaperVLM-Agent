"""Command-line entrypoint for evaluation error analysis."""

import argparse
import sys
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap_project

bootstrap_project(reexec_venv=True)

from src.eval.error_analysis import run_error_analysis  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate error analysis reports from evaluation result JSON files."
    )
    parser.add_argument(
        "--eval-result",
        required=True,
        help="Path to an evaluation result JSON.",
    )
    parser.add_argument(
        "--task-type",
        choices=["text_rag", "chartqa"],
        required=True,
        help="Evaluation task type.",
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
    if "page_hit_rate" in summary:
        print(f"page_hit_rate: {summary['page_hit_rate']:.4f}")
    if "exact_match_rate" in summary:
        print(f"exact_match_rate: {summary['exact_match_rate']:.4f}")
        print(f"relaxed_match_rate: {summary['relaxed_match_rate']:.4f}")
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
            task_type=args.task_type,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error analysis failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print_summary(output)


if __name__ == "__main__":
    main()
