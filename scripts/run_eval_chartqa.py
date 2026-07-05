"""Command-line entrypoint for ChartQA evaluation."""

import argparse
import sys
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap_project

bootstrap_project(reexec_venv=True)

from src.eval.evaluate_chartqa import (  # noqa: E402
    evaluate_chartqa_examples,
    load_chartqa_jsonl,
)
from src.eval.io_utils import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate qwen-vl on a small ChartQA JSONL set.")
    parser.add_argument("--eval-file", default="data/eval/chartqa_sample.jsonl")
    parser.add_argument("--output-dir", default="data/eval/results")
    parser.add_argument("--llm-model-name", default="qwen3-vl-flash")
    parser.add_argument(
        "--llm-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--image-scale",
        type=float,
        default=1.0,
        help="Optional image upscaling factor before sending charts to the VLM.",
    )
    parser.add_argument(
        "--preprocessed-image-dir",
        default="data/eval/preprocessed/chartqa",
        help="Directory for preprocessed chart images.",
    )
    return parser.parse_args()


def build_output_path(eval_file: str, output_dir: str) -> Path:
    """Build output path from eval file name."""
    eval_stem = Path(eval_file).stem
    return Path(output_dir) / f"{eval_stem}_results.json"


def print_summary(summary: dict[str, Any]) -> None:
    """Print evaluation summary."""
    print("ChartQA evaluation summary:")
    print(f"total: {summary['total']}")
    print(f"success: {summary['success']}")
    print(f"failed: {summary['failed']}")
    print(f"exact_match_rate: {summary['exact_match_rate']:.4f}")
    print(f"relaxed_match_rate: {summary['relaxed_match_rate']:.4f}")
    print("by_question_type:")
    for question_type, stats in summary["by_question_type"].items():
        print(
            f"  {question_type}: "
            f"success={stats['success']} "
            f"exact={stats['exact_match_count']} "
            f"relaxed={stats['relaxed_match_count']} "
            f"relaxed_match_rate={stats['relaxed_match_rate']:.4f}"
        )


def main() -> None:
    """Run ChartQA evaluation."""
    args = parse_args()
    try:
        examples = load_chartqa_jsonl(args.eval_file)
        output = evaluate_chartqa_examples(
            examples=examples,
            llm_model_name=args.llm_model_name,
            llm_base_url=args.llm_base_url,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            image_scale=args.image_scale,
            preprocessed_image_dir=args.preprocessed_image_dir,
        )
        output_path = build_output_path(args.eval_file, args.output_dir)
        save_json(output, str(output_path))
    except Exception as exc:
        print(f"ChartQA evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print_summary(output["summary"])
    print(f"Results saved to: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
