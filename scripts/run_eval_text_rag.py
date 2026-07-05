"""Command-line entrypoint for text RAG evaluation."""

import argparse
import sys
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap_project, default_bge_model

bootstrap_project(reexec_venv=True)

from src.eval.evaluate_text_rag import (  # noqa: E402
    evaluate_text_rag_examples,
    load_jsonl,
    save_json,
)


def default_retriever_model() -> str:
    """Prefer local fallback BGE model if available."""
    return default_bge_model("BAAI/bge-small-en-v1.5")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate text RAG on a JSONL QA set.")
    parser.add_argument("--eval-file", default="data/eval/example_text_qa.jsonl")
    parser.add_argument("--paper-id", default="")
    parser.add_argument("--output-dir", default="data/eval/results")
    parser.add_argument("--index-dir", default="data/extracted/faiss_index")
    parser.add_argument("--embedding-backend", default="local-bge")
    parser.add_argument("--retriever-model-name", default=default_retriever_model())
    parser.add_argument("--llm-backend", choices=["qwen-vl", "mock"], default="qwen-vl")
    parser.add_argument("--llm-model-name", default="qwen3-vl-flash")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def build_output_path(eval_file: str, output_dir: str) -> Path:
    """Build output path from eval file name."""
    eval_stem = Path(eval_file).stem
    return Path(output_dir) / f"{eval_stem}_results.json"


def print_summary(summary: dict[str, Any]) -> None:
    """Print evaluation summary."""
    print("Evaluation summary:")
    print(f"total: {summary['total']}")
    print(f"success: {summary['success']}")
    print(f"failed: {summary['failed']}")
    print(f"page_hit_rate: {summary['page_hit_rate']:.4f}")
    print("by_question_type:")
    for question_type, stats in summary["by_question_type"].items():
        print(
            f"  {question_type}: "
            f"success={stats['success']} "
            f"page_hit_count={stats['page_hit_count']} "
            f"page_hit_rate={stats['page_hit_rate']:.4f}"
        )


def main() -> None:
    """Run text RAG evaluation."""
    args = parse_args()
    try:
        examples = load_jsonl(args.eval_file)
        output = evaluate_text_rag_examples(
            examples=examples,
            paper_id=args.paper_id or None,
            index_dir=args.index_dir,
            embedding_backend=args.embedding_backend,
            retriever_model_name=args.retriever_model_name,
            llm_backend=args.llm_backend,
            llm_model_name=args.llm_model_name,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        output_path = build_output_path(args.eval_file, args.output_dir)
        save_json(output, str(output_path))
    except Exception as exc:
        print(f"Text RAG evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print_summary(output["summary"])
    print(f"Results saved to: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
