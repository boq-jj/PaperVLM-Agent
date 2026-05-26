"""Command-line entrypoint for text RAG evaluation."""

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

from src.eval.evaluate_text_rag import evaluate_examples, load_jsonl, save_json  # noqa: E402


def default_retriever_model() -> str:
    """Prefer local fallback BGE model if available."""
    local_model = PROJECT_ROOT / "models" / "bge-small-en-v1.5-hf-mirror"
    if local_model.exists():
        return str(local_model.relative_to(PROJECT_ROOT))
    return "BAAI/bge-small-en-v1.5"


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
        output = evaluate_examples(
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
