"""Command-line entrypoint for text RAG question answering."""

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
    """Restart this script with the project virtual environment when available."""
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

from src.agent.answer import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_RETRIEVER_MODEL,
    ask_with_rag,
)


def default_retriever_model() -> str:
    """Prefer the local downloaded embedding model if it exists."""
    local_model = PROJECT_ROOT / "models" / "bge-small-en-v1.5"
    if local_model.exists():
        return str(local_model.relative_to(PROJECT_ROOT))
    return DEFAULT_RETRIEVER_MODEL


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Ask a text RAG question about a paper.")
    parser.add_argument("--paper-id", required=True, help="Paper ID to query.")
    parser.add_argument("--query", required=True, help="Question to answer.")
    parser.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help="Directory containing FAISS index and metadata files.",
    )
    parser.add_argument(
        "--retriever-model-name",
        default=default_retriever_model(),
        help="SentenceTransformer retriever model name or local path.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["qwen-vl", "mock"],
        default="qwen-vl",
        help="LLM backend to use.",
    )
    parser.add_argument(
        "--llm-model-name",
        default=DEFAULT_LLM_MODEL,
        help="LLM model name.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=DEFAULT_LLM_BASE_URL,
        help="OpenAI-compatible LLM base URL.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=4000,
        help="Maximum context characters in the prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum answer token budget.",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature.")
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the final RAG prompt.",
    )
    return parser.parse_args()


def print_result(result: dict[str, Any], show_prompt: bool = False) -> None:
    """Print RAG QA result to the terminal."""
    print("Question:")
    print(result["question"])
    print(f"\nLLM backend: {result['llm_backend']}")
    print(f"LLM model: {result['llm_model_name']}")

    print("\nRetrieved chunks summary:")
    for chunk in result["retrieved_chunks"]:
        print(
            f"  rank={chunk['rank']} | "
            f"score={chunk['score']:.4f} | "
            f"chunk_id={chunk['chunk_id']} | "
            f"page_id={chunk['page_id']}"
        )
        print(f"  preview: {chunk['preview']}")

    print("\nAnswer:")
    print(result["answer"])

    if show_prompt:
        print("\nFinal Prompt:")
        print(result["prompt"])


def main() -> None:
    """Run text RAG question answering."""
    args = parse_args()

    try:
        result = ask_with_rag(
            question=args.query,
            paper_id=args.paper_id,
            index_dir=args.index_dir,
            retriever_model_name=args.retriever_model_name,
            llm_backend=args.llm_backend,
            llm_model_name=args.llm_model_name,
            llm_base_url=args.llm_base_url,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"RAG question answering failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print_result(result, show_prompt=args.show_prompt)


if __name__ == "__main__":
    main()
