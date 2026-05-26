"""Command-line entrypoint for visual page or figure RAG QA."""

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

from src.agent.vision_answer import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_RETRIEVER_MODEL,
    ask_with_visual_rag,
)


def default_retriever_model() -> str:
    """Prefer local fallback BGE model if available."""
    local_model = PROJECT_ROOT / "models" / "bge-small-en-v1.5-hf-mirror"
    if local_model.exists():
        return str(local_model.relative_to(PROJECT_ROOT))
    return DEFAULT_RETRIEVER_MODEL


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Ask a visual RAG question about a paper page.")
    parser.add_argument("--paper-id", required=True, help="Paper ID to query.")
    parser.add_argument("--query", required=True, help="Question to answer.")
    parser.add_argument("--image", default="", help="Optional image path.")
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--retriever-model-name", default=default_retriever_model())
    parser.add_argument("--llm-model-name", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=3000)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--show-prompt", action="store_true")
    return parser.parse_args()


def print_result(result: dict[str, Any], show_prompt: bool = False) -> None:
    """Print visual RAG QA result."""
    print("Question:")
    print(result["question"])
    print(f"\nPaper ID: {result['paper_id']}")
    print(f"Image path: {result['image_path']}")

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
        print("\nFinal Visual Prompt:")
        print(result["prompt"])


def main() -> None:
    """Run visual RAG question answering."""
    args = parse_args()
    try:
        result = ask_with_visual_rag(
            question=args.query,
            paper_id=args.paper_id,
            image_path=args.image,
            index_dir=args.index_dir,
            retriever_model_name=args.retriever_model_name,
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
        print(f"Visual RAG question answering failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print_result(result, show_prompt=args.show_prompt)


if __name__ == "__main__":
    main()
