"""Command-line entrypoint for text RAG question answering."""

import argparse
import sys
from typing import Any

from _bootstrap import bootstrap_project, default_bge_model

bootstrap_project(reexec_venv=True)

from src.agent.answer import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_RETRIEVER_MODEL,
    ask_with_rag,
)


def default_retriever_model() -> str:
    """Prefer the local downloaded embedding model if it exists."""
    return default_bge_model(DEFAULT_RETRIEVER_MODEL)


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
