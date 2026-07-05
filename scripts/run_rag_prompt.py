"""Command-line entrypoint for building a RAG prompt from retrieved chunks."""

import argparse
import sys
from pathlib import Path

from _bootstrap import bootstrap_project

bootstrap_project()

from src.agent.answer import build_rag_prompt, summarize_retrieval_results  # noqa: E402
from src.rag.retrieve import DEFAULT_RETRIEVER_MODEL, retrieve  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build a RAG prompt from retrieved chunks.")
    parser.add_argument(
        "--paper-id",
        required=True,
        help="Paper ID used to locate the FAISS index and metadata.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Question used for retrieval and prompt construction.",
    )
    parser.add_argument(
        "--index-dir",
        default="data/extracted/faiss_index",
        help="Directory containing FAISS indexes and metadata files.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_RETRIEVER_MODEL,
        help="SentenceTransformer embedding model name or local model path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks to include.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=4000,
        help="Maximum number of context characters in the final prompt.",
    )
    parser.add_argument(
        "--save-prompt",
        default="",
        help="Optional path for saving the final prompt as a text file.",
    )
    return parser.parse_args()


def save_prompt(prompt: str, output_path: str) -> Path:
    """Save the generated prompt to a UTF-8 text file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")
    return path


def print_summary(question: str, summaries: list[dict], prompt: str) -> None:
    """Print question, retrieval summary, and final prompt."""
    print("Question:")
    print(question)
    print("\nRetrieved chunks summary:")

    for item in summaries:
        print(
            f"  rank={item['rank']} | "
            f"score={item['score']:.4f} | "
            f"chunk_id={item['chunk_id']} | "
            f"page_id={item['page_id']}"
        )
        print(f"  preview: {item['preview']}")

    print("\nFinal prompt:")
    print(prompt)


def main() -> None:
    """Retrieve chunks and build a RAG prompt."""
    args = parse_args()
    index_dir = Path(args.index_dir)
    index_path = index_dir / f"{args.paper_id}.index"
    metadata_path = index_dir / f"{args.paper_id}_metadata.json"

    try:
        retrieved_chunks = retrieve(
            query=args.query,
            index_path=str(index_path),
            metadata_path=str(metadata_path),
            model_name=args.model_name,
            top_k=args.top_k,
        )
        prompt = build_rag_prompt(
            question=args.query,
            retrieved_chunks=retrieved_chunks,
            max_context_chars=args.max_context_chars,
        )
    except Exception as exc:
        print(f"Failed to build RAG prompt: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    summaries = summarize_retrieval_results(retrieved_chunks)
    print_summary(args.query, summaries, prompt)

    if args.save_prompt:
        try:
            saved_path = save_prompt(prompt, args.save_prompt)
        except Exception as exc:
            print(f"Failed to save prompt: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        print(f"\nPrompt saved to: {saved_path.as_posix()}")


if __name__ == "__main__":
    main()
