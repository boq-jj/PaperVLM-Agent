"""Command-line entrypoint for building a FAISS retrieval index."""

import argparse
import sys
from typing import Any

from _bootstrap import bootstrap_project

bootstrap_project()

from src.rag.build_index import DEFAULT_EMBEDDING_MODEL, build_index_from_chunks  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build a FAISS index from paper chunks.")
    parser.add_argument(
        "--chunks",
        required=True,
        help="Path to the chunks JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/extracted/faiss_index",
        help="Directory used to save the FAISS index and metadata.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer embedding model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size.",
    )
    return parser.parse_args()


def print_statistics(stats: dict[str, Any]) -> None:
    """Print index building statistics."""
    print("FAISS index building finished.")
    print(f"paper_id: {stats['paper_id']}")
    print(f"pdf_name: {stats['pdf_name']}")
    print(f"num_chunks: {stats['num_chunks']}")
    print(f"embedding_dim: {stats['embedding_dim']}")
    print(f"index_path: {stats['index_path']}")
    print(f"metadata_path: {stats['metadata_path']}")


def main() -> None:
    """Build the retrieval index from chunk JSON."""
    args = parse_args()

    try:
        stats = build_index_from_chunks(
            chunks_path=args.chunks,
            output_dir=args.output_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(f"Failed to build FAISS index: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print_statistics(stats)


if __name__ == "__main__":
    main()
