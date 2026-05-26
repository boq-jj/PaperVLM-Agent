"""Command-line entrypoint for retrieving relevant paper chunks."""

import argparse
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retrieve import DEFAULT_RETRIEVER_MODEL, retrieve  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Retrieve relevant chunks from a FAISS index.")
    parser.add_argument(
        "--paper-id",
        required=True,
        help="Paper ID used to locate the FAISS index and metadata.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Retrieval query.",
    )
    parser.add_argument(
        "--index-dir",
        default="data/extracted/faiss_index",
        help="Directory containing FAISS indexes and metadata files.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_RETRIEVER_MODEL,
        help="SentenceTransformer embedding model name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return.",
    )
    return parser.parse_args()


def _preview_text(text: str, max_chars: int = 300) -> str:
    """Create a compact one-line preview."""
    preview = text.replace("\r", " ").replace("\n", " ")
    preview = " ".join(preview.split())
    return preview[:max_chars]


def main() -> None:
    """Run FAISS retrieval for one query."""
    args = parse_args()
    index_dir = Path(args.index_dir)
    index_path = index_dir / f"{args.paper_id}.index"
    metadata_path = index_dir / f"{args.paper_id}_metadata.json"

    try:
        results = retrieve(
            query=args.query,
            index_path=str(index_path),
            metadata_path=str(metadata_path),
            model_name=args.model_name,
            top_k=args.top_k,
        )
    except Exception as exc:
        print(f"Failed to retrieve chunks: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Query: {args.query}")
    print(f"Top-{args.top_k} results:")
    for result in results:
        print(f"\nrank: {result['rank']}")
        print(f"score: {result['score']:.4f}")
        print(f"chunk_id: {result['chunk_id']}")
        print(f"page_id: {result['page_id']}")
        print(f"text preview: {_preview_text(result['text'])}")


if __name__ == "__main__":
    main()
