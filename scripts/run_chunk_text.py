"""Command-line entrypoint for chunking parsed paper text."""

import argparse
import sys
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap_project

bootstrap_project()

from src.rag.chunk_text import chunk_parsed_pdf  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Chunk parsed PDF text.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the parsed PDF JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/extracted/chunks",
        help="Directory used to save chunk JSON files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Maximum number of characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Number of overlapping characters between chunks.",
    )
    return parser.parse_args()


def _build_output_path(output_dir: str, paper_id: str) -> Path:
    """Build the chunk JSON output path."""
    return Path(output_dir) / f"{paper_id}_chunks.json"


def _preview_text(text: str, max_chars: int = 100) -> str:
    """Create a one-line text preview."""
    preview = text.replace("\r", " ").replace("\n", " ")
    preview = " ".join(preview.split())
    return preview[:max_chars]


def print_statistics(chunks_data: dict[str, Any], output_dir: str) -> None:
    """Print chunking statistics to the terminal."""
    paper_id = chunks_data["paper_id"]
    output_path = _build_output_path(output_dir, paper_id)

    print("Text chunking finished.")
    print(f"paper_id: {paper_id}")
    print(f"pdf_name: {chunks_data['pdf_name']}")
    print(f"num_pages: {chunks_data['num_pages']}")
    print(f"num_chunks: {chunks_data['num_chunks']}")
    print(f"chunks JSON saved to: {output_path.as_posix()}")
    print("First 5 chunks:")

    for chunk in chunks_data["chunks"][:5]:
        text = chunk.get("text", "")
        print(
            f"  {chunk['chunk_id']} | "
            f"page {chunk['page_id']} | "
            f"{len(text)} chars | "
            f"{_preview_text(text)}"
        )


def main() -> None:
    """Run parsed PDF text chunking."""
    args = parse_args()

    try:
        chunks_data = chunk_parsed_pdf(
            input_json_path=args.input,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except Exception as exc:
        print(f"Failed to chunk parsed PDF text: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print_statistics(chunks_data, args.output_dir)


if __name__ == "__main__":
    main()
