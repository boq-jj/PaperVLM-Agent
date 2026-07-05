"""Command-line entrypoint for parsing a paper PDF."""

import argparse
import sys
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap_project

bootstrap_project()

from src.pdf.parse_pdf import parse_pdf  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Parse a paper PDF.")
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file.")
    parser.add_argument(
        "--output-text-dir",
        default="data/extracted/text",
        help="Directory used to save the parsed JSON file.",
    )
    parser.add_argument(
        "--output-page-dir",
        default="data/extracted/pages",
        help="Directory used to save rendered page images.",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=2.0,
        help="Rendering zoom factor for page screenshots.",
    )
    return parser.parse_args()


def print_statistics(result: dict[str, Any], output_text_dir: str, output_page_dir: str) -> None:
    """Print PDF parsing statistics to the terminal."""
    paper_id = result["paper_id"]
    json_path = Path(output_text_dir) / f"{paper_id}.json"
    page_dir = Path(output_page_dir) / paper_id

    print("PDF parsing finished.")
    print(f"PDF file: {result['pdf_name']}")
    print(f"Pages: {result['num_pages']}")
    print(f"JSON saved to: {json_path.as_posix()}")
    print(f"Page images saved to: {page_dir.as_posix()}")
    print("Text length by page:")

    for page in result["pages"]:
        page_id = page["page_id"]
        text_length = len(page.get("text", ""))
        print(f"  page {page_id}: {text_length} chars")


def main() -> None:
    """Run the minimal PDF parsing pipeline."""
    args = parse_args()

    try:
        result = parse_pdf(
            pdf_path=args.pdf,
            output_text_dir=args.output_text_dir,
            output_page_dir=args.output_page_dir,
            zoom=args.zoom,
        )
    except Exception as exc:
        print(f"Failed to parse PDF: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print_statistics(
        result=result,
        output_text_dir=args.output_text_dir,
        output_page_dir=args.output_page_dir,
    )


if __name__ == "__main__":
    main()
