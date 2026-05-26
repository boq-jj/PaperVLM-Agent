"""Minimal PDF parsing utilities based on PyMuPDF."""

import json
from pathlib import Path
from typing import Any

import fitz


def load_pdf(pdf_path: str) -> fitz.Document:
    """Load a PDF file with PyMuPDF.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        An opened ``fitz.Document`` object.

    Raises:
        FileNotFoundError: If the PDF path does not exist.
        ValueError: If the path is invalid or the PDF requires a password.
        RuntimeError: If PyMuPDF fails to open the file.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise ValueError(f"PDF path is not a file: {path}")

    try:
        document = fitz.open(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {path}") from exc

    if document.needs_pass:
        document.close()
        raise ValueError(f"Password-protected PDFs are not supported: {path}")

    return document


def render_page_to_image(page: fitz.Page, output_path: str, zoom: float = 2.0) -> Path:
    """Render one PDF page to a PNG image.

    Args:
        page: PyMuPDF page object.
        output_path: Output PNG path.
        zoom: Rendering zoom factor. Larger values produce clearer images.

    Returns:
        Path to the saved PNG file.

    Raises:
        RuntimeError: If rendering or saving fails.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to render page image: {path}") from exc

    return path


def extract_text_from_page(page: fitz.Page) -> str:
    """Extract text from one PDF page.

    Args:
        page: PyMuPDF page object.

    Returns:
        Extracted page text with leading and trailing whitespace removed.
    """
    try:
        text = page.get_text("text")
    except Exception as exc:
        raise RuntimeError("Failed to extract text from PDF page.") from exc

    return (text or "").strip()


def save_json(data: dict[str, Any], output_path: str) -> Path:
    """Save a dictionary as a UTF-8 JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    except Exception as exc:
        raise RuntimeError(f"Failed to save JSON file: {path}") from exc

    return path


def parse_pdf(
    pdf_path: str,
    output_text_dir: str = "data/extracted/text",
    output_page_dir: str = "data/extracted/pages",
    zoom: float = 2.0,
) -> dict[str, Any]:
    """Parse a PDF into page text, page images, and a JSON metadata file.

    Args:
        pdf_path: Path to the input PDF file.
        output_text_dir: Directory used to save the parsed JSON file.
        output_page_dir: Directory used to save rendered page images.
        zoom: Rendering zoom factor for page screenshots.

    Returns:
        Parsed PDF result dictionary.
    """
    pdf_file = Path(pdf_path)
    paper_id = pdf_file.stem
    page_output_dir = Path(output_page_dir) / paper_id
    json_output_path = Path(output_text_dir) / f"{paper_id}.json"

    document = load_pdf(str(pdf_file))
    try:
        pages: list[dict[str, Any]] = []
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            page_id = page_index + 1
            page_image_path = page_output_dir / f"page_{page_id:03d}.png"

            text = extract_text_from_page(page)
            render_page_to_image(page, str(page_image_path), zoom=zoom)

            pages.append(
                {
                    "page_id": page_id,
                    "text": text,
                    "page_image": page_image_path.as_posix(),
                }
            )

        result: dict[str, Any] = {
            "pdf_name": pdf_file.name,
            "paper_id": paper_id,
            "num_pages": document.page_count,
            "pages": pages,
        }
        save_json(result, str(json_output_path))
        return result
    finally:
        document.close()
