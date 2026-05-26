"""Utilities for extracting candidate figures from PDF pages."""

from pathlib import Path
from typing import Any


def extract_figures_from_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    min_width: int = 120,
    min_height: int = 120,
) -> list[dict[str, Any]]:
    """Extract candidate figures from a PDF.

    Args:
        pdf_path: Path to the input PDF.
        output_dir: Directory used to save extracted figures.
        min_width: Minimum figure width in pixels.
        min_height: Minimum figure height in pixels.

    Returns:
        A list of figure metadata dictionaries.
    """
    raise NotImplementedError("Figure extraction will be implemented later.")
