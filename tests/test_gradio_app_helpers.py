from pathlib import Path

import pytest

from src.app.upload_utils import (
    coerce_non_negative_float,
    coerce_non_negative_int,
    coerce_positive_int,
    ensure_unique_path,
    safe_filename,
)


def test_safe_filename_removes_parent_paths_and_unsafe_chars() -> None:
    filename = safe_filename("../paper:figure?.pdf")

    assert filename == "paper_figure_.pdf"


def test_ensure_unique_path_appends_numeric_suffix() -> None:
    existing_path = Path(__file__).with_name("__tmp_paper.pdf")
    unique_path = Path(__file__).with_name("__tmp_paper_1.pdf")

    try:
        existing_path.write_text("old", encoding="utf-8")

        assert ensure_unique_path(existing_path) == unique_path
    finally:
        existing_path.unlink(missing_ok=True)
        unique_path.unlink(missing_ok=True)


def test_numeric_coercion_validates_ranges() -> None:
    assert coerce_positive_int("3", "top_k") == 3
    assert coerce_non_negative_int("0", "chunk_overlap") == 0
    assert coerce_non_negative_float("0.2", "temperature") == 0.2

    with pytest.raises(ValueError, match="top_k"):
        coerce_positive_int("0", "top_k")

    with pytest.raises(ValueError, match="chunk_overlap"):
        coerce_non_negative_int("-1", "chunk_overlap")

    with pytest.raises(ValueError, match="temperature"):
        coerce_non_negative_float("-0.1", "temperature")
