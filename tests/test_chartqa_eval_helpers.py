from src.eval.evaluate_chartqa import (
    exact_match,
    extract_final_answer,
    prepare_chartqa_image,
    relaxed_match,
)


def test_extract_final_answer_removes_reasoning_and_markdown() -> None:
    prediction = "Reasoning text\n\nAnswer: **2.72** (in thousands)"

    assert extract_final_answer(prediction) == "2.72"


def test_exact_match_accepts_numeric_spacing() -> None:
    assert exact_match("910 987", "910987")


def test_exact_match_accepts_numeric_unit_suffix() -> None:
    assert exact_match("76.6 billion GBP", "76.6")


def test_exact_match_accepts_harmless_label_prefix() -> None:
    assert exact_match("18-29", "Ages 18-29")


def test_relaxed_match_uses_extracted_final_answer() -> None:
    prediction = "Female total minus male total is 2.72.\nAnswer: **2.72**"

    assert relaxed_match(prediction, "2.72")


def test_exact_match_rejects_different_numbers() -> None:
    assert not exact_match("160 million", "162")


def test_prepare_chartqa_image_upscales_to_output_dir() -> None:
    from pathlib import Path

    from PIL import Image

    tmp_dir = Path(__file__).with_name("__tmp_chartqa_images")
    source_path = tmp_dir / "source.png"
    output_dir = tmp_dir / "processed"

    try:
        tmp_dir.mkdir(exist_ok=True)
        Image.new("RGB", (10, 8), "white").save(source_path)

        output_path = Path(
            prepare_chartqa_image(
                image_path=str(source_path),
                image_scale=2.0,
                output_dir=str(output_dir),
            )
        )

        assert output_path.is_file()
        with Image.open(output_path) as image:
            assert image.size == (20, 16)
    finally:
        for path in sorted(tmp_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        tmp_dir.rmdir()
