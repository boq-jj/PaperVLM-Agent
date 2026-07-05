"""Prepare a small official ChartQA Hugging Face subset for evaluation."""

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from _bootstrap import configure_stdio

configure_stdio()


QUESTION_FIELDS = ("question", "query")
ANSWER_FIELDS = ("answer", "label", "answers")
IMAGE_FIELDS = ("image",)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export a small Hugging Face ChartQA subset to the project JSONL format."
    )
    parser.add_argument("--dataset-name", default="HuggingFaceM4/ChartQA")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output-jsonl", default="data/eval/chartqa_hf_sample.jsonl")
    parser.add_argument("--output-image-dir", default="data/datasets/chartqa/images")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_hf_split(dataset_name: str, split: str) -> Any:
    """Load one split from a Hugging Face dataset with clear failure guidance."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        return load_dataset(dataset_name, split=split)
    except Exception as exc:
        raise RuntimeError(
            "\n".join(
                [
                    f"Failed to load Hugging Face dataset: {dataset_name} split={split}",
                    "Please check your network connection.",
                    "If Hugging Face is slow or blocked, try setting HF_ENDPOINT, for example:",
                    '  $env:HF_ENDPOINT="https://hf-mirror.com"',
                    "You can also use the synthetic ChartQA-style sample first:",
                    "  python scripts\\create_chartqa_sample_images.py",
                    "  python scripts\\run_eval_chartqa.py "
                    "--eval-file data\\eval\\chartqa_sample.jsonl",
                ]
            )
        ) from exc


def shuffled_examples(dataset: Any, seed: int) -> Iterable[dict[str, Any]]:
    """Return an iterable over dataset examples, shuffled when supported."""
    if hasattr(dataset, "shuffle"):
        try:
            dataset = dataset.shuffle(seed=seed)
        except TypeError:
            dataset = dataset.shuffle()
        except Exception:
            pass

    for example in dataset:
        if isinstance(example, dict):
            yield example


def get_first_present(example: dict[str, Any], field_names: tuple[str, ...]) -> Any:
    """Return the first present non-empty field value from an example."""
    for field_name in field_names:
        if field_name not in example:
            continue
        value = example[field_name]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, list) and not value:
            continue
        return value
    return None


def normalize_answer(value: Any) -> str:
    """Normalize an answer field to a string."""
    if isinstance(value, list):
        if not value:
            return ""
        value = value[0]
    return str(value).strip()


def load_image(value: Any) -> Image.Image | None:
    """Load an image from PIL, bytes, path, or a Hugging Face image dict."""
    if isinstance(value, Image.Image):
        return value

    if isinstance(value, dict):
        if value.get("bytes"):
            return image_from_bytes(value["bytes"])
        if value.get("path"):
            return image_from_path(value["path"])
        return None

    if isinstance(value, (bytes, bytearray)):
        return image_from_bytes(value)

    if isinstance(value, (str, Path)):
        return image_from_path(value)

    return None


def image_from_bytes(value: bytes | bytearray) -> Image.Image | None:
    """Load an image from bytes."""
    try:
        return Image.open(io.BytesIO(value))
    except Exception:
        return None


def image_from_path(value: str | Path) -> Image.Image | None:
    """Load an image from a local path."""
    path = Path(value)
    if not path.exists() or not path.is_file():
        return None

    try:
        return Image.open(path)
    except Exception:
        return None


def save_image(image: Image.Image, path: Path) -> Path:
    """Save an image as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG")
    return path


def export_chartqa_sample(
    dataset: Any,
    dataset_name: str,
    split: str,
    num_samples: int,
    output_jsonl: Path,
    output_image_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Export ChartQA examples and images to the project evaluation format."""
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than 0.")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    skipped_count = 0
    records: list[dict[str, Any]] = []

    for example in shuffled_examples(dataset, seed=seed):
        if saved_count >= num_samples:
            break

        question = get_first_present(example, QUESTION_FIELDS)
        answer = normalize_answer(get_first_present(example, ANSWER_FIELDS))
        image_value = get_first_present(example, IMAGE_FIELDS)
        image = load_image(image_value)

        if question is None or not str(question).strip() or not answer or image is None:
            skipped_count += 1
            continue

        sample_id = f"chartqa_hf_{saved_count + 1:06d}"
        image_path = output_image_dir / f"{sample_id}.png"
        save_image(image, image_path)

        records.append(
            {
                "id": sample_id,
                "image_path": image_path.as_posix(),
                "question": str(question).strip(),
                "answer": answer,
                "question_type": "unknown",
                "source": "official_chartqa_hf",
                "dataset_name": dataset_name,
                "split": split,
            }
        )
        saved_count += 1

    with output_jsonl.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "dataset_name": dataset_name,
        "split": split,
        "requested_num_samples": num_samples,
        "saved_count": saved_count,
        "skipped_count": skipped_count,
        "output_jsonl": output_jsonl.as_posix(),
        "output_image_dir": output_image_dir.as_posix(),
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Print export summary."""
    print(f"dataset_name: {summary['dataset_name']}")
    print(f"split: {summary['split']}")
    print(f"requested num_samples: {summary['requested_num_samples']}")
    print(f"saved count: {summary['saved_count']}")
    print(f"skipped count: {summary['skipped_count']}")
    print(f"output_jsonl: {summary['output_jsonl']}")
    print(f"output_image_dir: {summary['output_image_dir']}")


def main() -> None:
    """Prepare a small official ChartQA sample from Hugging Face."""
    args = parse_args()
    try:
        dataset = load_hf_split(dataset_name=args.dataset_name, split=args.split)
        summary = export_chartqa_sample(
            dataset=dataset,
            dataset_name=args.dataset_name,
            split=args.split,
            num_samples=args.num_samples,
            output_jsonl=Path(args.output_jsonl),
            output_image_dir=Path(args.output_image_dir),
            seed=args.seed,
        )
    except Exception as exc:
        print(f"Failed to prepare ChartQA HF sample: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print_summary(summary)


if __name__ == "__main__":
    main()
