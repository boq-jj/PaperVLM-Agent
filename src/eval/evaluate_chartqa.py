"""Small-scale ChartQA evaluation utilities."""

import re
import string
from pathlib import Path
from typing import Any

from src.eval.io_utils import load_jsonl, safe_rate as _safe_rate
from src.llm.qwen_vl_api import QwenVLAPILLM


DEFAULT_CHARTQA_PROMPT = (
    "Answer the chart question based on the image. Give a concise answer.\n"
    "Question: {question}\n"
    "Answer:"
)


def normalize_answer(answer: str) -> str:
    """Normalize an answer for exact and relaxed matching."""
    text = extract_final_answer(str(answer or "")).strip().lower()
    punctuation = string.punctuation + "，。！？；：“”‘’（）【】、"
    text = text.translate(str.maketrans("", "", punctuation))
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Return whether normalized prediction and ground truth are identical."""
    prediction_number = _parse_float(extract_final_answer(prediction))
    ground_truth_number = _parse_float(extract_final_answer(ground_truth))
    if prediction_number is not None and ground_truth_number is not None:
        return prediction_number == ground_truth_number

    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    return _labels_exact_match(normalized_prediction, normalized_ground_truth)


def relaxed_match(prediction: str, ground_truth: str) -> bool:
    """Return whether prediction and ground truth match with relaxed rules."""
    prediction_number = _parse_float(extract_final_answer(str(prediction or "")))
    ground_truth_number = _parse_float(extract_final_answer(str(ground_truth or "")))
    if prediction_number is not None and ground_truth_number is not None:
        tolerance = max(abs(ground_truth_number) * 0.05, 1e-12)
        return abs(prediction_number - ground_truth_number) <= tolerance

    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if not normalized_prediction or not normalized_ground_truth:
        return False

    return (
        normalized_ground_truth in normalized_prediction
        or normalized_prediction in normalized_ground_truth
    )


def extract_final_answer(prediction: str) -> str:
    """Extract a concise final answer from a model response."""
    text = str(prediction or "").strip()
    if not text:
        return ""

    text = text.replace("**", "").replace("__", "")
    answer_markers = [
        "final answer:",
        "answer:",
        "the answer is",
    ]
    lowered = text.lower()
    for marker in answer_markers:
        index = lowered.rfind(marker)
        if index >= 0:
            text = text[index + len(marker) :].strip()
            break

    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    if len(lines) == 1:
        text = lines[0]
    elif lines and len(lines[-1]) <= 80:
        text = lines[-1]

    text = text.strip().strip(".。")
    numeric = _parse_float(text)
    if numeric is not None:
        if numeric.is_integer():
            return str(int(numeric))
        return str(numeric)
    return text


def load_chartqa_jsonl(path: str) -> list[dict[str, Any]]:
    """Load a ChartQA JSONL file into a list of dictionaries."""
    return load_jsonl(path)


def prepare_chartqa_image(
    image_path: str,
    image_scale: float = 1.0,
    output_dir: str = "data/eval/preprocessed/chartqa",
) -> str:
    """Optionally upscale a chart image before sending it to the VLM."""
    if image_scale <= 0:
        raise ValueError("image_scale must be greater than 0.")
    if image_scale == 1.0:
        return image_path

    source_path = Path(image_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Image file not found: {source_path}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{source_path.stem}_scale{_scale_suffix(image_scale)}.png"
    if output_path.exists():
        return str(output_path)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image preprocessing.") from exc

    try:
        with Image.open(source_path) as image:
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            target_size = (
                max(1, int(round(width * image_scale))),
                max(1, int(round(height * image_scale))),
            )
            resampling = getattr(Image.Resampling, "LANCZOS", Image.BICUBIC)
            resized = rgb_image.resize(target_size, resampling)
            resized.save(output_path, format="PNG", optimize=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to preprocess chart image: {source_path}") from exc

    return str(output_path)


def evaluate_chartqa_examples(
    examples: list[dict[str, Any]],
    llm_model_name: str = "qwen3-vl-flash",
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    image_scale: float = 1.0,
    preprocessed_image_dir: str = "data/eval/preprocessed/chartqa",
) -> dict[str, Any]:
    """Evaluate qwen-vl chart question answering over a small JSONL set."""
    summary = _empty_summary(total=len(examples))
    type_stats: dict[str, dict[str, int]] = {}
    results: list[dict[str, Any]] = []
    llm: QwenVLAPILLM | None = None

    for example in examples:
        example_id = str(example.get("id", ""))
        image_path = str(example.get("image_path", "")).strip()
        question = str(example.get("question", "")).strip()
        ground_truth = str(example.get("answer", ""))
        question_type = str(example.get("question_type", "unknown") or "unknown")

        _ensure_type_stats(type_stats, question_type)
        type_stats[question_type]["total"] += 1

        record: dict[str, Any] = {
            "id": example_id,
            "image_path": image_path,
            "model_image_path": "",
            "question": question,
            "ground_truth": ground_truth,
            "raw_prediction": "",
            "prediction": "",
            "exact_match": False,
            "relaxed_match": False,
            "question_type": question_type,
            "error": "",
        }

        try:
            if not image_path:
                raise ValueError("image_path is empty.")
            if not Path(image_path).is_file():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            model_image_path = prepare_chartqa_image(
                image_path=image_path,
                image_scale=image_scale,
                output_dir=preprocessed_image_dir,
            )

            if llm is None:
                llm = QwenVLAPILLM(
                    model_name=llm_model_name,
                    base_url=llm_base_url,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    answer_language="en",
                )

            prompt = DEFAULT_CHARTQA_PROMPT.format(question=question)
            raw_prediction = llm.generate_with_image(prompt=prompt, image_path=model_image_path)
            prediction = extract_final_answer(raw_prediction)
            is_exact = exact_match(prediction, ground_truth)
            is_relaxed = relaxed_match(prediction, ground_truth)

            record.update(
                {
                    "model_image_path": model_image_path,
                    "raw_prediction": raw_prediction,
                    "prediction": prediction,
                    "exact_match": is_exact,
                    "relaxed_match": is_relaxed,
                }
            )
            summary["success"] += 1
            type_stats[question_type]["success"] += 1
            if is_exact:
                summary["exact_match_count"] += 1
                type_stats[question_type]["exact_match_count"] += 1
            if is_relaxed:
                summary["relaxed_match_count"] += 1
                type_stats[question_type]["relaxed_match_count"] += 1
        except Exception as exc:
            record["error"] = str(exc)
            summary["failed"] += 1
            type_stats[question_type]["failed"] += 1

        results.append(record)

    _finalize_summary(summary, type_stats)
    return {
        "summary": summary,
        "results": results,
    }


def _empty_summary(total: int) -> dict[str, Any]:
    """Create an initial ChartQA summary dictionary."""
    return {
        "total": total,
        "success": 0,
        "failed": 0,
        "exact_match_count": 0,
        "exact_match_rate": 0.0,
        "relaxed_match_count": 0,
        "relaxed_match_rate": 0.0,
        "by_question_type": {},
    }


def _ensure_type_stats(type_stats: dict[str, dict[str, int]], question_type: str) -> None:
    """Ensure per-question-type counters exist."""
    if question_type not in type_stats:
        type_stats[question_type] = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "exact_match_count": 0,
            "relaxed_match_count": 0,
        }


def _finalize_summary(
    summary: dict[str, Any],
    type_stats: dict[str, dict[str, int]],
) -> None:
    """Compute final ChartQA rates in-place."""
    success = summary["success"]
    summary["exact_match_rate"] = _safe_rate(summary["exact_match_count"], success)
    summary["relaxed_match_rate"] = _safe_rate(summary["relaxed_match_count"], success)

    by_type: dict[str, dict[str, float | int]] = {}
    for question_type, stats in type_stats.items():
        type_success = stats["success"]
        by_type[question_type] = {
            "total": stats["total"],
            "success": type_success,
            "failed": stats["failed"],
            "exact_match_count": stats["exact_match_count"],
            "exact_match_rate": _safe_rate(stats["exact_match_count"], type_success),
            "relaxed_match_count": stats["relaxed_match_count"],
            "relaxed_match_rate": _safe_rate(
                stats["relaxed_match_count"],
                type_success,
            ),
        }
    summary["by_question_type"] = by_type


def _parse_float(value: str) -> float | None:
    """Parse a float from an answer when the answer is numeric."""
    text = extract_final_answer(value) if "answer" in value.lower() else value
    text = text.strip().replace(",", "")
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    text = text.replace("−", "-")
    match = re.search(r"[-+]?\d+(?:\.\d+)?%?", text)
    if not match:
        return None
    prefix = text[: match.start()].strip()
    suffix = text[match.end() :].strip()
    if prefix:
        return None
    if suffix and not re.fullmatch(r"[%a-zA-Z\s/().-]+", suffix):
        return None
    try:
        return float(match.group(0).rstrip("%"))
    except ValueError:
        return None


def _scale_suffix(image_scale: float) -> str:
    """Format an image scale for stable output filenames."""
    suffix = f"{image_scale:.2f}".rstrip("0").rstrip(".")
    return suffix.replace(".", "p")


def _labels_exact_match(prediction: str, ground_truth: str) -> bool:
    """Compare normalized label strings with a few harmless descriptor prefixes."""
    if prediction == ground_truth:
        return True

    descriptor_prefixes = ("age ", "ages ", "category ", "group ")
    for prefix in descriptor_prefixes:
        if ground_truth.startswith(prefix) and ground_truth[len(prefix) :] == prediction:
            return True
        if prediction.startswith(prefix) and prediction[len(prefix) :] == ground_truth:
            return True
    return False
