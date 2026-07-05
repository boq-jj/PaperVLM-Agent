"""Error analysis utilities for text RAG and ChartQA evaluation results."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.eval.io_utils import (
    safe_rate as _safe_rate,
    save_csv as _save_csv,
    save_jsonl as _save_jsonl,
)


TASK_TYPES = {"text_rag", "chartqa"}


def load_eval_results(path: str) -> dict[str, Any]:
    """Load an evaluation result JSON file."""
    result_path = Path(path)
    if not result_path.exists():
        raise FileNotFoundError(f"Evaluation result file not found: {result_path}")
    if not result_path.is_file():
        raise ValueError(f"Evaluation result path is not a file: {result_path}")

    with result_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Evaluation result must be a JSON object.")
    if "summary" not in data or "results" not in data:
        raise ValueError("Evaluation result must contain 'summary' and 'results'.")
    if not isinstance(data["results"], list):
        raise ValueError("Evaluation result field 'results' must be a list.")
    return data


def classify_text_rag_error(result: dict[str, Any]) -> list[str]:
    """Assign diagnostic labels to one text RAG evaluation result."""
    labels: list[str] = []

    if result.get("error"):
        labels.append("runtime_error")
    if result.get("page_hit") is False:
        labels.append("retrieval_miss")

    retrieved_chunks = result.get("retrieved_chunks") or []
    if not retrieved_chunks:
        labels.append("no_retrieved_chunks")
    else:
        top_score = _safe_float(retrieved_chunks[0].get("score"))
        if top_score is not None and top_score < 0.3:
            labels.append("low_top_score")

    model_answer = str(result.get("model_answer") or "").strip()
    if len(model_answer) < 20:
        labels.append("empty_answer")
    elif 20 <= len(model_answer) <= 100:
        labels.append("short_answer")

    severe_labels = {
        "runtime_error",
        "retrieval_miss",
        "no_retrieved_chunks",
        "empty_answer",
        "low_top_score",
    }
    if not any(label in severe_labels for label in labels):
        labels.append("possible_success")
    return labels


def classify_chartqa_error(result: dict[str, Any]) -> list[str]:
    """Assign diagnostic labels to one ChartQA evaluation result."""
    labels: list[str] = []
    error = str(result.get("error") or "")
    error_lower = error.lower()

    if error:
        labels.append("runtime_error")
    if "image not found" in error_lower or "file not found" in error_lower:
        labels.append("image_missing")
    if result.get("exact_match") is False:
        labels.append("exact_miss")
    if result.get("relaxed_match") is False:
        labels.append("relaxed_miss")
    if not str(result.get("prediction") or "").strip():
        labels.append("empty_prediction")
    if result.get("relaxed_match") is True and not error:
        labels.append("possible_success")
    return labels


def classify_error_case(result: dict[str, Any]) -> list[str]:
    """Backward-compatible text RAG error classifier."""
    return classify_text_rag_error(result)


def build_case_records(eval_data: dict[str, Any], task_type: str) -> list[dict[str, Any]]:
    """Flatten evaluation results into records suitable for tables."""
    _validate_task_type(task_type)
    if task_type == "chartqa":
        return [
            _build_chartqa_record(result)
            for result in eval_data["results"]
            if isinstance(result, dict)
        ]
    return [
        _build_text_rag_record(result)
        for result in eval_data["results"]
        if isinstance(result, dict)
    ]


def summarize_errors(case_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize retrieval, answer, and chart QA diagnostics."""
    total = len(case_records)
    label_counts: Counter[str] = Counter()
    for record in case_records:
        labels = str(record.get("error_labels", "")).split(";")
        label_counts.update(label for label in labels if label)

    summary: dict[str, Any] = {
        "total": total,
        "error_label_counts": dict(sorted(label_counts.items())),
        "by_question_type": {},
    }

    if any("page_hit" in record for record in case_records):
        page_hit_count = sum(1 for record in case_records if record.get("page_hit"))
        summary["page_hit_count"] = page_hit_count
        summary["page_hit_rate"] = _safe_rate(page_hit_count, total)

    if any("exact_match" in record for record in case_records):
        exact_count = sum(1 for record in case_records if record.get("exact_match"))
        relaxed_count = sum(1 for record in case_records if record.get("relaxed_match"))
        summary["exact_match_count"] = exact_count
        summary["exact_match_rate"] = _safe_rate(exact_count, total)
        summary["relaxed_match_count"] = relaxed_count
        summary["relaxed_match_rate"] = _safe_rate(relaxed_count, total)

    summary["by_question_type"] = _summarize_by_question_type(case_records)
    return summary


def save_jsonl(records: list[dict[str, Any]], path: str) -> Path:
    """Save records as UTF-8 JSONL."""
    return _save_jsonl(records, path)


def save_csv(records: list[dict[str, Any]], path: str) -> Path:
    """Save records as CSV."""
    return _save_csv(records, path)


def save_summary_markdown(
    summary: dict[str, Any],
    case_records: list[dict[str, Any]],
    path: str,
) -> Path:
    """Save a Markdown report for README or project reports."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Evaluation Summary",
        "",
        "## Overall Metrics",
        "",
        f"- Total examples: {summary.get('total', 0)}",
    ]
    if "page_hit_rate" in summary:
        lines.extend(
            [
                f"- Page hit count: {summary.get('page_hit_count', 0)}",
                f"- Page hit rate: {summary.get('page_hit_rate', 0.0):.4f}",
            ]
        )
    if "exact_match_rate" in summary:
        lines.extend(
            [
                f"- Exact match count: {summary.get('exact_match_count', 0)}",
                f"- Exact match rate: {summary.get('exact_match_rate', 0.0):.4f}",
                f"- Relaxed match count: {summary.get('relaxed_match_count', 0)}",
                f"- Relaxed match rate: {summary.get('relaxed_match_rate', 0.0):.4f}",
            ]
        )

    lines.extend(["", "## Metrics by Question Type", ""])
    lines.extend(_markdown_by_type_table(summary.get("by_question_type", {})))

    lines.extend(
        [
            "",
            "## Error Label Counts",
            "",
            "| Error label | Count |",
            "| --- | ---: |",
        ]
    )
    for label, count in summary.get("error_label_counts", {}).items():
        lines.append(f"| {label} | {count} |")

    lines.extend(["", "## Representative Error Cases", ""])
    error_cases = [
        record
        for record in case_records
        if "possible_success" not in str(record.get("error_labels", "")).split(";")
    ][:5]
    if not error_cases:
        lines.append("- No representative error cases were found.")
    else:
        for record in error_cases:
            lines.extend(_format_error_case(record))

    lines.extend(
        [
            "",
            "## Notes for Report",
            "",
            (
                "These artifacts provide traceable evidence for retrieval, "
                "answer generation, chart QA accuracy, and error categories. "
                "They can be summarized in README files, reports, and experiment tables."
            ),
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_error_analysis(
    eval_result_path: str,
    task_type: str,
    output_dir: str = "data/eval/analysis",
) -> dict[str, Any]:
    """Run error analysis and save report/table artifacts."""
    _validate_task_type(task_type)
    eval_data = load_eval_results(eval_result_path)
    case_records = build_case_records(eval_data, task_type)
    summary = summarize_errors(case_records)

    output_root = Path(output_dir)
    base_name = _analysis_base_name(Path(eval_result_path).stem)
    summary_path = output_root / f"{base_name}_summary.md"
    cases_path = output_root / f"{base_name}_error_cases.jsonl"
    table_path = output_root / f"{base_name}_table.csv"
    by_type_path = output_root / f"{base_name}_by_type.csv"

    save_summary_markdown(summary, case_records, str(summary_path))
    save_jsonl(case_records, str(cases_path))
    save_csv(case_records, str(table_path))
    save_csv(_by_type_records(summary), str(by_type_path))

    return {
        "summary": summary,
        "output_paths": {
            "summary_md": str(summary_path),
            "error_cases_jsonl": str(cases_path),
            "table_csv": str(table_path),
            "by_type_csv": str(by_type_path),
        },
    }


def _build_text_rag_record(result: dict[str, Any]) -> dict[str, Any]:
    """Build one flattened text RAG error-analysis record."""
    retrieved_chunks = result.get("retrieved_chunks") or []
    top_chunk = retrieved_chunks[0] if retrieved_chunks else {}
    top_score = _safe_float(top_chunk.get("score")) if top_chunk else None
    model_answer = str(result.get("model_answer") or "")
    error_labels = classify_text_rag_error(result)

    return {
        "id": result.get("id", ""),
        "task_type": "text_rag",
        "question_type": result.get("question_type", ""),
        "question": result.get("question", ""),
        "expected_pages": _format_list(result.get("expected_pages", [])),
        "retrieved_pages": _format_list(result.get("retrieved_pages", [])),
        "page_hit": bool(result.get("page_hit", False)),
        "top_score": top_score,
        "top_chunk_id": top_chunk.get("chunk_id", "") if top_chunk else "",
        "answer_length": len(model_answer.strip()),
        "model_answer": model_answer,
        "error_labels": ";".join(error_labels),
        "error": result.get("error", "") or "",
    }


def _build_chartqa_record(result: dict[str, Any]) -> dict[str, Any]:
    """Build one flattened ChartQA error-analysis record."""
    prediction = str(result.get("prediction") or "")
    error_labels = classify_chartqa_error(result)

    return {
        "id": result.get("id", ""),
        "task_type": "chartqa",
        "question_type": result.get("question_type", ""),
        "image_path": result.get("image_path", ""),
        "question": result.get("question", ""),
        "ground_truth": result.get("ground_truth", ""),
        "prediction": prediction,
        "prediction_length": len(prediction.strip()),
        "exact_match": bool(result.get("exact_match", False)),
        "relaxed_match": bool(result.get("relaxed_match", False)),
        "error_labels": ";".join(error_labels),
        "error": result.get("error", "") or "",
    }


def _summarize_by_question_type(case_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize metrics grouped by question type."""
    raw: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(int))
    for record in case_records:
        question_type = str(record.get("question_type") or "unknown")
        stats = raw[question_type]
        stats["total"] += 1
        if record.get("page_hit"):
            stats["page_hit_count"] += 1
        if record.get("exact_match"):
            stats["exact_match_count"] += 1
        if record.get("relaxed_match"):
            stats["relaxed_match_count"] += 1

    by_type: dict[str, dict[str, Any]] = {}
    for question_type, stats in raw.items():
        total = int(stats["total"])
        row: dict[str, Any] = {"total": total}
        if any("page_hit" in record for record in case_records):
            row["page_hit_count"] = int(stats["page_hit_count"])
            row["page_hit_rate"] = _safe_rate(row["page_hit_count"], total)
        if any("exact_match" in record for record in case_records):
            row["exact_match_count"] = int(stats["exact_match_count"])
            row["exact_match_rate"] = _safe_rate(row["exact_match_count"], total)
            row["relaxed_match_count"] = int(stats["relaxed_match_count"])
            row["relaxed_match_rate"] = _safe_rate(row["relaxed_match_count"], total)
        by_type[question_type] = row
    return by_type


def _markdown_by_type_table(by_type: dict[str, Any]) -> list[str]:
    """Build a Markdown table for per-type metrics."""
    if not by_type:
        return ["No question-type metrics available."]

    metric_keys = sorted(
        {
            key
            for stats in by_type.values()
            for key in stats
            if key != "total"
        }
    )
    header = ["Question type", "Total", *metric_keys]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---", "---:", *["---:" for _ in metric_keys]]) + " |",
    ]
    for question_type, stats in by_type.items():
        values = [question_type, str(stats.get("total", 0))]
        values.extend(_format_metric(stats.get(key, "")) for key in metric_keys)
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _format_error_case(record: dict[str, Any]) -> list[str]:
    """Format one representative error case for Markdown."""
    lines = [
        f"- `{record.get('id', '')}` ({record.get('question_type', '')}): "
        f"{record.get('question', '')}",
        f"  - Labels: {record.get('error_labels', '')}",
    ]
    if record.get("task_type") == "text_rag":
        lines.append(
            "  - Expected pages: "
            f"{record.get('expected_pages', '')}; "
            f"retrieved pages: {record.get('retrieved_pages', '')}"
        )
        lines.append(
            "  - Top chunk: "
            f"{record.get('top_chunk_id', '')}; "
            f"top score: {_format_metric(record.get('top_score', ''))}"
        )
    if record.get("task_type") == "chartqa":
        lines.append(
            "  - Ground truth: "
            f"{record.get('ground_truth', '')}; "
            f"prediction: {record.get('prediction', '')}"
        )
        lines.append(f"  - Image: {record.get('image_path', '')}")
    if record.get("error"):
        lines.append(f"  - Runtime error: {record['error']}")
    return lines


def _analysis_base_name(stem: str) -> str:
    """Strip the result suffix from result file stems."""
    if stem.endswith("_results"):
        return stem[: -len("_results")]
    return stem


def _by_type_records(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert by-type metrics to CSV rows."""
    records: list[dict[str, Any]] = []
    for question_type, stats in summary.get("by_question_type", {}).items():
        record = {"question_type": question_type}
        record.update(stats)
        records.append(record)
    return records


def _validate_task_type(task_type: str) -> None:
    """Validate supported task types."""
    if task_type not in TASK_TYPES:
        raise ValueError("task_type must be either 'text_rag' or 'chartqa'.")


def _safe_float(value: Any) -> float | None:
    """Convert value to float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _format_list(value: Any) -> str:
    """Format list-like values as a compact semicolon-separated string."""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value)


def _format_metric(value: Any) -> str:
    """Format a metric value for Markdown output."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
