"""Error analysis utilities for text RAG evaluation results."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_eval_results(path: str) -> dict[str, Any]:
    """Load a text RAG evaluation result JSON file."""
    result_path = Path(path)
    if not result_path.exists():
        raise FileNotFoundError(f"Evaluation result file not found: {result_path}")

    with result_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Evaluation result must be a JSON object.")
    if "summary" not in data or "results" not in data:
        raise ValueError("Evaluation result must contain 'summary' and 'results'.")
    if not isinstance(data["results"], list):
        raise ValueError("Evaluation result field 'results' must be a list.")
    return data


def classify_error_case(result: dict[str, Any]) -> list[str]:
    """Assign simple diagnostic labels to one evaluation result."""
    labels: list[str] = []

    if result.get("error"):
        labels.append("runtime_error")

    if result.get("page_hit") is False:
        labels.append("retrieval_miss")

    model_answer = str(result.get("model_answer") or "")
    answer_length = len(model_answer.strip())
    if answer_length < 20:
        labels.append("empty_answer")
    elif answer_length <= 100:
        labels.append("short_answer")

    retrieved_chunks = result.get("retrieved_chunks") or []
    if not retrieved_chunks:
        labels.append("no_retrieved_chunks")
    else:
        top_score = _safe_float(retrieved_chunks[0].get("score"))
        if top_score is not None and top_score < 0.3:
            labels.append("low_top_score")

    if not labels:
        labels.append("possible_success")
    return labels


def build_case_records(eval_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten evaluation results into records suitable for tables."""
    records: list[dict[str, Any]] = []
    for result in eval_data["results"]:
        if not isinstance(result, dict):
            continue

        retrieved_chunks = result.get("retrieved_chunks") or []
        top_chunk = retrieved_chunks[0] if retrieved_chunks else {}
        top_score = _safe_float(top_chunk.get("score")) if top_chunk else None
        model_answer = str(result.get("model_answer") or "")
        error_labels = classify_error_case(result)

        records.append(
            {
                "id": result.get("id", ""),
                "question_type": result.get("question_type", ""),
                "question": result.get("question", ""),
                "expected_pages": _format_list(result.get("expected_pages", [])),
                "retrieved_pages": _format_list(result.get("retrieved_pages", [])),
                "page_hit": bool(result.get("page_hit", False)),
                "top_score": top_score,
                "top_chunk_id": top_chunk.get("chunk_id", "") if top_chunk else "",
                "answer_length": len(model_answer.strip()),
                "error_labels": ";".join(error_labels),
                "error": result.get("error", "") or "",
            }
        )
    return records


def summarize_errors(case_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize retrieval and answer-quality diagnostics."""
    total = len(case_records)
    page_hit_count = sum(1 for record in case_records if record.get("page_hit"))
    top_scores = [
        float(record["top_score"])
        for record in case_records
        if record.get("top_score") is not None
    ]
    answer_lengths = [
        int(record.get("answer_length", 0))
        for record in case_records
    ]

    label_counts: Counter[str] = Counter()
    for record in case_records:
        labels = str(record.get("error_labels", "")).split(";")
        label_counts.update(label for label in labels if label)

    by_type_raw: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "page_hit_count": 0, "top_scores": []}
    )
    for record in case_records:
        question_type = str(record.get("question_type") or "unknown")
        stats = by_type_raw[question_type]
        stats["total"] += 1
        if record.get("page_hit"):
            stats["page_hit_count"] += 1
        if record.get("top_score") is not None:
            stats["top_scores"].append(float(record["top_score"]))

    by_question_type: dict[str, dict[str, Any]] = {}
    for question_type, stats in by_type_raw.items():
        type_total = stats["total"]
        by_question_type[question_type] = {
            "total": type_total,
            "page_hit_count": stats["page_hit_count"],
            "page_hit_rate": _safe_rate(stats["page_hit_count"], type_total),
            "average_top_score": _average(stats["top_scores"]),
        }

    return {
        "total": total,
        "page_hit_count": page_hit_count,
        "page_hit_rate": _safe_rate(page_hit_count, total),
        "average_top_score": _average(top_scores),
        "average_answer_length": _average(answer_lengths),
        "error_label_counts": dict(sorted(label_counts.items())),
        "by_question_type": by_question_type,
    }


def save_jsonl(records: list[dict[str, Any]], path: str) -> Path:
    """Save records as UTF-8 JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def save_csv(records: list[dict[str, Any]], path: str) -> Path:
    """Save records as CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys()) if records else []

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(records)
    return output_path


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
        f"- Total examples: {summary['total']}",
        f"- Page hit count: {summary['page_hit_count']}",
        f"- Page hit rate: {summary['page_hit_rate']:.4f}",
        f"- Average top retrieval score: {summary['average_top_score']:.4f}",
        f"- Average answer length: {summary['average_answer_length']:.2f}",
        "",
        "## Metrics by Question Type",
        "",
        "| Question type | Total | Page hit count | Page hit rate | Average top score |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for question_type, stats in summary["by_question_type"].items():
        lines.append(
            f"| {question_type} | {stats['total']} | {stats['page_hit_count']} | "
            f"{stats['page_hit_rate']:.4f} | {stats['average_top_score']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Error Label Counts",
            "",
            "| Error label | Count |",
            "| --- | ---: |",
        ]
    )
    for label, count in summary["error_label_counts"].items():
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
            lines.extend(
                [
                    f"- `{record['id']}` ({record['question_type']}): {record['question']}",
                    f"  - Labels: {record['error_labels']}",
                    f"  - Expected pages: {record['expected_pages']}; retrieved pages: {record['retrieved_pages']}",
                    f"  - Top chunk: {record['top_chunk_id']}; top score: {_format_score(record['top_score'])}",
                ]
            )
            if record.get("error"):
                lines.append(f"  - Runtime error: {record['error']}")

    lines.extend(
        [
            "",
            "## Notes for Report",
            "",
            (
                "English: The evaluation focuses on retrieval grounding. "
                f"The current page hit rate is {summary['page_hit_rate']:.2%}, "
                "and error labels help identify retrieval misses, weak evidence, "
                "short answers, and runtime failures."
            ),
            "",
            (
                "中文：当前评测重点关注 RAG 检索是否命中参考页码。"
                f"本次 page hit rate 为 {summary['page_hit_rate']:.2%}，"
                "错误标签可用于定位检索未命中、证据分数较低、回答过短和运行失败等问题。"
            ),
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_error_analysis(
    eval_result_path: str,
    output_dir: str = "data/eval/analysis",
) -> dict[str, Any]:
    """Run error analysis and save report/table artifacts."""
    eval_data = load_eval_results(eval_result_path)
    case_records = build_case_records(eval_data)
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


def _analysis_base_name(stem: str) -> str:
    """Strip the run_eval_text_rag suffix from result file stems."""
    if stem.endswith("_results"):
        return stem[: -len("_results")]
    return stem


def _by_type_records(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert by-type metrics to CSV rows."""
    return [
        {
            "question_type": question_type,
            "total": stats["total"],
            "page_hit_count": stats["page_hit_count"],
            "page_hit_rate": stats["page_hit_rate"],
            "average_top_score": stats["average_top_score"],
        }
        for question_type, stats in summary["by_question_type"].items()
    ]


def _safe_float(value: Any) -> float | None:
    """Convert value to float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_rate(numerator: int, denominator: int) -> float:
    """Compute a rate while handling empty denominators."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _average(values: list[float] | list[int]) -> float:
    """Compute an average while handling empty lists."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _format_list(value: Any) -> str:
    """Format list-like values as a compact semicolon-separated string."""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value)


def _format_score(value: Any) -> str:
    """Format a score value for Markdown output."""
    score = _safe_float(value)
    if score is None:
        return ""
    return f"{score:.4f}"
