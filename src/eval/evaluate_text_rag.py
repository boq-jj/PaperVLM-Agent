"""Batch evaluation utilities for text RAG question answering."""

import json
from pathlib import Path
from typing import Any

from src.agent.answer import ask_with_rag


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"JSONL path is not a file: {file_path}")

    examples: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {file_path}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_number} must be a JSON object: {file_path}")
            examples.append(item)

    return examples


def save_json(data: dict[str, Any], path: str) -> Path:
    """Save a dictionary as a UTF-8 JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return output_path


def get_retrieved_pages(retrieved_chunks: list[dict[str, Any]]) -> list[int]:
    """Extract unique page IDs from retrieval results while preserving order."""
    pages: list[int] = []
    seen: set[int] = set()
    for chunk in retrieved_chunks:
        page_id = chunk.get("page_id")
        if page_id is None:
            continue
        try:
            page_number = int(page_id)
        except (TypeError, ValueError):
            continue
        if page_number not in seen:
            seen.add(page_number)
            pages.append(page_number)
    return pages


def compute_page_hit(expected_pages: list[int], retrieved_pages: list[int]) -> bool:
    """Return whether any expected page appears in retrieved pages."""
    expected = {int(page) for page in expected_pages}
    retrieved = {int(page) for page in retrieved_pages}
    return bool(expected & retrieved)


def _empty_summary(total: int) -> dict[str, Any]:
    """Create an initial summary dictionary."""
    return {
        "total": total,
        "success": 0,
        "failed": 0,
        "page_hit_count": 0,
        "page_hit_rate": 0.0,
        "by_question_type": {},
    }


def _update_type_stats(type_stats: dict[str, dict[str, int]], question_type: str, page_hit: bool) -> None:
    """Update per-question-type counters."""
    if question_type not in type_stats:
        type_stats[question_type] = {"total": 0, "success": 0, "page_hit_count": 0}
    type_stats[question_type]["total"] += 1
    type_stats[question_type]["success"] += 1
    if page_hit:
        type_stats[question_type]["page_hit_count"] += 1


def _finalize_summary(summary: dict[str, Any], type_stats: dict[str, dict[str, int]]) -> None:
    """Compute final rates in-place."""
    if summary["success"] > 0:
        summary["page_hit_rate"] = summary["page_hit_count"] / summary["success"]

    by_type: dict[str, dict[str, float | int]] = {}
    for question_type, stats in type_stats.items():
        success = stats["success"]
        by_type[question_type] = {
            "total": stats["total"],
            "success": success,
            "page_hit_count": stats["page_hit_count"],
            "page_hit_rate": stats["page_hit_count"] / success if success else 0.0,
        }
    summary["by_question_type"] = by_type


def evaluate_examples(
    examples: list[dict[str, Any]],
    paper_id: str | None = None,
    index_dir: str = "data/extracted/faiss_index",
    embedding_backend: str = "local-bge",
    retriever_model_name: str = "BAAI/bge-small-en-v1.5",
    llm_backend: str = "qwen-vl",
    llm_model_name: str = "qwen3-vl-flash",
    top_k: int = 5,
    max_context_chars: int = 4000,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Evaluate text RAG over a list of examples."""
    del embedding_backend  # Reserved for future embedding backend selection.

    summary = _empty_summary(total=len(examples))
    type_stats: dict[str, dict[str, int]] = {}
    results: list[dict[str, Any]] = []

    for example in examples:
        example_id = str(example.get("id", ""))
        current_paper_id = paper_id or str(example.get("paper_id", ""))
        question = str(example.get("question", ""))
        question_type = str(example.get("question_type", "unknown"))
        expected_pages = [int(page) for page in example.get("expected_pages", [])]
        reference_answer = str(example.get("reference_answer", ""))

        record: dict[str, Any] = {
            "id": example_id,
            "paper_id": current_paper_id,
            "question": question,
            "question_type": question_type,
            "expected_pages": expected_pages,
            "retrieved_pages": [],
            "page_hit": False,
            "reference_answer": reference_answer,
            "model_answer": "",
            "retrieved_chunks": [],
            "error": "",
        }

        try:
            rag_result = ask_with_rag(
                question=question,
                paper_id=current_paper_id,
                index_dir=index_dir,
                retriever_model_name=retriever_model_name,
                llm_backend=llm_backend,
                llm_model_name=llm_model_name,
                top_k=top_k,
                max_context_chars=max_context_chars,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            retrieved_chunks = rag_result["retrieved_chunks"]
            retrieved_pages = get_retrieved_pages(retrieved_chunks)
            page_hit = compute_page_hit(expected_pages, retrieved_pages)

            record.update(
                {
                    "retrieved_pages": retrieved_pages,
                    "page_hit": page_hit,
                    "model_answer": rag_result["answer"],
                    "retrieved_chunks": retrieved_chunks,
                }
            )
            summary["success"] += 1
            if page_hit:
                summary["page_hit_count"] += 1
            _update_type_stats(type_stats, question_type, page_hit)
        except Exception as exc:
            record["error"] = str(exc)
            summary["failed"] += 1

        results.append(record)

    _finalize_summary(summary, type_stats)
    return {
        "summary": summary,
        "results": results,
    }
