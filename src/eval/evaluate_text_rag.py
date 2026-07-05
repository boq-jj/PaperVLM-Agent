"""Batch evaluation utilities for text RAG question answering."""

from typing import Any

from src.eval.io_utils import load_jsonl, safe_rate as _safe_rate, save_json


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
    expected = {_safe_int(page) for page in expected_pages}
    retrieved = {_safe_int(page) for page in retrieved_pages}
    expected.discard(None)
    retrieved.discard(None)
    return bool(expected & retrieved)


def evaluate_text_rag_examples(
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
    """Evaluate text RAG over a list of examples.

    ``embedding_backend`` is accepted for CLI compatibility. The current
    retrieval path selects the embedding model through ``retriever_model_name``.
    """
    del embedding_backend

    summary = _empty_summary(total=len(examples))
    type_stats: dict[str, dict[str, int]] = {}
    results: list[dict[str, Any]] = []

    for example in examples:
        example_id = str(example.get("id", ""))
        current_paper_id = paper_id or str(example.get("paper_id", "")).strip()
        question = str(example.get("question", "")).strip()
        retrieval_query = str(example.get("retrieval_query", "") or question).strip()
        question_type = str(example.get("question_type", "unknown") or "unknown")
        expected_pages = _coerce_page_list(example.get("expected_pages", []))
        reference_answer = str(example.get("reference_answer", ""))

        _ensure_type_stats(type_stats, question_type)
        type_stats[question_type]["total"] += 1

        record: dict[str, Any] = {
            "id": example_id,
            "paper_id": current_paper_id,
            "question": question,
            "retrieval_query": retrieval_query,
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
            from src.agent.answer import ask_with_rag

            rag_result = ask_with_rag(
                question=question,
                retrieval_query=retrieval_query,
                paper_id=current_paper_id,
                index_dir=index_dir,
                retriever_model_name=retriever_model_name,
                llm_backend=llm_backend,
                llm_model_name=llm_model_name,
                top_k=top_k,
                max_context_chars=max_context_chars,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                answer_language="en",
            )
            retrieved_chunks = rag_result.get("retrieved_chunks", [])
            retrieved_pages = get_retrieved_pages(retrieved_chunks)
            page_hit = compute_page_hit(expected_pages, retrieved_pages)

            record.update(
                {
                    "retrieved_pages": retrieved_pages,
                    "page_hit": page_hit,
                    "model_answer": str(rag_result.get("answer", "")),
                    "retrieved_chunks": retrieved_chunks,
                }
            )
            summary["success"] += 1
            type_stats[question_type]["success"] += 1
            if page_hit:
                summary["page_hit_count"] += 1
                type_stats[question_type]["page_hit_count"] += 1
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
    """Backward-compatible alias for older scripts."""
    return evaluate_text_rag_examples(
        examples=examples,
        paper_id=paper_id,
        index_dir=index_dir,
        embedding_backend=embedding_backend,
        retriever_model_name=retriever_model_name,
        llm_backend=llm_backend,
        llm_model_name=llm_model_name,
        top_k=top_k,
        max_context_chars=max_context_chars,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


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


def _ensure_type_stats(type_stats: dict[str, dict[str, int]], question_type: str) -> None:
    """Ensure per-question-type counters exist."""
    if question_type not in type_stats:
        type_stats[question_type] = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "page_hit_count": 0,
        }


def _finalize_summary(
    summary: dict[str, Any],
    type_stats: dict[str, dict[str, int]],
) -> None:
    """Compute final rates in-place."""
    summary["page_hit_rate"] = _safe_rate(summary["page_hit_count"], summary["success"])

    by_type: dict[str, dict[str, float | int]] = {}
    for question_type, stats in type_stats.items():
        success = stats["success"]
        by_type[question_type] = {
            "total": stats["total"],
            "success": success,
            "failed": stats["failed"],
            "page_hit_count": stats["page_hit_count"],
            "page_hit_rate": _safe_rate(stats["page_hit_count"], success),
        }
    summary["by_question_type"] = by_type


def _coerce_page_list(value: Any) -> list[int]:
    """Coerce expected_pages to a list of integers."""
    if not isinstance(value, list):
        return []

    pages: list[int] = []
    for item in value:
        page = _safe_int(item)
        if page is not None:
            pages.append(page)
    return pages


def _safe_int(value: Any) -> int | None:
    """Convert value to int when possible."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
