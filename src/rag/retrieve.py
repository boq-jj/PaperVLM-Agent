"""FAISS retrieval utilities for paper chunks."""

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import faiss
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


DEFAULT_RETRIEVER_MODEL = "BAAI/bge-small-en-v1.5"


def load_index(index_path: str) -> faiss.Index:
    """Load a FAISS index from disk."""
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    if not path.is_file():
        raise ValueError(f"FAISS index path is not a file: {path}")

    try:
        return faiss.read_index(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load FAISS index: {path}") from exc


def load_metadata(metadata_path: str) -> dict[str, Any]:
    """Load chunk metadata saved with the FAISS index."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {path}")
    if not path.is_file():
        raise ValueError(f"Metadata path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid metadata JSON file: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load metadata JSON: {path}") from exc

    if not isinstance(metadata, dict):
        raise ValueError("Metadata JSON must contain an object.")
    if "chunks" not in metadata or not isinstance(metadata["chunks"], list):
        raise ValueError("Metadata JSON must contain a list field named 'chunks'.")

    return metadata


@lru_cache(maxsize=4)
def load_retriever_model(model_name: str = DEFAULT_RETRIEVER_MODEL) -> "SentenceTransformer":
    """Load a SentenceTransformer model for query encoding."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to load retriever model: {model_name}") from exc


def encode_query(model: "SentenceTransformer", query: str) -> np.ndarray:
    """Encode a retrieval query as a normalized float32 vector."""
    if not query.strip():
        raise ValueError("query must not be empty.")

    try:
        embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise RuntimeError("Failed to encode query.") from exc

    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError(f"Expected query embedding shape (1, dim), got {embedding.shape}.")

    return embedding


def is_reference_like_text(text: str) -> bool:
    """Return whether text looks like references instead of paper content."""
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    first_lines = lines[:8]
    if any(line in {"references", "bibliography"} for line in first_lines):
        return True

    citation_lines = sum(1 for line in lines if line.startswith("[") or line[:4].isdigit())
    citation_markers = sum(text.count(f"[{index}]") for index in range(1, 30))
    return citation_lines >= max(6, len(lines) // 3) or citation_markers >= 8


def compact_query_terms(query: str) -> set[str]:
    """Extract simple lexical terms for lightweight reranking."""
    stopwords = {
        "about",
        "and",
        "are",
        "bibliography",
        "does",
        "from",
        "into",
        "over",
        "paper",
        "prefer",
        "references",
        "section",
        "sections",
        "terms",
        "the",
        "this",
        "what",
        "with",
    }
    normalized = []
    for char in query.lower():
        normalized.append(char if char.isalnum() else " ")

    terms: set[str] = set()
    for token in "".join(normalized).split():
        if len(token) < 4 or token in stopwords:
            continue
        terms.add(token)
    return terms


def lexical_overlap_bonus(query: str, text: str) -> float:
    """Compute a small bonus for query terms found in a chunk."""
    terms = compact_query_terms(query)
    if not terms:
        return 0.0

    lowered_text = text.lower()
    matches = sum(1 for term in terms if term in lowered_text)
    return min(matches * 0.012, 0.08)


def section_heading_bonus(query: str, text: str) -> float:
    """Boost chunks that contain section headings relevant to the query intent."""
    lowered_query = query.lower()
    if any(term in lowered_query for term in {"motivation", "problem", "abstract", "introduction"}):
        headings = {"abstract", "introduction"}
    elif any(term in lowered_query for term in {"method", "model", "architecture", "approach"}):
        headings = {"method", "methods", "approach", "model", "architecture"}
    elif any(term in lowered_query for term in {"experiment", "evaluation", "dataset", "result"}):
        headings = {"experiment", "experiments", "evaluation", "results"}
    else:
        headings = set()

    if not headings:
        return 0.0

    for line in text.splitlines()[:20]:
        normalized = line.strip().lower().rstrip(":")
        if normalized in headings or any(normalized.startswith(f"{heading} ") for heading in headings):
            return 0.08
    return 0.0


def page_prior_bonus(query: str, page_id: Any) -> float:
    """Apply a conservative page-position prior for broad paper overview questions."""
    try:
        page_number = int(page_id)
    except (TypeError, ValueError):
        return 0.0

    lowered_query = query.lower()
    if any(term in lowered_query for term in {"motivation", "problem", "abstract", "introduction"}):
        if page_number == 1:
            return 0.10
        if page_number <= 3:
            return 0.06
    if any(term in lowered_query for term in {"method", "model", "architecture", "approach"}):
        if page_number <= 3:
            return 0.05
        if page_number <= 6:
            return 0.03
    return 0.0


def rerank_retrieval_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Rerank FAISS candidates with lightweight paper-structure heuristics."""
    adjusted: list[dict[str, Any]] = []
    for candidate in candidates:
        text = str(candidate.get("text", "") or "")
        original_score = float(candidate.get("score", 0.0))
        adjusted_score = original_score
        adjusted_score += lexical_overlap_bonus(query, text)
        adjusted_score += section_heading_bonus(query, text)
        adjusted_score += page_prior_bonus(query, candidate.get("page_id"))
        if is_reference_like_text(text):
            adjusted_score -= 0.25

        item = dict(candidate)
        item["original_score"] = original_score
        item["score"] = adjusted_score
        adjusted.append(item)

    adjusted.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return diversify_pages(adjusted, top_k)


def diversify_pages(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Prefer page diversity while filling all requested retrieval slots."""
    selected: list[dict[str, Any]] = []
    seen_pages: set[Any] = set()

    for candidate in candidates:
        page_id = candidate.get("page_id")
        if page_id in seen_pages:
            continue
        selected.append(candidate)
        seen_pages.add(page_id)
        if len(selected) >= top_k:
            return _rerank_output(selected)

    for candidate in candidates:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break

    return _rerank_output(selected)


def _rerank_output(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize ranks after reranking."""
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results


def retrieve(
    query: str,
    index_path: str,
    metadata_path: str,
    model_name: str = DEFAULT_RETRIEVER_MODEL,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant chunks for a query."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    index = load_index(index_path)
    metadata = load_metadata(metadata_path)
    chunks = metadata["chunks"]
    if not chunks:
        return []

    model = load_retriever_model(model_name)
    query_embedding = encode_query(model, query)

    search_k = min(max(top_k * 8, top_k), len(chunks), index.ntotal)
    if search_k <= 0:
        return []

    scores, indices = index.search(query_embedding, search_k)

    candidates: list[dict[str, Any]] = []
    for rank, (score, chunk_index) in enumerate(zip(scores[0], indices[0]), start=1):
        if chunk_index == -1:
            continue
        if chunk_index < 0 or chunk_index >= len(chunks):
            continue

        chunk = chunks[int(chunk_index)]
        candidates.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk.get("chunk_id", ""),
                "page_id": chunk.get("page_id", None),
                "text": chunk.get("text", ""),
                "page_image": chunk.get("page_image", ""),
            }
        )

    return rerank_retrieval_candidates(query=query, candidates=candidates, top_k=top_k)
