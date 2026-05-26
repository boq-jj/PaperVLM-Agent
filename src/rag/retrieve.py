"""FAISS retrieval utilities for paper chunks."""

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
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


def load_retriever_model(model_name: str = DEFAULT_RETRIEVER_MODEL) -> SentenceTransformer:
    """Load a SentenceTransformer model for query encoding."""
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to load retriever model: {model_name}") from exc


def encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
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

    search_k = min(top_k, len(chunks), index.ntotal)
    if search_k <= 0:
        return []

    scores, indices = index.search(query_embedding, search_k)

    results: list[dict[str, Any]] = []
    for rank, (score, chunk_index) in enumerate(zip(scores[0], indices[0]), start=1):
        if chunk_index == -1:
            continue
        if chunk_index < 0 or chunk_index >= len(chunks):
            continue

        chunk = chunks[int(chunk_index)]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk.get("chunk_id", ""),
                "page_id": chunk.get("page_id", None),
                "text": chunk.get("text", ""),
                "page_image": chunk.get("page_image", ""),
            }
        )

    return results
