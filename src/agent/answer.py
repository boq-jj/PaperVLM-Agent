"""Answer and RAG prompt assembly utilities for PaperVLM-Agent."""

from pathlib import Path
from typing import Any

from src.llm import MockLLM, QwenVLAPILLM
from src.rag.retrieve import retrieve


DEFAULT_INDEX_DIR = "data/extracted/faiss_index"
DEFAULT_RETRIEVER_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM_MODEL = "qwen3-vl-flash"
DEFAULT_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def normalize_answer_language(answer_language: str = "zh") -> str:
    """Normalize answer language options to ``zh`` or ``en``."""
    language = (answer_language or "zh").strip().lower()
    if language in {"en", "english", "英文"}:
        return "en"
    return "zh"


def _compact_preview(text: str, max_chars: int) -> str:
    """Create a one-line preview by replacing whitespace."""
    preview = text.replace("\r", " ").replace("\n", " ")
    preview = " ".join(preview.split())
    return preview[:max_chars]


def format_evidence(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as evidence text for a prompt.

    Args:
        retrieved_chunks: Retrieval results returned by ``src.rag.retrieve.retrieve``.

    Returns:
        A formatted evidence block.
    """
    evidence_blocks: list[str] = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        score = float(chunk.get("score", 0.0))
        page_id = chunk.get("page_id", "unknown")
        chunk_id = chunk.get("chunk_id", "")
        text = str(chunk.get("text", "") or "").strip()

        evidence_blocks.append(
            "\n".join(
                [
                    f"[证据 {index}]",
                    f"page_id: {page_id}",
                    f"chunk_id: {chunk_id}",
                    f"score: {score:.4f}",
                    "text:",
                    text,
                ]
            )
        )

    return "\n\n".join(evidence_blocks)


def build_rag_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    max_context_chars: int = 4000,
    answer_language: str = "zh",
) -> str:
    """Build a RAG prompt from retrieved evidence.

    Args:
        question: User question.
        retrieved_chunks: Retrieval results returned by the retriever.
        max_context_chars: Maximum number of evidence characters in the prompt.

    Returns:
        A prompt string ready for a large language model.
    """
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be greater than 0.")

    evidence = format_evidence(retrieved_chunks)
    if len(evidence) > max_context_chars:
        evidence = evidence[:max_context_chars].rstrip()
        evidence += "\n\n[Context truncated due to max_context_chars limit.]"

    language = normalize_answer_language(answer_language)
    if language == "en":
        return "\n".join(
            [
                "You are a research paper assistant.",
                "Answer the question only based on the evidence provided below.",
                "Please answer in English.",
                "If the evidence is insufficient, clearly say \"insufficient evidence\".",
                "Do not use outside knowledge or invent information not found in the paper.",
                "",
                "Evidence:",
                evidence if evidence else "No evidence retrieved.",
                "",
                "Question:",
                question,
                "",
                "Please answer with:",
                "Answer:",
                "Supporting pages:",
                "Reasoning:",
            ]
        )

    return "\n".join(
        [
            "你是一个科研论文问答助手。",
            "请仅根据下面给定的证据回答用户问题。",
            "请使用中文回答，即使证据文本是英文，也要用中文总结。",
            "如果证据不足，请明确说明“证据不足”。",
            "不要使用外部知识，不要编造论文中没有出现的信息。",
            "",
            "证据：",
            evidence if evidence else "未检索到证据。",
            "",
            "问题：",
            question,
            "",
            "请按以下格式回答：",
            "回答：",
            "支持页码：",
            "推理：",
        ]
    )


def summarize_retrieval_results(
    retrieved_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize retrieval results for terminal display."""
    summaries: list[dict[str, Any]] = []
    for chunk in retrieved_chunks:
        text = str(chunk.get("text", "") or "")
        summaries.append(
            {
                "rank": chunk.get("rank"),
                "score": float(chunk.get("score", 0.0)),
                "chunk_id": chunk.get("chunk_id", ""),
                "page_id": chunk.get("page_id"),
                "preview": _compact_preview(text, 200),
            }
        )

    return summaries


def ask_with_rag(
    question: str,
    paper_id: str,
    retrieval_query: str | None = None,
    index_dir: str = DEFAULT_INDEX_DIR,
    retriever_model_name: str = DEFAULT_RETRIEVER_MODEL,
    llm_backend: str = "qwen-vl",
    llm_model_name: str = DEFAULT_LLM_MODEL,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    top_k: int = 5,
    max_context_chars: int = 4000,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    answer_language: str = "zh",
) -> dict[str, Any]:
    """Answer a paper question with retrieval-augmented generation.

    Args:
        question: User question.
        paper_id: Paper ID used to locate index and metadata files.
        retrieval_query: Optional query used only for retrieval. If omitted,
            the user-facing question is used for retrieval.
        index_dir: Directory containing FAISS index files.
        retriever_model_name: SentenceTransformer model name or local path.
        llm_backend: LLM backend name, either ``qwen-vl`` or ``mock``.
        llm_model_name: LLM model name.
        llm_base_url: OpenAI-compatible base URL for the LLM API.
        top_k: Number of chunks to retrieve.
        max_context_chars: Maximum context length in the prompt.
        max_new_tokens: Maximum answer token budget.
        temperature: LLM sampling temperature.

    Returns:
        A dictionary containing retrieval summary, prompt, and answer.
    """
    index_root = Path(index_dir)
    index_path = index_root / f"{paper_id}.index"
    metadata_path = index_root / f"{paper_id}_metadata.json"

    retrieved_chunks = retrieve(
        query=retrieval_query or question,
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        model_name=retriever_model_name,
        top_k=top_k,
    )
    prompt = build_rag_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
        max_context_chars=max_context_chars,
        answer_language=answer_language,
    )
    normalized_language = normalize_answer_language(answer_language)

    if llm_backend == "qwen-vl":
        llm = QwenVLAPILLM(
            model_name=llm_model_name,
            base_url=llm_base_url,
            max_tokens=max_new_tokens,
            temperature=temperature,
            answer_language=normalized_language,
        )
    elif llm_backend == "mock":
        llm = MockLLM(answer_language=normalized_language)
    else:
        raise ValueError("llm_backend must be either 'qwen-vl' or 'mock'.")

    answer = llm.generate(prompt)

    return {
        "question": question,
        "paper_id": paper_id,
        "llm_backend": llm_backend,
        "llm_model_name": llm_model_name,
        "retrieved_chunks": summarize_retrieval_results(retrieved_chunks),
        "prompt": prompt,
        "answer": answer,
    }


def answer_question(
    question: str,
    pdf_path: str | Path | None = None,
    image_path: str | Path | None = None,
) -> str:
    """Answer a user question using PDF, image, and retrieval tools.

    Args:
        question: User question.
        pdf_path: Optional uploaded paper PDF path.
        image_path: Optional uploaded figure image path.

    Returns:
        Generated answer.
    """
    raise NotImplementedError("End-to-end answering will be implemented later.")
