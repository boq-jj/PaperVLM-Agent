"""Visual RAG question answering for rendered paper pages or figures."""

from pathlib import Path
from typing import Any

from src.agent.answer import (
    format_evidence,
    normalize_answer_language,
    summarize_retrieval_results,
)
from src.llm import QwenVLAPILLM
from src.rag.retrieve import retrieve


DEFAULT_INDEX_DIR = "data/extracted/faiss_index"
DEFAULT_RETRIEVER_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM_MODEL = "qwen3-vl-flash"
DEFAULT_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def build_visual_rag_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    image_path: str,
    max_context_chars: int = 3000,
    answer_language: str = "zh",
) -> str:
    """Build a prompt for visual RAG question answering."""
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
                "You are analyzing a research paper page image with retrieved text evidence.",
                "Please answer in English.",
                "Use both the page image and the retrieved text evidence.",
                "If the image and text evidence are insufficient, clearly say \"insufficient evidence\".",
                "Do not use outside knowledge or invent information not found in the paper.",
                "",
                "Retrieved text evidence:",
                evidence if evidence else "No text evidence retrieved.",
                "",
                f"Image path: {image_path}",
                "Image:",
                "The image is a rendered paper page or an uploaded figure image.",
                "",
                "Question:",
                question,
                "",
                "Please answer with:",
                "Answer:",
                "Supporting pages:",
                "Visual evidence:",
                "Reasoning:",
            ]
        )

    return "\n".join(
        [
            "你正在分析一页科研论文页面图像，并结合检索到的文本证据回答问题。",
            "请使用中文回答。",
            "请同时参考页面图像和文本证据。",
            "如果图像和文本证据不足，请明确说明“证据不足”。",
            "不要使用外部知识，不要编造论文中没有出现的信息。",
            "",
            "检索到的文本证据：",
            evidence if evidence else "未检索到文本证据。",
            "",
            f"图像路径：{image_path}",
            "图像说明：",
            "该图像是论文页面截图或用户上传的图表图片。",
            "",
            "问题：",
            question,
            "",
            "请按以下格式回答：",
            "回答：",
            "支持页码：",
            "视觉证据：",
            "推理：",
        ]
    )


def select_page_image_from_retrieval(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Select the first existing page image from retrieved chunks."""
    for chunk in retrieved_chunks:
        page_image = str(chunk.get("page_image", "") or "").strip()
        if not page_image:
            continue

        image_path = Path(page_image)
        if image_path.exists() and image_path.is_file():
            return str(image_path)

    return ""


def ask_with_visual_rag(
    question: str,
    paper_id: str,
    image_path: str | None = None,
    index_dir: str = DEFAULT_INDEX_DIR,
    retriever_model_name: str = DEFAULT_RETRIEVER_MODEL,
    llm_model_name: str = DEFAULT_LLM_MODEL,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    top_k: int = 5,
    max_context_chars: int = 3000,
    max_new_tokens: int = 768,
    temperature: float = 0.2,
    answer_language: str = "zh",
) -> dict[str, Any]:
    """Answer a question using retrieved text chunks plus a page or figure image."""
    question = question.strip()
    paper_id = paper_id.strip()
    if not question:
        raise ValueError("question must not be empty.")
    if not paper_id:
        raise ValueError("paper_id must not be empty.")

    index_root = Path(index_dir)
    index_path = index_root / f"{paper_id}.index"
    metadata_path = index_root / f"{paper_id}_metadata.json"

    retrieved_chunks = retrieve(
        query=question,
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        model_name=retriever_model_name,
        top_k=top_k,
    )

    selected_image = str(image_path or "").strip()
    if not selected_image:
        selected_image = select_page_image_from_retrieval(retrieved_chunks)
    if not selected_image:
        raise ValueError("No page image found. Provide --image or ensure retrieved chunks contain page_image.")

    if not Path(selected_image).exists():
        raise FileNotFoundError(f"Image file not found: {selected_image}")

    prompt = build_visual_rag_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
        image_path=selected_image,
        max_context_chars=max_context_chars,
        answer_language=answer_language,
    )

    llm = QwenVLAPILLM(
        model_name=llm_model_name,
        base_url=llm_base_url,
        max_tokens=max_new_tokens,
        temperature=temperature,
        answer_language=normalize_answer_language(answer_language),
    )
    answer = llm.generate_with_image(prompt, selected_image)

    return {
        "question": question,
        "paper_id": paper_id,
        "image_path": selected_image,
        "retrieved_chunks": summarize_retrieval_results(retrieved_chunks),
        "prompt": prompt,
        "answer": answer,
    }
