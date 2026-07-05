"""FastAPI backend used by the React frontend."""

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent.answer import ask_with_rag
from src.agent.vision_answer import ask_with_visual_rag
from src.app.upload_utils import (
    coerce_non_negative_float,
    coerce_non_negative_int,
    coerce_positive_int,
    ensure_unique_path,
    safe_filename,
)
from src.pdf.parse_pdf import parse_pdf
from src.rag.chunk_text import chunk_parsed_pdf


RAW_PAPERS_DIR = Path("data/raw_papers")
TEXT_DIR = Path("data/extracted/text")
PAGE_DIR = Path("data/extracted/pages")
CHUNKS_DIR = Path("data/extracted/chunks")
INDEX_DIR = Path("data/extracted/faiss_index")
UPLOADED_IMAGES_DIR = Path("data/extracted/uploaded_images")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM_MODEL = "qwen3-vl-flash"
DEFAULT_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


class AskRequest(BaseModel):
    """Text RAG request payload."""

    paper_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    llm_backend: str = "qwen-vl"
    llm_model_name: str = DEFAULT_LLM_MODEL
    answer_language: str = "zh"
    top_k: int = 5
    max_context_chars: int = 4000
    max_new_tokens: int = 512
    temperature: float = 0.2
    retriever_model_name: str = DEFAULT_EMBEDDING_MODEL
    llm_base_url: str = DEFAULT_LLM_BASE_URL


def create_app() -> FastAPI:
    """Create the PaperVLM-Agent API application."""
    app = FastAPI(title="PaperVLM-Agent API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_get_cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return a lightweight health check response."""
        return {"status": "ok"}

    @app.post("/api/process-pdf")
    async def process_pdf_endpoint(
        pdf: UploadFile = File(...),
        chunk_size: int = Form(800),
        chunk_overlap: int = Form(150),
        embedding_model: str = Form(DEFAULT_EMBEDDING_MODEL),
    ) -> dict[str, Any]:
        """Parse an uploaded PDF, chunk text, and build its FAISS index."""
        try:
            chunk_size = coerce_positive_int(chunk_size, "chunk_size")
            chunk_overlap = coerce_non_negative_int(chunk_overlap, "chunk_overlap")
            if chunk_overlap >= chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size.")

            pdf_path = await _save_upload(pdf, RAW_PAPERS_DIR, {".pdf"})
            parsed_pdf = parse_pdf(
                pdf_path=str(pdf_path),
                output_text_dir=str(TEXT_DIR),
                output_page_dir=str(PAGE_DIR),
            )
            paper_id = parsed_pdf["paper_id"]
            chunks_data = chunk_parsed_pdf(
                input_json_path=str(TEXT_DIR / f"{paper_id}.json"),
                output_dir=str(CHUNKS_DIR),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            from src.rag.build_index import build_index_from_chunks

            index_stats = build_index_from_chunks(
                chunks_path=str(CHUNKS_DIR / f"{paper_id}_chunks.json"),
                output_dir=str(INDEX_DIR),
                model_name=(embedding_model or DEFAULT_EMBEDDING_MODEL).strip(),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        status = "\n".join(
            [
                "PDF 处理完成。",
                f"paper_id: {paper_id}",
                f"PDF 页数: {parsed_pdf['num_pages']}",
                f"文本块数量: {chunks_data['num_chunks']}",
                f"索引路径: {index_stats['index_path']}",
            ]
        )
        return {
            "paper_id": paper_id,
            "status": status,
            "num_pages": parsed_pdf["num_pages"],
            "num_chunks": chunks_data["num_chunks"],
            "index_path": index_stats["index_path"],
            "metadata_path": index_stats["metadata_path"],
        }

    @app.post("/api/ask")
    def ask_endpoint(request: AskRequest) -> dict[str, Any]:
        """Answer a text RAG question for an indexed paper."""
        try:
            result = ask_with_rag(
                question=request.query,
                paper_id=request.paper_id,
                index_dir=str(INDEX_DIR),
                retriever_model_name=request.retriever_model_name,
                llm_backend=request.llm_backend,
                llm_model_name=request.llm_model_name or DEFAULT_LLM_MODEL,
                llm_base_url=request.llm_base_url or DEFAULT_LLM_BASE_URL,
                top_k=coerce_positive_int(request.top_k, "top_k"),
                max_context_chars=coerce_positive_int(
                    request.max_context_chars,
                    "max_context_chars",
                ),
                max_new_tokens=coerce_positive_int(
                    request.max_new_tokens,
                    "max_new_tokens",
                ),
                temperature=coerce_non_negative_float(
                    request.temperature,
                    "temperature",
                ),
                answer_language=request.answer_language,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return result

    @app.post("/api/ask-visual")
    async def ask_visual_endpoint(
        paper_id: str = Form(...),
        query: str = Form(...),
        answer_language: str = Form("zh"),
        top_k: int = Form(5),
        max_context_chars: int = Form(3000),
        max_new_tokens: int = Form(768),
        temperature: float = Form(0.2),
        retriever_model_name: str = Form(DEFAULT_EMBEDDING_MODEL),
        image: UploadFile | None = File(None),
    ) -> dict[str, Any]:
        """Answer a visual RAG question with an optional uploaded image."""
        try:
            image_path = ""
            if image is not None and image.filename:
                image_path = str(
                    await _save_upload(
                        image,
                        UPLOADED_IMAGES_DIR,
                        SUPPORTED_IMAGE_SUFFIXES,
                    )
                )

            result = ask_with_visual_rag(
                question=query,
                paper_id=paper_id,
                image_path=image_path,
                index_dir=str(INDEX_DIR),
                retriever_model_name=(
                    retriever_model_name or DEFAULT_EMBEDDING_MODEL
                ).strip(),
                llm_model_name=DEFAULT_LLM_MODEL,
                llm_base_url=DEFAULT_LLM_BASE_URL,
                top_k=coerce_positive_int(top_k, "top_k"),
                max_context_chars=coerce_positive_int(
                    max_context_chars,
                    "max_context_chars",
                ),
                max_new_tokens=coerce_positive_int(
                    max_new_tokens,
                    "max_new_tokens",
                ),
                temperature=coerce_non_negative_float(temperature, "temperature"),
                answer_language=answer_language,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return result

    return app


async def _save_upload(
    upload: UploadFile,
    target_dir: Path,
    suffixes: set[str],
) -> Path:
    """Persist an uploaded file after suffix validation."""
    filename = safe_filename(upload.filename or "")
    target_path = ensure_unique_path(target_dir / filename)
    if target_path.suffix.lower() not in suffixes:
        suffix_text = ", ".join(sorted(suffixes))
        raise ValueError(f"Unsupported file type. Expected one of: {suffix_text}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    content = await upload.read()
    if not content:
        raise ValueError(f"Uploaded file is empty: {filename}")
    target_path.write_bytes(content)
    return target_path


def _get_cors_origins() -> list[str]:
    """Read allowed CORS origins from the environment."""
    raw_origins = os.getenv("PAPERVLM_CORS_ORIGINS", "*")
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return origins or ["*"]


app = create_app()
