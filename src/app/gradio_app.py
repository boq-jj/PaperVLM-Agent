"""Gradio demo for PaperVLM-Agent."""

from pathlib import Path
from typing import Any

import gradio as gr

from src.agent.answer import ask_with_rag
from src.agent.vision_answer import ask_with_visual_rag
from src.app.upload_utils import (
    coerce_non_negative_float,
    coerce_non_negative_int,
    coerce_positive_int,
    copy_uploaded_file,
    resolve_uploaded_file,
)
from src.pdf.parse_pdf import parse_pdf
from src.rag.build_index import build_index_from_chunks
from src.rag.chunk_text import chunk_parsed_pdf


RAW_PAPERS_DIR = Path("data/raw_papers")
TEXT_DIR = Path("data/extracted/text")
PAGE_DIR = Path("data/extracted/pages")
CHUNKS_DIR = Path("data/extracted/chunks")
INDEX_DIR = Path("data/extracted/faiss_index")
UPLOADED_IMAGES_DIR = Path("data/extracted/uploaded_images")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM_MODEL = "qwen3-vl-flash"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
APP_CSS = """
.gradio-container {
    background: #f6f5f1;
}
#hero_screen {
    min-height: 88vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 28px;
}
#hero_title {
    text-align: center;
}
#hero_title h1 {
    font-size: 44px;
    line-height: 1.05;
    margin-bottom: 8px;
    color: #1d1d1f;
    letter-spacing: 0;
}
#hero_title p {
    color: #62646a;
    font-size: 16px;
    margin: 0;
}
.paper-composer {
    position: relative;
    width: min(820px, calc(100vw - 36px));
    margin: 0 auto;
}
#hero_composer,
#bottom_composer {
    border: 1px solid #d7d4cd;
    background: #ffffff;
    border-radius: 999px;
    box-shadow: 0 18px 45px rgba(31, 31, 31, 0.10);
    padding: 10px 12px 10px 24px;
    min-height: 68px;
    align-items: center;
}
#bottom_bar {
    position: sticky;
    bottom: 0;
    z-index: 20;
    padding: 18px 0 10px;
    background: linear-gradient(180deg, rgba(246, 245, 241, 0), #f6f5f1 32%);
}
#question_box textarea {
    border: 0 !important;
    box-shadow: none !important;
    background: transparent !important;
    resize: none;
    font-size: 15px;
}
#hero_pdf_dropzone {
    position: absolute;
    inset: 0;
    opacity: 0;
    z-index: 2;
}
#hero_pdf_dropzone * {
    cursor: pointer !important;
}
.round-upload button,
.round-action button {
    border-radius: 999px !important;
    min-width: 46px !important;
    height: 46px !important;
    padding: 0 !important;
    font-size: 24px !important;
}
.ask-action button {
    border-radius: 999px !important;
    min-width: 92px !important;
    height: 46px !important;
}
#workspace {
    max-width: 1180px;
    margin: 0 auto;
    padding: 28px 16px 0;
}
#workspace_header {
    margin-bottom: 16px;
}
#workspace_header h2 {
    font-size: 28px;
    color: #1d1d1f;
    margin: 0;
}
.qa-panel {
    border: 1px solid #ddd8cf;
    border-radius: 18px;
    background: #ffffff;
    padding: 16px;
    min-height: 540px;
}
.settings-panel {
    border-radius: 14px;
}
"""


def default_embedding_model() -> str:
    """Prefer the local mirror-downloaded BGE model if it exists."""
    local_model = Path("models/bge-small-en-v1.5-hf-mirror")
    if local_model.exists():
        return local_model.as_posix()
    return DEFAULT_EMBEDDING_MODEL


def _normalize_embedding_model(embedding_model: str) -> str:
    """Normalize an embedding model input from the UI."""
    model_name = (embedding_model or default_embedding_model()).strip()
    if not model_name:
        raise ValueError("embedding_model 不能为空。")
    return model_name


def _copy_uploaded_image(image_file: Any) -> str:
    """Copy an uploaded image to the project data directory."""
    if image_file is None:
        return ""

    source_path = resolve_uploaded_file(image_file, "请先上传图片文件。")
    target_path = copy_uploaded_file(
        source_path=source_path,
        target_dir=UPLOADED_IMAGES_DIR,
        suffixes=SUPPORTED_IMAGE_SUFFIXES,
    )
    return str(target_path)


def process_pdf_for_demo(
    pdf_file: Any,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> tuple[str, str, str]:
    """Parse an uploaded PDF, build chunks, and create a FAISS index."""
    try:
        source_path = resolve_uploaded_file(pdf_file, "请先上传 PDF 文件。")
        chunk_size = coerce_positive_int(chunk_size, "chunk_size")
        chunk_overlap = coerce_non_negative_int(chunk_overlap, "chunk_overlap")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须大于等于 0，并且小于 chunk_size。")
        embedding_model = _normalize_embedding_model(embedding_model)
        target_path = copy_uploaded_file(source_path, RAW_PAPERS_DIR, {".pdf"})

        parsed_pdf = parse_pdf(
            pdf_path=str(target_path),
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
        index_stats = build_index_from_chunks(
            chunks_path=str(CHUNKS_DIR / f"{paper_id}_chunks.json"),
            output_dir=str(INDEX_DIR),
            model_name=embedding_model,
        )

        status_text = "\n".join(
            [
                "PDF 处理成功。",
                f"paper_id: {paper_id}",
                f"PDF 页数: {parsed_pdf['num_pages']}",
                f"文本块数量: {chunks_data['num_chunks']}",
                f"embedding 模型: {embedding_model}",
                f"索引路径: {index_stats['index_path']}",
                f"元数据路径: {index_stats['metadata_path']}",
            ]
        )
        return status_text, paper_id, embedding_model
    except Exception as exc:
        return f"PDF 处理失败：{exc}", "", ""


def process_pdf_and_open_workspace(
    pdf_file: Any,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> tuple[str, str, str, Any, Any, Any]:
    """Process a PDF and switch from the landing view to the QA workspace."""
    status_text, paper_id, resolved_embedding_model = process_pdf_for_demo(
        pdf_file=pdf_file,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
    )
    if paper_id:
        return (
            status_text,
            paper_id,
            resolved_embedding_model,
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
        )

    return (
        status_text,
        "",
        embedding_model or default_embedding_model(),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def format_retrieved_chunks_markdown(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunk summaries as Markdown."""
    if not chunks:
        return "未检索到证据。"

    sections: list[str] = []
    for chunk in chunks:
        sections.append(
            "\n".join(
                [
                    f"### 第 {chunk.get('rank')} 条证据",
                    f"- 相似度分数: {float(chunk.get('score', 0.0)):.4f}",
                    f"- chunk_id: `{chunk.get('chunk_id', '')}`",
                    f"- 页码: {chunk.get('page_id')}",
                    "",
                    str(chunk.get("preview", "")),
                ]
            )
        )

    return "\n\n".join(sections)


def ask_question_for_demo(
    paper_id: str,
    retriever_model_name: str,
    question: str,
    llm_backend: str,
    llm_model_name: str,
    answer_language: str,
    top_k: int,
    max_context_chars: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, str]:
    """Ask a question against the current paper index."""
    try:
        paper_id = (paper_id or "").strip()
        question = (question or "").strip()
        if not paper_id:
            raise ValueError("请先上传并处理 PDF。")
        if not question:
            raise ValueError("请输入问题。")

        retriever_model_name = (
            retriever_model_name or default_embedding_model()
        ).strip()
        top_k = coerce_positive_int(top_k, "top_k")
        max_context_chars = coerce_positive_int(max_context_chars, "max_context_chars")
        max_new_tokens = coerce_positive_int(max_new_tokens, "max_new_tokens")
        temperature = coerce_non_negative_float(temperature, "temperature")

        result = ask_with_rag(
            question=question,
            paper_id=paper_id,
            index_dir=str(INDEX_DIR),
            retriever_model_name=retriever_model_name,
            llm_backend=llm_backend,
            llm_model_name=llm_model_name or DEFAULT_LLM_MODEL,
            top_k=top_k,
            max_context_chars=max_context_chars,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            answer_language=answer_language,
        )

        evidence_markdown = format_retrieved_chunks_markdown(result["retrieved_chunks"])
        return result["answer"], evidence_markdown
    except Exception as exc:
        return f"回答失败：{exc}", ""


def ask_visual_question_for_demo(
    paper_id: str,
    retriever_model_name: str,
    image_file: Any,
    question: str,
    answer_language: str,
    top_k: int,
    max_context_chars: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, str]:
    """Ask a visual question using an uploaded image or retrieved page screenshot."""
    try:
        paper_id = (paper_id or "").strip()
        question = (question or "").strip()
        if not paper_id:
            raise ValueError("请先上传并处理 PDF，再进行视觉问答。")
        if not question:
            raise ValueError("请输入视觉问答问题。")

        retriever_model_name = (
            retriever_model_name or default_embedding_model()
        ).strip()
        top_k = coerce_positive_int(top_k, "visual_top_k")
        max_context_chars = coerce_positive_int(
            max_context_chars,
            "visual_max_context_chars",
        )
        max_new_tokens = coerce_positive_int(max_new_tokens, "visual_max_new_tokens")
        temperature = coerce_non_negative_float(temperature, "visual_temperature")
        image_path = _copy_uploaded_image(image_file)

        result = ask_with_visual_rag(
            question=question,
            paper_id=paper_id,
            image_path=image_path,
            index_dir=str(INDEX_DIR),
            retriever_model_name=retriever_model_name,
            llm_model_name=DEFAULT_LLM_MODEL,
            top_k=top_k,
            max_context_chars=max_context_chars,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            answer_language=answer_language,
        )

        evidence_markdown = "\n\n".join(
            [
                f"**使用图像：** `{result['image_path']}`",
                format_retrieved_chunks_markdown(result["retrieved_chunks"]),
            ]
        )
        return result["answer"], evidence_markdown
    except Exception as exc:
        return f"视觉问答失败：{exc}", ""


def build_demo() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""
    with gr.Blocks(title="PaperVLM-Agent") as demo:
        current_paper_id = gr.State("")
        current_embedding_model = gr.State(default_embedding_model())

        with gr.Column(visible=True, elem_id="hero_screen") as hero_screen:
            gr.Markdown(
                "\n".join(
                    [
                        "# PaperVLM-Agent",
                        "科研论文中文问答助手",
                    ]
                ),
                elem_id="hero_title",
            )
            with gr.Column(elem_classes=["paper-composer"]):
                with gr.Row(elem_id="hero_composer"):
                    gr.Markdown("上传一篇论文 PDF 开始分析。")
                    hero_upload_button = gr.UploadButton(
                        "＋",
                        file_types=[".pdf"],
                        file_count="single",
                        type="filepath",
                        elem_classes=["round-upload"],
                    )
                hero_dropzone = gr.File(
                    label="",
                    show_label=False,
                    file_types=[".pdf"],
                    type="filepath",
                    interactive=True,
                    elem_id="hero_pdf_dropzone",
                )
            hero_status_output = gr.Textbox(
                label="状态",
                lines=6,
                visible=True,
                interactive=False,
            )

        with gr.Column(visible=False, elem_id="workspace") as workspace:
            gr.Markdown("## PaperVLM-Agent", elem_id="workspace_header")
            status_output = gr.Textbox(label="当前论文状态", lines=6, interactive=False)

            with gr.Row(equal_height=True):
                with gr.Column(elem_classes=["qa-panel"]):
                    gr.Markdown("### 文本问答")
                    with gr.Accordion("文本问答设置", open=False, elem_classes=["settings-panel"]):
                        with gr.Row():
                            llm_backend_input = gr.Dropdown(
                                label="llm_backend",
                                choices=["qwen-vl", "mock"],
                                value="qwen-vl",
                            )
                            llm_model_input = gr.Textbox(label="llm_model_name", value=DEFAULT_LLM_MODEL)
                        answer_language_input = gr.Dropdown(
                            label="回答语言",
                            choices=["中文", "English"],
                            value="中文",
                        )
                        with gr.Row():
                            top_k_input = gr.Slider(
                                label="top_k",
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                            )
                            max_context_chars_input = gr.Number(
                                label="max_context_chars",
                                value=4000,
                                precision=0,
                            )
                        with gr.Row():
                            max_new_tokens_input = gr.Number(
                                label="max_new_tokens",
                                value=512,
                                precision=0,
                            )
                            temperature_input = gr.Number(label="temperature", value=0.2)
                    answer_output = gr.Textbox(label="回答", lines=12)
                    evidence_output = gr.Markdown(label="检索证据")

                with gr.Column(elem_classes=["qa-panel"]):
                    gr.Markdown("### 视觉问答")
                    with gr.Accordion("视觉问答设置", open=False, elem_classes=["settings-panel"]):
                        image_upload = gr.File(
                            label="可选：上传页面或图表图片",
                            file_types=[".png", ".jpg", ".jpeg", ".webp"],
                            type="filepath",
                        )
                        visual_answer_language_input = gr.Dropdown(
                            label="回答语言",
                            choices=["中文", "English"],
                            value="中文",
                        )
                        with gr.Row():
                            visual_top_k_input = gr.Slider(
                                label="visual_top_k",
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                            )
                            visual_max_context_chars_input = gr.Number(
                                label="visual_max_context_chars",
                                value=3000,
                                precision=0,
                            )
                        with gr.Row():
                            visual_max_new_tokens_input = gr.Number(
                                label="visual_max_new_tokens",
                                value=768,
                                precision=0,
                            )
                            visual_temperature_input = gr.Number(label="visual_temperature", value=0.2)
                    visual_answer_output = gr.Textbox(label="视觉回答", lines=12)
                    visual_evidence_output = gr.Markdown(label="视觉证据")

        with gr.Column(visible=False, elem_id="bottom_bar") as bottom_bar:
            with gr.Row(elem_id="bottom_composer", elem_classes=["paper-composer"]):
                bottom_upload_button = gr.UploadButton(
                    "＋",
                    file_types=[".pdf"],
                    file_count="single",
                    type="filepath",
                    elem_classes=["round-upload"],
                )
                question_input = gr.Textbox(
                    label="",
                    placeholder="请输入关于这篇论文的问题...",
                    lines=1,
                    max_lines=4,
                    show_label=False,
                    scale=8,
                    elem_id="question_box",
                )
                ask_button = gr.Button("文本", variant="primary", elem_classes=["ask-action"])
                visual_ask_button = gr.Button("图像", variant="secondary", elem_classes=["ask-action"])

        with gr.Accordion("索引构建设置", open=False, visible=False) as build_settings:
            with gr.Row():
                chunk_size_input = gr.Number(label="chunk_size", value=800, precision=0)
                chunk_overlap_input = gr.Number(label="chunk_overlap", value=150, precision=0)
            embedding_model_input = gr.Textbox(
                label="embedding_model",
                value=default_embedding_model(),
            )

        process_outputs = [
            status_output,
            current_paper_id,
            current_embedding_model,
            hero_screen,
            workspace,
            bottom_bar,
        ]
        process_inputs = [
            hero_dropzone,
            chunk_size_input,
            chunk_overlap_input,
            embedding_model_input,
        ]

        hero_dropzone.upload(
            fn=process_pdf_and_open_workspace,
            inputs=process_inputs,
            outputs=process_outputs,
        ).then(
            fn=lambda status: status,
            inputs=status_output,
            outputs=hero_status_output,
        )

        hero_upload_button.upload(
            fn=process_pdf_and_open_workspace,
            inputs=[
                hero_upload_button,
                chunk_size_input,
                chunk_overlap_input,
                embedding_model_input,
            ],
            outputs=process_outputs,
        ).then(
            fn=lambda status: status,
            inputs=status_output,
            outputs=hero_status_output,
        )

        bottom_upload_button.upload(
            fn=process_pdf_and_open_workspace,
            inputs=[
                bottom_upload_button,
                chunk_size_input,
                chunk_overlap_input,
                embedding_model_input,
            ],
            outputs=process_outputs,
        )

        ask_button.click(
            fn=ask_question_for_demo,
            inputs=[
                current_paper_id,
                current_embedding_model,
                question_input,
                llm_backend_input,
                llm_model_input,
                answer_language_input,
                top_k_input,
                max_context_chars_input,
                max_new_tokens_input,
                temperature_input,
            ],
            outputs=[answer_output, evidence_output],
        )

        visual_ask_button.click(
            fn=ask_visual_question_for_demo,
            inputs=[
                current_paper_id,
                current_embedding_model,
                image_upload,
                question_input,
                visual_answer_language_input,
                visual_top_k_input,
                visual_max_context_chars_input,
                visual_max_new_tokens_input,
                visual_temperature_input,
            ],
            outputs=[visual_answer_output, visual_evidence_output],
        )

    return demo


def launch_demo(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    """Launch the Gradio demo."""
    demo = build_demo()
    demo.launch(server_name=host, server_port=port, share=share, css=APP_CSS)
