"""Qwen-VL API backend using DashScope OpenAI-compatible interface."""

import os
import re

from src.llm.base import BaseLLM
from src.vision.image_utils import check_image_size, image_to_data_url


DEFAULT_QWEN_VL_MODEL = "qwen3-vl-flash"
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def sanitize_error_message(message: str) -> str:
    """Remove likely API keys from an error message before displaying it."""
    return re.sub(r"sk-[A-Za-z0-9_\-]+", "sk-***", message)


class QwenVLAPILLM(BaseLLM):
    """Text generation backend for qwen3-vl-flash via DashScope."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_QWEN_VL_MODEL,
        base_url: str = DEFAULT_DASHSCOPE_BASE_URL,
        max_tokens: int = 512,
        temperature: float = 0.2,
        answer_language: str = "zh",
    ) -> None:
        """Initialize the OpenAI-compatible DashScope client."""
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set. Please set it in your environment.")

        try:
            import httpx
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is not installed. Please install requirements.txt.") from exc

        self.model_name = model_name
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.answer_language = self._normalize_answer_language(answer_language)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(trust_env=False),
        )

    @staticmethod
    def _normalize_answer_language(answer_language: str) -> str:
        """Normalize language options to ``zh`` or ``en``."""
        language = (answer_language or "zh").strip().lower()
        if language in {"en", "english", "英文"}:
            return "en"
        return "zh"

    def _text_system_prompt(self) -> str:
        """Return the system prompt for text RAG generation."""
        if self.answer_language == "en":
            return (
                "You are a rigorous research paper assistant. "
                "Answer strictly based on the provided evidence and respond in English. "
                "If the evidence is insufficient, clearly say insufficient evidence."
            )
        return (
            "你是一个严谨的科研论文问答助手。"
            "请严格根据给定证据回答，并使用中文输出。"
            "如果证据不足，请明确说明证据不足。"
        )

    def _vision_system_prompt(self) -> str:
        """Return the system prompt for visual RAG generation."""
        if self.answer_language == "en":
            return (
                "You are a rigorous research paper visual question answering assistant. "
                "Answer based on the paper page image and retrieved evidence, and respond in English. "
                "If the evidence is insufficient, clearly say insufficient evidence."
            )
        return (
            "你是一个严谨的科研论文图文问答助手。"
            "请根据论文页面图像和检索证据回答，并使用中文输出。"
            "如果证据不足，请明确说明证据不足。"
        )

    def generate(self, prompt: str) -> str:
        """Generate an answer using qwen3-vl-flash."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._text_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            detail = sanitize_error_message(str(exc))
            raise RuntimeError(f"Qwen-VL API call failed: {detail}") from exc

        content = response.choices[0].message.content
        return (content or "").strip()

    def generate_with_image(self, prompt: str, image_path: str) -> str:
        """Generate an answer using text evidence and one image."""
        check_image_size(image_path, max_mb=9.0)
        image_data_url = image_to_data_url(image_path)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._vision_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url,
                                },
                            },
                        ],
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            detail = sanitize_error_message(str(exc))
            raise RuntimeError(f"Qwen-VL image API call failed: {detail}") from exc

        content = response.choices[0].message.content
        return (content or "").strip()
