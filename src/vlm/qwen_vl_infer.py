"""Qwen-VL inference wrapper."""

from pathlib import Path


def load_vlm_model(model_name: str, device: str = "auto") -> object:
    """Load a multimodal large language model.

    Args:
        model_name: Hugging Face model name or local model path.
        device: Device placement strategy.

    Returns:
        Loaded model wrapper or pipeline.
    """
    raise NotImplementedError("VLM loading will be implemented later.")


def answer_image_question(
    image_path: str | Path,
    question: str,
    model: object | None = None,
) -> str:
    """Answer a question about an image.

    Args:
        image_path: Path to an input image.
        question: User question.
        model: Optional preloaded VLM model.

    Returns:
        Generated answer text.
    """
    raise NotImplementedError("Image-question answering will be implemented later.")
