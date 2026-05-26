"""Image helper functions for Qwen-VL API calls."""

import base64
from pathlib import Path


def get_image_mime_type(image_path: str) -> str:
    """Return the MIME type for an image path based on file suffix."""
    suffix = Path(image_path).suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return a base64 string."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")

    return base64.b64encode(path.read_bytes()).decode("utf-8")


def image_to_data_url(image_path: str) -> str:
    """Convert an image file to a data URL."""
    mime_type = get_image_mime_type(image_path)
    image_base64 = encode_image_to_base64(image_path)
    return f"data:{mime_type};base64,{image_base64}"


def check_image_size(image_path: str, max_mb: float = 9.0) -> None:
    """Validate image file size for DashScope base64 image input."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(
            "DashScope base64 image input should be smaller than 10 MB, "
            "please reduce page render zoom or compress image."
        )
