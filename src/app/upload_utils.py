"""Upload and UI parameter helpers for application backends."""

import shutil
from pathlib import Path
from typing import Any


def safe_filename(filename: str) -> str:
    """Return a safe file name without any parent directory."""
    name = Path(filename or "").name.strip()
    if not name:
        return "uploaded_paper.pdf"

    safe_chars = []
    for char in name:
        if char.isalnum() or char in {"-", "_", ".", " "}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    safe_name = "".join(safe_chars).strip(" .")
    return safe_name or "uploaded_paper.pdf"


def resolve_uploaded_file(uploaded_file: Any, empty_message: str) -> Path:
    """Resolve an uploaded file object to an existing local file."""
    if uploaded_file is None:
        raise ValueError(empty_message)

    if isinstance(uploaded_file, str):
        file_path = Path(uploaded_file)
    else:
        file_name = getattr(uploaded_file, "name", None)
        if not file_name:
            raise ValueError("不支持的上传文件对象。")
        file_path = Path(file_name)

    if not file_path.exists():
        raise FileNotFoundError(f"上传文件不存在：{file_path}")
    if not file_path.is_file():
        raise ValueError(f"上传路径不是文件：{file_path}")

    return file_path


def ensure_unique_path(path: Path) -> Path:
    """Return a non-existing path by appending a numeric suffix when needed."""
    if not path.exists():
        return path

    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"无法生成不重复的文件名：{path.name}")


def copy_uploaded_file(source_path: Path, target_dir: Path, suffixes: set[str]) -> Path:
    """Copy an uploaded file into a data directory after suffix validation."""
    if source_path.suffix.lower() not in suffixes:
        suffix_text = "、".join(sorted(suffix.strip(".") for suffix in suffixes))
        raise ValueError(f"仅支持 {suffix_text} 文件。")

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = ensure_unique_path(target_dir / safe_filename(source_path.name))
    shutil.copy2(source_path, target_path)
    return target_path


def coerce_positive_int(value: Any, field_name: str) -> int:
    """Convert a UI numeric value to a positive integer."""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是整数。") from exc

    if number <= 0:
        raise ValueError(f"{field_name} 必须大于 0。")

    return number


def coerce_non_negative_int(value: Any, field_name: str) -> int:
    """Convert a UI numeric value to a non-negative integer."""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是整数。") from exc

    if number < 0:
        raise ValueError(f"{field_name} 必须大于等于 0。")

    return number


def coerce_non_negative_float(value: Any, field_name: str) -> float:
    """Convert a UI numeric value to a non-negative float."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是数字。") from exc

    if number < 0:
        raise ValueError(f"{field_name} 必须大于等于 0。")

    return number
