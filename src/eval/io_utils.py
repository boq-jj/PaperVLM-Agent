"""Shared file I/O helpers for evaluation modules."""

import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load a UTF-8 JSONL file into a list of dictionaries."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"JSONL path is not a file: {file_path}")

    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {file_path}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_number} must be a JSON object: {file_path}")
            records.append(item)
    return records


def save_json(data: dict[str, Any], path: str) -> Path:
    """Save a dictionary as formatted UTF-8 JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return output_path


def save_jsonl(records: list[dict[str, Any]], path: str) -> Path:
    """Save records as UTF-8 JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def save_csv(records: list[dict[str, Any]], path: str) -> Path:
    """Save records as a UTF-8 CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = collect_fieldnames(records)

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        if not fieldnames:
            return output_path
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: csv_value(record.get(key)) for key in fieldnames})

    return output_path


def collect_fieldnames(records: list[dict[str, Any]]) -> list[str]:
    """Collect CSV field names while preserving first-seen order."""
    fieldnames: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def csv_value(value: Any) -> Any:
    """Convert nested values into stable CSV cell strings."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def safe_rate(numerator: int, denominator: int) -> float:
    """Compute a rate while handling empty denominators."""
    if denominator == 0:
        return 0.0
    return numerator / denominator
