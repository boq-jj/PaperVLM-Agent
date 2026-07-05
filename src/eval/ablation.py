"""Lightweight ablation study helpers."""

from collections import OrderedDict
from typing import Any

from src.eval.io_utils import load_jsonl, safe_rate, save_csv, save_json


def summarize_by_setting(
    results: list[dict[str, Any]],
    setting_key: str,
) -> dict[str, Any]:
    """Summarize ablation results grouped by a setting key."""
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for result in results:
        setting = str(result.get(setting_key, "unknown"))
        grouped.setdefault(setting, []).append(result)

    by_setting: dict[str, dict[str, Any]] = {}
    for setting, records in grouped.items():
        total = len(records)
        failed = sum(1 for record in records if str(record.get("error", "") or "").strip())
        success = total - failed
        summary: dict[str, Any] = {
            setting_key: setting,
            "total": total,
            "success": success,
            "failed": failed,
        }

        if any("page_hit" in record for record in records):
            page_records = [record for record in records if "page_hit" in record]
            page_hit_count = sum(1 for record in page_records if bool(record.get("page_hit")))
            summary["page_hit_count"] = page_hit_count
            summary["page_hit_rate"] = safe_rate(page_hit_count, len(page_records))

        if any("exact_match" in record for record in records):
            exact_records = [record for record in records if "exact_match" in record]
            exact_count = sum(1 for record in exact_records if bool(record.get("exact_match")))
            summary["exact_match_count"] = exact_count
            summary["exact_match_rate"] = safe_rate(exact_count, len(exact_records))

        if any("relaxed_match" in record for record in records):
            relaxed_records = [record for record in records if "relaxed_match" in record]
            relaxed_count = sum(1 for record in relaxed_records if bool(record.get("relaxed_match")))
            summary["relaxed_match_count"] = relaxed_count
            summary["relaxed_match_rate"] = safe_rate(relaxed_count, len(relaxed_records))

        by_setting[setting] = summary

    return {
        "total": len(results),
        "setting_key": setting_key,
        "by_setting": by_setting,
    }


def coerce_int_list(value: Any) -> list[int]:
    """Coerce a list-like value into integers, skipping invalid items."""
    if not isinstance(value, list):
        return []

    numbers: list[int] = []
    for item in value:
        try:
            numbers.append(int(item))
        except (TypeError, ValueError):
            continue
    return numbers
