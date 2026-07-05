"""Run visual input ablation for paper visual QA."""

import argparse
import sys
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap_project, default_bge_model

DEFAULT_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

bootstrap_project(reexec_venv=True)

from src.agent.answer import ask_with_rag  # noqa: E402
from src.agent.vision_answer import ask_with_visual_rag  # noqa: E402
from src.eval.ablation import (  # noqa: E402
    coerce_int_list,
    load_jsonl,
    save_csv,
    save_json,
    summarize_by_setting,
)
from src.eval.evaluate_text_rag import (  # noqa: E402
    compute_page_hit,
    get_retrieved_pages,
)
from src.llm import QwenVLAPILLM  # noqa: E402


def default_retriever_model() -> str:
    """Prefer the local BGE fallback model if it exists."""
    return default_bge_model("BAAI/bge-small-en-v1.5")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run visual input ablation.")
    parser.add_argument("--eval-file", default="data/eval/example_visual_qa.jsonl")
    parser.add_argument("--paper-id", default="example")
    parser.add_argument("--output-dir", default="data/eval/ablation")
    parser.add_argument("--llm-model-name", default="qwen3-vl-flash")
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--index-dir", default="data/extracted/faiss_index")
    parser.add_argument("--retriever-model-name", default=default_retriever_model())
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=3000)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def get_image_path(example: dict[str, Any]) -> str:
    """Read image_path with a page_image fallback."""
    return str(example.get("image_path") or example.get("page_image") or "").strip()


def get_reference_answer(example: dict[str, Any]) -> str:
    """Read reference answer from common field names."""
    return str(example.get("reference_answer") or example.get("answer") or "")


def base_record(example: dict[str, Any], args: argparse.Namespace, mode: str) -> dict[str, Any]:
    """Build the common visual ablation result record."""
    return {
        "id": str(example.get("id", "")),
        "paper_id": args.paper_id or str(example.get("paper_id", "")).strip(),
        "question": str(example.get("question", "")).strip(),
        "question_type": str(example.get("question_type", "unknown") or "unknown"),
        "mode": mode,
        "image_path": get_image_path(example),
        "answer": "",
        "reference_answer": get_reference_answer(example),
        "retrieved_pages": [],
        "page_hit": False,
        "retrieved_chunks": [],
        "error": "",
    }


def run_text_only(example: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Run text-only RAG without image input."""
    record = base_record(example, args, "text_only")
    try:
        rag_result = ask_with_rag(
            question=record["question"],
            paper_id=record["paper_id"],
            index_dir=args.index_dir,
            retriever_model_name=args.retriever_model_name,
            llm_backend="qwen-vl",
            llm_model_name=args.llm_model_name,
            llm_base_url=args.llm_base_url,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            answer_language="en",
        )
        retrieved_chunks = rag_result.get("retrieved_chunks", [])
        retrieved_pages = get_retrieved_pages(retrieved_chunks)
        record.update(
            {
                "answer": str(rag_result.get("answer", "")),
                "retrieved_pages": retrieved_pages,
                "page_hit": compute_page_hit(
                    coerce_int_list(example.get("expected_pages", [])),
                    retrieved_pages,
                ),
                "retrieved_chunks": retrieved_chunks,
            }
        )
    except Exception as exc:
        record["error"] = str(exc)
    return record


def run_image_only(example: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Run image-only QA without retrieved text evidence."""
    record = base_record(example, args, "image_only")
    try:
        image_path = record["image_path"]
        if not image_path:
            raise ValueError("image_path is empty.")
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        prompt = "\n".join(
            [
                "Answer the research paper visual question based only on the image.",
                "Do not use retrieved text evidence.",
                "Give a concise answer in English.",
                "",
                "Question:",
                record["question"],
                "",
                "Answer:",
            ]
        )
        llm = QwenVLAPILLM(
            model_name=args.llm_model_name,
            base_url=args.llm_base_url,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            answer_language="en",
        )
        record["answer"] = llm.generate_with_image(prompt=prompt, image_path=image_path)
    except Exception as exc:
        record["error"] = str(exc)
    return record


def run_text_image(example: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Run visual RAG with both retrieved text and image input."""
    record = base_record(example, args, "text_image")
    try:
        image_path = record["image_path"] or None
        visual_result = ask_with_visual_rag(
            question=record["question"],
            paper_id=record["paper_id"],
            image_path=image_path,
            index_dir=args.index_dir,
            retriever_model_name=args.retriever_model_name,
            llm_model_name=args.llm_model_name,
            llm_base_url=args.llm_base_url,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            answer_language="en",
        )
        retrieved_chunks = visual_result.get("retrieved_chunks", [])
        retrieved_pages = get_retrieved_pages(retrieved_chunks)
        record.update(
            {
                "image_path": str(visual_result.get("image_path", record["image_path"])),
                "answer": str(visual_result.get("answer", "")),
                "retrieved_pages": retrieved_pages,
                "page_hit": compute_page_hit(
                    coerce_int_list(example.get("expected_pages", [])),
                    retrieved_pages,
                ),
                "retrieved_chunks": retrieved_chunks,
            }
        )
    except Exception as exc:
        record["error"] = str(exc)
    return record


def build_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert setting summary into CSV rows."""
    table: list[dict[str, Any]] = []
    for mode, stats in summary["by_setting"].items():
        row = {
            "mode": mode,
            "total": stats.get("total", 0),
            "success": stats.get("success", 0),
            "failed": stats.get("failed", 0),
        }
        if "page_hit_rate" in stats:
            row["page_hit_rate"] = stats["page_hit_rate"]
        table.append(row)
    return table


def print_missing_eval_file(eval_file: str) -> None:
    """Print a clear message for the optional visual QA eval file."""
    print(f"Visual QA eval file not found: {eval_file}", file=sys.stderr)
    print("Create a JSONL file with fields such as:", file=sys.stderr)
    print(
        '{"id":"v001","paper_id":"example","question":"What is shown in Figure 1?",'
        '"image_path":"data/extracted/pages/example_page_3.png",'
        '"reference_answer":"...","expected_pages":[3],"question_type":"figure"}',
        file=sys.stderr,
    )


def print_summary(table: list[dict[str, Any]]) -> None:
    """Print a compact visual ablation summary."""
    print("Visual ablation summary:")
    for row in table:
        message = (
            f"mode={row['mode']} "
            f"total={row['total']} "
            f"success={row['success']} "
            f"failed={row['failed']}"
        )
        if "page_hit_rate" in row:
            message += f" page_hit_rate={float(row['page_hit_rate']):.4f}"
        print(message)


def main() -> None:
    """Run visual input ablation."""
    args = parse_args()
    if not Path(args.eval_file).exists():
        print_missing_eval_file(args.eval_file)
        raise SystemExit(1)

    examples = load_jsonl(args.eval_file)
    output_dir = Path(args.output_dir)

    records: list[dict[str, Any]] = []
    for example in examples:
        records.append(run_text_only(example, args))
        records.append(run_image_only(example, args))
        records.append(run_text_image(example, args))

    summary = summarize_by_setting(records, "mode")
    table = build_table(summary)
    result_path = output_dir / "visual_ablation_results.json"
    table_path = output_dir / "visual_ablation_table.csv"

    output = {
        "task": "visual_ablation",
        "eval_file": args.eval_file,
        "paper_id": args.paper_id,
        "summary": summary,
        "results": records,
    }
    save_json(output, str(result_path))
    save_csv(table, str(table_path))

    print_summary(table)
    print(f"Results saved to: {result_path.as_posix()}")
    print(f"Table saved to: {table_path.as_posix()}")


if __name__ == "__main__":
    main()
