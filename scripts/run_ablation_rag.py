"""Run no-RAG versus text-RAG ablation."""

import argparse
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap_project, default_bge_model

DEFAULT_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

bootstrap_project(reexec_venv=True)

from src.agent.answer import ask_with_rag  # noqa: E402
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
from src.llm import MockLLM, QwenVLAPILLM  # noqa: E402


def default_retriever_model() -> str:
    """Prefer the local BGE fallback model if it exists."""
    return default_bge_model("BAAI/bge-small-en-v1.5")


def direct_answer_without_rag(
    question: str,
    llm_backend: str = "qwen-vl",
    llm_model_name: str = "qwen3-vl-flash",
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """Answer directly with only the question and no retrieved evidence."""
    prompt = "\n".join(
        [
            "You are a research paper assistant.",
            "Answer the question directly without retrieved evidence.",
            "If the question cannot be answered from the question alone, say insufficient evidence.",
            "",
            "Question:",
            question,
            "",
            "Answer:",
        ]
    )

    if llm_backend == "mock":
        llm = MockLLM(answer_language="en")
    elif llm_backend == "qwen-vl":
        llm = QwenVLAPILLM(
            model_name=llm_model_name,
            base_url=llm_base_url,
            max_tokens=max_new_tokens,
            temperature=temperature,
            answer_language="en",
        )
    else:
        raise ValueError("llm_backend must be either 'qwen-vl' or 'mock'.")

    return llm.generate(prompt)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run no-RAG versus text-RAG ablation.")
    parser.add_argument("--eval-file", default="data/eval/example_text_qa.jsonl")
    parser.add_argument("--paper-id", default="example")
    parser.add_argument("--output-dir", default="data/eval/ablation")
    parser.add_argument("--llm-model-name", default="qwen3-vl-flash")
    parser.add_argument("--llm-backend", choices=["qwen-vl", "mock"], default="qwen-vl")
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--index-dir", default="data/extracted/faiss_index")
    parser.add_argument("--retriever-model-name", default=default_retriever_model())
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def run_no_rag_record(
    example: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one no-RAG ablation example."""
    question = str(example.get("question", "")).strip()
    record = {
        "id": str(example.get("id", "")),
        "paper_id": args.paper_id or str(example.get("paper_id", "")).strip(),
        "question": question,
        "question_type": str(example.get("question_type", "unknown") or "unknown"),
        "method": "no_rag",
        "answer": "",
        "reference_answer": str(example.get("reference_answer", "")),
        "error": "",
    }

    try:
        record["answer"] = direct_answer_without_rag(
            question=question,
            llm_backend=args.llm_backend,
            llm_model_name=args.llm_model_name,
            llm_base_url=args.llm_base_url,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    except Exception as exc:
        record["error"] = str(exc)

    return record


def run_text_rag_record(
    example: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one text-RAG ablation example."""
    current_paper_id = args.paper_id or str(example.get("paper_id", "")).strip()
    question = str(example.get("question", "")).strip()
    expected_pages = coerce_int_list(example.get("expected_pages", []))
    record: dict[str, Any] = {
        "id": str(example.get("id", "")),
        "paper_id": current_paper_id,
        "question": question,
        "question_type": str(example.get("question_type", "unknown") or "unknown"),
        "method": "text_rag",
        "answer": "",
        "reference_answer": str(example.get("reference_answer", "")),
        "retrieved_pages": [],
        "page_hit": False,
        "retrieved_chunks": [],
        "error": "",
    }

    try:
        rag_result = ask_with_rag(
            question=question,
            paper_id=current_paper_id,
            index_dir=args.index_dir,
            retriever_model_name=args.retriever_model_name,
            llm_backend=args.llm_backend,
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
                "page_hit": compute_page_hit(expected_pages, retrieved_pages),
                "retrieved_chunks": retrieved_chunks,
            }
        )
    except Exception as exc:
        record["error"] = str(exc)

    return record


def build_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert setting summary into CSV rows."""
    table: list[dict[str, Any]] = []
    for method, stats in summary["by_setting"].items():
        row = {
            "method": method,
            "total": stats.get("total", 0),
            "success": stats.get("success", 0),
            "failed": stats.get("failed", 0),
        }
        if "page_hit_rate" in stats:
            row["page_hit_rate"] = stats["page_hit_rate"]
        table.append(row)
    return table


def print_summary(table: list[dict[str, Any]]) -> None:
    """Print a compact RAG ablation summary."""
    print("RAG ablation summary:")
    for row in table:
        message = (
            f"method={row['method']} "
            f"total={row['total']} "
            f"success={row['success']} "
            f"failed={row['failed']}"
        )
        if "page_hit_rate" in row:
            message += f" page_hit_rate={float(row['page_hit_rate']):.4f}"
        print(message)


def main() -> None:
    """Run no-RAG and text-RAG ablation."""
    args = parse_args()
    examples = load_jsonl(args.eval_file)
    output_dir = Path(args.output_dir)

    records: list[dict[str, Any]] = []
    for example in examples:
        records.append(run_no_rag_record(example, args))
        records.append(run_text_rag_record(example, args))

    summary = summarize_by_setting(records, "method")
    table = build_table(summary)
    result_path = output_dir / "rag_ablation_results.json"
    table_path = output_dir / "rag_ablation_table.csv"

    output = {
        "task": "rag_ablation",
        "eval_file": args.eval_file,
        "paper_id": args.paper_id,
        "llm_backend": args.llm_backend,
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
