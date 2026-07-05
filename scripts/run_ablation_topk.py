"""Run top-k ablation for text RAG evaluation."""

import argparse
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap_project, default_bge_model

bootstrap_project(reexec_venv=True)

from src.eval.ablation import load_jsonl, save_csv, save_json  # noqa: E402
from src.eval.evaluate_text_rag import evaluate_text_rag_examples  # noqa: E402


def default_retriever_model() -> str:
    """Prefer the local BGE fallback model if it exists."""
    return default_bge_model("BAAI/bge-small-en-v1.5")


def parse_top_k_list(value: str) -> list[int]:
    """Parse a comma-separated top-k list."""
    top_k_values: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            top_k = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid top_k value: {item}") from exc
        if top_k <= 0:
            raise argparse.ArgumentTypeError("top_k values must be positive integers.")
        top_k_values.append(top_k)
    if not top_k_values:
        raise argparse.ArgumentTypeError("--top-k-list must contain at least one value.")
    return top_k_values


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run top-k ablation for text RAG.")
    parser.add_argument("--eval-file", default="data/eval/example_text_qa.jsonl")
    parser.add_argument("--paper-id", default="example")
    parser.add_argument(
        "--top-k-list",
        type=parse_top_k_list,
        default=parse_top_k_list("1,3,5,10"),
    )
    parser.add_argument("--output-dir", default="data/eval/ablation")
    parser.add_argument("--index-dir", default="data/extracted/faiss_index")
    parser.add_argument("--llm-backend", choices=["mock", "qwen-vl"], default="mock")
    parser.add_argument("--llm-model-name", default="qwen3-vl-flash")
    parser.add_argument("--embedding-backend", default="local-bge")
    parser.add_argument("--retriever-model-name", default=default_retriever_model())
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def build_table_row(top_k: int, summary: dict[str, Any]) -> dict[str, Any]:
    """Build one CSV-ready top-k summary row."""
    return {
        "top_k": top_k,
        "total": summary.get("total", 0),
        "success": summary.get("success", 0),
        "failed": summary.get("failed", 0),
        "page_hit_rate": summary.get("page_hit_rate", 0.0),
    }


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print a compact top-k ablation summary."""
    print("Top-k ablation summary:")
    for row in rows:
        print(
            f"top_k={row['top_k']} "
            f"total={row['total']} "
            f"success={row['success']} "
            f"failed={row['failed']} "
            f"page_hit_rate={float(row['page_hit_rate']):.4f}"
        )


def main() -> None:
    """Run top-k ablation."""
    args = parse_args()
    examples = load_jsonl(args.eval_file)
    output_dir = Path(args.output_dir)

    setting_outputs: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []

    for top_k in args.top_k_list:
        try:
            evaluation = evaluate_text_rag_examples(
                examples=examples,
                paper_id=args.paper_id or None,
                index_dir=args.index_dir,
                embedding_backend=args.embedding_backend,
                retriever_model_name=args.retriever_model_name,
                llm_backend=args.llm_backend,
                llm_model_name=args.llm_model_name,
                top_k=top_k,
                max_context_chars=args.max_context_chars,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            summary = evaluation["summary"]
            row = build_table_row(top_k, summary)
            setting_outputs.append(
                {
                    "top_k": top_k,
                    "summary": summary,
                    "results": evaluation["results"],
                    "error": "",
                }
            )
        except Exception as exc:
            row = {
                "top_k": top_k,
                "total": len(examples),
                "success": 0,
                "failed": len(examples),
                "page_hit_rate": 0.0,
                "error": str(exc),
            }
            setting_outputs.append(
                {
                    "top_k": top_k,
                    "summary": row,
                    "results": [],
                    "error": str(exc),
                }
            )

        table_rows.append(row)

    result_path = output_dir / "topk_ablation_results.json"
    table_path = output_dir / "topk_ablation_table.csv"
    output = {
        "task": "topk_ablation",
        "eval_file": args.eval_file,
        "paper_id": args.paper_id,
        "llm_backend": args.llm_backend,
        "retriever_model_name": args.retriever_model_name,
        "summary": {
            "by_top_k": table_rows,
        },
        "results": setting_outputs,
    }
    save_json(output, str(result_path))
    save_csv(table_rows, str(table_path))

    print_summary(table_rows)
    print(f"Results saved to: {result_path.as_posix()}")
    print(f"Table saved to: {table_path.as_posix()}")


if __name__ == "__main__":
    main()
