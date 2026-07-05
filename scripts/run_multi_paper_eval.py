"""Download several papers and run a lightweight multi-paper RAG evaluation."""

import argparse
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap_project, default_bge_model
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _bootstrap import bootstrap_project, default_bge_model


DEFAULT_PAPERS = [
    {
        "paper_id": "attention_is_all_you_need",
        "title": "Attention Is All You Need",
        "pdf_url": "https://arxiv.org/pdf/1706.03762",
    },
    {
        "paper_id": "layoutlm",
        "title": "LayoutLM: Pre-training of Text and Layout for Document Image Understanding",
        "pdf_url": "https://arxiv.org/pdf/1912.13318",
    },
    {
        "paper_id": "chartqa",
        "title": "ChartQA: A Benchmark for Question Answering about Charts",
        "pdf_url": "https://arxiv.org/pdf/2203.10244",
    },
]

QUESTION_TEMPLATES = [
    {
        "suffix": "motivation",
        "question": "What is the main problem or motivation addressed by this paper?",
        "question_type": "motivation",
        "keywords": ["abstract", "introduction", "motivation", "problem", "challenge", "limitation"],
        "retrieval_terms": ["abstract", "introduction", "problem", "motivation", "contribution"],
        "section_headings": ["abstract", "introduction"],
        "fallback_pages": [1],
    },
    {
        "suffix": "method",
        "question": "What method, model, architecture, or benchmark does the paper propose?",
        "question_type": "method",
        "keywords": [
            "we propose",
            "we introduce",
            "model",
            "architecture",
            "method",
            "approach",
            "pre-training",
            "fine-tuning",
            "benchmark",
            "encoder",
            "decoder",
        ],
        "retrieval_terms": [
            "proposed method",
            "model architecture",
            "approach",
            "implementation",
            "training objective",
        ],
        "section_headings": ["method", "methods", "approach", "model", "architecture"],
        "fallback_pages": [1, 2],
    },
    {
        "suffix": "experiments",
        "question": "What datasets, experiments, evaluations, or results are reported?",
        "question_type": "experiments",
        "keywords": [
            "experiment",
            "experiments",
            "evaluation",
            "evaluated",
            "dataset",
            "datasets",
            "results",
            "baseline",
            "table",
            "accuracy",
            "performance",
        ],
        "retrieval_terms": ["experiments", "evaluation", "datasets", "results", "baselines", "metrics"],
        "section_headings": ["experiment", "experiments", "evaluation", "results"],
        "fallback_pages": [1, 2, 3],
    },
]


@dataclass(frozen=True)
class PaperSpec:
    """Paper download metadata."""

    paper_id: str
    title: str
    pdf_url: str = ""
    pdf_path: str = ""


def safe_paper_id(value: str) -> str:
    """Convert a title, URL fragment, or ID into a filesystem-safe paper ID."""
    cleaned = []
    for char in value.strip().lower():
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("_")
    paper_id = "".join(cleaned).strip("_")
    while "__" in paper_id:
        paper_id = paper_id.replace("__", "_")
    return paper_id or "paper"


def derive_paper_id_from_url(url: str) -> str:
    """Build a stable paper ID from a PDF URL."""
    parsed = urllib.parse.urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return safe_paper_id(parsed.netloc)
    last = parts[-1]
    if last.lower().endswith(".pdf"):
        last = last[:-4]
    if last.lower() == "pdf" and len(parts) >= 2:
        last = parts[-2]
    return safe_paper_id(last)


def parse_paper_arg(value: str) -> PaperSpec:
    """Parse ``paper_id=url`` or a bare PDF URL from the CLI."""
    if "=" in value:
        paper_id, url = value.split("=", 1)
        return PaperSpec(
            paper_id=safe_paper_id(paper_id),
            title=paper_id.strip() or url,
            pdf_url=url.strip(),
        )

    url = value.strip()
    paper_id = derive_paper_id_from_url(url)
    return PaperSpec(paper_id=paper_id, title=paper_id, pdf_url=url)


def parse_pdf_arg(value: str) -> PaperSpec:
    """Parse ``paper_id=path`` or a bare local PDF path from the CLI."""
    if "=" in value:
        paper_id, path = value.split("=", 1)
        return PaperSpec(
            paper_id=safe_paper_id(paper_id),
            title=paper_id.strip() or Path(path).stem,
            pdf_path=path.strip(),
        )

    path = Path(value)
    paper_id = safe_paper_id(path.stem)
    return PaperSpec(paper_id=paper_id, title=path.stem, pdf_path=str(path))


def collect_pdf_dir(pdf_dir: str) -> list[PaperSpec]:
    """Collect local PDF files recursively from a directory."""
    root = Path(pdf_dir)
    if not root.exists():
        raise FileNotFoundError(f"PDF directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"PDF directory path is not a directory: {root}")

    papers: list[PaperSpec] = []
    for pdf_path in sorted(root.rglob("*.pdf")):
        paper_id = safe_paper_id(pdf_path.stem)
        papers.append(
            PaperSpec(
                paper_id=paper_id,
                title=pdf_path.stem,
                pdf_path=str(pdf_path),
            )
        )
    return papers


def default_papers() -> list[PaperSpec]:
    """Return the built-in paper set."""
    return [
        PaperSpec(
            paper_id=item["paper_id"],
            title=item["title"],
            pdf_url=item["pdf_url"],
        )
        for item in DEFAULT_PAPERS
    ]


def ensure_pdf_downloaded(
    paper: PaperSpec,
    download_dir: Path,
    timeout: int,
    force: bool = False,
) -> Path:
    """Download one PDF unless it already exists."""
    download_dir.mkdir(parents=True, exist_ok=True)
    output_path = download_dir / f"{paper.paper_id}.pdf"
    if output_path.exists() and output_path.stat().st_size > 1024 and not force:
        return output_path

    tmp_path = output_path.with_suffix(".pdf.tmp")
    request = urllib.request.Request(
        paper.pdf_url,
        headers={"User-Agent": "PaperVLM-Agent/0.1"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with tmp_path.open("wb") as file:
                shutil.copyfileobj(response, file)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {paper.pdf_url}") from exc

    if tmp_path.stat().st_size <= 1024:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is too small to be a PDF: {paper.pdf_url}")

    with tmp_path.open("rb") as file:
        header = file.read(5)
    if header != b"%PDF-":
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not a PDF: {paper.pdf_url}")

    tmp_path.replace(output_path)
    return output_path


def infer_expected_pages(
    parsed_pdf: dict[str, Any],
    keywords: list[str],
    fallback_pages: list[int],
    section_headings: list[str] | None = None,
    max_pages: int = 3,
) -> list[int]:
    """Infer expected pages for a smoke-test question by page-level scoring."""
    lowered_keywords = [keyword.lower() for keyword in keywords]
    lowered_headings = [heading.lower() for heading in section_headings or []]
    scored_pages: list[tuple[int, int]] = []

    for page in parsed_pdf.get("pages", []):
        text = str(page.get("text", "") or "").lower()
        page_id = int(page.get("page_id", 0) or 0)
        if page_id <= 0:
            continue
        if is_likely_reference_page(text):
            continue

        score = score_page_for_keywords(
            text=text,
            keywords=lowered_keywords,
            section_headings=lowered_headings,
        )
        if score > 0:
            scored_pages.append((score, page_id))

    if scored_pages:
        scored_pages.sort(key=lambda item: (-item[0], item[1]))
        return sorted(page_id for _, page_id in scored_pages[:max_pages])
    return fallback_pages[:max_pages]


def is_likely_reference_page(text: str) -> bool:
    """Return whether a page is mostly a references or bibliography page."""
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    first_lines = lines[:8]
    has_reference_heading = any(line in {"references", "bibliography"} for line in first_lines)
    citation_like_lines = sum(1 for line in lines if line.startswith("[") or line[:4].isdigit())
    return has_reference_heading or citation_like_lines >= max(8, len(lines) // 3)


def score_page_for_keywords(
    text: str,
    keywords: list[str],
    section_headings: list[str],
) -> int:
    """Score how likely a page is to answer a smoke-test question."""
    score = 0
    for keyword in keywords:
        if not keyword:
            continue
        count = text.count(keyword)
        score += min(count, 5)

    for line in text.splitlines():
        line = line.strip().lower().rstrip(":")
        if any(line == heading or line.startswith(f"{heading} ") for heading in section_headings):
            score += 6

    return score


def extract_title_from_first_page(parsed_pdf: dict[str, Any]) -> str:
    """Extract a compact title hint from the first parsed page."""
    pages = parsed_pdf.get("pages", [])
    if not pages:
        return ""
    first_text = str(pages[0].get("text", "") or "")
    lines = [line.strip() for line in first_text.splitlines() if line.strip()]
    title_lines: list[str] = []
    for line in lines[:5]:
        lowered = line.lower()
        if lowered in {"abstract", "introduction"} or "@" in line:
            break
        title_lines.append(line)
        if len(" ".join(title_lines)) >= 120:
            break
    return " ".join(title_lines).strip()


def build_retrieval_query(
    question: str,
    paper_title: str,
    terms: list[str],
) -> str:
    """Build a focused retrieval query while keeping the answer question unchanged."""
    query_parts = [question]
    if paper_title:
        query_parts.append(f"Paper title: {paper_title}.")
    if terms:
        query_parts.append("Relevant section terms: " + ", ".join(terms) + ".")
    query_parts.append("Prefer content sections over references or bibliography.")
    return " ".join(query_parts)


def build_smoke_examples(parsed_pdf: dict[str, Any], paper_title: str = "") -> list[dict[str, Any]]:
    """Build a small QA set for checking retrieval over one parsed paper."""
    paper_id = str(parsed_pdf.get("paper_id", "")).strip()
    if not paper_id:
        raise ValueError("parsed_pdf is missing paper_id.")

    title_hint = paper_title.strip() or extract_title_from_first_page(parsed_pdf)
    examples: list[dict[str, Any]] = []
    for template in QUESTION_TEMPLATES:
        expected_pages = infer_expected_pages(
            parsed_pdf=parsed_pdf,
            keywords=list(template["keywords"]),
            fallback_pages=list(template["fallback_pages"]),
            section_headings=list(template["section_headings"]),
        )
        retrieval_query = build_retrieval_query(
            question=str(template["question"]),
            paper_title=title_hint,
            terms=list(template["retrieval_terms"]),
        )
        examples.append(
            {
                "id": f"{paper_id}_{template['suffix']}",
                "paper_id": paper_id,
                "question": template["question"],
                "retrieval_query": retrieval_query,
                "expected_pages": expected_pages,
                "reference_answer": "Smoke-test reference; answer quality is not manually annotated.",
                "question_type": template["question_type"],
            }
        )
    return examples


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download multiple paper PDFs and run a lightweight RAG smoke evaluation.",
    )
    parser.add_argument(
        "--paper",
        action="append",
        default=[],
        help="Paper spec as paper_id=url, or a bare PDF URL. Repeat for multiple papers.",
    )
    parser.add_argument(
        "--pdf",
        action="append",
        default=[],
        help="Local PDF spec as paper_id=path, or a bare PDF path. Repeat for multiple PDFs.",
    )
    parser.add_argument(
        "--pdf-dir",
        action="append",
        default=[],
        help="Directory containing local PDF files. Repeat to include multiple directories.",
    )
    parser.add_argument("--download-dir", default="data/raw_papers/batch_eval")
    parser.add_argument("--eval-file", default="data/eval/multi_paper_smoke_qa.jsonl")
    parser.add_argument("--output-file", default="data/eval/results/multi_paper_smoke_results.json")
    parser.add_argument("--index-dir", default="data/extracted/faiss_index")
    parser.add_argument("--text-dir", default="data/extracted/text")
    parser.add_argument("--page-dir", default="data/extracted/pages")
    parser.add_argument("--chunk-dir", default="data/extracted/chunks")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--zoom", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--llm-backend", choices=["qwen-vl", "mock"], default="mock")
    parser.add_argument("--llm-model-name", default="qwen3-vl-flash")
    parser.add_argument("--retriever-model-name", default=default_bge_model("BAAI/bge-small-en-v1.5"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def print_paper_summary(record: dict[str, Any]) -> None:
    """Print one paper pipeline summary."""
    print(
        "  "
        f"{record['paper_id']}: pages={record['num_pages']} "
        f"chunks={record['num_chunks']} pdf={record['pdf_path']}"
    )


def main() -> None:
    """Run the full multi-paper smoke evaluation pipeline."""
    bootstrap_project(reexec_venv=True)

    from src.eval.evaluate_text_rag import evaluate_text_rag_examples
    from src.eval.io_utils import save_json, save_jsonl
    from src.pdf.parse_pdf import parse_pdf
    from src.rag.build_index import build_index_from_chunks
    from src.rag.chunk_text import chunk_parsed_pdf

    args = parse_args()
    papers: list[PaperSpec] = []
    papers.extend(parse_paper_arg(value) for value in args.paper)
    papers.extend(parse_pdf_arg(value) for value in args.pdf)
    for pdf_dir in args.pdf_dir:
        papers.extend(collect_pdf_dir(pdf_dir))
    if not papers:
        papers = default_papers()

    all_examples: list[dict[str, Any]] = []
    paper_records: list[dict[str, Any]] = []

    print("Running multi-paper pipeline:")
    for paper in papers:
        if paper.pdf_path:
            pdf_path = Path(paper.pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"Local PDF not found: {pdf_path}")
            if not pdf_path.is_file():
                raise ValueError(f"Local PDF path is not a file: {pdf_path}")
        elif args.skip_download:
            pdf_path = Path(args.download_dir) / f"{paper.paper_id}.pdf"
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found with --skip-download: {pdf_path}")
        else:
            pdf_path = ensure_pdf_downloaded(
                paper=paper,
                download_dir=Path(args.download_dir),
                timeout=args.timeout,
                force=args.force_download,
            )

        parsed_pdf = parse_pdf(
            pdf_path=str(pdf_path),
            output_text_dir=args.text_dir,
            output_page_dir=args.page_dir,
            zoom=args.zoom,
        )
        chunks = chunk_parsed_pdf(
            input_json_path=str(Path(args.text_dir) / f"{parsed_pdf['paper_id']}.json"),
            output_dir=args.chunk_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        if not args.skip_index:
            build_index_from_chunks(
                chunks_path=str(Path(args.chunk_dir) / f"{parsed_pdf['paper_id']}_chunks.json"),
                output_dir=args.index_dir,
                model_name=args.retriever_model_name,
            )

        examples = build_smoke_examples(parsed_pdf, paper_title=paper.title)
        all_examples.extend(examples)
        paper_records.append(
            {
                "paper_id": parsed_pdf["paper_id"],
                "title": paper.title,
                "pdf_url": paper.pdf_url,
                "pdf_path": pdf_path.as_posix(),
                "num_pages": parsed_pdf["num_pages"],
                "num_chunks": chunks["num_chunks"],
                "num_examples": len(examples),
            }
        )
        print_paper_summary(paper_records[-1])

    save_jsonl(all_examples, args.eval_file)
    output = evaluate_text_rag_examples(
        examples=all_examples,
        index_dir=args.index_dir,
        retriever_model_name=args.retriever_model_name,
        llm_backend=args.llm_backend,
        llm_model_name=args.llm_model_name,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    output["papers"] = paper_records
    output["notes"] = [
        "This is a lightweight smoke evaluation.",
        "Expected pages are inferred from keyword matches, not human annotation.",
        "Answer quality is not a benchmark metric in this output.",
    ]
    save_json(output, args.output_file)

    summary = output["summary"]
    print("Evaluation summary:")
    print(f"  papers: {len(paper_records)}")
    print(f"  total: {summary['total']}")
    print(f"  success: {summary['success']}")
    print(f"  failed: {summary['failed']}")
    print(f"  page_hit_rate: {summary['page_hit_rate']:.4f}")
    print(f"Eval file saved to: {Path(args.eval_file).as_posix()}")
    print(f"Results saved to: {Path(args.output_file).as_posix()}")


if __name__ == "__main__":
    main()
