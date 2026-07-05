from pathlib import Path

from scripts.run_multi_paper_eval import (
    build_smoke_examples,
    collect_pdf_dir,
    extract_title_from_first_page,
    derive_paper_id_from_url,
    infer_expected_pages,
    is_likely_reference_page,
    parse_pdf_arg,
    parse_paper_arg,
    safe_paper_id,
)


def test_safe_paper_id_normalizes_unsafe_characters() -> None:
    assert safe_paper_id("LayoutLM: Paper v1.0!") == "layoutlm_paper_v1_0"


def test_parse_paper_arg_accepts_explicit_id_and_url() -> None:
    paper = parse_paper_arg("chartqa=https://arxiv.org/pdf/2203.10244")

    assert paper.paper_id == "chartqa"
    assert paper.pdf_url == "https://arxiv.org/pdf/2203.10244"


def test_parse_pdf_arg_accepts_explicit_id_and_path() -> None:
    paper = parse_pdf_arg("local_paper=data/raw_papers/example.pdf")

    assert paper.paper_id == "local_paper"
    assert paper.pdf_path == "data/raw_papers/example.pdf"


def test_collect_pdf_dir_discovers_pdfs_recursively() -> None:
    tmp_root = Path(__file__).with_name("__tmp_pdf_dir")
    nested_dir = tmp_root / "nested"
    pdf_path = nested_dir / "Paper One.pdf"

    try:
        nested_dir.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-")

        papers = collect_pdf_dir(str(tmp_root))

        assert len(papers) == 1
        assert papers[0].paper_id == "paper_one"
        assert papers[0].pdf_path.endswith("Paper One.pdf")
    finally:
        pdf_path.unlink(missing_ok=True)
        nested_dir.rmdir()
        tmp_root.rmdir()


def test_derive_paper_id_from_arxiv_pdf_url() -> None:
    assert derive_paper_id_from_url("https://arxiv.org/pdf/1912.13318") == "1912_13318"


def test_infer_expected_pages_prefers_keyword_matches() -> None:
    parsed_pdf = {
        "pages": [
            {"page_id": 1, "text": "Abstract and introduction"},
            {"page_id": 2, "text": "Experiments\nExperiments and results results results"},
        ],
    }

    assert infer_expected_pages(parsed_pdf, ["results"], [1], ["experiments"]) == [2]


def test_infer_expected_pages_skips_reference_pages() -> None:
    parsed_pdf = {
        "pages": [
            {"page_id": 1, "text": "Experiments\nResults and evaluation"},
            {"page_id": 9, "text": "References\n[1] results benchmark method\n[2] evaluation dataset"},
        ],
    }

    assert infer_expected_pages(parsed_pdf, ["results", "evaluation"], [1]) == [1]
    assert is_likely_reference_page("References\n[1] A paper\n[2] Another paper")


def test_extract_title_from_first_page_uses_early_lines() -> None:
    parsed_pdf = {
        "pages": [
            {
                "page_id": 1,
                "text": "BERT: Pre-training of Deep Bidirectional Transformers\nfor Language Understanding\nAuthor Name\nAbstract\nWe introduce...",
            }
        ],
    }

    assert "BERT" in extract_title_from_first_page(parsed_pdf)


def test_build_smoke_examples_uses_paper_id_and_three_templates() -> None:
    parsed_pdf = {
        "paper_id": "example",
        "pages": [{"page_id": 1, "text": "Abstract. We propose a benchmark with results."}],
    }

    examples = build_smoke_examples(parsed_pdf)

    assert len(examples) == 3
    assert {example["paper_id"] for example in examples} == {"example"}
    assert all(example["retrieval_query"] for example in examples)
