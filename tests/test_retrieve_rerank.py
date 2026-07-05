from src.rag.retrieve import (
    diversify_pages,
    is_reference_like_text,
    lexical_overlap_bonus,
    page_prior_bonus,
    rerank_retrieval_candidates,
    section_heading_bonus,
)


def test_is_reference_like_text_detects_bibliography() -> None:
    text = "\n".join(["References", "[1] First paper", "[2] Second paper"])

    assert is_reference_like_text(text)


def test_page_prior_bonus_prefers_early_pages_for_motivation() -> None:
    query = "What is the main problem or motivation addressed by this paper?"

    assert page_prior_bonus(query, 1) > page_prior_bonus(query, 8)


def test_section_heading_bonus_matches_method_heading() -> None:
    query = "What method, model, architecture, or benchmark does the paper propose?"
    text = "Method\nWe propose a new model architecture."

    assert section_heading_bonus(query, text) > 0


def test_lexical_overlap_bonus_rewards_matching_terms() -> None:
    query = "model architecture training objective"
    text = "The model uses an architecture with a training objective."

    assert lexical_overlap_bonus(query, text) > 0


def test_rerank_penalizes_reference_candidates() -> None:
    query = "What method, model, architecture, or benchmark does the paper propose?"
    candidates = [
        {
            "rank": 1,
            "score": 0.70,
            "chunk_id": "ref",
            "page_id": 9,
            "text": "References\n[1] model architecture paper\n[2] method paper\n[3] benchmark paper",
        },
        {
            "rank": 2,
            "score": 0.66,
            "chunk_id": "method",
            "page_id": 2,
            "text": "Method\nWe propose a model architecture and training objective.",
        },
    ]

    reranked = rerank_retrieval_candidates(query, candidates, top_k=2)

    assert reranked[0]["chunk_id"] == "method"


def test_diversify_pages_prefers_unique_pages() -> None:
    candidates = [
        {"rank": 1, "score": 0.9, "page_id": 1},
        {"rank": 2, "score": 0.8, "page_id": 1},
        {"rank": 3, "score": 0.7, "page_id": 2},
    ]

    diversified = diversify_pages(candidates, top_k=2)

    assert [item["page_id"] for item in diversified] == [1, 2]
