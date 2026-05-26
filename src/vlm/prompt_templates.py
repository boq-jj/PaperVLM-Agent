"""Prompt templates for paper and figure understanding."""


FIGURE_QA_PROMPT = (
    "You are a research paper figure understanding assistant. "
    "Answer the question based on the given figure and available context."
)

PAPER_QA_PROMPT = (
    "You are a research paper reading assistant. "
    "Use the retrieved paper context to answer the question accurately."
)


def build_figure_qa_prompt(question: str, context: str | None = None) -> str:
    """Build a prompt for figure question answering.

    Args:
        question: User question.
        context: Optional retrieved paper context.

    Returns:
        Prompt string.
    """
    if context:
        return f"{FIGURE_QA_PROMPT}\nContext:\n{context}\nQuestion: {question}"
    return f"{FIGURE_QA_PROMPT}\nQuestion: {question}"


def build_paper_qa_prompt(question: str, context: str) -> str:
    """Build a prompt for text-based paper question answering."""
    return f"{PAPER_QA_PROMPT}\nContext:\n{context}\nQuestion: {question}"
