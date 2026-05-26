"""Simple planning utilities for selecting paper understanding tools."""


def plan_tools(question: str, has_image: bool = False, has_pdf: bool = False) -> list[str]:
    """Plan which tools should be used for a user question.

    Args:
        question: User question.
        has_image: Whether an image input is available.
        has_pdf: Whether a PDF input is available.

    Returns:
        Ordered tool names to execute.
    """
    raise NotImplementedError("Agent planning will be implemented later.")
