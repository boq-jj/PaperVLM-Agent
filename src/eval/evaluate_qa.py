"""Question-answering evaluation utilities."""


def evaluate_predictions(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Evaluate generated answers against references.

    Args:
        predictions: Model-generated answers.
        references: Ground-truth answers.

    Returns:
        Metric names and scores.
    """
    raise NotImplementedError("QA evaluation will be implemented later.")
