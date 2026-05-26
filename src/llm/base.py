"""Base interface for LLM backends."""


class BaseLLM:
    """Minimal text generation interface."""

    def generate(self, prompt: str) -> str:
        """Generate a text answer from a prompt."""
        raise NotImplementedError
