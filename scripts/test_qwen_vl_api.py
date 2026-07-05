"""Test qwen3-vl-flash API availability through DashScope."""

import argparse
import os
import sys

from _bootstrap import bootstrap_project

bootstrap_project()

from src.llm.qwen_vl_api import (  # noqa: E402
    DEFAULT_DASHSCOPE_BASE_URL,
    DEFAULT_QWEN_VL_MODEL,
    sanitize_error_message,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test DashScope qwen3-vl-flash API.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_QWEN_VL_MODEL,
        help="DashScope model name.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_DASHSCOPE_BASE_URL,
        help="OpenAI-compatible DashScope base URL.",
    )
    return parser.parse_args()


def main() -> None:
    """Send a minimal test request to qwen3-vl-flash."""
    args = parse_args()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print(
            "DASHSCOPE_API_KEY is not set. Please set it in your environment.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        import httpx
        from openai import OpenAI
    except ImportError as exc:
        print("openai package is not installed. Please install requirements.txt.", file=sys.stderr)
        raise SystemExit(1) from exc

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=args.base_url,
            http_client=httpx.Client(trust_env=False),
        )
        response = client.chat.completions.create(
            model=args.model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Say hello in one sentence.",
                }
            ],
            temperature=0.2,
            max_tokens=64,
        )
    except Exception as exc:
        detail = sanitize_error_message(str(exc))
        print(f"Qwen-VL API test failed: {type(exc).__name__}: {detail}", file=sys.stderr)
        raise SystemExit(1) from exc

    content = response.choices[0].message.content
    print((content or "").strip())


if __name__ == "__main__":
    main()
