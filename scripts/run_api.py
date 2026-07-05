"""Command-line entrypoint for the PaperVLM-Agent FastAPI backend."""

import argparse
import sys

from _bootstrap import bootstrap_project

bootstrap_project(reexec_venv=True)

import uvicorn  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the PaperVLM-Agent REST API.")
    parser.add_argument("--host", default="127.0.0.1", help="API bind host.")
    parser.add_argument("--port", type=int, default=8000, help="API bind port.")
    parser.add_argument("--reload", action="store_true", help="Enable reload mode.")
    return parser.parse_args()


def main() -> None:
    """Run the FastAPI backend with Uvicorn."""
    args = parse_args()
    try:
        uvicorn.run(
            "src.app.api_app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    except Exception as exc:
        print(f"Failed to start API server: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
