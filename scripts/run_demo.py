"""Command-line entrypoint for launching the Gradio demo."""

import argparse

from _bootstrap import bootstrap_project

bootstrap_project(reexec_venv=True)

from src.app.gradio_app import APP_CSS, build_demo  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Launch the PaperVLM-Agent Gradio demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser.parse_args()


def main() -> None:
    """Launch the local Gradio application."""
    args = parse_args()
    demo = build_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
