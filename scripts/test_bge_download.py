"""Test downloading and loading the BGE embedding model."""

import os
import sys
import traceback
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_NAME = "BAAI/bge-small-en-v1.5"
LOCAL_MIRROR_MODEL_DIR = PROJECT_ROOT / "models" / "bge-small-en-v1.5-hf-mirror"
REQUIRED_MODEL_FILES = [
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
]


def print_environment() -> None:
    """Print Python and Hugging Face environment configuration."""
    print(f"Python executable: {sys.executable}")
    for name in [
        "HF_ENDPOINT",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "HF_HUB_DISABLE_XET",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    ]:
        print(f"{name}: {os.getenv(name, '<not set>')}")


def configure_hf_http_client() -> None:
    """Force Hugging Face Hub to ignore broken proxy environment variables."""
    import httpx
    import huggingface_hub

    def client_factory() -> httpx.Client:
        return httpx.Client(trust_env=False, timeout=60.0)

    huggingface_hub.set_client_factory(client_factory)


def check_mirror_connection() -> None:
    """Check whether the mirror endpoint is reachable without environment proxies."""
    import httpx

    endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/{MODEL_NAME}/resolve/main/modules.json"
    print(f"\nTesting mirror URL: {url}")
    with httpx.Client(trust_env=False, timeout=30.0, follow_redirects=True) as client:
        response = client.get(url)
        print(f"mirror GET status: {response.status_code}")
        response.raise_for_status()


def download_files_from_mirror(files: Iterable[str], output_dir: Path) -> None:
    """Download model files from the configured HF mirror using direct GET requests."""
    import httpx

    endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    output_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(trust_env=False, timeout=120.0, follow_redirects=True) as client:
        for relative_path in files:
            url = f"{endpoint}/{MODEL_NAME}/resolve/main/{relative_path}"
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Already exists: {output_path}")
                continue

            print(f"Downloading {relative_path} -> {output_path}")
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with output_path.open("wb") as file:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        file.write(chunk)


def load_and_encode(model_name_or_path: str) -> None:
    """Load a SentenceTransformer model and print embedding diagnostics."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {model_name_or_path}")
    model = SentenceTransformer(model_name_or_path)

    texts = [
        "The Transformer is based on attention mechanisms.",
        "Scaled dot-product attention computes attention weights.",
    ]
    print("Encoding test sentences...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"embedding shape: {embeddings.shape}")
    print(f"dtype: {embeddings.dtype}")
    print(f"first 5 values: {embeddings[0][:5].tolist()}")


def main() -> None:
    """Load BGE and run a minimal embedding test."""
    print_environment()

    try:
        configure_hf_http_client()
        check_mirror_connection()

        print("\nImporting sentence_transformers...")
        try:
            load_and_encode(MODEL_NAME)
        except Exception as direct_exc:
            print(
                "\nDirect SentenceTransformer load failed. "
                "Trying direct file download from HF mirror into a local model directory..."
            )
            print(f"Direct load error: {type(direct_exc).__name__}: {direct_exc}")
            download_files_from_mirror(REQUIRED_MODEL_FILES, LOCAL_MIRROR_MODEL_DIR)
            load_and_encode(str(LOCAL_MIRROR_MODEL_DIR))
            print(f"\nLocal mirror model is ready: {LOCAL_MIRROR_MODEL_DIR}")
            print(
                "Use this model path when building an index: "
                "models/bge-small-en-v1.5-hf-mirror"
            )
    except Exception as exc:
        print(f"\nBGE download/load test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("\nFull traceback:", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
