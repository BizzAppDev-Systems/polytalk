"""Download all Indic Parler assets required for offline runtime inference."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--description-tokenizer-dir", required=True)
    parser.add_argument("--token-file", required=True)
    args = parser.parse_args()

    token = Path(args.token_file).read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is empty")

    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.model_dir,
        token=token,
    )

    config_path = Path(args.model_dir) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    description_model = config.get("text_encoder", {}).get("_name_or_path")
    if not description_model:
        raise RuntimeError("Model config does not identify its description tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(description_model, token=token)
    tokenizer.save_pretrained(args.description_tokenizer_dir)


if __name__ == "__main__":
    main()
