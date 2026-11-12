"""Download the selected STT model into the image cache at build time."""

import os


def main() -> None:
    provider = os.environ["STT_PROVIDER"]
    model_name = os.environ["STT_PROVIDER_MODEL"]
    if provider == "indicconformer":
        from transformers import AutoModel

        AutoModel.from_pretrained(model_name, trust_remote_code=True)
    elif provider == "parakeet":
        from nemo.collections.asr.models import ASRModel

        ASRModel.from_pretrained(model_name=model_name, map_location="cpu")
    elif provider == "sensevoice":
        from funasr import AutoModel

        AutoModel(model=model_name, device="cpu", disable_update=True)
    else:
        raise ValueError(f"Unsupported STT_PROVIDER: {provider}")


if __name__ == "__main__":
    main()
