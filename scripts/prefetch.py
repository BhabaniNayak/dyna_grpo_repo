"""Pre-download all heavy assets to the network volume from a cheap CPU pod.

Run this ONCE before claiming a GPU instance, with the volume attached at /workspace.
"""
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from datasets import load_dataset

CACHE = Path(os.environ.get("DYNA_GRPO_WORKSPACE", "/workspace/dyna_grpo"))
HF_CACHE = CACHE / "data" / "hf_cache"
MODEL_CACHE = CACHE / "models"
HF_CACHE.mkdir(parents=True, exist_ok=True)
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

MODELS = ["Qwen/Qwen3-4B-Instruct", "Qwen/Qwen3-0.6B"]
DATASETS = [
    ("Maxwell-Jia/AIME_2024", None),
    ("AI-MO/NuminaMath-CoT", None),
    ("mbpp", "sanitized"),
]


def main():
    for m in MODELS:
        print(f"Downloading model: {m}")
        snapshot_download(m, cache_dir=str(MODEL_CACHE),
                           allow_patterns=["*.json", "*.txt", "*.safetensors",
                                            "*.model", "*tokenizer*"])
    for ds in DATASETS:
        name, config = ds
        print(f"Downloading dataset: {name} ({config})")
        try:
            if config:
                load_dataset(name, config, cache_dir=str(HF_CACHE))
            else:
                load_dataset(name, cache_dir=str(HF_CACHE))
        except Exception as e:
            print(f"  Failed: {e}")
    print("Prefetch done.")


if __name__ == "__main__":
    main()
