"""Dataset loaders. All data pulled from HF Hub on first call, cached on volume."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Iterator

from datasets import load_dataset, Dataset

from .config import PATHS

_CACHE_DIR = PATHS["data"] / "hf_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load(name: str, *args, **kwargs) -> Dataset:
    return load_dataset(name, *args, cache_dir=str(_CACHE_DIR), **kwargs)


# ---------------- Benchmarks ----------------
def aime_2024() -> Dataset:
    """AIME 2024 (30 problems). Fields: problem (str), answer (int 0-999)."""
    ds = _load("Maxwell-Jia/AIME_2024", split="train")
    return ds.rename_columns({"Problem": "problem", "Answer": "answer"}) \
        if "Problem" in ds.column_names else ds


def aime_2025() -> Dataset:
    """AIME 2025 problems for held-out eval."""
    try:
        ds = _load("opencompass/AIME2025", "AIME2025-I", split="test")
    except Exception:
        ds = _load("yentinglin/aime_2025", split="train")
    return ds


def gpqa_diamond(limit: int | None = None) -> Dataset:
    ds = _load("Idavidrein/gpqa", "gpqa_diamond", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def livecodebench(limit: int | None = None) -> Dataset:
    """LiveCodeBench v6 problems (subset for our short window)."""
    try:
        ds = _load("livecodebench/code_generation_lite",
                   version_tag="release_v6", split="test")
    except Exception:
        ds = _load("livecodebench/code_generation_lite", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


# ---------------- Trace-collection prompts ----------------
def numina_math_subset(n: int = 3000) -> Dataset:
    ds = _load("AI-MO/NuminaMath-CoT", split="train")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    # Field: 'problem'. Expose as ('prompt', 'kind').
    return ds.map(lambda r: {"prompt": r["problem"], "kind": "math"},
                  remove_columns=[c for c in ds.column_names if c not in {"problem"}])


def collection_prompts(n_math: int = 3000, n_code: int = 1500,
                        n_knowledge: int = 500) -> list[dict]:
    """Mixed prompt pool used for trace collection (Step 2)."""
    rows: list[dict] = []
    try:
        for r in numina_math_subset(n_math):
            rows.append({"prompt": r["prompt"], "kind": "math"})
    except Exception:
        pass
    try:
        # MBPP problems for code-stressed traces
        mbpp = _load("mbpp", "sanitized", split="train").shuffle(seed=1).select(
            range(min(n_code, 974)))
        for r in mbpp:
            rows.append({"prompt": r["text"] + "\n\nWrite Python that solves this.",
                         "kind": "code"})
    except Exception:
        pass
    try:
        gpqa = _load("Idavidrein/gpqa", "gpqa_main", split="train").shuffle(seed=2)
        for r in gpqa.select(range(min(n_knowledge, len(gpqa)))):
            rows.append({"prompt": r["Question"], "kind": "knowledge"})
    except Exception:
        pass
    return rows


# ---------------- Reward extraction ----------------
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None
