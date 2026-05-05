"""Rule-based reward functions per benchmark. No learned reward model."""
from __future__ import annotations
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from .data import extract_answer


def _normalize_num(s: str) -> Optional[float]:
    s = s.strip().replace(",", "").replace("$", "")
    s = re.sub(r"\\boxed\{?", "", s).rstrip("}")
    try:
        return float(s)
    except ValueError:
        return None


# ---------------- AIME ----------------
def reward_aime(generation: str, gold_answer: int | str) -> float:
    """1.0 if numeric answer matches; 0.5 partial if format ok; 0 otherwise."""
    ans = extract_answer(generation)
    if ans is None:
        return 0.0
    n = _normalize_num(ans)
    if n is None:
        return 0.0
    try:
        gold = float(gold_answer)
    except (ValueError, TypeError):
        return 0.0
    return 1.0 if abs(n - gold) < 1e-3 else 0.0


# ---------------- GPQA ----------------
def reward_gpqa(generation: str, correct_answer: str,
                 distractors: list[str]) -> float:
    """Match against the correct multiple-choice option (free-form)."""
    ans = extract_answer(generation)
    if ans is None:
        return 0.0
    a_low = ans.lower().strip()
    c_low = correct_answer.lower().strip()
    if c_low and (c_low in a_low or a_low in c_low):
        for d in distractors:
            if d and d.lower().strip() in a_low:
                return 0.0
        return 1.0
    return 0.0


# ---------------- LiveCodeBench ----------------
def reward_lcb(generation: str, sample: dict, timeout_s: float = 6.0) -> float:
    """Extract code from generation, run against test cases."""
    code = _extract_code_block(generation)
    if not code:
        ans = extract_answer(generation)
        code = _extract_code_block(ans or "")
    if not code:
        return 0.0
    tests = sample.get("public_test_cases") or sample.get("test") or []
    if isinstance(tests, str):
        try:
            import json
            tests = json.loads(tests)
        except Exception:
            tests = []
    if not tests:
        return 0.5  # syntax-pass partial
    n_pass = 0
    for t in tests[:5]:  # cap to 5 tests for budget
        ok = _run_one(code, t.get("input", ""), t.get("output", ""), timeout_s)
        n_pass += int(ok)
    return n_pass / max(1, min(5, len(tests)))


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _extract_code_block(text: str) -> str:
    m = _CODE_BLOCK_RE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def _run_one(code: str, stdin: str, expected: str, timeout: float) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        proc = subprocess.run(
            [sys.executable, fname], input=stdin, capture_output=True,
            text=True, timeout=timeout,
        )
        return proc.stdout.strip() == expected.strip()
    except Exception:
        return False
    finally:
        try:
            Path(fname).unlink()
        except OSError:
            pass


# ---------------- Format reward ----------------
def reward_format(generation: str) -> float:
    """Small format bonus: did the model emit <answer>...</answer>?"""
    return 0.1 if extract_answer(generation) is not None else 0.0


# ---------------- Dispatcher ----------------
def reward_for(kind: str, generation: str, sample: dict) -> float:
    """Return final reward in [0, 1.1] for a single trajectory."""
    base = 0.0
    if kind == "aime":
        base = reward_aime(generation, sample["answer"])
    elif kind == "gpqa":
        base = reward_gpqa(generation, sample.get("Correct Answer", ""),
                            [sample.get(f"Incorrect Answer {i}", "") for i in (1, 2, 3)])
    elif kind == "lcb":
        base = reward_lcb(generation, sample)
    return base + reward_format(generation)
