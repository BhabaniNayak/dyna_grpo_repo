"""Tool harness: code interpreter, calculator, web search.

All tools share the contract:
    call(args: dict) -> dict(output=str, latency_ms=float, error=str|None)

Outputs are cached in sqlite keyed by (tool_name, args_json) so re-runs are free.
"""
from __future__ import annotations
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import TOOL


@dataclass
class ToolResult:
    output: str
    latency_ms: float
    error: Optional[str] = None
    tool: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


# ---------------- sqlite cache ----------------
class ToolCache:
    def __init__(self, path: Path = TOOL.cache_db):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(tool TEXT, args TEXT, output TEXT, latency REAL, error TEXT, "
            "PRIMARY KEY(tool, args))"
        )
        self._conn.commit()

    def get(self, tool: str, args: dict) -> Optional[ToolResult]:
        key = json.dumps(args, sort_keys=True)
        row = self._conn.execute(
            "SELECT output, latency, error FROM cache WHERE tool=? AND args=?",
            (tool, key),
        ).fetchone()
        if row:
            return ToolResult(output=row[0], latency_ms=row[1], error=row[2], tool=tool)
        return None

    def put(self, tool: str, args: dict, result: ToolResult) -> None:
        key = json.dumps(args, sort_keys=True)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)",
            (tool, key, result.output, result.latency_ms, result.error),
        )
        self._conn.commit()

    def all_for(self, tool: str):
        for row in self._conn.execute(
            "SELECT args, output FROM cache WHERE tool=?", (tool,)
        ):
            yield json.loads(row[0]), row[1]


_cache = ToolCache()


# ---------------- calculator ----------------
def _calc(args: dict) -> ToolResult:
    """args = {'expression': str}. Uses sympy for safe evaluation."""
    import sympy
    expr = args.get("expression", "").strip()
    t0 = time.perf_counter()
    try:
        # restrict to numeric/symbolic math
        result = sympy.sympify(expr, evaluate=True)
        out = str(sympy.N(result, 20)) if result.free_symbols == set() else str(result)
        return ToolResult(out, (time.perf_counter() - t0) * 1000, None, "calc")
    except Exception as e:
        return ToolResult("", (time.perf_counter() - t0) * 1000, repr(e), "calc")


# ---------------- code interpreter ----------------
_CODE_HEADER = (
    "import math, statistics, itertools, functools, collections, re\n"
    "import numpy as np\n"
)


def _code(args: dict) -> ToolResult:
    """args = {'code': str}. Runs in subprocess with timeout."""
    code = args.get("code", "")
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_CODE_HEADER + code)
        fname = f.name
    try:
        proc = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=TOOL.code_timeout_s,
        )
        out = proc.stdout
        err = proc.stderr if proc.returncode != 0 else None
        # truncate to keep predictor targets bounded
        out = out[-4000:]
        return ToolResult(out, (time.perf_counter() - t0) * 1000, err, "code")
    except subprocess.TimeoutExpired:
        return ToolResult("", (time.perf_counter() - t0) * 1000,
                          f"Timeout after {TOOL.code_timeout_s}s", "code")
    except Exception as e:
        return ToolResult("", (time.perf_counter() - t0) * 1000, repr(e), "code")
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


# ---------------- web search (free, via duckduckgo) ----------------
def _search(args: dict) -> ToolResult:
    """args = {'query': str, 'k': int=3}. Returns top-k titles+snippets, structured."""
    try:
        from ddgs import DDGS                       # new package name
    except ImportError:
        from duckduckgo_search import DDGS         # fallback to old name
    query = args.get("query", "")
    k = int(args.get("k", TOOL.search_top_k))
    t0 = time.perf_counter()
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=k))
        items = []
        for h in hits[:k]:
            title = (h.get("title") or "")[:120]
            snippet = (h.get("body") or "")[:300]
            items.append({"title": title, "snippet": snippet})
        out = json.dumps(items, ensure_ascii=False)
        return ToolResult(out, (time.perf_counter() - t0) * 1000, None, "search")
    except Exception as e:
        return ToolResult("", (time.perf_counter() - t0) * 1000, repr(e), "search")


# ---------------- public API ----------------
_TOOL_FNS = {"calc": _calc, "code": _code, "search": _search}


def call_tool(tool: str, args: dict, use_cache: bool = True) -> ToolResult:
    if tool not in _TOOL_FNS:
        return ToolResult("", 0.0, f"Unknown tool: {tool}", tool)
    if use_cache:
        cached = _cache.get(tool, args)
        if cached:
            return cached
    result = _TOOL_FNS[tool](args)
    if use_cache:
        _cache.put(tool, args, result)
    return result


# ---------------- ReAct-style tool-call parsing ----------------
# Format we ask the model to emit:
#   <tool name="calc"><args>{"expression": "..."}</args></tool>
TOOL_RE = re.compile(
    r'<tool\s+name="(?P<tool>code|calc|search)"\s*>\s*<args>(?P<args>.*?)</args>\s*</tool>',
    re.DOTALL,
)


def parse_tool_call(text: str) -> Optional[tuple[str, dict, tuple[int, int]]]:
    """Find the FIRST tool call in `text`. Returns (tool, args_dict, (start, end))."""
    m = TOOL_RE.search(text)
    if not m:
        return None
    tool = m.group("tool")
    raw = m.group("args").strip()
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return tool, args, m.span()


def format_observation(result: ToolResult) -> str:
    if result.error:
        body = f"ERROR: {result.error}"
    else:
        body = result.output
    return f"<obs>{body}</obs>"


SYSTEM_PROMPT = """You are a reasoning assistant with access to three tools.

To call a tool, emit:
<tool name="TOOL"><args>JSON</args></tool>
where TOOL is one of: code, calc, search.

- code: runs Python. args = {"code": "..."}. Use print() to output.
- calc: evaluates a math expression with sympy. args = {"expression": "..."}.
- search: web search returning top-3 (title, snippet) JSON. args = {"query": "..."}.

After each tool call you will receive <obs>...</obs>. You may call up to {MAX_CALLS} tools.
When you have the final answer, write it as: <answer>...</answer>
"""


def system_prompt(max_calls: int) -> str:
    return SYSTEM_PROMPT.replace("{MAX_CALLS}", str(max_calls))
