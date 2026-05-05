"""ReAct-style multi-tool rollout. Used for trace collection AND for GRPO rollouts.

A 'trajectory' is a list of segments:
  [{"type": "gen", "text": ...},
   {"type": "tool_call", "tool": ..., "args": ..., "obs": ..., "obs_source": "real"|"sim"},
   ...]

We keep token-level slices so credit assignment can target tool-call spans precisely.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import GRPO
from .tools import call_tool, parse_tool_call, format_observation, system_prompt


@dataclass
class Segment:
    type: str                       # "gen" or "tool_call"
    text: str = ""
    tool: Optional[str] = None
    args: Optional[dict] = None
    obs: Optional[str] = None
    obs_source: Optional[str] = None  # "real" or "sim"
    uncertainty: Optional[float] = None
    span: Optional[tuple[int, int]] = None  # token span in flattened sequence


@dataclass
class Trajectory:
    prompt: str
    segments: list[Segment] = field(default_factory=list)
    reward: float = 0.0
    finished: bool = False

    def flat_text(self) -> str:
        chunks = []
        for s in self.segments:
            if s.type == "gen":
                chunks.append(s.text)
            else:
                chunks.append(s.text)  # the model-emitted <tool ...> string
                if s.obs is not None:
                    chunks.append(f"<obs>{s.obs}</obs>")
        return "".join(chunks)


# ---------------- Rollout function ----------------
def rollout(
    prompt: str,
    generate_fn: Callable[[str, int], str],
    tool_fn: Callable[[str, dict], dict] = None,   # default: real tools
    max_tool_calls: int = GRPO.max_tool_calls,
    max_new_tokens: int = GRPO.max_gen_tokens,
) -> Trajectory:
    """Generic rollout used by both real (trace collection) and mixed (Dyna) rollouts.

    `generate_fn(context, max_new_tokens) -> generated_text`
    `tool_fn(tool, args) -> {"output": str, "source": "real"|"sim", "uncertainty": float}`
    """
    if tool_fn is None:
        def tool_fn(tool, args):
            r = call_tool(tool, args)
            return {"output": r.output if not r.error else f"ERROR:{r.error}",
                    "source": "real", "uncertainty": 0.0, "latency_ms": r.latency_ms}

    sys = system_prompt(max_tool_calls)
    context = f"{sys}\n\nUser: {prompt}\nAssistant: "

    traj = Trajectory(prompt=prompt)
    n_calls = 0
    remaining = max_new_tokens
    while remaining > 0 and n_calls < max_tool_calls:
        gen = generate_fn(context, min(remaining, 1024))
        remaining -= len(gen.split())  # rough token budget
        parsed = parse_tool_call(gen)
        if parsed is None:
            traj.segments.append(Segment(type="gen", text=gen))
            traj.finished = "<answer>" in gen
            break
        tool, args, span = parsed
        # Split: text up to and including the tool tag is one tool_call segment
        pre = gen[: span[1]]
        traj.segments.append(Segment(type="tool_call", text=pre, tool=tool, args=args))
        out = tool_fn(tool, args)
        traj.segments[-1].obs = out["output"]
        traj.segments[-1].obs_source = out["source"]
        traj.segments[-1].uncertainty = out.get("uncertainty", 0.0)
        n_calls += 1
        context += pre + f"<obs>{out['output']}</obs>"

    return traj
