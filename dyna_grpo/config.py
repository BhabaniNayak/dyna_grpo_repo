"""Centralized config. Override via env vars when running on RunPod."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


WORKSPACE = Path(_env("DYNA_GRPO_WORKSPACE", "/workspace/dyna_grpo"))
WORKSPACE.mkdir(parents=True, exist_ok=True)

PATHS = {
    "models": WORKSPACE / "models",
    "data": WORKSPACE / "data",
    "traces": WORKSPACE / "traces",
    "ckpts": WORKSPACE / "ckpts",
    "logs": WORKSPACE / "logs",
    "figures": WORKSPACE / "figures",
    "tool_cache": WORKSPACE / "tool_cache.sqlite",
}
for p in PATHS.values():
    if not p.suffix:
        p.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    actor_name: str = "Qwen/Qwen3-4B-Instruct"
    predictor_base: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj")


@dataclass
class GRPOConfig:
    group_size: int = 8
    kl_coef: float = 0.001
    clip_ratio: float = 0.2
    max_tool_calls: int = 4
    max_gen_tokens: int = 4096
    learning_rate: float = 5e-6
    batch_size: int = 4              # prompts per gradient step
    grad_accum: int = 2
    total_steps: int = 500
    log_every: int = 5
    save_every: int = 50
    seed: int = 42


@dataclass
class DynaConfig:
    eta: float = 0.2                 # anchor-group fraction (real-tools-only rollouts)
    lambda_cf: float = 0.5           # weight on CF correction
    cf_topk: int = 3                 # alternative tool actions per decision
    fidelity_target: float = 0.92    # min real-vs-simulated agreement
    ece_target: float = 0.05
    refresh_predictor_every: int = 200  # steps; world-model drift mitigation


@dataclass
class ToolConfig:
    code_timeout_s: float = 8.0
    code_mem_mb: int = 512
    search_top_k: int = 3
    cache_db: Path = PATHS["tool_cache"]


@dataclass
class HubConfig:
    push_to_hub: bool = bool(int(_env("PUSH_TO_HUB", "0")))
    hf_user: str = _env("HF_USER", "genaiquest")
    private: bool = True


MODEL = ModelConfig()
GRPO = GRPOConfig()
DYNA = DynaConfig()
TOOL = ToolConfig()
HUB = HubConfig()

TOOL_NAMES = ("code", "calc", "search")


def repo_id(suffix: str) -> str:
    return f"{HUB.hf_user}/{suffix}"
