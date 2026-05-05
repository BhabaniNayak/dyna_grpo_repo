"""Minimal GRPO trainer. ~300 lines. Plugs into TRL only for tokenization helpers.

Loop:
  for step in range(total_steps):
    sample G prompts × group_size rollouts
    compute group-normalized advantages
    backprop on PPO-style clipped surrogate with KL to reference
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .config import GRPO, MODEL, PATHS
from .trace_collector import Trajectory, rollout
from .utils import logger, save_metrics


@dataclass
class GRPOBatch:
    prompts: list[str]                       # B prompts
    trajectories: list[list[Trajectory]]     # B × G
    rewards: torch.Tensor                    # (B, G)
    advantages: torch.Tensor                 # (B, G)


def compute_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Group-relative normalization: (r - mean) / std within each group."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True).clamp(min=eps)
    return (rewards - mean) / std


def trajectory_logprobs(model, tokenizer, full_text: str, prompt: str,
                          device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token logprobs of the *generated* portion, plus the corresponding token ids."""
    enc_full = tokenizer(full_text, return_tensors="pt").to(device)
    enc_prompt = tokenizer(prompt, return_tensors="pt").to(device)
    L_p = enc_prompt["input_ids"].size(1)
    out = model(**enc_full)
    logits = out.logits[0, :-1]                         # (T-1, V)
    targets = enc_full["input_ids"][0, 1:]              # (T-1,)
    logp = F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(-1)).squeeze(-1)
    # Slice generation portion only
    gen_logp = logp[L_p - 1:]
    gen_tok = targets[L_p - 1:]
    return gen_logp, gen_tok


def grpo_loss(actor_logp: torch.Tensor, ref_logp: torch.Tensor,
               old_logp: torch.Tensor, advantages: torch.Tensor,
               clip: float = GRPO.clip_ratio, kl_coef: float = GRPO.kl_coef,
               token_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """PPO-style clipped surrogate with KL penalty.

    actor_logp / ref_logp / old_logp: (T,) per-token logprobs of taken actions
    advantages: scalar per token, broadcasted by caller
    token_mask: optional (T,) mask to zero-out tool-observation tokens, etc.
    """
    if token_mask is None:
        token_mask = torch.ones_like(actor_logp)

    ratio = torch.exp(actor_logp - old_logp)
    s1 = ratio * advantages
    s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
    pg = -torch.min(s1, s2)

    # Low-variance KL estimator (Schulman): exp(d) - d - 1, d = ref - actor
    d = ref_logp - actor_logp
    kl = torch.exp(d) - d - 1.0

    loss = ((pg + kl_coef * kl) * token_mask).sum() / token_mask.sum().clamp(min=1)
    info = {
        "pg_loss": (pg * token_mask).sum().item() / token_mask.sum().clamp(min=1).item(),
        "kl": (kl * token_mask).sum().item() / token_mask.sum().clamp(min=1).item(),
        "ratio_mean": ratio.mean().item(),
    }
    return loss, info


# ---------------- Token-mask helpers ----------------
def build_token_mask(tokenizer, full_text: str, prompt: str,
                     trajectory: Trajectory, device: str = "cuda",
                     mask_observations: bool = True) -> torch.Tensor:
    """1 for model-generated tokens, 0 for tool-observation tokens.

    Implementation is approximate: we tokenize the prompt and walk segments,
    marking tokens that fall inside <obs>...</obs> spans as 0.
    """
    enc = tokenizer(full_text, return_tensors="pt").to(device)
    L = enc["input_ids"].size(1)
    L_p = tokenizer(prompt, return_tensors="pt").input_ids.size(1)
    mask = torch.zeros(L - 1, device=device)
    mask[L_p - 1:] = 1.0
    if not mask_observations:
        return mask

    # Find <obs>...</obs> char spans, map to token spans
    obs_open = tokenizer.encode("<obs>", add_special_tokens=False)
    obs_close = tokenizer.encode("</obs>", add_special_tokens=False)
    ids = enc["input_ids"][0].tolist()
    in_obs = False
    for i in range(len(ids) - len(obs_open)):
        if ids[i:i + len(obs_open)] == obs_open:
            in_obs = True
        if i < L - 1 and in_obs and i >= L_p:
            mask[i - 1] = 0.0
        if ids[i:i + len(obs_close)] == obs_close:
            in_obs = False
    return mask


# ---------------- High-level training step ----------------
class GRPOTrainer:
    def __init__(self, actor, ref, tokenizer, optimizer, generate_fn,
                 reward_fn: Callable[[Trajectory, dict], float],
                 device: str = "cuda", config=GRPO):
        self.actor = actor
        self.ref = ref
        self.tok = tokenizer
        self.opt = optimizer
        self.generate_fn = generate_fn
        self.reward_fn = reward_fn
        self.device = device
        self.cfg = config

    def collect_group(self, prompt: str, sample: dict,
                      tool_fn=None) -> tuple[list[Trajectory], torch.Tensor]:
        trajs = []
        for _ in range(self.cfg.group_size):
            t = rollout(prompt, self.generate_fn, tool_fn=tool_fn,
                        max_tool_calls=self.cfg.max_tool_calls)
            t.reward = self.reward_fn(t, sample)
            trajs.append(t)
        rewards = torch.tensor([t.reward for t in trajs], device=self.device)
        return trajs, rewards

    def step(self, prompts_with_samples: list[tuple[str, dict]],
              tool_fn=None) -> dict:
        all_rewards, all_groups = [], []
        for prompt, sample in prompts_with_samples:
            trajs, rewards = self.collect_group(prompt, sample, tool_fn=tool_fn)
            all_groups.append((prompt, trajs))
            all_rewards.append(rewards)

        rewards = torch.stack(all_rewards)             # (B, G)
        advantages = compute_advantages(rewards)        # (B, G)

        # ----- Optimization -----
        self.opt.zero_grad()
        total_loss = 0.0
        info_acc = {"pg_loss": 0.0, "kl": 0.0, "ratio_mean": 0.0}
        n = 0
        for (prompt, trajs), adv_row in zip(all_groups, advantages):
            for traj, adv in zip(trajs, adv_row):
                full = self.tok(prompt + traj.flat_text(), return_tensors="pt").to(
                    self.device)
                # logprobs for actor, ref, old (== detached actor here for one-pass)
                actor_logp, _ = trajectory_logprobs(
                    self.actor, self.tok, prompt + traj.flat_text(), prompt, self.device)
                with torch.no_grad():
                    ref_logp, _ = trajectory_logprobs(
                        self.ref, self.tok, prompt + traj.flat_text(), prompt, self.device)
                old_logp = actor_logp.detach()
                mask = build_token_mask(self.tok, prompt + traj.flat_text(), prompt,
                                         traj, self.device)[: actor_logp.size(0)]
                adv_t = adv.detach() * torch.ones_like(actor_logp)
                loss, info = grpo_loss(actor_logp, ref_logp, old_logp, adv_t,
                                        token_mask=mask)
                (loss / max(1, len(prompts_with_samples) * self.cfg.group_size)).backward()
                total_loss += loss.item()
                for k in info_acc:
                    info_acc[k] += info[k]
                n += 1

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt.step()

        return {
            "loss": total_loss / max(1, n),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            **{k: v / max(1, n) for k, v in info_acc.items()},
        }
