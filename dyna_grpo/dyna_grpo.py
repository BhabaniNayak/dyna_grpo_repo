"""Dyna-GRPO: mixed real/simulated rollouts + counterfactual per-tool credit assignment.

Two main extensions over the vanilla GRPO trainer:

1) `mixed_tool_fn(predictor_pool, eta)`: a `tool_fn` factory that returns a callable
   compatible with `trace_collector.rollout`. It either calls a real tool or queries
   the predictor pool, gated by uncertainty thresholds. A fraction `eta` of rollouts
   are forced to use real tools only (the anchor group).

2) `counterfactual_advantages(...)`: for each tool-call segment in a trajectory,
   sample top-K alternative actions, simulate the suffix via the predictor pool +
   actor, and compute Δ_t^CF as the mean (real_R - sim_R_alt) gap. Returns a
   per-segment correction that gets applied to tokens inside the tool-call span.
"""
from __future__ import annotations
import copy
import random
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .config import DYNA, GRPO
from .grpo import compute_advantages, GRPOTrainer, trajectory_logprobs, grpo_loss, build_token_mask
from .predictor import PredictorPool
from .tools import call_tool
from .trace_collector import Trajectory, Segment, rollout
from .utils import logger


# ---------------- Mixed-rollout tool fn ----------------
def make_mixed_tool_fn(pool: PredictorPool, force_real: bool = False) -> Callable:
    """Return a callable compatible with rollout(tool_fn=...)."""

    def tool_fn(tool: str, args: dict) -> dict:
        if force_real:
            r = call_tool(tool, args)
            return {"output": r.output if not r.error else f"ERROR:{r.error}",
                    "source": "real", "uncertainty": 0.0,
                    "latency_ms": r.latency_ms}
        # First, query predictor for uncertainty
        try:
            pred = pool.query(tool, args)
        except Exception as e:
            r = call_tool(tool, args)
            return {"output": r.output, "source": "real",
                    "uncertainty": 1.0, "latency_ms": r.latency_ms}
        if pool.should_simulate(tool, pred.uncertainty):
            return {"output": pred.text, "source": "sim",
                    "uncertainty": pred.uncertainty, "latency_ms": 0.0}
        r = call_tool(tool, args)
        return {"output": r.output if not r.error else f"ERROR:{r.error}",
                "source": "real", "uncertainty": pred.uncertainty,
                "latency_ms": r.latency_ms}

    return tool_fn


# ---------------- Counterfactual credit assignment ----------------
@dataclass
class CFCorrection:
    seg_idx: int                      # index in traj.segments
    delta: float                      # Δ_t^CF: R(τ) - mean(R(τ<t, a'_t))


def sample_alternative_actions(actor, tokenizer, prefix_text: str,
                                 current_tool: str, K: int = DYNA.cf_topk,
                                 device: str = "cuda") -> list[tuple[str, dict]]:
    """Sample K alternative tool calls from the policy at the same decision point.

    Strategy: re-prompt the actor with `prefix_text` and ask it to emit a different
    tool. We use temperature sampling and parse the first valid tool tag.
    """
    from .tools import parse_tool_call
    alts: list[tuple[str, dict]] = []
    seen = set()
    for _ in range(K * 3):                       # over-sample to dedupe
        if len(alts) >= K:
            break
        enc = tokenizer(prefix_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = actor.generate(
                **enc, max_new_tokens=128, do_sample=True, top_p=0.9,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][enc["input_ids"].size(1):],
                                skip_special_tokens=True)
        parsed = parse_tool_call(gen)
        if parsed is None:
            continue
        tool, args, _ = parsed
        if tool == current_tool:
            continue
        key = (tool, str(sorted(args.items()) if isinstance(args, dict) else args))
        if key in seen:
            continue
        seen.add(key)
        alts.append((tool, args))
    return alts


def counterfactual_advantages(traj: Trajectory, prefix_text_at: list[str],
                               actor, tokenizer, pool: PredictorPool,
                               reward_fn, sample: dict,
                               generate_fn, device: str = "cuda",
                               K: int = DYNA.cf_topk) -> list[CFCorrection]:
    """For each tool_call segment, compute the CF correction Δ_t^CF.

    `prefix_text_at[i]` is the cumulative text up to the start of segment i.
    Suffix rollouts use the predictor pool (simulated) for tool calls — cheap.
    """
    real_R = traj.reward
    corrections: list[CFCorrection] = []
    sim_tool_fn = make_mixed_tool_fn(pool, force_real=False)

    for i, seg in enumerate(traj.segments):
        if seg.type != "tool_call":
            continue
        prefix = prefix_text_at[i]
        alts = sample_alternative_actions(actor, tokenizer, prefix, seg.tool, K=K,
                                           device=device)
        if not alts:
            continue
        sim_rewards = []
        for tool_alt, args_alt in alts:
            # Replace this segment with the alternative; resimulate suffix
            cf_traj = _replay_with_alternative(traj, i, tool_alt, args_alt,
                                                 generate_fn, sim_tool_fn)
            cf_traj.reward = reward_fn(cf_traj, sample)
            sim_rewards.append(cf_traj.reward)
        baseline = sum(sim_rewards) / len(sim_rewards)
        delta = real_R - baseline
        corrections.append(CFCorrection(seg_idx=i, delta=delta))
    return corrections


def _replay_with_alternative(traj: Trajectory, swap_idx: int,
                              tool_alt: str, args_alt: dict,
                              generate_fn, tool_fn) -> Trajectory:
    """Build a CF trajectory by swapping segment[swap_idx] for an alternative tool call,
    then continuing the rollout from that point under sim tool_fn."""
    new = Trajectory(prompt=traj.prompt)
    # Copy prefix
    for seg in traj.segments[:swap_idx]:
        new.segments.append(copy.deepcopy(seg))
    # Apply alternative tool call (use predictor)
    obs = tool_fn(tool_alt, args_alt)
    swap_seg = Segment(type="tool_call",
                        text=f'<tool name="{tool_alt}"><args>{args_alt}</args></tool>',
                        tool=tool_alt, args=args_alt, obs=obs["output"],
                        obs_source=obs["source"], uncertainty=obs.get("uncertainty"))
    new.segments.append(swap_seg)
    # Continue rollout from here using generate_fn + sim tool_fn
    from .tools import system_prompt, parse_tool_call
    sys = system_prompt(GRPO.max_tool_calls)
    context = f"{sys}\n\nUser: {traj.prompt}\nAssistant: " + new.flat_text()
    remaining_calls = GRPO.max_tool_calls - sum(1 for s in new.segments if s.type == "tool_call")
    remaining_tokens = GRPO.max_gen_tokens // 2  # tighter budget for CF
    while remaining_calls > 0 and remaining_tokens > 0:
        gen = generate_fn(context, min(remaining_tokens, 512))
        remaining_tokens -= len(gen.split())
        parsed = parse_tool_call(gen)
        if parsed is None:
            new.segments.append(Segment(type="gen", text=gen))
            new.finished = "<answer>" in gen
            break
        tool, args, span = parsed
        pre = gen[: span[1]]
        new.segments.append(Segment(type="tool_call", text=pre, tool=tool, args=args))
        out = tool_fn(tool, args)
        new.segments[-1].obs = out["output"]
        new.segments[-1].obs_source = out["source"]
        new.segments[-1].uncertainty = out.get("uncertainty", 0.0)
        remaining_calls -= 1
        context += pre + f"<obs>{out['output']}</obs>"
    return new


# ---------------- Dyna-GRPO trainer ----------------
class DynaGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, predictor_pool: PredictorPool,
                 lambda_cf: float = DYNA.lambda_cf,
                 eta: float = DYNA.eta, use_cf: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = predictor_pool
        self.lambda_cf = lambda_cf
        self.eta = eta
        self.use_cf = use_cf

    def step(self, prompts_with_samples: list[tuple[str, dict]],
              **kwargs) -> dict:
        # Build mixed tool_fn for non-anchor rollouts
        tool_fn_mixed = make_mixed_tool_fn(self.pool, force_real=False)
        tool_fn_real = make_mixed_tool_fn(self.pool, force_real=True)

        all_rewards, all_groups = [], []
        sim_calls = real_calls = 0
        for prompt, sample in prompts_with_samples:
            trajs, rewards = [], []
            n_anchor = max(1, int(self.cfg.group_size * self.eta))
            for j in range(self.cfg.group_size):
                tf = tool_fn_real if j < n_anchor else tool_fn_mixed
                t = rollout(prompt, self.generate_fn, tool_fn=tf,
                            max_tool_calls=self.cfg.max_tool_calls)
                t.reward = self.reward_fn(t, sample)
                trajs.append(t)
                rewards.append(t.reward)
                for s in t.segments:
                    if s.type == "tool_call":
                        if s.obs_source == "sim":
                            sim_calls += 1
                        else:
                            real_calls += 1
            all_groups.append((prompt, sample, trajs))
            all_rewards.append(torch.tensor(rewards, device=self.device))

        rewards = torch.stack(all_rewards)
        advantages = compute_advantages(rewards)

        # ---- Compute CF corrections (only for non-anchor trajectories) ----
        cf_corrections_per_traj: dict[tuple[int, int], list] = {}
        if self.use_cf and self.lambda_cf > 0:
            for bi, (prompt, sample, trajs) in enumerate(all_groups):
                for gi, t in enumerate(trajs):
                    n_anchor = max(1, int(self.cfg.group_size * self.eta))
                    if gi < n_anchor:
                        continue
                    prefix_text_at = self._segment_prefixes(prompt, t)
                    corrections = counterfactual_advantages(
                        t, prefix_text_at, self.actor, self.tok, self.pool,
                        self.reward_fn, sample, self.generate_fn, self.device,
                        K=DYNA.cf_topk,
                    )
                    cf_corrections_per_traj[(bi, gi)] = corrections

        # ---- Backward ----
        self.opt.zero_grad()
        info_acc = {"pg_loss": 0.0, "kl": 0.0, "ratio_mean": 0.0,
                    "cf_delta_mean": 0.0, "cf_count": 0}
        n_traj = 0
        for bi, (prompt, sample, trajs) in enumerate(all_groups):
            for gi, t in enumerate(trajs):
                full_text = prompt + t.flat_text()
                actor_logp, _ = trajectory_logprobs(
                    self.actor, self.tok, full_text, prompt, self.device)
                with torch.no_grad():
                    ref_logp, _ = trajectory_logprobs(
                        self.ref, self.tok, full_text, prompt, self.device)
                old_logp = actor_logp.detach()
                base_mask = build_token_mask(self.tok, full_text, prompt, t,
                                              self.device)[: actor_logp.size(0)]
                adv_base = advantages[bi, gi].detach()
                adv_per_token = adv_base * torch.ones_like(actor_logp)

                # CF correction: only inside tool-call spans
                corr = cf_corrections_per_traj.get((bi, gi), [])
                if corr:
                    span_mask = self._tool_call_token_spans(
                        full_text, prompt, t, self.tok, self.device)
                    for c in corr:
                        if c.seg_idx >= len(span_mask):
                            continue
                        m = span_mask[c.seg_idx][: actor_logp.size(0)]
                        adv_per_token = adv_per_token + self.lambda_cf * c.delta * m
                        info_acc["cf_delta_mean"] += abs(c.delta)
                        info_acc["cf_count"] += 1

                loss, info = grpo_loss(actor_logp, ref_logp, old_logp, adv_per_token,
                                        token_mask=base_mask)
                (loss / max(1, len(prompts_with_samples) * self.cfg.group_size)).backward()
                for k in ("pg_loss", "kl", "ratio_mean"):
                    info_acc[k] += info[k]
                n_traj += 1

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt.step()

        out = {
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "sim_call_frac": sim_calls / max(1, sim_calls + real_calls),
            **{k: v / max(1, n_traj) for k, v in info_acc.items() if k != "cf_count"},
            "cf_count": info_acc["cf_count"],
        }
        return out

    def _segment_prefixes(self, prompt: str, traj: Trajectory) -> list[str]:
        """Cumulative text up to (and excluding) each segment."""
        from .tools import system_prompt
        sys = system_prompt(self.cfg.max_tool_calls)
        base = f"{sys}\n\nUser: {prompt}\nAssistant: "
        out = []
        cur = base
        for seg in traj.segments:
            out.append(cur)
            if seg.type == "gen":
                cur += seg.text
            else:
                cur += seg.text + (f"<obs>{seg.obs}</obs>" if seg.obs is not None else "")
        return out

    def _tool_call_token_spans(self, full_text: str, prompt: str, traj: Trajectory,
                                 tokenizer, device: str) -> list[torch.Tensor]:
        """For each segment, return a 0/1 mask over the generated tokens marking the
        tool-call span."""
        L_full = tokenizer(full_text, return_tensors="pt").input_ids.size(1)
        L_prompt = tokenizer(prompt, return_tensors="pt").input_ids.size(1)
        T_gen = L_full - L_prompt
        masks = []
        cumulative = prompt
        for seg in traj.segments:
            mask = torch.zeros(T_gen, device=device)
            if seg.type == "tool_call":
                start_char = len(cumulative)
                end_char = start_char + len(seg.text)
                # Approximate: tokens that fall in [start_char, end_char] of full_text
                start_tok = len(tokenizer(full_text[:start_char]).input_ids) - L_prompt
                end_tok = len(tokenizer(full_text[:end_char]).input_ids) - L_prompt
                start_tok = max(0, start_tok)
                end_tok = min(T_gen, end_tok)
                if end_tok > start_tok:
                    mask[start_tok:end_tok] = 1.0
            masks.append(mask)
            cumulative += seg.text + (f"<obs>{seg.obs}</obs>"
                                       if seg.type == "tool_call" and seg.obs is not None
                                       else "")
        return masks
