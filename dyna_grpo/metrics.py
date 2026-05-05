"""Evaluation metrics: pass@1, TS-KL, predictor fidelity, ECE."""
from __future__ import annotations
import math
from collections import Counter
from typing import Iterable

import torch
import torch.nn.functional as F


def pass_at_1(rewards_per_problem: list[list[float]]) -> float:
    """Mean over problems of P(reward==1)."""
    if not rewards_per_problem:
        return 0.0
    success = [int(any(r > 0.999 for r in rs)) for rs in rewards_per_problem]
    return sum(success) / len(success)


def tool_selection_kl(predicted_tools: list[str], oracle_tools: list[str],
                       all_tools: tuple = ("code", "calc", "search", "none")) -> float:
    """KL(π_θ(k|s) || δ_{k*}) averaged across the set.

    Approximated as: -log p(k* | s). With a hard prediction we use a smoothed
    one-hot over `all_tools`.
    """
    eps = 1e-3
    kl_sum = 0.0
    n = 0
    for pred, gold in zip(predicted_tools, oracle_tools):
        if gold not in all_tools:
            continue
        # Smoothed empirical distribution treating prediction as 1-eps spike
        probs = {t: eps / (len(all_tools) - 1) for t in all_tools}
        probs[pred] = 1.0 - eps
        kl_sum += -math.log(max(eps, probs.get(gold, eps)))
        n += 1
    return kl_sum / max(1, n)


def per_tool_invocation_freq(trajectories) -> dict:
    counter: Counter = Counter()
    total = 0
    for t in trajectories:
        for s in t.segments:
            if s.type == "tool_call":
                counter[s.tool] += 1
                total += 1
    return {k: v / max(1, total) for k, v in counter.items()}


def reliability_diagram_bins(probs: torch.Tensor, labels: torch.Tensor,
                              n_bins: int = 15) -> tuple[list, list, list]:
    """Returns (mean_conf, mean_acc, count) per bin for plotting."""
    bins = torch.linspace(0, 1, n_bins + 1)
    mean_conf, mean_acc, count = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i].item(), bins[i + 1].item()
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0:
            mean_conf.append((lo + hi) / 2)
            mean_acc.append(0.0)
            count.append(0)
            continue
        mean_conf.append(probs[mask].mean().item())
        mean_acc.append(labels[mask].float().mean().item())
        count.append(int(mask.sum().item()))
    return mean_conf, mean_acc, count
