"""Per-tool world-models with calibrated uncertainty.

Architecture: Qwen3-0.6B base + LoRA, fine-tuned to predict the cached tool output
from (tool_name, args_json). An auxiliary scalar uncertainty head (single Linear over
the last hidden state) is trained against an exact-match label, then temperature-scaled
on the validation split until ECE < 0.05.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import MODEL, PATHS, DYNA, TOOL_NAMES


# ---------------- Dataset ----------------
class ToolPairDataset(Dataset):
    """(tool, args_json) -> output_str. Loaded from a jsonl trace dump."""

    def __init__(self, rows: list[dict], tokenizer, max_len: int = 1024):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        prompt = self._format_prompt(r["tool"], r["args"])
        target = (r.get("output") or "")[:2000]
        full = prompt + target + self.tok.eos_token
        enc = self.tok(full, truncation=True, max_length=self.max_len, return_tensors="pt")
        ids = enc["input_ids"][0]
        labels = ids.clone()
        # Mask the prompt portion in labels
        prompt_len = len(self.tok(prompt, truncation=True, max_length=self.max_len)["input_ids"])
        labels[:prompt_len] = -100
        return {"input_ids": ids, "attention_mask": enc["attention_mask"][0],
                "labels": labels, "prompt_len": prompt_len, "target": target}

    @staticmethod
    def _format_prompt(tool: str, args) -> str:
        if isinstance(args, str):
            args_str = args
        else:
            args_str = json.dumps(args, ensure_ascii=False)
        return f"<tool>{tool}</tool>\n<args>{args_str}</args>\n<output>"


def collate(batch, pad_id: int):
    maxlen = max(b["input_ids"].size(0) for b in batch)
    ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        ids[i, :L] = b["input_ids"]
        mask[i, :L] = b["attention_mask"]
        labels[i, :L] = b["labels"]
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# ---------------- Predictor wrapper ----------------
@dataclass
class PredictorOutput:
    text: str
    uncertainty: float    # 0..1, calibrated post-temperature
    raw_score: float      # pre-calibration


class ToolPredictor(nn.Module):
    """Wraps a HF causal LM with a small uncertainty head over the final hidden state.

    During training, the LM is fine-tuned with LoRA. The uncertainty head is a 1-layer
    linear projection trained with BCE against an exact-match label gathered post-hoc.
    """

    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base = base_model
        self.unc_head = nn.Linear(hidden_size, 1)
        self.temperature = nn.Parameter(torch.ones(1))   # post-hoc scaling

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]
        # Pool: mean over non-padded tokens
        m = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        unc_logit = self.unc_head(pooled).squeeze(-1)
        return out, unc_logit

    @torch.no_grad()
    def predict(self, tokenizer, tool: str, args: dict, max_new_tokens: int = 256,
                device: str = "cuda") -> PredictorOutput:
        prompt = ToolPairDataset._format_prompt(tool, args)
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        gen_ids = self.base.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        text = tokenizer.decode(gen_ids[0][enc["input_ids"].size(1):],
                                 skip_special_tokens=True)
        text = text.split("</output>")[0]

        # Uncertainty pass over the full (prompt+gen) sequence
        full = enc["input_ids"]
        full_attn = enc["attention_mask"]
        feat = self.base(input_ids=full, attention_mask=full_attn,
                          output_hidden_states=True).hidden_states[-1]
        pooled = feat.mean(dim=1)
        logit = self.unc_head(pooled).squeeze(-1)
        scaled = logit / self.temperature
        prob_correct = torch.sigmoid(scaled).item()
        unc = 1.0 - prob_correct
        return PredictorOutput(text=text, uncertainty=unc, raw_score=logit.item())


# ---------------- Calibration ----------------
def fit_temperature(model: ToolPredictor, val_logits: torch.Tensor,
                    val_labels: torch.Tensor, max_iter: int = 200) -> float:
    """LBFGS over temperature only. Minimizes BCE on (logit/T, label)."""
    T = torch.ones(1, requires_grad=True, device=val_logits.device)
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(val_logits / T, val_labels.float())
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        model.temperature.copy_(T.detach().clamp(min=0.01))
    return float(T.detach().item())


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor,
                                n_bins: int = 15) -> float:
    """Standard ECE."""
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = probs.size(0)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].float().mean().item()
        conf = probs[mask].mean().item()
        ece += (mask.sum().item() / N) * abs(acc - conf)
    return ece


# ---------------- Fidelity probe ----------------
def fidelity_score(predicted: str, gold: str, tool: str) -> float:
    """Tool-specific match scoring: exact, numeric-tolerant, or BLEU-ish."""
    p, g = (predicted or "").strip(), (gold or "").strip()
    if tool == "calc":
        try:
            return 1.0 if abs(float(p) - float(g)) < 1e-3 else 0.0
        except ValueError:
            return float(p == g)
    if tool == "code":
        # accept exact-line match on stripped stdout
        return float(p.strip() == g.strip())
    # search: top-3 title overlap
    try:
        pj, gj = json.loads(p), json.loads(g)
        pt = {x.get("title", "").lower() for x in pj if isinstance(x, dict)}
        gt = {x.get("title", "").lower() for x in gj if isinstance(x, dict)}
        if not gt:
            return 0.0
        return len(pt & gt) / len(gt)
    except Exception:
        return 0.0


def pick_threshold(uncertainties: list[float], fidelities: list[float],
                    target: float = DYNA.fidelity_target) -> float:
    """Highest τ s.t. mean fidelity for u<τ ≥ target. Falls back to 0 if impossible."""
    pairs = sorted(zip(uncertainties, fidelities))
    best = 0.0
    for k in range(10, len(pairs) + 1, 10):
        sub = pairs[:k]
        u_max = sub[-1][0]
        fid = sum(p[1] for p in sub) / len(sub)
        if fid >= target:
            best = u_max
    return best


# ---------------- Predictor pool ----------------
class PredictorPool:
    """Holds three trained predictors + their gating thresholds. Used at rollout time."""

    def __init__(self, predictors: dict, tokenizers: dict, thresholds: dict):
        self.predictors = predictors        # tool -> ToolPredictor
        self.tokenizers = tokenizers        # tool -> tokenizer
        self.thresholds = thresholds        # tool -> float

    def query(self, tool: str, args: dict, device: str = "cuda") -> PredictorOutput:
        return self.predictors[tool].predict(self.tokenizers[tool], tool, args,
                                              device=device)

    def should_simulate(self, tool: str, uncertainty: float) -> bool:
        return uncertainty < self.thresholds.get(tool, 0.0)
