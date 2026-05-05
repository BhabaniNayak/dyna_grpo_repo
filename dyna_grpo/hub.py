"""Push trained checkpoints + model cards to Hugging Face Hub."""
from __future__ import annotations
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

from .config import HUB, repo_id
from .utils import logger


_MODEL_CARD_TEMPLATE = """---
license: apache-2.0
base_model: {base_model}
tags:
  - dyna-grpo
  - reinforcement-learning
  - tool-use
  - reasoning
language: en
---

# {repo_short}

LoRA adapter trained as part of the **Dyna-GRPO** research project.

## Description
{description}

## Usage
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo_full}")
tok = AutoTokenizer.from_pretrained("{base_model}")
```

## Reported metrics
{metrics_block}

## Citation
Paper: *Dyna-GRPO: World-Model-Augmented Rollouts and Counterfactual Credit
Assignment for Multi-Tool Agentic RL.* 2026.
"""


def _model_card(repo_short: str, base_model: str, description: str,
                 metrics: dict) -> str:
    lines = "\n".join(f"- **{k}**: {v}" for k, v in metrics.items()) or "_pending_"
    return _MODEL_CARD_TEMPLATE.format(
        repo_short=repo_short, base_model=base_model,
        description=description, metrics_block=lines,
        repo_full=f"{HUB.hf_user}/{repo_short}",
    )


def push(local_dir: str | Path, repo_short: str, base_model: str,
          description: str, metrics: dict | None = None,
          token: str | None = None) -> str:
    """Create the repo (idempotent) and upload the folder."""
    if not HUB.push_to_hub:
        logger.info(f"PUSH_TO_HUB=0; skipping push of {repo_short}")
        return ""

    full = repo_id(repo_short)
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")

    api = HfApi(token=token)
    create_repo(full, repo_type="model", private=HUB.private,
                exist_ok=True, token=token)

    # Write model card
    card = _model_card(repo_short, base_model, description, metrics or {})
    card_path = Path(local_dir) / "README.md"
    card_path.write_text(card)

    api.upload_folder(folder_path=str(local_dir), repo_id=full, repo_type="model",
                       commit_message=f"Upload {repo_short}")
    url = f"https://huggingface.co/{full}"
    logger.info(f"Pushed: {url}")
    return url
