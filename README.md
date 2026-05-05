# Dyna-GRPO

Reproducible implementation of **Dyna-GRPO: World-Model-Augmented Rollouts and Counterfactual Credit Assignment for Multi-Tool Agentic RL**.

> **Start here**: read [`INSTRUCTIONS.md`](INSTRUCTIONS.md) for the 2-week execution playbook on RunPod.

## Layout
```
dyna_grpo/         # core package — tools, predictors, GRPO, Dyna-GRPO, metrics
notebooks/         # 8 modular notebooks (01-08) — orchestration + viz
paper/             # NeurIPS LaTeX draft with placeholder macros
scripts/prefetch.py # CPU-pod download script (run before claiming GPUs)
INSTRUCTIONS.md    # daily checklist / run log
```

## Quick start
```bash
# On RunPod (volume mounted at /workspace)
git clone <this-repo> /workspace/dyna_grpo_repo
cd /workspace/dyna_grpo_repo
pip install -r requirements.txt
export DYNA_GRPO_WORKSPACE=/workspace/dyna_grpo
export HF_TOKEN=<your token>     # if pushing to hub
export PUSH_TO_HUB=1             # opt-in
# then run notebooks/01_setup_and_tools.ipynb → 08 in order
```

## Citation
See `paper/main.tex`.
