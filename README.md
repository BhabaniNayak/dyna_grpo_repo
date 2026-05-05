# Dyna-GRPO

Reproducible implementation of **Dyna-GRPO: World-Model-Augmented Rollouts and Counterfactual Credit Assignment for Multi-Tool Agentic RL**.

> **Start here**: read [`INSTRUCTIONS.md`](INSTRUCTIONS.md) for the 2-week execution playbook on RunPod.

## Layout
```
dyna_grpo/                 # core package — tools, predictors, GRPO, Dyna-GRPO, metrics
notebooks/                 # 8 modular notebooks (01–08) — orchestration + viz
paper/                     # NeurIPS LaTeX draft with placeholder macros
scripts/prefetch.py        # CPU-pod download script (run before claiming GPUs)
requirements.txt           # full GPU-pod stack
requirements-prefetch.txt  # minimal CPU-pod deps for downloads only
INSTRUCTIONS.md            # daily checklist / run log / troubleshooting
```

## Two-pod workflow

This project runs on **two pod types** to save cost.

### 1. CPU pod (one-time, ~30 min, ~$0.05)
Used only to pre-populate the network volume with model weights and datasets so you don't burn GPU $ on downloads.
- Image: any standard Ubuntu / Python image
- Storage: attach the 200 GB network volume at `/workspace`
- Run only `requirements-prefetch.txt` and `scripts/prefetch.py`. **Do not** install `requirements.txt` here — it pulls vLLM and bitsandbytes which require CUDA and stable Python wheels.

### 2. GPU pod (most of the work)
- Template: **RunPod "PyTorch 2.5.1" — Python 3.11, CUDA 12.1**
- GPU: 1× A100-80G (Days 1–7) → 2× A100-80G (Days 8–13)
- Attach the same network volume at `/workspace`
- Install `requirements.txt`; vLLM resolves to a prebuilt CUDA wheel
- Stop the pod overnight; the volume persists at storage rates only

## Quick start (GPU pod)
```bash
# After provisioning the GPU pod with volume attached
cd /workspace/dyna_grpo_repo
git pull

# Install torch FIRST from the CUDA 12.4 index (works with driver 12.4-12.8)
pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

# Then the rest of the pinned stack
pip install --no-cache-dir -r requirements.txt

# Persist the HF cache redirects so nothing fills up the small root disk
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/workspace/dyna_grpo/hf_home
export HF_HUB_CACHE=/workspace/dyna_grpo/hf_home/hub
export HF_DATASETS_CACHE=/workspace/dyna_grpo/hf_home/datasets
export HF_XET_CACHE=/workspace/dyna_grpo/hf_home/xet
export DYNA_GRPO_WORKSPACE=/workspace/dyna_grpo
export HF_HUB_ENABLE_HF_TRANSFER=1
# Optional — for HF Hub push at end of run:
# export HF_TOKEN=hf_...
# export PUSH_TO_HUB=1
EOF
source ~/.bashrc

# Launch Jupyter inside tmux (survives SSH disconnects)
tmux new -s jupyter
jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser
# Detach: Ctrl+b then d ; reattach: tmux attach -t jupyter
```

Then open `notebooks/01_setup_and_tools.ipynb` and run sequentially through `08`.

## Models / datasets
- Actor: `Qwen/Qwen3-4B-Instruct-2507`
- Predictor base: `Qwen/Qwen3-0.6B`
- Training: `AI-MO/NuminaMath-CoT` (subset)
- Eval: `Maxwell-Jia/AIME_2024`, AIME 2025, GPQA-Diamond, LiveCodeBench-v6
- Tools: subprocess Python sandbox, Sympy calculator, DuckDuckGo (no API key needed)

## HF Hub push (optional)
At the end of Notebook 08, set `HF_TOKEN` and `PUSH_TO_HUB=1` to publish 5 repos under `genaiquest/`:
- `dyna-grpo-qwen3-4b-final`, `grpo-baseline-qwen3-4b`
- `dyna-grpo-tool-predictor-{calc,code,search}`

Each gets a generated model card with eval numbers, license, and load snippet.

## Troubleshooting
See the [Troubleshooting](INSTRUCTIONS.md#troubleshooting) section in `INSTRUCTIONS.md` for issues you may hit on RunPod (Python version mismatches, broken `six`, full root disk, gated repo 401s, volume not actually attached).

## Citation
Paper: `paper/main.tex`. Compile with the NeurIPS 2024 style file from
https://media.neurips.cc/Conferences/NeurIPS2024/Styles/neurips_2024.sty
