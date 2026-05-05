# Dyna-GRPO: 2-Week Execution Playbook

This file is your operational checklist. Work through it top-to-bottom, recording timing, cost, and results in the **Run Log** section at the bottom as you go. Every notebook expects `/workspace/dyna_grpo` (your network volume mount) to persist between sessions.

---

## 0. Architecture Overview

| Component | Purpose | Where it lives |
|---|---|---|
| `dyna_grpo/` package | All reusable logic (tools, predictors, GRPO, CF, metrics) | Repo root, importable from notebooks |
| Notebooks 01-08 | Orchestration + visualization, one per pipeline stage | `notebooks/` |
| `paper/main.tex` | NeurIPS-format paper draft with `\PLACEHOLDER{...}` macros | `paper/` |
| `/workspace/dyna_grpo/{models,data,traces,ckpts,logs,figures}` | Persistent artifacts on network volume | RunPod NV |

**Design principle**: notebooks are thin orchestrators. Heavy logic is in the `dyna_grpo` package so you can unit-test, re-run, and edit without re-executing entire notebooks.

---

## 1. RunPod setup (do this once)

### 1a. Provision the network volume
- **Size**: 200 GB
- **Region**: same region as your future GPU pods
- **Mount path**: `/workspace`

### 1b. Pre-download from a CPU pod (saves GPU $)
Spin up a **CPU-only pod** (e.g., 4 vCPU / 8 GB RAM, ~$0.05/hr) with the volume attached.
```bash
# On the CPU pod — install ONLY the lightweight download deps.
# Do NOT install requirements.txt here (it pulls vLLM/bitsandbytes which need a GPU).
cd /workspace
git clone <your-repo> dyna_grpo_repo
cd dyna_grpo_repo
pip install --no-cache-dir -r requirements-prefetch.txt
export HF_HUB_ENABLE_HF_TRANSFER=1            # fast parallel downloads
export DYNA_GRPO_WORKSPACE=/workspace/dyna_grpo
python3 scripts/prefetch.py                    # ~25 GB, takes 15-40 min
```

**Pod image note**: when you provision the GPU pod later, pick a template with **Python 3.10/3.11/3.12** (e.g., RunPod's "PyTorch 2.5.1" template). Python 3.13 is too new — vLLM and bitsandbytes don't ship wheels for it yet, and source builds will fail.
Stop the CPU pod once download completes. The volume now has all weights.

### 1c. HF Hub setup (for checkpoint publishing)
- Create a HF account / log in to `genaiquest`
- Generate a write token: https://huggingface.co/settings/tokens
- On the GPU pod, `export HF_TOKEN=hf_...` and `export PUSH_TO_HUB=1`
- Repos are auto-created **private** by default (configurable in `dyna_grpo/config.py`)
- Notebook 08 will publish 5 repos under `genaiquest/`:
  - `dyna-grpo-qwen3-4b-final`, `grpo-baseline-qwen3-4b`
  - `dyna-grpo-tool-predictor-{calc,code,search}`

### 1d. Spin up the GPU pod for compute work
| Stage | Pod recommendation | Hourly $ (Community Cloud, approx) |
|---|---|---|
| Days 1-2 (trace collection, baseline GRPO debug) | 1× A100-80G | ~$1.10/hr |
| Days 3-4 (predictor training) | 1× A100-80G or 1× A6000-48G | ~$0.50-1.10/hr |
| Days 5-7 (Dyna-GRPO + ablations + eval) | 2× A100-80G | ~$2.20/hr |
| Day 8+ (writing, figures) | CPU-only or local | ~$0.05/hr |

**Stop the pod overnight when not running**. Your volume persists. Resume cost = $0.

Pod environment: PyTorch 2.5+ container, CUDA 12.1+. The `requirements.txt` adds the rest.

---

## 2. Daily checklist

Follow this. Each item links to a notebook or shell command. **Tick the box and record results in the Run Log.**

### Week 1

#### Day 1 — Environment & tool harness
- [ ] Spin up 1× A100-80G with the volume attached
- [ ] `cd /workspace/dyna_grpo_repo && pip install -r requirements.txt`
- [ ] Run `notebooks/01_setup_and_tools.ipynb` end-to-end
- [ ] Verify: tool harness returns expected outputs for a calc, code, and search probe
- [ ] Verify: Qwen3-4B loads in bf16, generates one response under 30 s
- [ ] **Record**: GPU mem usage at idle and post-load

#### Day 2 — Trace collection
- [ ] Run `notebooks/02_collect_traces.ipynb`
- [ ] Target: 50-100K cached `(tool, args, output)` tuples
- [ ] **Record**: total wall-clock, # of tool calls per tool, sqlite cache size

#### Day 3 — Baseline GRPO
- [ ] Run `notebooks/03_baseline_grpo.ipynb`
- [ ] Short run first (50 steps) to confirm convergence; then full run (≥500 steps or wall-clock budget)
- [ ] **Record**: AIME-2024 pass@1 at init / mid / end; tool-time vs gen-time breakdown

#### Day 4 — Predictor data + calculator predictor
- [ ] Run first half of `notebooks/04_train_predictors.ipynb` (calc only)
- [ ] **Record**: calc predictor exact-match on val

#### Day 5 — Code + search predictors
- [ ] Run remainder of `notebooks/04_train_predictors.ipynb`
- [ ] **Record**: code predictor BLEU + numeric-tolerance match; search predictor top-3 snippet recall

#### Day 6 — Calibration + gating
- [ ] Run `notebooks/05_fidelity_and_gating.ipynb`
- [ ] Iterate temperature scaling until ECE < 0.05 per tool
- [ ] **Record**: per-tool gating threshold τ_k, fidelity at chosen τ

#### Day 7 — Buffer / writing
- [ ] Re-run anything that needs it
- [ ] Draft Section 1-2 of paper while results are fresh
- [ ] If on schedule, switch pod off

### Week 2

#### Day 8 — Switch to 2× A100, sanity Dyna-GRPO
- [ ] Move to 2× A100-80G pod
- [ ] Run first cells of `notebooks/06_dyna_grpo.ipynb` (sanity: 5 steps with mixed rollouts only, no CF)
- [ ] Confirm gradients finite, no OOM

#### Day 9 — Dyna-GRPO no-CF training
- [ ] Run `notebooks/06_dyna_grpo.ipynb` ablation A: world-model only, no CF correction
- [ ] **Record**: wall-clock per step, tool-call substitution rate

#### Day 10 — Full Dyna-GRPO (CF on)
- [ ] Run `notebooks/06_dyna_grpo.ipynb` full method
- [ ] **Record**: training curves, CF advantage magnitude distribution

#### Day 11 — Evaluation
- [ ] Run `notebooks/07_eval_and_ablations.ipynb` evaluation cells
- [ ] AIME-2024, AIME-2025, GPQA-Diamond (subset), LCB-v6 (subset)
- [ ] **Record**: pass@1 for all baselines and Dyna-GRPO variants

#### Day 12 — Ablations
- [ ] Run ablation cells: η ∈ {0, 0.2, 1.0}, no-CF, predictor-fidelity stress test
- [ ] **Record**: ablation table

#### Day 13 — Figures + paper
- [ ] Run `notebooks/08_figures_and_writeup.ipynb` to generate all figures and tables
- [ ] Populate `paper/main.tex` placeholders
- [ ] Compile PDF (locally or via Overleaf)

#### Day 14 — Polish
- [ ] Re-runs for any unstable seeds
- [ ] Final paper read-through

---

## 3. Troubleshooting

This section captures every issue actually hit during setup. If you see one of these, jump to the fix.

### CPU pod (prefetch)

**`pip install -r requirements.txt` tries to source-build vLLM and hangs**
The CPU pod doesn't need vLLM. Use the slim deps file instead:
```bash
python3.13 -m pip install --no-cache-dir -r requirements-prefetch.txt
```

**`ModuleNotFoundError: No module named 'huggingface_hub'` after pip says "already installed"**
Python version mismatch. The image often ships `python3` → 3.8 while `pip` writes to 3.13's site-packages. Always use the same interpreter for install and run:
```bash
python3.13 -m pip install --no-cache-dir <pkg>
python3.13 scripts/prefetch.py
```

**`ModuleNotFoundError: No module named 'six.moves'`**
Broken `six` on the base image. Force-reinstall:
```bash
python3.13 -m pip install --no-cache-dir --force-reinstall six python-dateutil
```

**`401 Unauthorized` / `RepositoryNotFoundError` for a public Qwen repo**
HF rate-limits anonymous metadata calls. Authenticate:
```bash
huggingface-cli login   # paste a READ token from https://huggingface.co/settings/tokens
```

**`No space left on device (os error 28)` mid-download**
HF's intermediate cache is filling the small root disk. Redirect every cache to the network volume **before** running anything:
```bash
export HF_HOME=/workspace/dyna_grpo/hf_home
export HF_HUB_CACHE=/workspace/dyna_grpo/hf_home/hub
export HF_DATASETS_CACHE=/workspace/dyna_grpo/hf_home/datasets
export HF_XET_CACHE=/workspace/dyna_grpo/hf_home/xet
mkdir -p $HF_HOME $HF_HUB_CACHE $HF_DATASETS_CACHE $HF_XET_CACHE
```
The latest `scripts/prefetch.py` already sets these, but verify with `df -h /workspace`.

**`/workspace` shows < 200 GB total in `df -h`**
Your network volume isn't attached, OR was provisioned smaller. Stop the pod, edit pod settings → attach the 200 GB volume at `/workspace`, restart. The volume is independent of the pod, so files persist.

**SSH dropped while `prefetch.py` was running**
Re-run inside `tmux`:
```bash
tmux new -s prefetch
python3.13 scripts/prefetch.py 2>&1 | tee /workspace/dyna_grpo/prefetch.log
# Detach: Ctrl+b then d ;  Reattach: tmux attach -t prefetch
```
HF downloads are resumable; just re-run the script.

### GPU pod (training)

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM during rollout | KV cache too big | Lower `max_seq_len` to 2048 or `max_tool_calls` to 3 |
| OOM during backward | LoRA rank too high | Drop `lora_r` from 64 → 32 |
| GRPO loss explodes | Reward scale or KL coef | Cap reward to `[-1,1]`, raise KL coef to 0.005 |
| Predictor ECE > 0.05 | Bad temp scaling fit | Re-fit on 5K validation samples; switch to focal loss |
| Code predictor BLEU low | Code outputs too variable | Restrict targets to `(stdout, error_type)`, not full traceback |
| Pod killed mid-train | Spot interrupt | Notebooks checkpoint every N steps to volume; resume by re-running |
| `vllm` install fails on Python 3.13 | Wrong pod template | Re-provision with **PyTorch 2.5.1 / Python 3.11** template |
| `torch.cuda.is_available()` False, "driver too old" warning | Pip pulled torch built for CUDA 13 (e.g. `torch==2.11.0+cu130`); your driver is 12.x | Force-reinstall on cu124: `pip install --no-cache-dir --force-reinstall torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124` |
| Loose pip resolves to bleeding-edge `transformers==5.x`, `trl==1.x`, etc. | `requirements.txt` had `>=` only | Pull latest `requirements.txt` (now pinned) and `pip install --no-cache-dir -r requirements.txt` |

---

## 4. Run Log

Fill this in as you go. **Date/time, what was run, key numbers, any deviations from plan.**

```
============================================================
  DATE: 2026-05-05  POD: CPU-only  TIME: ~30 min  COST: ~$0.163
  NOTEBOOK: scripts/prefetch.py
  KEY METRICS:
    - Models cached: 9.0 GB (Qwen3-4B-Instruct-2507 + Qwen3-0.6B)
    - Datasets cached: 2.4 GB (NuminaMath-CoT, AIME-2024, MBPP)
    - Volume free: ample
  NOTES / DEVIATIONS:
    - python mismatch 
        alias python=python3.13
        alias python3=python3.13 
    - Issue with correct model name change 
    - workspace network drive not mounted properly. recreated the pod with network drive mounted
    - Force install of six 
      python3.13 -m pip install --no-cache-dir --force-reinstall six python-dateutil 
============================================================

DATE: ____________  POD: ___________  GPU-HRS USED: ________
NOTEBOOK: _________________________
KEY METRICS:
  - 
  -
NOTES / DEVIATIONS:
============================================================

```

(Copy the block above for each session.)

---

## 5. Final deliverables checklist

- [ ] Codebase clean and documented
- [ ] `paper/main.tex` compiles with all placeholders replaced
- [ ] All figures in `paper/figures/`
- [ ] Final pass@1 / TS-KL / wall-clock numbers in `paper/main.tex` Section 5
- [ ] Predictor fidelity report in `paper/appendix.tex`
- [ ] Two trained checkpoints (baseline GRPO + Dyna-GRPO) on volume
- [ ] All notebooks re-runnable end-to-end on a clean pod
