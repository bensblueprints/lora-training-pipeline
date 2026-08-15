# LoRA Training Pipeline — FLUX Kontext + Krea 2

A production-ready pipeline for generating **consistent character images** and training **face-identity LoRAs** for the NightTale game series (87 characters across 14 games). Designed for a single **RTX 3090 24GB** on **pop-os Linux** at the edge of VRAM.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Phase 1: FLUX Kontext — Dataset Generation](#phase-1-flux-kontext--dataset-generation)
  - [How Kontext Works](#how-kontext-works)
  - [Installation](#installation)
  - [Batch Generation Script](#batch-generation-script)
  - [VRAM Strategy](#vram-strategy)
  - [The 20-Pose Training Set](#the-20-pose-training-set)
- [Phase 2: ArcFace Identity Verification](#phase-2-arcface-identity-verification)
- [Phase 3: Krea 2 LoRA Training](#phase-3-krea-2-lora-training)
  - [Model Setup](#model-setup)
  - [Training Configuration](#training-configuration)
  - [Training Script](#training-script)
- [ComfyUI Integration](#comfyui-integration)
- [Directory Structure](#directory-structure)
- [Monitoring & Debugging](#monitoring--debugging)
- [Pitfalls & Troubleshooting](#pitfalls--troubleshooting)
- [Performance Benchmarks](#performance-benchmarks)
- [References](#references)

---

## Overview

This pipeline solves the hardest problem in AI story-game character generation: **keeping the same person's face consistent across every image, from every angle**.

**Two models, two roles:**

| Model | Role | Why |
|-------|------|-----|
| **FLUX.1-Kontext-dev** | Dataset generator | In-context learning preserves face identity across all angles — no LoRA needed for dataset creation |
| **Krea 2 (int8)** | LoRA base model | Fast inference at int8 precision, purpose-built for character consistency |

**Two machines, two roles** (revised Aug 2026):

| Machine | GPU | Role | Why |
|---------|-----|------|-----|
| **pop-os** | RTX 3090 24GB (**Ampere**) | **LoRA training** | 24GB VRAM required for bf16 training; Ampere has **no fp8 tensor cores** so fp8 must dequantize |
| **5060 Ti box** | RTX 5060 Ti 16GB (**Blackwell**) | **fp8 inference / dataset generation** | Blackwell has **native fp8 tensor cores** → runs FLUX fp8 natively, faster per-image than the 3090 (16GB is too tight for full LoRA training) |

**Scale**: 87 characters × 20 images = **1,740 training images** (headshots 1024×1024, full-body 832×1248).

---

## Architecture

![Pipeline Architecture](docs/architecture.svg)


## Prerequisites

### Hardware
- **GPU (training)**: NVIDIA RTX 3090 24GB on pop-os (single card — pop-os motherboard only supports one)
- **GPU (inference/generation)**: RTX 5060 Ti 16GB (Blackwell) — native fp8, faster than the 3090 for fp8 FLUX
- **RAM**: 64GB system RAM recommended (Kontext CPU offload uses ~30GB)
- **Storage**: ~60GB for FLUX Kontext model, ~10GB for Krea 2, ~5GB for datasets

### Software
- **OS**: pop-os Linux 22.04 (or any Ubuntu derivative)
- **Python**: 3.10+ with `uv` package manager
- **CUDA**: 12.1+ (required by bfloat16 support)
- **ComfyUI**: Running on port 8188 for inference

### Models to Download

| Model | Size | Location | Source |
|-------|------|----------|--------|
| FLUX.1-Kontext-dev | 54 GB | `~/ComfyUI/models/diffusers/FLUX.1-Kontext-dev/` | `black-forest-labs/FLUX.1-Kontext-dev` (HuggingFace) |
| Krea 2 Turbo int8 | ~5 GB | `~/ComfyUI/models/diffusion_models/krea2_turbo_int8_convrot.safetensors` | Krea AI (gated) |
| FLUX.1-dev | 12 GB | `~/ComfyUI/models/unet/flux1-dev.safetensors` | `Comfy-Org/flux1-dev` (HuggingFace) |
| ArcFace (buffalo_l) | ~350 MB | `~/.insightface/models/buffalo_l/` | Auto-downloaded by insightface |

---

## Phase 1: FLUX Kontext — Dataset Generation

### How Kontext Works

FLUX Kontext uses **in-context learning** — you give it a reference image of a character and it generates new images of **that same person** from any angle, with any expression, lighting, or pose. No LoRA needed for dataset creation.

**Key behaviors:**
- The reference image acts as an identity anchor
- The prompt controls pose, angle, lighting, expression
- `guidance_scale=2.5` (lower than FLUX.dev's typical 3.5-5.0 — Kontext is more responsive)
- Sequential CPU offload keeps VRAM within 24GB

### Installation

**⚠️ Use ComfyUI with the fp8 checkpoint — NOT diffusers.** (See ["Why ComfyUI, not diffusers"](#why-comfyui-not-diffusers) below.)

```bash
# ComfyUI fp8 checkpoint (11.9 GB) + text encoders + VAE
# diffusion_models/flux1-kontext-dev-fp8-e4m3fn.safetensors
# clip/clip_l.safetensors  +  clip/t5xxl_fp16.safetensors
# vae/ae.safetensors
```

The production orchestrator is **`/home/ben/batch_comfy.py`** — it drives ComfyUI over HTTP (`127.0.0.1:8188`), generates the 20-pose set per character with resume logic, and bakes the full-body/tall-canvas handling described below. See the `flux-kontext-pose-sets` skill for the exact workflow graph.

### Why ComfyUI, not diffusers (CRITICAL)

The RTX 3090 is **Ampere** — it has **no fp8 tensor cores** (only Ada/Blackwell: 40xx/50xx have them).

- **diffusers fp8** → `mat1 and mat2 must have the same dtype (BFloat16 vs Float8_e4m3fn)` — hard fail.
- **diffusers bnb int8** → `quantization_config` is **silently ignored** (loads bf16 anyway, OOM).
- **ComfyUI's fp8 loader dequantizes fp8→fp16 natively on Ampere** — the only path that works on the 3090.

**Speed:** ComfyUI fp8 @ 12 steps = **32s/img** vs diffusers bf16 = **153.7s/img** (~4.8× faster).

**The 5060 Ti (Blackwell) runs FLUX fp8 NATIVELY** — no dequantize penalty, so it's faster still. Use the **3090 for training** (VRAM-bound) and the **5060 Ti for generation** (fp8-speed-bound).

### Batch Generation Script

The production script is **`/home/ben/batch_comfy.py`** (drives ComfyUI over HTTP). It generates 20 images per character from a `selections.json` file.

**Usage:**

```bash
# 1. Prepare face references
mkdir -p /home/ben/ComfyUI/input/cast_faces/
# Place one reference image per character

# 2. Create selections.json (or use the cast page picker)
# Format: [{"character": "Roxanne", "img": "/static/cast_faces/roxanne_anchor.png"}, ...]

# 3. Run generation
python3 scripts/kontext_batch.py
```

**What it does:**
1. Loads `FluxKontextPipeline` with sequential CPU offload (fits 3090)
2. Reads character list from `selections.json`
3. For each character, generates 20 images with varied prompts
4. Skips already-generated images (resumable — safe to interrupt)
5. Saves to `/home/ben/ComfyUI/output/lora_kontext/{character}_{00-19}.png`

**Performance:** ~32s per image @ 12 steps (ComfyUI fp8). Full run (87 chars × 20 images = 1,740 images): **~15 hours** on a single 3090.

### VRAM Strategy

ComfyUI fp8 fits entirely in the 3090's 24GB — **no CPU offload needed** (the old diffusers path required `enable_sequential_cpu_offload()` and OOM'd without it). Peak VRAM: ~22.7 GB during generation.

**Before starting, free VRAM:**
```bash
pkill -9 -f "VLLM|vllm|EngineCore"
pkill -9 -f "ComfyUI"
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # should show ~1.5GB used
```

### The 20-Pose Training Set

Each character gets **20 images**: **12 tight headshots** (neck-up) + **8 full-body shots** (tall 832×1248 canvas, genre outfit).

**Hard rules (learned the hard way):**
- **NO hands** — FLUX renders hands badly and identity training doesn't need them. Never use hand-involving poses (looking-down-at-hands, arms-crossed, chin-on-hand); bake "hands out of frame" into every prompt.
- **NO wind/hair changes** — hair is identity; "blowing in wind" alters it and breaks consistency. Use "hair unchanged".
- **Tight headshots** for face identity — `preserve the FACE IDENTICALLY` (not "keep the person EXACTLY the same", which glitches accessories).
- **Full-body on a TALL canvas** — a square 1024×1024 + face-only reference produces "midget" shots (giant head, stunted body). Fix: **832×1248 canvas + composite the face small (~35% width) at the top** so the model builds the body beneath it (`make_fullbody_ref()` in `batch_comfy.py`).
- **Genre outfit** for full-body consistency (biker leather, pirate coat, knight armor…) via a `{OUTFIT}` placeholder mapped from the game.

**Why this split:** headshots teach face geometry; full-body teaches body/outfit. Together they prevent the LoRA from learning just the reference image's lighting.

---

## Full-Body Generation: Head-Proportion + Identity Fix (CRITICAL)

FLUX Kontext **cannot** reliably generate a natural full-body from a face-only reference — the head size is dominated by the model's own prior, NOT the composite size. The composite face fraction controls **identity**, not head size.

**Measured (ArcFace identity sim vs reference, on the 5060 Ti fp8 node):**

| Composite | ArcFace identity | Head/body ratio | Verdict |
|-----------|------------------|-----------------|---------|
| 35% (original) | strong | 0.107–0.140 | head too big / variable |
| 15% | **0.170** ❌ | 0.099 | face too small → loses identity |
| **20%** | **0.687** ✅ | 0.101 | ✅ **sweet spot** |
| 25% | 0.720 ✅ | 0.108 | slightly bigger head |

**Locked config:** `FACE_FRAC = 0.20` composite + `832×1248` tall canvas (see `scripts/batch_fullbody_5060.py`). Below 20% the face is too small to carry identity (ArcFace < 0.3 = different person); above 25% the head grows. `guidance` 2.5 vs 4.0 made no meaningful difference.

**How to measure identity/proportion correctly:** use `insightface` (ArcFace `buffalo_l`) — face bbox height ÷ image height for proportion (natural ≈ 0.08), and `normed_embedding` cosine similarity to the reference for identity. Do **NOT** use a skin-tone/column heuristic — it falsely reports "tiny head".

**Two-stage alternative** (when reference-only full-body fails): train the LoRA on headshots (identity), then generate full-body *with the LoRA* + a "full body" prompt — no face reference to fight.

---

## ai-toolkit FLUX LoRA Training on 24GB (the OOM fix)

`quantize: true` **alone OOMs** — it loads the full 23 GB bf16 transformer into VRAM *before* quantizing, peaking past 24 GB. Add **`low_vram: true`**:

```yaml
model:
  name_or_path: "black-forest-labs/FLUX.1-dev"
  is_flux: true
  quantize: true
  low_vram: true   # ← required on a 24 GB card
```

Result: ~12.6 GB VRAM after quantize (vs 23.55 GB OOM without it). Full working config in `config/nita_lora.yaml`.

**FLUX.1-dev is license-gated** — needs a HF token at `~/.cache/huggingface/token` (accept the license first). Training setup on pop-os: `~/ai-toolkit/` venv, run `./venv/bin/python run.py config/nita_lora.yaml`.

**Machine split (locked):** **3090 (24 GB) = LoRA training** (FLUX.1-dev base ~23 GB + optimizer states). **5060 Ti (16 GB) = fp8 generation** (Blackwell fp8 native, ~50 s/img at 832×1248, ~24 s at 1024²).

---

## Phase 2: ArcFace Identity Verification

Before training a LoRA, verify the generated images actually depict the same person.

```bash
uv pip install insightface onnxruntime
```

**Gate thresholds:**
- **Mean cosine similarity** across all 20 images: **≥ 0.78**
- **Minimum pairwise similarity**: **≥ 0.65** (no single probe is an impostor)

```python
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

def verify_identity(image_paths):
    embeddings = []
    for path in image_paths:
        img = np.array(Image.open(path).convert('RGB'))
        faces = app.get(img)
        if len(faces) != 1:
            return False, f"Expected 1 face, got {len(faces)} in {path}"
        embeddings.append(faces[0].normed_embedding)

    embs = np.array(embeddings)
    sim_matrix = embs @ embs.T
    mean_sim = sim_matrix.mean()
    min_sim = sim_matrix[~np.eye(len(embs), dtype=bool)].min()

    passed = mean_sim >= 0.78 and min_sim >= 0.65
    return passed, f"mean={mean_sim:.3f} min={min_sim:.3f}"
```

**If it fails:** Regenerate the specific poses that are outliers. Usually they're extreme angles (pure profile, looking up/down) where Kontext struggles.

### Artifact QC (complement to ArcFace)

ArcFace verifies *identity* (same person), but not *quality* — it won't catch mangled hands, hair artifacts, or face glitches. Run the **Qwen2.5-VL-7B** artifact gate for that:

```bash
/home/ben/clip_audit/qc_lora.py   # flags hands/hair/face/artifacts → lora_qc_report.json
```

It's VRAM-gated (waits for the FLUX batch to finish) and resume-safe. A character's 20 images must pass **both** gates: ArcFace (`mean ≥ 0.78`, `min ≥ 0.65`) AND zero `usable:false` artifacts.

---

## Phase 3: Krea 2 LoRA Training

### Model Setup

Krea 2 Turbo int8 is a quantized version optimized for character consistency. The model file:

```
/home/ben/ComfyUI/models/diffusion_models/krea2_turbo_int8_convrot.safetensors
```

**⚠️ Status**: Krea 2 is **downloaded but NOT yet tested for LoRA training**. The training configuration below is based on the FLUX architecture family and will need validation.

### Training Configuration

Using **kohya_ss** or **ai-toolkit**:

```toml
# lora_config.toml
[model]
pretrained_model_name_or_path = "/home/ben/ComfyUI/models/diffusion_models/krea2_turbo_int8_convrot.safetensors"
vae = null  # bundled
clip_l = null  # bundled
t5xxl = null  # bundled

[network]
type = "lora"
rank = 16
alpha = 16
module_dropout = 0.0
conv_rank = 0
conv_alpha = 0

[training]
resolution = 768                    # Krea 2 native
batch_size = 1                      # 3090 limit
max_train_steps = 2000
learning_rate = 1e-4
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 100
optimizer = "adamw8bit"             # bitsandbytes for VRAM savings
mixed_precision = "bf16"
fp8_base = true                     # Additional VRAM savings
gradient_checkpointing = true
gradient_accumulation_steps = 4     # Effective batch size = 4
save_every_n_steps = 500

[dataset]
dataset_dir = "/home/ben/ComfyUI/output/lora_kontext/"
caption_extension = ".txt"
num_repeats = 10                    # 20 images × 10 = 200 steps/epoch, 10 epochs
```

**Why these parameters:**
- **rank 16 / alpha 16**: Proven sweet spot for face identity LoRAs — captures enough detail without overfitting
- **resolution 768**: Krea 2 native resolution; 1024 training on 3090 would OOM
- **2000 steps**: Empirically determined for 20-image datasets with 10 repeats
- **adamw8bit + fp8_base**: Cuts VRAM from ~22GB to ~16GB during training
- **45-60 min per character**: Full pipeline for 87 characters ≈ 65-87 hours

### Training Script

See [`scripts/train_lora_krea2.py`](scripts/train_lora_krea2.py) for the complete training orchestrator.

**Usage:**

```bash
# Train a single character
python3 scripts/train_lora_krea2.py --character "Roxanne" --images-dir /home/ben/ComfyUI/output/lora_kontext/roxanne/

# Train all characters (87 × ~50 min = ~72 hours)
python3 scripts/train_lora_krea2.py --all --images-dir /home/ben/ComfyUI/output/lora_kontext/

# Resume from checkpoint
python3 scripts/train_lora_krea2.py --character "Roxanne" --resume
```

---

## ComfyUI Integration

### Launch ComfyUI

```bash
cd /home/ben/ComfyUI
python main.py --listen 127.0.0.1 --port 8188 --enable-cors-header --disable-pinned-memory
```

**⚠️ `--disable-pinned-memory` is MANDATORY on the 3090** — without it, jobs silently fail with no API error. The OOM only appears in server console output.

### IP-Adapter Workflow (Alternative to Kontext)

For faster generation (25s vs 5min per image), use IP-Adapter+ with Juggernaut XL instead of FLUX Kontext:

| Parameter | Value | Why |
|-----------|-------|-----|
| **weight** | `0.75` | Sweet spot — face consistency without overpowering pose |
| **end_at** | `0.8` | IP-Adapter stops at 80% — prompt controls final composition |
| **cfg** | `8` | Higher CFG gives prompt more control over direction/angle |
| **weight_type** | `"linear"` | Works better than ease-in for face preservation |
| **embeds_scaling** | `"V only"` | Only scale value embedding, not key/query |

**Trade-off**: IP-Adapter is 12× faster but produces SDXL-quality images (vs Kontext's FLUX quality). For LoRA training datasets, FLUX Kontext quality is worth the time.

### Queue Monitoring

```bash
# Queue status
curl -s http://127.0.0.1:8188/queue | python3 -c "
import json,sys; d=json.load(sys.stdin)
print(f\"{len(d['queue_running'])} running, {len(d['queue_pending'])} pending\")
"

# GPU status
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used --format=csv

# Output images
ls -lh /home/ben/ComfyUI/output/
```

---

## Directory Structure

### On-Disk (pop-os)

Models live on the 16 TB Synology drive at `/mnt/syno1/models/` — symlinked into ComfyUI so paths don't change.

```text
/mnt/syno1/models/                        # 16 TB HDD — model storage (8.7 TB free)
├── FLUX.1-Kontext-dev/                   # 54 GB — FLUX Kontext pipeline
├── MiniMax-H3/                           # 89 GB — MiniMax H3 video model
├── FLUX.2-klein-4B/                      # 23 GB — FLUX.2 klein fast image
├── LTX-2.5/                              # ~40 GB — LTX video model (in progress)
├── checkpoints/                          # 40 GB — ComfyUI checkpoints
│   ├── Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors
│   └── flux1-kontext-dev-fp8-e4m3fn.safetensors
├── loras/                                # 54 GB — trained LoRA files
├── diffusion_models/
│   └── krea2_turbo_int8_convrot.safetensors  # ~5 GB
└── text_encoders/

/home/ben/ComfyUI/models/                # Symlinks → /mnt/syno1/models/
/home/ben/ComfyUI/input/
└── cast_faces/                           # Reference anchor faces (1 per character)
/home/ben/ComfyUI/output/
└── lora_kontext/                         # Generated training sets
    ├── roxanne_00.png
    └── ...
```

### This Repo

```text
lora-training-pipeline/
├── README.md                             # You are here
├── docs/
│   └── architecture.svg                  # Pipeline architecture diagram
├── scripts/
│   ├── batch_comfy.py                    # FLUX Kontext batch generator (ComfyUI fp8 — production)
│   ├── batch_fullbody_5060.py            # 5060 Ti full-body orchestrator (20% composite fix)
│   ├── qc_lora.py                        # Qwen2.5-VL artifact QC (hands/hair/face)
│   ├── train_lora_krea2.py               # Krea 2 LoRA training orchestrator
│   ├── arcface_verify.py                 # ArcFace identity verification gate
│   ├── comfyui_queue.py                  # ComfyUI API queue helper
│   └── kontext_batch.py                  # DEPRECATED — old diffusers path (fails on Ampere)
└── config/
    ├── prompts.json                      # 20 training pose prompts (12 headshots + 8 full-body)
    ├── nita_lora.yaml                    # ai-toolkit FLUX config (low_vram: true)
    └── lora_config.toml                  # Kohya_ss training config
```

---

## Monitoring & Debugging

### Batch Manager Dashboard

A web dashboard for monitoring generation progress:
- **URL**: `http://100.72.70.38:8400`
- **Features**: Character grid, progress bars (X/20), queue status, GPU stats, ETA, pause/resume
- **Start**: `python3 /home/ben/batch_manager.py`

**⚠️ Pause button clears the ComfyUI queue** — requires re-running the queue script to resume.

### GPU Temperature & Utilization

The 3090 runs at **75-78°C at 100% utilization** during sustained ComfyUI workloads. This is normal. Idle after batch completion: 32-39°C.

### Common VRAM States

| State | VRAM Used | Action |
|-------|-----------|--------|
| Idle (clean) | ~1.5 GB | Ready to start |
| vLLM running | ~17 GB | `pkill -9 -f VLLM` |
| ComfyUI idle | ~4-5 GB | Normal — models cached |
| FLUX generation | ~20 GB peak | Sequential offload working |

---

## Pitfalls & Troubleshooting

### FLUX Kontext

| Symptom | Cause | Fix |
|---------|-------|-----|
| Generation 10× slow (2–6 min/**step**, GPU idle at ~1%) | diffusers mmaps the model to disk — every denoise step re-reads ~24 GB off a SATA/HDD (disk pinned at ~100% util, hundreds of GB `read_bytes`) | Add `disable_mmap=True` to `from_pretrained()` so weights load into RAM instead of mmap |
| CUDA OOM at load | Offload not enabled | Add `pipe.enable_sequential_cpu_offload()` |
| CUDA OOM mid-generation | Ghost processes holding VRAM | Kill stale Python processes: see VRAM cleanup section |
| Bad face consistency on extreme angles | Kontext struggles with pure 90° profiles | Increase `guidance_scale` to 3.0-3.5 for those prompts |
| "Connection refused" to ComfyUI | Server not running or wrong port | Check `ps aux | grep ComfyUI`, verify port 8188 |
| Silent job failure | Missing `--disable-pinned-memory` flag | Always launch with this flag on 3090 |

### LoRA Training

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM during training | batch_size too high | Use batch_size=1 + gradient_accumulation_steps=4 |
| LoRA produces "same face" for all characters | Overfitting / rank too low | Increase rank to 32, add module_dropout=0.1 |
| LoRA doesn't capture identity | Underfitting / insufficient steps | Increase to 3000-4000 steps |
| Training loss spikes | Learning rate too high | Reduce lr to 5e-5, add more warmup |
| Krea 2 model not found | Wrong path or not downloaded | Verify safetensors file exists at the expected path |

### General

| Symptom | Cause | Fix |
|---------|-------|-----|
| ComfyUI won't start | System libraries missing | See `nighttale-game-dev` skill: references/comfyui-system-deps.md |
| "KeyError: 'character'" | selections.json wrong format | Verify format: list of `{"character": "...", "img": "..."}` |
| Images have wrong colors | VAE mismatch (16ch vs 4ch) | FLUX VAE is 16-channel. Don't use SDXL VAE with FLUX models. |
| Model drive unmounts after reboot | No fstab entry — session-only udisks mount evaporates on power-off | Mount permanently: fstab entry, or a systemd `.mount` unit via `systemctl link`+`enable` (works even with passwordless `systemctl` only) |
| 5060 Ti BSOD `0x1E` (nvlddmkm.sys) under CUDA training | NVIDIA Blackwell driver bug under sustained compute load | Update to latest driver; if it persists lock GPU to base clock (`nvidia-smi -lgc 0,<base>`), reset with `-rgc` |

### No-Dumbledore Rule

Dark fantasy wizard characters MUST NOT have round spectacles, half-moon glasses, or grandfatherly appearance. Use sharp angular faces, scarred skin, glowing arcane eyes, braided beards, ritual scarification. Add `glasses spectacles Dumbledore` to negative prompt.

---

## Performance Benchmarks

All measurements on **RTX 3090 24GB**, pop-os Linux, CUDA 12.1.

### Dataset Generation

| Method | Time/Image | Quality | VRAM Peak |
|--------|------------|---------|-----------|
| **FLUX Kontext (ComfyUI fp8)** | ~32 sec | ⭐⭐⭐⭐⭐ | ~22.7 GB |
| IP-Adapter + Juggernaut XL | ~25 sec | ⭐⭐⭐ | ~8 GB |
| FLUX.2-klein (4B) | ~9 sec | ⭐⭐⭐ | 8.4 GB |

### LoRA Training

| Model | Time/Character | VRAM | Quality |
|-------|---------------|------|---------|
| **Krea 2 int8** | ~45-60 min | ~16 GB | TBD (untested) |
| FLUX.1-dev (bf16) | ~60-90 min | ~22 GB | ⭐⭐⭐⭐ |

### Full Pipeline Estimates

| Scale | Characters | Images | Generation | ArcFace | Training | **Total** |
|-------|-----------|--------|------------|---------|----------|-----------|
| Small test | 1 | 20 | ~11 min | 30 sec | 1 hr | **~1.3 hrs** |
| One game | 6 | 120 | ~1.1 hrs | 3 min | 6 hrs | **~7 hrs** |
| Full roster | 87 | 1,740 | 15 hrs | 45 min | 72 hrs | **~88 hrs (~3.7 days)** |

**Pipeline is fully resumable** — images and checkpoints are saved incrementally. Interrupt at any point and resume without data loss.

---

## References

- **FLUX.1-Kontext-dev**: [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- **FLUX.1-dev (ComfyUI)**: [Comfy-Org/flux1-dev](https://huggingface.co/Comfy-Org/flux1-dev)
- **Krea 2**: Gated model from Krea AI — request access at [krea.ai](https://www.krea.ai/)
- **kohya_ss**: [github.com/kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
- **ai-toolkit**: [github.com/ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
- **NightTale Games**: [play.nighttalegames.com](https://play.nighttalegames.com)
- **Cast Page**: [play.nighttalegames.com/static/cast.html](https://play.nighttalegames.com/static/cast.html)

---

## License

MIT — use it, fork it, train your own character LoRAs.

---

*Pipeline by [bensblueprints](https://github.com/bensblueprints). Built for the NightTale game series (EVERMORE-GAMES).*
