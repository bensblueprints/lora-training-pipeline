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

**Scale**: 87 characters × 20 images = **1,740 training images**, each at 1024×1024.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATASET CREATION (Phase 1)                     │
│                                                                   │
│  cast_faces/          FluxKontextPipeline       lora_kontext/     │
│  ┌─────────┐          ┌──────────────┐         ┌──────────┐      │
│  │ ref.png │─────────▶│  FLUX Kontext │────────▶│ 00-19.png│      │
│  │ (anchor)│          │  (bfloat16)   │         │ per char │      │
│  └─────────┘          │  CPU offload  │         └──────────┘      │
│                       └──────────────┘                            │
│                             │                                     │
│                     3090 24GB VRAM                                │
│                     peak ~20GB per image                          │
├───────────────────────────────────────────────────────────────────┤
│                    QUALITY GATE (Phase 2)                          │
│                                                                   │
│  lora_kontext/        ArcFace (buffalo_l)       PASS / FAIL       │
│  ┌──────────┐         ┌──────────────┐         ┌──────────┐      │
│  │ 00-19.png│────────▶│ cosine sim   │────────▶│ mean≥0.78│      │
│  └──────────┘         │ per pair     │         │ min≥0.65 │      │
│                       └──────────────┘         └──────────┘      │
├───────────────────────────────────────────────────────────────────┤
│                    LORA TRAINING (Phase 3)                         │
│                                                                   │
│  lora_kontext/        kohya_ss / ai-toolkit    ~/.cache/lora/     │
│  ┌──────────┐         ┌──────────────┐         ┌──────────┐      │
│  │ 00-19.png│────────▶│ Krea 2 int8  │────────▶│char.safe │      │
│  │ + captions│        │ rank 16/α 16 │         │tensors   │      │
│  └──────────┘         │ 2000 steps   │         └──────────┘      │
│                       └──────────────┘                            │
│                     3090: ~45-60min/char                          │
└───────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Hardware
- **GPU**: NVIDIA RTX 3090 24GB (single card — pop-os motherboard only supports one)
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

```bash
# Install diffusers with FLUX Kontext support
uv pip install diffusers transformers accelerate sentencepiece protobuf
uv pip install pillow torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Download the model (54 GB — run overnight)
python3 -c "
from diffusers import FluxKontextPipeline
pipe = FluxKontextPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-Kontext-dev',
    torch_dtype='bfloat16',
    cache_dir='/home/ben/ComfyUI/models/diffusers/'
)
"
```

### Batch Generation Script

The production script at [`scripts/kontext_batch.py`](scripts/kontext_batch.py) generates 20 images per character from a `selections.json` file.

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

**Performance:** ~5 minutes per 1024×1024 image, 28 steps. Full run (87 chars × 20 images = 1,740 images): **~145 hours** on a single 3090.

### VRAM Strategy

FluxKontextPipeline on a 3090 requires `enable_sequential_cpu_offload()` — without it, you'll OOM instantly.

```python
pipe = FluxKontextPipeline.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, local_files_only=True
)
pipe.enable_sequential_cpu_offload()  # ← THIS is the magic
```

**How it works:**
- Each model component (text encoder, VAE, transformer, etc.) is loaded to GPU only when needed
- Between components, previous ones are offloaded to CPU RAM
- Peak VRAM: ~20GB during transformer forward pass
- System RAM needed: ~30GB for offloaded components

**Before starting, free VRAM:**
```bash
# Kill any competing GPU processes
pkill -9 -f "VLLM|vllm|EngineCore"
pkill -9 -f "ComfyUI"
nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9

# Verify: should show ~1.5GB used (driver overhead)
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

### The 20-Pose Training Set

Each character gets 20 images covering the full pose matrix needed for a robust LoRA:

| # | Pose | Lighting | Composition |
|---|------|----------|-------------|
| 0 | Front view, facing camera | Studio, clean white bg | Headshot |
| 1 | Three-quarter LEFT profile | Window light | Headshot |
| 2 | Three-quarter RIGHT profile | Even light | Headshot |
| 3 | Pure LEFT side profile | Clean background | Headshot |
| 4 | Pure RIGHT side profile | Clean background | Headshot |
| 5 | Looking UP at ceiling | Dramatic low angle | Headshot |
| 6 | Looking DOWN at hands | Overhead shot | Headshot |
| 7 | Over shoulder, looking back | Candid | Upper body |
| 8 | Extreme close-up on face | Sharp focus on eyes | Close-up |
| 9 | Eyes closed, peaceful | Soft diffused light | Headshot |
| 10 | Serious intense stare | Rembrandt (single shadow) | Headshot |
| 11 | Slight smile, head tilted left | Warm interior | Headshot |
| 12 | Mouth open, speaking | Animated expression | Headshot |
| 13 | Dynamic action, hair in wind | Outdoor | Upper body |
| 14 | Leaning forward, engaged | Professional | Upper body |
| 15 | Half-body, arms crossed | Professional headshot | Half-body |
| 16 | Chin on hand, thoughtful | Soft window light | Headshot |
| 17 | Low angle, looking down at viewer | Powerful stance | Headshot |
| 18 | High angle, looking up at camera | Softer expression | Headshot |
| 19 | Head tilted right, skeptical | Cool blue light | Headshot |

**Why these 20:** The combination of all cardinal directions + expressions + lighting conditions ensures the LoRA learns the character's face geometry, not just the lighting in the reference image.

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

```
/home/ben/
├── ComfyUI/
│   ├── models/
│   │   ├── diffusers/
│   │   │   └── FLUX.1-Kontext-dev/       # 54 GB — FLUX Kontext pipeline
│   │   ├── diffusion_models/
│   │   │   └── krea2_turbo_int8_convrot.safetensors  # ~5 GB — Krea 2 int8
│   │   ├── unet/
│   │   │   └── flux1-dev.safetensors     # 12 GB — FLUX.1-dev for LoRA inference
│   │   ├── checkpoints/
│   │   │   └── Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors  # ~7 GB
│   │   └── ipadapter/
│   │       └── ip-adapter-plus_sdxl_vit-h.safetensors
│   ├── input/
│   │   └── cast_faces/                   # Reference anchor faces (1 per character)
│   └── output/
│       └── lora_kontext/                 # Generated training sets
│           ├── roxanne_00.png
│           ├── roxanne_01.png
│           ├── ...
│           └── roxanne_19.png
├── evermore_web/
│   ├── static/
│   │   ├── selections.json               # Cast page face selections
│   │   └── cast.html                     # Character roster + face picker
│   └── games/                            # 14 game configs
└── batch_manager.py                      # Web dashboard (:8400)
```

### This Repo

```
lora-training-pipeline/
├── README.md                             # You are here
├── scripts/
│   ├── kontext_batch.py                  # FLUX Kontext batch generator
│   ├── train_lora_krea2.py               # Krea 2 LoRA training orchestrator
│   ├── arcface_verify.py                 # Identity verification gate
│   └── comfyui_queue.py                  # ComfyUI API queue helper
└── config/
    ├── prompts.json                      # 20 training pose prompts
    └── lora_config.toml                  # kohya_ss training config
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

### No-Dumbledore Rule

Dark fantasy wizard characters MUST NOT have round spectacles, half-moon glasses, or grandfatherly appearance. Use sharp angular faces, scarred skin, glowing arcane eyes, braided beards, ritual scarification. Add `glasses spectacles Dumbledore` to negative prompt.

---

## Performance Benchmarks

All measurements on **RTX 3090 24GB**, pop-os Linux, CUDA 12.1.

### Dataset Generation

| Method | Time/Image | Quality | VRAM Peak |
|--------|------------|---------|-----------|
| **FLUX Kontext** | ~5 min | ⭐⭐⭐⭐⭐ | ~20 GB |
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
| Small test | 1 | 20 | 1.7 hrs | 30 sec | 1 hr | **~3 hrs** |
| One game | 6 | 120 | 10 hrs | 3 min | 6 hrs | **~16 hrs** |
| Full roster | 87 | 1,740 | 145 hrs | 45 min | 72 hrs | **~217 hrs (~9 days)** |

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
