#!/usr/bin/env python3
"""
FLUX Kontext batch generator for LoRA training datasets.

Generates 20 varied-pose images per character using FLUX.1-Kontext-dev
with sequential CPU offload (fits single RTX 3090 24GB).

Usage:
    python3 kontext_batch.py                           # All characters from selections.json
    python3 kontext_batch.py --character "Roxanne"     # Single character
    python3 kontext_batch.py --dry-run                 # Show what would be generated

Environment:
    - RTX 3090 24GB with CUDA 12.1+
    - 64GB system RAM (CPU offload uses ~30GB)
    - FLUX.1-Kontext-dev downloaded to ~/ComfyUI/models/diffusers/FLUX.1-Kontext-dev/
"""

import torch
import os
import json
import time
import glob
import argparse
from pathlib import Path
from diffusers import FluxKontextPipeline
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────

MODEL_PATH = os.path.expanduser("~/ComfyUI/models/diffusers/FLUX.1-Kontext-dev")
FACE_DIR = os.path.expanduser("~/ComfyUI/input/cast_faces/")
OUT_DIR = os.path.expanduser("~/ComfyUI/output/lora_kontext/")
SELECTIONS_PATH = os.path.expanduser("~/evermore_web/static/selections.json")

# 20 training poses covering all angles, lighting, and expressions
PROMPTS = [
    "keep the person EXACTLY the same, front view facing camera, studio lighting, clean white background",
    "keep the person EXACTLY the same, turn head to the LEFT showing three-quarter left profile, window light",
    "keep the person EXACTLY the same, turn head to the RIGHT showing three-quarter right profile, even light",
    "keep the person EXACTLY the same, pure LEFT side profile, head turned 90 degrees, clean background",
    "keep the person EXACTLY the same, pure RIGHT side profile, head turned 90 degrees, clean background",
    "keep the person EXACTLY the same, looking UP at ceiling, chin raised, dramatic low angle",
    "keep the person EXACTLY the same, looking DOWN at hands, chin tucked, overhead shot",
    "keep the person EXACTLY the same, over shoulder looking back, candid moment",
    "keep the person EXACTLY the same, extreme close-up on face, sharp focus on eyes",
    "keep the person EXACTLY the same, eyes closed peaceful expression, soft diffused light",
    "keep the person EXACTLY the same, serious intense stare, dramatic single shadow, Rembrandt lighting",
    "keep the person EXACTLY the same, slight smile, head tilted slightly left, warm interior light",
    "keep the person EXACTLY the same, mouth slightly open speaking, animated expression",
    "keep the person EXACTLY the same, dynamic action, hair blowing in wind, outdoor setting",
    "keep the person EXACTLY the same, leaning forward toward camera, engaged expression, professional",
    "keep the person EXACTLY the same, half-body shot, arms crossed, professional headshot style",
    "keep the person EXACTLY the same, chin resting on hand, thoughtful pose, soft window light",
    "keep the person EXACTLY the same, low angle from below, looking down at viewer, powerful stance",
    "keep the person EXACTLY the same, high angle from above, looking up at camera, softer expression",
    "keep the person EXACTLY the same, head tilted right, skeptical expression, cool blue light",
]

# Generation parameters
GUIDANCE_SCALE = 2.5
INFERENCE_STEPS = 28
BASE_SEED = 42


def sanitize_filename(name: str) -> str:
    """Convert character name to filesystem-safe slug."""
    slug = (
        name.replace(" ", "_")
        .replace("-", "_")
        .replace('"', "")
        .replace("'", "")
        .replace("/", "_")
    )[:30]
    return slug


def load_characters(selections_path: str, char_filter: str | None = None) -> dict[str, str]:
    """
    Load character-to-image mapping from selections.json.

    Returns dict of {character_name: path_to_reference_image}
    """
    if not os.path.exists(selections_path):
        raise FileNotFoundError(
            f"selections.json not found at {selections_path}. "
            "Create it from the cast page or manually."
        )

    with open(selections_path) as f:
        selections = json.load(f)

    chars = {}
    for s in selections:
        name = s["character"]
        if char_filter and name != char_filter:
            continue
        if name in chars:
            continue
        fn = os.path.basename(s["img"])
        path = os.path.join(FACE_DIR, fn)
        if os.path.exists(path):
            chars[name] = path
        else:
            print(f"  ⚠ Missing face image for {name}: {path}")

    return chars


def generate_dataset(
    pipe: FluxKontextPipeline,
    chars: dict[str, str],
    out_dir: str,
    dry_run: bool = False,
) -> dict:
    """
    Generate 20 images per character. Resumable — skips existing images.

    Returns stats dict with counts.
    """
    os.makedirs(out_dir, exist_ok=True)

    total = 0
    skipped = 0
    failed = 0
    failed_details = []

    for char_name, img_path in sorted(chars.items()):
        slug = sanitize_filename(char_name)
        existing = glob.glob(os.path.join(out_dir, f"{slug}_*"))
        if len(existing) >= 20:
            print(f"  ✓ SKIP {char_name} (already has {len(existing)} images)")
            skipped += 20
            continue

        ref = Image.open(img_path).convert("RGB")

        for i, prompt in enumerate(PROMPTS):
            out_path = os.path.join(out_dir, f"{slug}_{i:02d}.png")
            if os.path.exists(out_path):
                skipped += 1
                continue

            if dry_run:
                print(f"  [DRY RUN] {char_name} #{i:02d}: {prompt[:60]}...")
                continue

            try:
                result = pipe(
                    image=ref,
                    prompt=prompt,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=torch.Generator().manual_seed(BASE_SEED + i),
                    num_inference_steps=INFERENCE_STEPS,
                ).images[0]
                result.save(out_path)
                total += 1
                print(f"  ✓ {char_name} {i+1:02d}/20 — {out_path}")
            except Exception as e:
                failed += 1
                failed_details.append(f"{char_name} #{i}: {e}")
                print(f"  ✗ FAIL {char_name} #{i}: {e}")
                torch.cuda.empty_cache()
                time.sleep(2)

    return {
        "generated": total,
        "skipped": skipped,
        "failed": failed,
        "failed_details": failed_details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="FLUX Kontext batch generator for LoRA datasets"
    )
    parser.add_argument(
        "--character", "-c",
        help="Generate for a single character only",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be generated without actually generating",
    )
    parser.add_argument(
        "--model", default=MODEL_PATH,
        help=f"Path to FLUX.1-Kontext-dev diffusers model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--selections", default=SELECTIONS_PATH,
        help=f"Path to selections.json (default: {SELECTIONS_PATH})",
    )
    parser.add_argument(
        "--out-dir", default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FLUX Kontext — LoRA Dataset Generator")
    print("=" * 60)
    print(f"Model:   {args.model}")
    print(f"Output:  {args.out_dir}")
    print(f"Config:  {args.selections}")
    print(f"Poses:   {len(PROMPTS)} per character")
    print()

    # Load characters
    chars = load_characters(args.selections, args.character)
    if not chars:
        print("No characters found. Check selections.json and face images.")
        return
    print(f"Characters: {len(chars)}")
    total_images = len(chars) * len(PROMPTS)
    print(f"Total images to generate: {total_images}")
    print(f"Estimated time: ~{total_images * 5 / 60:.0f} hours on RTX 3090")
    print()

    if args.dry_run:
        print("── DRY RUN — no images will be generated ──")
        pipe = None
    else:
        print("Loading FluxKontextPipeline with sequential CPU offload...")
        pipe = FluxKontextPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe.enable_sequential_cpu_offload()
        print("Pipeline loaded. Starting generation...")
        print()

    stats = generate_dataset(pipe, chars, args.out_dir, dry_run=args.dry_run)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Generated: {stats['generated']}")
    print(f"  Skipped:   {stats['skipped']} (already exist)")
    print(f"  Failed:    {stats['failed']}")
    if stats["failed_details"]:
        print("\n  Failures:")
        for detail in stats["failed_details"]:
            print(f"    - {detail}")
    print(f"\n  Output:    {args.out_dir}")


if __name__ == "__main__":
    main()
