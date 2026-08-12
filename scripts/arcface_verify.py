#!/usr/bin/env python3
"""
ArcFace identity verification for LoRA training datasets.

Verifies that all 20 generated images for a character depict the same person.
Uses insightface (buffalo_l) for face embedding extraction.

Gate thresholds:
    - Mean cosine similarity ≥ 0.78
    - Minimum pairwise similarity ≥ 0.65

Usage:
    python3 arcface_verify.py --dir /path/to/character/images/
    python3 arcface_verify.py --all --base-dir ~/ComfyUI/output/lora_kontext/
    python3 arcface_verify.py --dir . --save-embeddings  # Save embeddings for later use
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_face_analyzer(device: str = "cuda"):
    """Load insightface with buffalo_l model."""
    from insightface.app import FaceAnalysis

    ctx_id = 0 if device == "cuda" else -1
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=ctx_id)
    return app


def extract_embeddings(
    app, image_paths: list[str], require_single_face: bool = True
) -> tuple[np.ndarray, list[str]]:
    """
    Extract face embeddings from a list of images.

    Returns (embeddings_array, warnings_list).
    embeddings_array shape: (n_images, 512)
    """
    embeddings = []
    warnings = []

    for path in image_paths:
        img = np.array(Image.open(path).convert("RGB"))
        faces = app.get(img)

        if len(faces) == 0:
            warnings.append(f"No face detected in {path}")
            continue
        if require_single_face and len(faces) > 1:
            warnings.append(f"Multiple faces ({len(faces)}) in {path} — using largest")
            # Use largest face (by bounding box area)
            faces.sort(
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )

        embeddings.append(faces[0].normed_embedding)

    if not embeddings:
        raise RuntimeError("No faces detected in any image")

    return np.array(embeddings), warnings


def verify_identity(
    embeddings: np.ndarray,
    mean_threshold: float = 0.78,
    min_threshold: float = 0.65,
) -> dict:
    """
    Verify identity consistency across a set of face embeddings.

    Returns dict with:
        - passed: bool
        - mean_similarity: float
        - min_similarity: float
        - sim_matrix: np.ndarray (for debugging)
        - outliers: list of (idx1, idx2, sim) for pairs below min_threshold
    """
    n = len(embeddings)
    if n < 2:
        return {
            "passed": False,
            "mean_similarity": 0.0,
            "min_similarity": 0.0,
            "sim_matrix": np.array([[]]),
            "outliers": [],
            "error": "Need at least 2 images",
        }

    sim_matrix = embeddings @ embeddings.T  # (n, n) cosine similarities

    # Exclude diagonal (self-similarity = 1.0)
    mask = ~np.eye(n, dtype=bool)
    pairwise_sims = sim_matrix[mask]

    mean_sim = pairwise_sims.mean()
    min_sim = pairwise_sims.min()

    # Find outlier pairs
    outliers = []
    if min_sim < min_threshold:
        low_mask = (sim_matrix < min_threshold) & mask
        for i, j in zip(*np.where(low_mask)):
            if i < j:  # only upper triangle
                outliers.append((int(i), int(j), float(sim_matrix[i, j])))

    passed = mean_sim >= mean_threshold and min_sim >= min_threshold

    return {
        "passed": passed,
        "mean_similarity": float(mean_sim),
        "min_similarity": float(min_sim),
        "sim_matrix": sim_matrix,
        "outliers": outliers,
    }


def format_report(result: dict, image_paths: list[str], warnings: list[str]) -> str:
    """Format verification results as a readable report."""
    status = "✅ PASS" if result["passed"] else "❌ FAIL"
    lines = [
        "=" * 50,
        f"ArcFace Identity Verification: {status}",
        "=" * 50,
        f"Images tested:      {len(image_paths)}",
        f"Mean similarity:    {result['mean_similarity']:.4f} (threshold: 0.78)",
        f"Min similarity:     {result['min_similarity']:.4f} (threshold: 0.65)",
    ]

    if result["outliers"]:
        lines.append(f"\nOutlier pairs ({len(result['outliers'])} below 0.65):")
        for i, j, sim in sorted(result["outliers"], key=lambda x: x[2]):
            lines.append(
                f"  [{i:02d}] ↔ [{j:02d}]: {sim:.4f}  "
                f"({os.path.basename(image_paths[i])} ↔ {os.path.basename(image_paths[j])})"
            )

    if warnings:
        lines.append(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            lines.append(f"  ⚠ {w}")

    if not result["passed"]:
        lines.append("\nACTION REQUIRED: Regenerate outlier images before training.")
    else:
        lines.append("\nDataset ready for LoRA training. ✓")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="ArcFace identity verification for LoRA datasets"
    )
    parser.add_argument(
        "--dir", "-d",
        help="Directory containing character images to verify",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Verify all character subdirectories in --base-dir",
    )
    parser.add_argument(
        "--base-dir",
        default=os.path.expanduser("~/ComfyUI/output/lora_kontext/"),
        help="Base directory for --all mode",
    )
    parser.add_argument(
        "--mean-threshold", type=float, default=0.78,
        help="Minimum mean cosine similarity (default: 0.78)",
    )
    parser.add_argument(
        "--min-threshold", type=float, default=0.65,
        help="Minimum pairwise cosine similarity (default: 0.65)",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Run on CPU instead of CUDA",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    print(f"Loading ArcFace (buffalo_l, {device})...")
    app = load_face_analyzer(device)

    if args.all:
        base = Path(args.base_dir)
        if not base.exists():
            print(f"Error: Base directory not found: {base}")
            sys.exit(1)

        all_results = {}
        all_passed = True
        for subdir in sorted(base.iterdir()):
            if not subdir.is_dir():
                continue
            images = sorted(subdir.glob("*.png")) + sorted(subdir.glob("*.jpg"))
            if len(images) < 2:
                continue

            print(f"\n── {subdir.name} ({len(images)} images) ──")
            embeddings, warnings = extract_embeddings(app, [str(p) for p in images])
            result = verify_identity(embeddings, args.mean_threshold, args.min_threshold)
            all_results[subdir.name] = {
                **{k: v for k, v in result.items() if k != "sim_matrix"},
                "n_images": len(images),
                "warnings": warnings,
            }
            if not result["passed"]:
                all_passed = False
            print(format_report(result, [str(p) for p in images], warnings))

        if args.json:
            print(json.dumps({"passed": all_passed, "characters": all_results}, indent=2))
        else:
            print(f"\n{'=' * 50}")
            print(f"Overall: {'✅ ALL PASS' if all_passed else '❌ SOME FAIL'}")

    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"Error: Directory not found: {dir_path}")
            sys.exit(1)

        images = sorted(dir_path.glob("*.png")) + sorted(dir_path.glob("*.jpg"))
        if len(images) < 2:
            print(f"Error: Need at least 2 images, found {len(images)} in {dir_path}")
            sys.exit(1)

        image_paths = [str(p) for p in images]
        embeddings, warnings = extract_embeddings(app, image_paths)
        result = verify_identity(embeddings, args.mean_threshold, args.min_threshold)

        print(format_report(result, image_paths, warnings))

        if args.json:
            output = {
                "passed": result["passed"],
                "mean_similarity": result["mean_similarity"],
                "min_similarity": result["min_similarity"],
                "n_images": len(images),
                "outliers": result["outliers"],
                "warnings": warnings,
            }
            print(json.dumps(output, indent=2))

        sys.exit(0 if result["passed"] else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
