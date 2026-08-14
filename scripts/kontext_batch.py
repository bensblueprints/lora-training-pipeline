#!/usr/bin/env python3
"""DEPRECATED — see scripts/batch_comfy.py

This script used the diffusers FluxKontextPipeline, which FAILS on the RTX 3090
(Ampere has no fp8 tensor cores):

  - diffusers fp8  -> "mat1 and mat2 must have the same dtype (BFloat16 vs Float8_e4m3fn)"
  - diffusers bnb int8 -> quantization_config silently ignored (loads bf16, OOM)

The working path is ComfyUI's fp8 loader (dequantizes fp8->fp16 natively on
Ampere). The production orchestrator is scripts/batch_comfy.py — 32s/img @ 12
steps vs this script's ~153.7s/img.

Kept for historical reference only.
"""
raise RuntimeError("Deprecated. Use scripts/batch_comfy.py (ComfyUI fp8).")
