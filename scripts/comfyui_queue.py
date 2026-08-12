#!/usr/bin/env python3
"""
ComfyUI API queue helper for batch image generation.

Submits prompts to a running ComfyUI instance and monitors progress.
Used as a faster alternative to FLUX Kontext for initial prototyping
or when using IP-Adapter + Juggernaut XL.

Usage:
    python3 comfyui_queue.py --workflow workflow.json --prompts prompts.json
    python3 comfyui_queue.py --server 100.72.70.38:8188

Requires:
    - ComfyUI running with --enable-cors-header --disable-pinned-memory
"""

import argparse
import json
import os
import sys
import time
import uuid
from urllib.request import Request, urlopen
from urllib.error import URLError


def submit_prompt(server: str, workflow: dict) -> str | None:
    """
    Submit a workflow to ComfyUI. Returns prompt_id or None on failure.
    """
    payload = json.dumps({"prompt": workflow}).encode("utf-8")
    req = Request(
        f"http://{server}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            if "prompt_id" in result:
                return result["prompt_id"]
            if "node_errors" in result:
                print(f"  Node errors: {json.dumps(result['node_errors'], indent=2)}")
            return None
    except URLError as e:
        print(f"  Connection error: {e}")
        return None


def get_queue_status(server: str) -> dict:
    """Get current queue status from ComfyUI."""
    try:
        with urlopen(f"http://{server}/queue", timeout=5) as resp:
            return json.loads(resp.read())
    except URLError:
        return {"queue_running": [], "queue_pending": []}


def get_history(server: str, prompt_id: str) -> dict | None:
    """Get execution history for a prompt."""
    try:
        with urlopen(f"http://{server}/history/{prompt_id}", timeout=5) as resp:
            return json.loads(resp.read())
    except URLError:
        return None


def wait_for_completion(server: str, prompt_ids: list[str], poll_interval: float = 2.0):
    """Wait for all submitted prompts to complete."""
    pending = set(prompt_ids)
    completed = 0
    failed = 0
    total = len(prompt_ids)

    print(f"\nWaiting for {total} jobs to complete...")
    print(f"Monitor: http://{server}")

    while pending:
        time.sleep(poll_interval)
        status = get_queue_status(server)
        running = len(status.get("queue_running", []))
        queued = len(status.get("queue_pending", []))

        newly_done = []
        for pid in list(pending):
            history = get_history(server, pid)
            if history and pid in history:
                newly_done.append(pid)
                pending.remove(pid)
                completed += 1

        if newly_done:
            progress = completed / total * 100
            print(
                f"  [{completed}/{total}] {progress:.0f}% — "
                f"{running} running, {queued} queued, {len(pending)} waiting"
            )

    print(f"\nDone: {completed} completed, {failed} failed, {total} total")


def build_simple_workflow(
    prompt: str,
    negative_prompt: str = "",
    checkpoint: str = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    width: int = 1024,
    height: int = 1024,
    steps: int = 25,
    cfg: float = 7.0,
    seed: int | None = None,
) -> dict:
    """
    Build a minimal CheckpointLoaderSimple → CLIPTextEncode → KSampler → VAEDecode → SaveImage workflow.
    Node IDs match the standard ComfyUI default numbering.
    """
    import random

    if seed is None:
        seed = random.randint(1, 999999999)

    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt
                or "hat helmet mask glasses spectacles sunglasses face paint makeup beard cartoon anime painting 3d render plastic deformed watermark blurry low quality",
                "clip": ["4", 1],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "comfyui_batch", "images": ["8", 0]},
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="ComfyUI API queue helper for batch generation"
    )
    parser.add_argument(
        "--server", "-s",
        default="127.0.0.1:8188",
        help="ComfyUI server address (default: 127.0.0.1:8188)",
    )
    parser.add_argument(
        "--workflow", "-w",
        help="Path to ComfyUI workflow JSON file",
    )
    parser.add_argument(
        "--prompts", "-p",
        help="Path to prompts JSON file (list of {prefix: str, prompt: str})",
    )
    parser.add_argument(
        "--checkpoint",
        default="Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        help="Checkpoint name for simple workflow (default: Juggernaut XL v9)",
    )
    parser.add_argument(
        "--steps", type=int, default=25,
        help="Inference steps (default: 25)",
    )
    parser.add_argument(
        "--cfg", type=float, default=7.0,
        help="CFG scale (default: 7.0)",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Submit and exit without waiting for completion",
    )
    args = parser.parse_args()

    # Check server is reachable
    print(f"Checking ComfyUI at http://{args.server}...")
    status = get_queue_status(args.server)
    if not status:
        print(f"Error: Cannot reach ComfyUI at {args.server}")
        print("Make sure ComfyUI is running with --enable-cors-header")
        sys.exit(1)

    running = len(status.get("queue_running", []))
    pending_q = len(status.get("queue_pending", []))
    print(f"Server OK — {running} running, {pending_q} queued")

    # Load workflow or prompts
    if args.workflow:
        with open(args.workflow) as f:
            base_workflow = json.load(f)
        print(f"Loaded workflow: {args.workflow}")
    elif args.prompts:
        with open(args.prompts) as f:
            prompts_list = json.load(f)
        print(f"Loaded {len(prompts_list)} prompts from {args.prompts}")
        base_workflow = None  # Will build per-prompt
    else:
        print("Error: Provide --workflow or --prompts")
        sys.exit(1)

    # Submit jobs
    prompt_ids = []

    if args.prompts and not args.workflow:
        for item in prompts_list:
            prefix = item.get("prefix", "batch")
            prompt_text = item["prompt"]

            wf = build_simple_workflow(
                prompt=prompt_text,
                checkpoint=args.checkpoint,
                steps=args.steps,
                cfg=args.cfg,
            )
            # Customize filename prefix
            wf["9"]["inputs"]["filename_prefix"] = prefix

            pid = submit_prompt(args.server, wf)
            if pid:
                prompt_ids.append(pid)
                print(f"  Submitted: {prefix} → {pid}")
            else:
                print(f"  Failed to submit: {prefix}")
            time.sleep(0.2)  # Throttle submissions
    else:
        # Submit workflow as-is, possibly with modifications
        pid = submit_prompt(args.server, base_workflow)
        if pid:
            prompt_ids.append(pid)
            print(f"  Submitted workflow → {pid}")

    print(f"\nSubmitted {len(prompt_ids)} jobs")

    if not args.no_wait and prompt_ids:
        wait_for_completion(args.server, prompt_ids)


if __name__ == "__main__":
    main()
