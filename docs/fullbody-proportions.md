# Full-Body Generation: Fixing the "Big Head" / "Midget" Proportion Problem

> **Status:** Active lesson — hit this again on the Ghost Protocol (Hacktivist Studios) character pipeline, 2026-08-16.
> **Applies to:** FLUX.1 dev / Kontext (ComfyUI + diffusers), and any SD-family model.

---

## The Problem

When you generate a **full-body** shot, the subject comes out looking like a **midget** —
head too big, body too short, top-heavy, squat. Same character, same prompt, but the
head-to-body ratio is wrong.

**Natural human proportions:** head ≈ **1/8** of total body height (7.5–8 heads tall).

**What FLUX gives you by default:** head ≈ **1/4** of body height (a "4 heads tall" figure
— looks like a child or a dwarf).

## Root Cause

FLUX was trained mostly on **portraits and headshots**. Its prior for "a person" is a
close-up face filling the frame. When you ask for "full body," it doesn't shrink the head —
it just stuffs a big head onto a compressed body. This is a model **prior**, not a prompt
bug, so you have to fight it explicitly every time.

## The Fix (verified)

### 1. Use a TALL canvas
| Shot type | Canvas |
|---|---|
| Headshot / portrait | 1024×1024 (or 768×1024) |
| **Full body** | **768×1344 or 832×1216** (never square) |

The square canvas compresses the body; the tall canvas gives the body room and lets the
model place a correctly-proportioned figure.

### 2. Positive prompt — state the proportions explicitly
```
full body shot, head to toe visible, natural human proportions,
small head relative to tall body, realistic height, slim athletic build
```

### 3. Negative prompt — kill the big-head prior
```
big head, large head, oversized head, disproportionate body, midget, dwarf,
short stature, top-heavy, wide-angle distortion, warped proportions, tiny body
```

### 4. Kontext (reference-conditioned) specifics
The reference image gets scaled to **1024×1024** by `FluxKontextImageScale` (that's the
*reference*, not the output). Set the **target** canvas (`EmptyLatentImage`) to the tall
size for full-body output. The reference's identity carries over; the proportions are
controlled by the target canvas + prompts above.

---

## Companion Lesson: Don't Make People Look Cartoonish

The default "cinematic" style block triggers a **3D-render / cartoon** look. The culprits
are the render keywords. What to do:

**❌ Remove:** `unreal engine 5 render`, `8k render`, `ray-traced reflections`,
`subsurface scattering`, `cinematic lighting`, `synthwave color palette`, `volumetric fog`

**✅ Use instead:**
```
photorealistic photograph, real photograph, photojournalism, documentary photography,
natural human skin texture with visible pores, realistic facial features,
natural lighting, sharp focus, 35mm film grain
```

**❌ Negative:** `cartoon, anime, illustration, painting, 3d render, cgi, video game
screenshot, unreal engine, airbrushed, plastic, wax figure, smooth skin, stylized`

Also: the `dirty future` / `used-world` / `oil stains` / `broken capillaries` / `weathered`
keywords paint **random dirt and smudges** onto faces. If the client wants clean faces,
drop those and add `clean clear skin` + negative `dirt on face, grime, smudges, blemishes`.

---

## Character LoRA Training Dataset (proven recipe)

Per character, build **~20–25 images**, all face-consistent (Kontext, same reference):

1. **Hood/hat UP and DOWN** — the face MUST be visible in most shots or the LoRA can't learn identity.
2. **Angles:** front, ¾, profile, looking up, looking down, high angle, low angle, extreme close-up, from behind.
3. **Emotions:** neutral, angry, sad, scared, surprised, laughing, determined, worried.
4. **Full body** (with the proportion fix above): front, ¾, walking, sitting, from behind.
5. **Clean faces** (no random marks) unless a scar is a deliberate design feature.

### Kontext workflow gotcha (ComfyUI)
The reference image is injected via the **`ReferenceLatent`** node
(`VAEEncode(reference)` → `ReferenceLatent(conditioning, latent)`), **NOT** `LatentConcat`
(which only concatenates along spatial axes and will throw `dim not in ['x','-x','y','-y','t','-t']`).

Workflow skeleton:
```
UNETLoader(flux1-kontext-dev-fp8) → DualCLIPLoader(clip_l + t5xxl, flux) → VAELoader(ae)
LoadImage(ref) → FluxKontextImageScale → VAEEncode → ReferenceLatent(conditioning=FluxGuidance(CLIPTextEncode(prompt)), latent)
EmptyLatentImage(W×H) → KSampler → VAEDecode → SaveImage
```

### ComfyUI install gotchas (fresh GPU box)
- Pin to a commit **before** the `comfy_aimdo` / `comfy_kitchen` experimental nodes:
  `git checkout $(git rev-list -1 --before="2026-07-25" HEAD)` — those need a newer torch than the standard pytorch image ships and will crash on import.
- `pip install sqlalchemy` — the new ComfyUI asset DB needs it but it's not always in requirements.txt.
- FLUX UNET goes in **`models/diffusion_models/`** (UNETLoader), text encoders in `models/clip/`, VAE in `models/vae/`. Not `models/checkpoints/`.

### vast.ai disk gotcha
The "disk_space" on an offer is the **host's** disk. Your container gets an **overlay** sized
by the `disk` param you pass at rental. A 24 GB model set (dev + kontext + t5xxl) needs
**≥ 150 GB** of container disk. Always rent with `disk: 200` for LoRA work or you'll hit
"not enough free disk space" mid-download.
