import json, os, time, glob, shutil, urllib.request, urllib.error
from PIL import Image

COMFY = "http://127.0.0.1:8188"
FACE_DIR = "/home/ben/ComfyUI/input/cast_faces/"
OUT_BASE = "/mnt/ai-models/ComfyUI-output/lora_kontext/"
SELS = "/home/ben/evermore_web/static/selections.json"
STEPS = 12
GUIDANCE = 2.5
CLIENT_ID = "batch_comfy"

PROMPTS = [
    "keep the person EXACTLY the same, full body shot, standing straight, front view, wearing {OUTFIT}, studio lighting, clean white background",
    "keep the person EXACTLY the same, turn head LEFT, three-quarter left profile, tight headshot, no hands, window light",
    "keep the person EXACTLY the same, turn head RIGHT, three-quarter right profile, tight headshot, no hands, even light",
    "keep the person EXACTLY the same, pure LEFT side profile, head turned 90 degrees, tight headshot, no hands, clean bg",
    "keep the person EXACTLY the same, pure RIGHT side profile, head turned 90 degrees, tight headshot, no hands, clean bg",
    "keep the person EXACTLY the same, looking UP at ceiling, chin raised, tight headshot, no hands, dramatic low angle",
    "keep the person EXACTLY the same, full body shot, walking forward, wearing {OUTFIT}, natural candid pose",
    "keep the person EXACTLY the same, full body shot, looking back over shoulder, wearing {OUTFIT}, candid moment",
    "keep the person EXACTLY the same, extreme close-up on face, sharp focus on eyes, no hands",
    "keep the person EXACTLY the same, full body shot, relaxed standing, casual pose, wearing {OUTFIT}, soft diffused light",
    "keep the person EXACTLY the same, full body shot, confident stance, arms relaxed, wearing {OUTFIT}, soft studio lighting, clean neutral background",
    "keep the person EXACTLY the same, slight smile, head tilted left, tight headshot, no hands, warm interior light",
    "keep the person EXACTLY the same, mouth slightly open animated, speaking expression, tight headshot, no hands",
    "keep the person EXACTLY the same, full body shot, dynamic action pose, energetic, wearing {OUTFIT}, studio lighting",
    "keep the person EXACTLY the same, full body shot, leaning against a wall, casual, wearing {OUTFIT}",
    "keep the person EXACTLY the same, full body shot, professional standing pose, wearing {OUTFIT}, clean neutral background",
    "keep the person EXACTLY the same, thoughtful expression, chin slightly tucked, tight headshot, no hands, soft window light",
    "keep the person EXACTLY the same, low angle from below, tight headshot, no hands",
    "keep the person EXACTLY the same, high angle from above, tight headshot, no hands",
    "keep the person EXACTLY the same, head tilted right, skeptical, tight headshot, no hands, studio lighting, neutral color balance",
]

GENRE_OUTFIT = {
    "ancient_egypt": "ancient Egyptian robes and jewelry",
    "biker_club": "biker leather jacket and denim",
    "cyberpunk": "cyberpunk neon streetwear",
    "dark_fantasy": "dark fantasy robes and armor",
    "horror_mansion": "1920s period attire",
    "jungle_expedition": "explorer safari gear",
    "medieval_knights": "medieval knight armor and tunic",
    "pirate_adventure": "pirate coat and boots",
    "post_apocalyptic": "wasteland survival gear",
    "samurai_japan": "samurai kimono and armor",
    "signalfire": "wildland firefighter gear",
    "space_opera": "futuristic space uniform",
    "steampunk": "steampunk Victorian attire",
    "superhero": "superhero costume",
    "underwater_atlantis": "underwater diving suit",
    "wild_west": "frontier western attire",
    "zombie_apocalypse": "survivalist gear",
}

SLUG_TO_GAME = {}
def load_games():
    global SLUG_TO_GAME
    try:
        d = json.load(open("/home/ben/cast_games.json"))
        SLUG_TO_GAME = d.get("slug_to_game", {})
    except Exception:
        pass

# Full-body poses need a TALL canvas with the face small at the top, so Kontext
# generates a proportionate body instead of a giant head on a square canvas.
FULL_W, FULL_H = 832, 1248
HEAD_W, HEAD_H = 1024, 1024
FULL_INDICES = {0, 6, 7, 9, 10, 13, 14, 15}

def make_fullbody_ref(face_path, out_path):
    face = Image.open(face_path).convert("RGB")
    W, H = FULL_W, FULL_H
    face_w = int(W * 0.35)
    face_h = int(face_w * face.height / face.width)
    face = face.resize((face_w, face_h), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), (250, 250, 252))
    x = (W - face_w) // 2
    y = int(H * 0.04)
    canvas.paste(face, (x, y))
    canvas.save(out_path)

BIKER_CHARS = [
    "Roxanne \"Roxie\" Vance",
    "Ray \"Rayzor\" Martinez",
    "Dale \"Diesel\" Kowalski",
    "Junior Vance",
    "Cheryl Vance",
    "Wes Bulle",
    "Bird Tubbs",
    "Sonny Kiel",
    "Akane Matsuda",
    "Alita Kyubey",
]

def build_workflow(image_name, prompt, seed, prefix, W=1024, H=1024):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "diffusion_models/flux1-kontext-dev-fp8-e4m3fn.safetensors", "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip/clip_l.safetensors", "clip_name2": "clip/t5xxl_fp16.safetensors", "type": "flux"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "vae/ae.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["4", 0]}},
        "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "8": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["6", 0]}},
        "9": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["8", 0], "guidance": GUIDANCE}},
        "10": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7", 0]}},
        "11": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["9", 0], "negative": ["10", 0], "latent_image": ["14", 0], "seed": seed, "steps": STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
        "14": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
    }

def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())

def get(url):
    return json.loads(urllib.request.urlopen(url, timeout=30).read())

def queue(wf):
    r = post(COMFY + "/prompt", {"prompt": wf, "client_id": CLIENT_ID})
    return r["prompt_id"]

def wait_done(pid, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = get(COMFY + "/history/" + pid)
        except Exception:
            time.sleep(2); continue
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                msgs = h[pid].get("messages", [])
                raise RuntimeError("ComfyUI error: " + json.dumps(msgs)[:400])
            if st.get("completed"):
                return h[pid]
        time.sleep(2)
    raise TimeoutError("prompt %s timed out" % pid)

def main():
    load_games()
    with open(SELS) as f:
        selections = json.load(f)
    chars = {}
    for s in selections:
        name = s["character"]
        if name in chars:
            continue
        fn = os.path.basename(s["img"])
        if os.path.exists(os.path.join(FACE_DIR, fn)):
            chars[name] = fn
    ordered = BIKER_CHARS + [c for c in sorted(chars) if c not in BIKER_CHARS]

    total = 0
    for ci, char_name in enumerate(ordered):
        if char_name not in chars:
            continue
        fn = chars[char_name]
        image_name = "cast_faces/" + fn
        slug = char_name.replace(" ", "_").replace("-", "_").replace('"', "").replace("'", "")[:30]
        gt = SLUG_TO_GAME.get(slug, "")
        outfit = GENRE_OUTFIT.get(gt, "casual everyday clothing")
        full_ref_name = f"cast_faces/fullbody_{slug}.png"
        full_ref_path = os.path.join(FACE_DIR, f"fullbody_{slug}.png")
        if not os.path.exists(full_ref_path):
            make_fullbody_ref(os.path.join(FACE_DIR, fn), full_ref_path)
        out_dir = os.path.join(OUT_BASE, slug)
        os.makedirs(out_dir, exist_ok=True)
        existing = len(glob.glob(os.path.join(out_dir, "*.png")))
        if existing >= 20:
            print(f"  SKIP {char_name} ({existing}/20)", flush=True)
            total += existing
            continue
        for i, prompt in enumerate(PROMPTS):
            out_path = os.path.join(out_dir, f"{i:02d}.png")
            if os.path.exists(out_path):
                continue
            t0 = time.time()
            try:
                prompt_final = prompt.replace("{OUTFIT}", outfit)
                if i in FULL_INDICES:
                    ref_img, W, H = full_ref_name, FULL_W, FULL_H
                else:
                    ref_img, W, H = image_name, HEAD_W, HEAD_H
                wf = build_workflow(ref_img, prompt_final, 42 + i, f"lora_kontext/{slug}/{i:02d}", W, H)
                pid = queue(wf)
                h = wait_done(pid)
                imgs = h["outputs"].get("13", {}).get("images", [])
                if not imgs:
                    raise RuntimeError("no output image")
                out = imgs[0]
                src = os.path.join("/home/ben/ComfyUI/output", out["subfolder"], out["filename"])
                shutil.copy(src, out_path)
                dt = time.time() - t0
                total += 1
                print(f"  [{ci+1}/{len(ordered)}] {char_name}: {i+1}/20 ({dt:.0f}s)", flush=True)
            except Exception as e:
                print(f"  FAIL {char_name} #{i}: {e}", flush=True)
                time.sleep(3)
    print(f"\nDONE: {total} images @ {STEPS} steps", flush=True)

if __name__ == "__main__":
    main()
