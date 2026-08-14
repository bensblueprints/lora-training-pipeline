#!/usr/bin/env python
"""LoRA training-data QC: flag bad hands / hair / face / artifacts in the
generated character pose images using Qwen2.5-VL-7B.

Resume-safe (skips files already in the report). Waits for free VRAM so it
never collides with the FLUX batch. Output: lora_qc_report.json + a human
summary on stdout.
"""
import os, json, time, logging, sys, torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

ROOT = "/mnt/ai-models/ComfyUI-output/lora_kontext"
OUT = "/home/ben/clip_audit/lora_qc_report.json"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MIN_FREE_VRAM = 7 * 2**30  # 7 GiB

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.info

PROMPT = (
    "You are a LoRA training-data quality reviewer. Look at this character reference "
    "image and flag ONLY defects that would hurt a character-identity LoRA. Respond with "
    "JSON and nothing else, using exactly these keys: "
    '{"hands_ok": bool, "hair_ok": bool, "face_ok": bool, "usable": bool, "issues": str}. '
    "hands_ok=false if hands are mangled, have extra/missing fingers, or are badly drawn. "
    "hair_ok=false if the hair has artifacts, is cut off at the frame edge, or looks inconsistent. "
    "face_ok=false if the face is distorted, asymmetric, or glitched. "
    "usable=true only if the image is clean enough to train on. "
    'issues = a short phrase of the problems, or "none".'
)


def wait_for_gpu():
    while True:
        free, total = torch.cuda.mem_get_info()
        log("VRAM free %.1f / %.1f GiB" % (free / 2**30, total / 2**30))
        if free >= MIN_FREE_VRAM:
            return
        time.sleep(60)


def collect_files():
    files = []
    for d in sorted(os.listdir(ROOT)):
        dd = os.path.join(ROOT, d)
        if os.path.isdir(dd):
            for f in sorted(os.listdir(dd)):
                if f.endswith(".png"):
                    files.append(os.path.join(d, f))  # "slug/NN.png"
    return files


def main():
    wait_for_gpu()
    report = {}
    if os.path.exists(OUT):
        try:
            report = json.load(open(OUT))
        except Exception:
            report = {}
    files = collect_files()
    todo = [f for f in files if f not in report]
    log("%d images total, %d already checked, %d to review" %
        (len(files), len(report), len(todo)))
    if not todo:
        log("nothing to do")
        return

    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=bnb, device_map="cuda",
        dtype=torch.float16, attn_implementation="sdpa")
    model.eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    log("model loaded; vram %.1f GiB" % (torch.cuda.memory_allocated() / 2**30))

    t0 = time.time()
    bad = []
    for i, rel in enumerate(todo):
        p = os.path.join(ROOT, rel)
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            report[rel] = {"hands_ok": False, "hair_ok": False, "face_ok": False,
                           "usable": False, "issues": "unreadable: %s" % e}
            bad.append(rel)
            continue
        img.thumbnail((768, 768))
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        resp = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0].strip()
        # try to parse a JSON object out of the response
        obj = None
        try:
            s = resp[resp.find("{"): resp.rfind("}") + 1]
            obj = json.loads(s)
        except Exception:
            obj = {"hands_ok": True, "hair_ok": True, "face_ok": True,
                   "usable": True, "issues": "unparseable: " + resp[:80]}
        report[rel] = obj
        if not obj.get("usable", True):
            bad.append(rel)
        if (i + 1) % 5 == 0:
            json.dump(report, open(OUT, "w"), indent=1)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            log("%d/%d  %.2f img/s  flagged=%d  ETA %.0f min" %
                (len(report), len(files), (i + 1) / el, len(bad),
                 (len(todo) - i - 1) / ((i + 1) / el) / 60))
        del inputs, out, img
    json.dump(report, open(OUT, "w"), indent=1)
    log("DONE %d checked, %d flagged, in %.0f min" %
        (len(report), len(bad), (time.time() - t0) / 60))
    log("FLAGGED: %s" % json.dumps(bad))


if __name__ == "__main__":
    main()
