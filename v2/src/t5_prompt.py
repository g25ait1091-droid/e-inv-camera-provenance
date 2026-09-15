"""Tier 5 — E-PROMPT: the diverse prompt bank on the archived v1 adapters (no retraining).

Uses the six archived adapters (A_raw s0-s2, B_raw s0-s2) and the base model, and generates
500 images each from v1's DIVERSE_PROMPTS bank with the same seed bank as the uniform
generations (GEN_SEED_BASE + j, prompt j % 5), 28 steps, CFG 4.5, 1024^2, PNG.
Then measures every image with the v1 core against K_A_E2 / K_B_E2 and reports the paired
contrast per arm, theta_sym, and the base-model reading. Resumable.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, glob, csv, time, json
import numpy as np, torch
from PIL import Image
sys.path.insert(0, EINV.SRC)

V2 = EINV.V2; OUT = os.path.join(V2, "out", "t5"); os.makedirs(os.path.join(OUT, "gens"), exist_ok=True)
FP = os.path.join(V2, "out", "fp"); ADAPTERS = os.path.join(V2, "data", "adapters")
HF_MODEL = "stabilityai/stable-diffusion-3.5-medium"; MEAS = 1024
G, SEED_BASE, STEPS, CFG, BATCH = 500, 770000, 28, 4.5, 4
DIVERSE_PROMPTS = ["a photograph of a street", "a photograph of a room interior", "a photograph of trees",
                   "a photograph of a building facade", "a photograph of a table with objects"]
ARMS = ["base", "A_raw_s0_r16", "A_raw_s1_r16", "A_raw_s2_r16", "B_raw_s0_r16", "B_raw_s1_r16", "B_raw_s2_r16"]

def log(*a): print(time.strftime("[%H:%M:%S]"), *a, flush=True)

def generate(tag):
    outd = os.path.join(OUT, "gens", tag); os.makedirs(outd, exist_ok=True)
    bank = [(DIVERSE_PROMPTS[j % 5], SEED_BASE + j) for j in range(G)]
    have = len(glob.glob(os.path.join(outd, "*.png")))
    if have >= G: log("skip", tag); return
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(HF_MODEL, torch_dtype=torch.bfloat16).to("cuda")
    if tag != "base": pipe.load_lora_weights(os.path.join(ADAPTERS, tag))
    pipe.set_progress_bar_config(disable=True); i = have; t0 = time.time()
    while i < G:
        chunk = bank[i:i+BATCH]; gs = [torch.Generator(device="cuda").manual_seed(s) for _, s in chunk]
        with torch.no_grad():
            imgs = pipe(prompt=[p for p, _ in chunk], num_inference_steps=STEPS, guidance_scale=CFG, height=MEAS, width=MEAS, generator=gs).images
        for j, im in enumerate(imgs): im.save(os.path.join(outd, f"{i+j:05d}.png"), format="PNG")
        i += len(chunk)
        if i % 100 < BATCH: log(f"{tag}: {i}/{G} ({(time.time()-t0)/60:.1f} min)")
    del pipe; torch.cuda.empty_cache()

_G = {}
def _init():
    torch.set_num_threads(1)
    from fingerprints import wavelet_residual
    _G.update(wr=wavelet_residual, KA=np.load(os.path.join(FP, "K_A_E2.npy")), KB=np.load(os.path.join(FP, "K_B_E2.npy")))
def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
def _m(task):
    arm, path = task; a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    Y = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(np.float32); W = _G["wr"](Y)
    return [arm, os.path.basename(path), _ncc(W, Y*_G["KA"]), _ncc(W, Y*_G["KB"])]

def measure():
    from multiprocessing import Pool
    rows_p = os.path.join(OUT, "rows.csv"); done = set()
    if os.path.exists(rows_p):
        for r in csv.DictReader(open(rows_p)): done.add((r["arm"], r["image"]))
    tasks = [(arm, f) for arm in ARMS for f in sorted(glob.glob(os.path.join(OUT, "gens", arm, "*.png"))) if (arm, os.path.basename(f)) not in done]
    new = not os.path.exists(rows_p); fh = open(rows_p, "a", newline=""); wr = csv.writer(fh)
    if new: wr.writerow(["arm", "image", "rho_KA", "rho_KB"])
    with Pool(6, initializer=_init) as pool:
        for row in pool.imap_unordered(_m, tasks, chunksize=4): wr.writerow(row)
    fh.close()
    import pandas as pd
    d = pd.read_csv(rows_p); S = {"arms": {}}
    for arm in ARMS:
        g = d[d.arm == arm]
        if not len(g): continue
        own = (g.rho_KA - g.rho_KB) if arm.startswith("A_") else (g.rho_KB - g.rho_KA) if arm.startswith("B_") else None
        S["arms"][arm] = {"n": int(len(g)), "mean_KA": float(g.rho_KA.mean()), "mean_KB": float(g.rho_KB.mean()),
                          "paired_contrast": None if own is None else float(own.mean()), "se": None if own is None else float(own.std(ddof=1)/np.sqrt(len(g)))}
    tA = [S["arms"][a]["paired_contrast"] for a in ARMS if a.startswith("A_") and a in S["arms"]]
    tB = [S["arms"][a]["paired_contrast"] for a in ARMS if a.startswith("B_") and a in S["arms"]]
    if tA and tB: S["theta_A"], S["theta_B"], S["theta_sym"] = float(np.mean(tA)), float(np.mean(tB)), float(0.5*(np.mean(tA)+np.mean(tB)))
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1); print(json.dumps(S, indent=1))

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("generate", "all"):
        for tag in ARMS: generate(tag)
    if what in ("measure", "all"): measure()
    log("done", what)
