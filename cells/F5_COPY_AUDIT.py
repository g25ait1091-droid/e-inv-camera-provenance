# %% F5 — copy-detection audit on the full fine-tuning generations
#
# Paste into E_FULLFT_FINAL.ipynb after cell 2 (forensic core). Needs no GPU training;
# DINOv2 inference over ~3400 images takes roughly 15 min on the GPU, longer on CPU.
#
# WHY THIS MATTERS HERE MORE THAN ANYWHERE ELSE
#   2.24B parameters fine-tuned on 50 images is the most memorisation-prone configuration
#   in the study. LoRA gave a 0.0% flag rate. If full fine-tuning does not, then instance
#   reproduction — not device transfer — could be carrying the measured statistic, and the
#   bound in Section V-G is not interpretable.
#
# THRESHOLDS: read directly out of 01_pilot.ipynb's CFG and hard-coded below, so this
# audit and the LoRA one use identical cut points. Do not change them.
#
# BASE-STUDY REFERENCE (same thresholds, 500 generations per adapter):
#   A_raw_s0/s1/s2_r16   0.0%  0.0%  0.0%      <- the arms carrying the bound
#   D_raw_s0/s1/s2_r16   0.4%  0.2%  0.6%      <- cross-model arms; every flag was resid_z
# So the honest comparison is "0.0% on the arms that carry the bound", and a small
# resid_z-only rate is within what this ensemble produces on adapted generations.
import os, glob, gc, csv as _csv
import numpy as np, torch
import torch.nn.functional as tF
from PIL import Image

COPY_DINO    = 0.90   # verified against 01_pilot.ipynb CFG
COPY_CROP    = 0.92   # verified
COPY_PHASH   = 8      # verified   (<= flags)
COPY_RESID_Z = 6.0    # verified

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ImageHash"], check=False)
import imagehash
from torchvision import transforms as T2

print(f"[F5] thresholds: dino>={COPY_DINO}  crop>={COPY_CROP}  "
      f"phash<={COPY_PHASH}  resid_z>={COPY_RESID_Z}")

# ---- held-out real images per device, from the base study's manifest ----
MAN = {}
with open(os.path.join(BASE, "manifest", "manifest.csv")) as fh:
    for r in _csv.DictReader(fh):
        MAN.setdefault((r["role"], r["split"]), []).append(r["path"])

def _rgb(p):
    with Image.open(p) as im:
        return im.convert("RGB").copy()

def _ncc0(a, b):
    a = a - a.mean(); b = b - b.mean()
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    return 0.0 if na == 0 or nb == 0 else float((a * b).sum() / (na * nb))

def _quadrants(im):
    w, h = im.size
    return [im.crop((0, 0, w//2, h//2)), im.crop((w//2, 0, w, h//2)),
            im.crop((0, h//2, w//2, h)), im.crop((w//2, h//2, w, h))]

dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                      verbose=False).to(DEV).eval()
pre = T2.Compose([T2.Resize(256), T2.CenterCrop(224), T2.ToTensor(),
                  T2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

@torch.no_grad()
def _enc(images):
    out = []
    for s in range(0, len(images), 16):
        b = torch.stack([pre(x) for x in images[s:s+16]]).to(DEV)
        out.append(tF.normalize(dino(b), dim=-1).cpu())
    return torch.cat(out)

csvp = os.path.join(ROOT, "csv", "f5_copy.csv")
hdr = ["tag", "gen_idx", "dino_sim", "crop_sim", "phash_min", "resid_z", "copy_flag"]
done = done_keys(csvp, ["tag", "gen_idx"])

# one reference bank per arm: its 50 training images plus its held-out real split
groups = {}
for tag, role, _ in all_arms():
    groups.setdefault(role, []).append(tag)

for role, tags in groups.items():
    if all((t, str(i)) in done for t in tags for i in range(C.G_PER)):
        print(f"[F5] skip reference {role} — all tags complete")
        continue
    train_paths = sorted(glob.glob(os.path.join(tdir(role), "*.png")))
    held_paths  = MAN[(role, "E2")]
    if not train_paths:
        HALT(f"[F5] no training PNGs at {tdir(role)}")
    hashes = [imagehash.phash(_rgb(p)) for p in train_paths + held_paths]
    resid  = np.stack([wres(load_lum(p, cache=False)).astype(np.float16)
                       for p in train_paths])
    refemb = torch.cat([_enc([_rgb(p) for p in (train_paths + held_paths)[s:s+16]])
                        for s in range(0, len(train_paths) + len(held_paths), 16)])
    print(f"[F5] reference {role}: {len(train_paths)} train + {len(held_paths)} held-out")

    for tag in tags:
        paths = [os.path.join(gdir(tag), f"{i:05d}.png") for i in range(C.G_PER)]
        todo = [(i, p) for i, p in enumerate(paths) if (tag, str(i)) not in done]
        if not todo:
            print(f"[F5] {tag}: already complete"); continue
        for s in range(0, len(todo), 32):
            chunk = todo[s:s+32]
            imgs = [_rgb(p) for _, p in chunk]
            dsim = (_enc(imgs) @ refemb.T).max(dim=1).values.numpy()
            csim = (_enc([q for im in imgs for q in _quadrants(im)]) @ refemb.T) \
                     .max(dim=1).values.numpy().reshape(len(chunk), 4).max(axis=1)
            for k, (gi, p) in enumerate(chunk):
                ph = min(int(imagehash.phash(imgs[k]) - h) for h in hashes)
                W = wres(load_lum(p, cache=False))
                cc = np.array([_ncc0(W, r.astype(np.float32)) for r in resid])
                med = float(np.median(cc))
                mad = float(np.median(np.abs(cc - med))) / 0.6745 + 1e-12
                rz = float((cc.max() - med) / mad)
                flag = int(dsim[k] >= COPY_DINO or csim[k] >= COPY_CROP
                           or ph <= COPY_PHASH or rz >= COPY_RESID_Z)
                append_row(csvp, hdr, [tag, gi, f"{dsim[k]:.4f}", f"{csim[k]:.4f}",
                                       ph, f"{rz:.2f}", flag])
        print(f"[F5] {tag}: {len(todo)} newly audited")
    del resid, refemb, hashes
    gc.collect()
    if DEV == "cuda": torch.cuda.empty_cache()

del dino
gc.collect()
if DEV == "cuda": torch.cuda.empty_cache()

# ---- verdict ----
import pandas as pd
d = pd.read_csv(csvp)
n, exp = len(d), len(all_tags()) * C.G_PER
if n != exp: HALT(f"[F5] incomplete: {n}/{exp} rows")
rate = 100.0 * d["copy_flag"].mean()
print("\n" + "=" * 72)
print(f"[F5] {n}/{exp} generations audited")
print(f"{'tag':14s} {'flags':>6s} {'max dino':>9s} {'max crop':>9s} "
      f"{'min phash':>10s} {'max resid_z':>12s}")
for t in all_tags():
    g = d[d.tag == t]
    print(f"{t:14s} {int(g['copy_flag'].sum()):6d} {g['dino_sim'].max():9.4f} "
          f"{g['crop_sim'].max():9.4f} {int(g['phash_min'].min()):10d} "
          f"{g['resid_z'].max():12.2f}")
print(f"\nOVERALL COPY-FLAG RATE: {rate:.2f}%")
print("   base study, same thresholds: A arms 0.0% / 0.0% / 0.0%; D arms 0.4% / 0.2% / 0.6%")
comp = {c: int(d[d.copy_flag == 1][c].notna().sum()) for c in ()} if False else None
fired = {c: int(((d.copy_flag == 1) & (
            (d[c] >= COPY_DINO) if c == "dino_sim" else
            (d[c] >= COPY_CROP) if c == "crop_sim" else
            (d[c] <= COPY_PHASH) if c == "phash_min" else
            (d[c] >= COPY_RESID_Z))).sum())
         for c in ("dino_sim", "crop_sim", "phash_min", "resid_z")}
print(f"   which detector fired: {fired}")
if rate == 0.0:
    print("VERDICT: no evidence of memorisation under these thresholds. The Section V-G")
    print("         bound is interpretable. Replace the TODO-AUTHOR sentence with the rate.")
elif rate < 1.0:
    print("VERDICT: low but non-zero. Report the exact rate and inspect the flagged images;")
    print("         do NOT report 0.0% by rounding.")
else:
    print("VERDICT: HIGH. Instance reproduction may be carrying the statistic. Report the")
    print("         rate, inspect the flagged images, and call the arm INCONCLUSIVE rather")
    print("         than reporting a bound from memorised generations.")
print("=" * 72)
import json
json.dump({"rate_pct": rate, "n": int(n),
           "thresholds": {"dino": COPY_DINO, "crop": COPY_CROP,
                          "phash": COPY_PHASH, "resid_z": COPY_RESID_Z},
           "per_tag_flags": {t: int(d[d.tag == t]["copy_flag"].sum()) for t in all_tags()}},
          open(os.path.join(ROOT, "F5_copy_audit.json"), "w"), indent=2)
