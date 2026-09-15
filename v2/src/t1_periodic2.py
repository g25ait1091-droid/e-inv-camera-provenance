"""Entry 51 (review follow-up) — tile replicates, grid-separating tiles, independent decoys.

    python t1_periodic2.py fields | materialise | vae | measure

New tiles (additive, 4.0 gray RMS, rounded once, crops -> train_png/per{p}_a4), seeds as listed; the
comparator of each is the same construction with seed + 100:
    per24 (8 x 3: on the 8-px latent grid, not a multiple of the 16-px patch)
    per40 (8 x 5: on the latent grid, not a multiple of the patch)
    per48 (16 x 3: a multiple of both)
    per28 (a multiple of neither; a second off-grid period besides 36)
Existing tiles per32 / per36 (Entry 40) are reused unchanged from out/t1/fields.
Decoys: 30 INDEPENDENT random tiles of the same period, never injected (seeds 20261000 + 100 j + p),
replacing the circular rolls of Entry 40, which the never-injected arms showed are not exchangeable
for grid-aligned periodic fields. Decoy correlations are computed by folding the residual onto the
period, which is exact for a tiled field.
Summary is per adapter: excess_i = C_i - offset (never-injected arms), transmission_i = excess_i / R,
then the mean over adapters with the adapter-level SE (image-level SE flagged when one adapter).
T1P2_LIMIT=n measures only the first n images per arm and writes *_smoke outputs.
Outputs under out/t1/: periodic2_fields.json, periodic2_materialise.json, periodic2_vae.json,
periodic2_rows.npz, periodic2_summary.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json, glob, time
import numpy as np
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, EINV.SRC)

V2 = EINV.V2; T1 = os.path.join(V2, "out", "t1"); FIELDS = os.path.join(T1, "fields")
MEAS = 1024; AMP = 4.0; N_DECOY = 30; DECOY_SEED0 = 20261000
OLD = {"per32": (32, 20260914), "per36": (36, 20260915)}
NEW = {"per24": (24, 20260920), "per28": (28, 20260921), "per40": (40, 20260922), "per48": (48, 20260923)}
ALL = {**OLD, **NEW}
ARM_FIELD = {"per32_s0": "per32", "per32_s1": "per32", "per32_s2": "per32", "per36_s0": "per36", "per36_s1": "per36", "per36_s2": "per36",
             "per24_s0": "per24", "per28_s0": "per28", "per40_s0": "per40", "per48_s0": "per48"}
NEVER = ["nomark_s0", "nomark_s1", "nomark_s2", "nomarkB_s0", "nomarkB_s1", "nomarkB_s2"]
HF_MODEL = "stabilityai/stable-diffusion-3.5-medium"
LIMIT = int(os.environ.get("T1P2_LIMIT", "0")); SFX = "_smoke" if LIMIT else ""

def lum(a): return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
def ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def tile_raw(p, seed): return np.random.default_rng(seed).standard_normal((p, p)).astype(np.float32)
def tiled(p, seed):
    t = tile_raw(p, seed); n = -(-MEAS // p)
    f = np.tile(t, (n, n))[:MEAS, :MEAS].astype(np.float32); f -= f.mean(); f /= f.std(); return f
def paths(k): p = ALL[k][0]; return os.path.join(FIELDS, f"P{p}.npy"), os.path.join(FIELDS, f"P{p}p.npy")
def fold(Wc, p):
    A1 = np.stack([Wc[r::p].sum(0) for r in range(p)]); return np.stack([A1[:, c::p].sum(1) for c in range(p)], 1)

_G = {}
def _init():
    import torch as _t; _t.set_num_threads(1)
    from fingerprints import wavelet_residual
    _G["wr"] = wavelet_residual; _G["F"] = {}
    for k, (p, _) in ALL.items():
        f, fp = paths(k); dec = []
        for j in range(N_DECOY):
            s = DECOY_SEED0 + 100 * j + p; t = tile_raw(p, s); full = np.tile(t, (-(-MEAS // p),) * 2)[:MEAS, :MEAS]
            dec.append((t, float(full.std())))
        _G["F"][k] = (np.load(f), np.load(fp), dec, p)

def stats_Y(Y, keys=None):
    W = _G["wr"](Y); Wc = W - W.mean(); nW = float(np.linalg.norm(Wc)); out = {}
    for k in (keys or ALL):
        F, Fp, dec, p = _G["F"][k]; t = ncc(W, F); Wf = fold(Wc, p)
        out[k] = [t - ncc(W, Fp), t] + [float((Wf * d).sum() / (s_ * nW * MEAS + 1e-12)) for d, s_ in dec]
    return out

def _task(t): return stats_Y(lum(np.asarray(Image.open(t[1]).convert("RGB"), np.float32)))

def stage_fields():
    import t1_band as B
    M = B.masks(); rep = {}
    for k, (p, seed) in NEW.items():
        F, Fp = tiled(p, seed), tiled(p, seed + 100)
        np.save(os.path.join(FIELDS, f"P{p}.npy"), F); np.save(os.path.join(FIELDS, f"P{p}p.npy"), Fp)
        rep[k] = {"period_px": p, "on_latent_grid_8px": p % 8 == 0, "on_patch_grid_16px": p % 16 == 0,
                  "cross_ncc_F_Fp": ncc(F, Fp), "energy_fraction_by_band_b0_b5": B.band_energy(F, M)}
    json.dump(rep, open(os.path.join(T1, "periodic2_fields.json"), "w"), indent=1); print(json.dumps(rep, indent=1))

def stage_materialise():
    import t1_ladder as L
    from fingerprints import dv, splits
    _init(); T = splits(dv["Nikon_D200_1"])["T"]; assert len(T) == 50
    none_files = sorted(glob.glob(os.path.join(T1, "train_png", "none_a0", "*.png"))); assert len(none_files) == 50
    base = [stats_Y(lum(np.asarray(Image.open(f).convert("RGB"), np.float32)), list(NEW)) for f in none_files]
    meta = {"uninjected": {k: float(np.mean([b[k][0] for b in base])) for k in NEW}}
    old = json.load(open(os.path.join(T1, "periodic_materialise.json")))
    for k in OLD: meta[k] = {"period_px": OLD[k][0], "R": old[k]["R"], "R_se": old[k]["R_se"], "source": "periodic_materialise.json"}
    for k, (p, _) in NEW.items():
        F = _G["F"][k][0]; d = os.path.join(T1, "train_png", f"per{p}_a4"); os.makedirs(d, exist_ok=True); st, clip, rms = [], [], []
        for i, fp in enumerate(T):
            rgb = L.load_rgb_crop(fp); Y = rgb + AMP * F[..., None]
            clip.append(float(((Y < 0) | (Y > 255)).mean())); inj = np.clip(np.rint(Y), 0, 255).astype(np.uint8)
            rms.append(float(np.sqrt(((inj.astype(np.float32) - np.rint(rgb)) ** 2).mean())))
            Image.fromarray(inj).save(os.path.join(d, f"{i:04d}.png"), compress_level=1)
            st.append(stats_Y(lum(inj.astype(np.float32)), [k])[k][0])
        meta[k] = {"period_px": p, "clip_fraction": float(np.mean(clip)), "stored_change_rms_gray": float(np.mean(rms)),
                   "R": float(np.mean(st) - meta["uninjected"][k]), "R_se": float(np.std(st, ddof=1) / np.sqrt(len(st)))}
        print(k, meta[k], flush=True)
    json.dump(meta, open(os.path.join(T1, "periodic2_materialise.json"), "w"), indent=1)

def stage_vae():
    import torch
    from diffusers import AutoencoderKL
    _init(); dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    vae = AutoencoderKL.from_pretrained(HF_MODEL, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
    out = {}
    for name, d in [("none", "none_a0")] + [(k, f"per{NEW[k][0]}_a4") for k in NEW]:
        files = sorted(glob.glob(os.path.join(T1, "train_png", d, "*.png"))); assert len(files) == 50, (name, len(files))
        before, after = [], []
        for f in files:
            a = np.asarray(Image.open(f).convert("RGB"), np.float32)
            x = torch.from_numpy(a / 127.5 - 1.0).permute(2, 0, 1)[None].to(dev)
            with torch.no_grad():
                y = vae.decode(vae.encode(x).latent_dist.mode()).sample
            r = np.clip(np.rint((y[0].permute(1, 2, 0).float().cpu().numpy() + 1.0) * 127.5), 0, 255).astype(np.float32)
            sb, sa = stats_Y(lum(a), list(NEW)), stats_Y(lum(r), list(NEW))
            before.append({k: sb[k][0] for k in NEW}); after.append({k: sa[k][0] for k in NEW})
        out[name] = {"before": {k: float(np.mean([b[k] for b in before])) for k in NEW}, "after": {k: float(np.mean([a_[k] for a_ in after])) for k in NEW}}
        print(name, out[name], flush=True)
    T = {k: (out[k]["after"][k] - out["none"]["after"][k]) / (out[k]["before"][k] - out["none"]["before"][k]) for k in NEW}
    json.dump({"sets": out, "T_vae": T}, open(os.path.join(T1, "periodic2_vae.json"), "w"), indent=1); print("T_vae", T)

def stage_measure():
    arms = [a for a in list(ARM_FIELD) + NEVER if os.path.isdir(os.path.join(T1, "gens", a))]
    tasks = [(arm, f) for arm in arms for f in sorted(glob.glob(os.path.join(T1, "gens", arm, "*.png")))[: (LIMIT or None)]]
    print(f"[per2] {len(tasks)} images over {len(arms)} arms", flush=True); t0 = time.time()
    with Pool(6, initializer=_init) as pool:
        res = pool.map(_task, tasks, chunksize=8)
    print(f"[per2] measured in {(time.time() - t0) / 60:.1f} min", flush=True)
    rows = {}
    for (arm, _), s in zip(tasks, res):
        for k in ALL: rows.setdefault(f"{arm}|{k}", []).append(s[k])
    rows = {k: np.array(v) for k, v in rows.items()}
    np.savez_compressed(os.path.join(T1, f"periodic2_rows{SFX}.npz"), **{k.replace("|", "__"): v for k, v in rows.items()})
    mat = json.load(open(os.path.join(T1, "periodic2_materialise.json")))
    def rank(a): return int(1 + (a[:, 2:].mean(0) > a[:, 1].mean()).sum())
    summ = {"n_decoys": N_DECOY, "decoys": "independent never-injected tiles of the same period", "limit": LIMIT, "fields": {}}
    for k in ALL:
        ads = [a for a, f in ARM_FIELD.items() if f == k and f"{a}|{k}" in rows]
        if not ads: continue
        nev = [n for n in NEVER if f"{n}|{k}" in rows]; nm = np.array([rows[f"{n}|{k}"][:, 0].mean() for n in nev])
        off, off_se = float(nm.mean()), float(nm.std(ddof=1) / np.sqrt(len(nm)))
        R = mat[k]["R"]; per = []
        for a in ads:
            x = rows[f"{a}|{k}"][:, 0]; C, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
            per.append({"arm": a, "C": C, "C_se": se, "excess": C - off, "lambda_pct": 100 * (C - off) / R,
                        "lambda_image_se_pct": 100 * float(np.sqrt(se ** 2 + off_se ** 2)) / R, "decoy_rank": rank(rows[f"{a}|{k}"])})
        lam = np.array([p_["lambda_pct"] for p_ in per])
        if len(lam) >= 2:
            se_ad = float(lam.std(ddof=1) / np.sqrt(len(lam))); se_tot = float(np.sqrt(se_ad ** 2 + (100 * off_se / R) ** 2)); kind = "adapter"
        else:
            se_tot = per[0]["lambda_image_se_pct"]; kind = "image (one adapter)"
        summ["fields"][k] = {"period_px": ALL[k][0], "on_latent_grid_8px": ALL[k][0] % 8 == 0, "on_patch_grid_16px": ALL[k][0] % 16 == 0,
                             "R": R, "offset": off, "offset_se": off_se, "n_adapters": len(per), "adapters": per,
                             "lambda_mean_pct": float(lam.mean()), "lambda_se_pct": se_tot, "se_kind": kind,
                             "never_injected_ranks": [rank(rows[f"{n}|{k}"]) for n in nev]}
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in summ["fields"][k].items() if kk != "adapters"}, flush=True)
    json.dump(summ, open(os.path.join(T1, f"periodic2_summary{SFX}.json"), "w"), indent=1)

if __name__ == "__main__":
    {"fields": stage_fields, "materialise": stage_materialise, "vae": stage_vae, "measure": stage_measure}[sys.argv[1]]()
