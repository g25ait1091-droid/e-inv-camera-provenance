"""Entry 40 — why does the published watermark transfer? Periodicity versus latent-grid alignment.

    python t1_periodic.py fields | materialise | vae | measure

Fields: P32 = random Gaussian 32x32 tile (seed 20260914) tiled over 1024^2 (period divides the 8-px
latent grid and 16-px transformer patch); P36 = random 36x36 tile (seed 20260915) tiled and cropped
(periodic, aligned with neither); comparators from seeds +100, same period. Zero-mean, unit RMS,
additive at 4.0 gray RMS on all channels, rounded once -> train_png/per32_a4, per36_a4.
Statistic (Entry 33's): rho_add = NCC(W, F) - NCC(W, F'), W the wavelet residual of the luminance;
30 decoys = circular rolls with row and column shifts not multiples of the period. The band0 arm of
Entry 37 is read with the same statistic as the non-periodic comparator.
Outputs under out/t1/: periodic_fields.json, periodic_materialise.json, periodic_vae.json,
periodic_rows.npz, periodic_summary.json.
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
MEAS = 1024; AMP = 4.0; N_DECOY = 30; DECOY_SEED = 20260916
PERIODS = {"per32": (32, 20260914), "per36": (36, 20260915)}
FIELD_SET = {"per32": ("P32.npy", "P32p.npy", 32), "per36": ("P36.npy", "P36p.npy", 36), "band0": ("F_band0.npy", "Fp_band0.npy", None)}
ARM_FIELD = {"per32_s0": "per32", "per36_s0": "per36", "band0_s0": "band0"}
TRAIN_DIR = {"per32": "per32_a4", "per36": "per36_a4", "band0": "band0_a4"}
NEVER = ["nomark_s0", "nomark_s1", "nomark_s2", "nomarkB_s0", "nomarkB_s1", "nomarkB_s2"]
HF_MODEL = "stabilityai/stable-diffusion-3.5-medium"

def lum(a): return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
def ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def tiled(p, seed):
    t = np.random.default_rng(seed).standard_normal((p, p)).astype(np.float32); n = -(-MEAS // p)
    f = np.tile(t, (n, n))[:MEAS, :MEAS].astype(np.float32); f -= f.mean(); f /= f.std(); return f

_G = {}
def _init():
    import torch as _t; _t.set_num_threads(1)
    from fingerprints import wavelet_residual
    _G["wr"] = wavelet_residual; rng = np.random.default_rng(DECOY_SEED); _G["F"] = {}
    for k, (f, fp, p) in FIELD_SET.items():
        F = np.load(os.path.join(FIELDS, f)); Fp = np.load(os.path.join(FIELDS, fp)); shifts = []
        while len(shifts) < N_DECOY:
            dr, dc = int(rng.integers(40, MEAS - 40)), int(rng.integers(40, MEAS - 40))
            if p is None or (dr % p and dc % p): shifts.append((dr, dc))
        _G["F"][k] = (F, Fp, [np.roll(np.roll(F, dr, 0), dc, 1) for dr, dc in shifts])

def stats_Y(Y):
    W = _G["wr"](Y); out = {}
    for k, (F, Fp, dec) in _G["F"].items():
        t = ncc(W, F); out[k] = [t - ncc(W, Fp), t] + [ncc(W, D) for D in dec]
    return out

def _task(t):
    return stats_Y(lum(np.asarray(Image.open(t[1]).convert("RGB"), np.float32)))

def stage_fields():
    import t1_band as B
    M = B.masks(); rep = {}
    for k, (p, seed) in PERIODS.items():
        F, Fp = tiled(p, seed), tiled(p, seed + 100)
        np.save(os.path.join(FIELDS, f"P{p}.npy"), F); np.save(os.path.join(FIELDS, f"P{p}p.npy"), Fp)
        rep[k] = {"period_px": p, "cross_ncc_F_Fp": ncc(F, Fp), "energy_fraction_by_band_b0_b5": B.band_energy(F, M)}
    json.dump(rep, open(os.path.join(T1, "periodic_fields.json"), "w"), indent=1); print(json.dumps(rep, indent=1))

def stage_materialise():
    import t1_ladder as L
    from fingerprints import dv, splits
    _init(); T = splits(dv["Nikon_D200_1"])["T"]; assert len(T) == 50
    none_files = sorted(glob.glob(os.path.join(T1, "train_png", "none_a0", "*.png"))); assert len(none_files) == 50
    base = [stats_Y(lum(np.asarray(Image.open(f).convert("RGB"), np.float32))) for f in none_files]
    meta = {"uninjected": {k: float(np.mean([b[k][0] for b in base])) for k in FIELD_SET}}
    for k, (p, _) in PERIODS.items():
        F = _G["F"][k][0]; d = os.path.join(T1, "train_png", TRAIN_DIR[k]); os.makedirs(d, exist_ok=True); st, clip, rms = [], [], []
        for i, fp in enumerate(T):
            rgb = L.load_rgb_crop(fp); Y = rgb + AMP * F[..., None]
            clip.append(float(((Y < 0) | (Y > 255)).mean())); inj = np.clip(np.rint(Y), 0, 255).astype(np.uint8)
            rms.append(float(np.sqrt(((inj.astype(np.float32) - np.rint(rgb)) ** 2).mean())))
            Image.fromarray(inj).save(os.path.join(d, f"{i:04d}.png"), compress_level=1)
            st.append(stats_Y(lum(inj.astype(np.float32)))[k][0])
        meta[k] = {"period_px": p, "clip_fraction": float(np.mean(clip)), "stored_change_rms_gray": float(np.mean(rms)),
                   "R": float(np.mean(st) - meta["uninjected"][k]), "R_se": float(np.std(st, ddof=1) / np.sqrt(len(st)))}
        print(k, meta[k], flush=True)
    b0 = sorted(glob.glob(os.path.join(T1, "train_png", "band0_a4", "*.png"))); assert len(b0) == 50, "run t1_band.py materialise first"
    st = [stats_Y(lum(np.asarray(Image.open(f).convert("RGB"), np.float32)))["band0"][0] for f in b0]
    meta["band0"] = {"period_px": None, "R": float(np.mean(st) - meta["uninjected"]["band0"]), "R_se": float(np.std(st, ddof=1) / np.sqrt(len(st)))}
    print("band0", meta["band0"], flush=True)
    json.dump(meta, open(os.path.join(T1, "periodic_materialise.json"), "w"), indent=1)

def stage_vae():
    import torch
    from diffusers import AutoencoderKL
    _init(); dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    vae = AutoencoderKL.from_pretrained(HF_MODEL, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
    out = {}
    for name, d in [("none", "none_a0")] + [(k, TRAIN_DIR[k]) for k in FIELD_SET]:
        files = sorted(glob.glob(os.path.join(T1, "train_png", d, "*.png"))); assert len(files) == 50, (name, len(files))
        before, after = [], []
        for f in files:
            a = np.asarray(Image.open(f).convert("RGB"), np.float32)
            x = torch.from_numpy(a / 127.5 - 1.0).permute(2, 0, 1)[None].to(dev)
            with torch.no_grad():
                y = vae.decode(vae.encode(x).latent_dist.mode()).sample
            r = np.clip(np.rint((y[0].permute(1, 2, 0).float().cpu().numpy() + 1.0) * 127.5), 0, 255).astype(np.float32)
            sb, sa = stats_Y(lum(a)), stats_Y(lum(r)); before.append({k: sb[k][0] for k in FIELD_SET}); after.append({k: sa[k][0] for k in FIELD_SET})
        out[name] = {"before": {k: float(np.mean([b[k] for b in before])) for k in FIELD_SET}, "after": {k: float(np.mean([a_[k] for a_ in after])) for k in FIELD_SET}}
        print(name, out[name], flush=True)
    T = {k: (out[k]["after"][k] - out["none"]["after"][k]) / (out[k]["before"][k] - out["none"]["before"][k]) for k in FIELD_SET}
    json.dump({"sets": out, "T_vae": T}, open(os.path.join(T1, "periodic_vae.json"), "w"), indent=1); print("T_vae", T)

def stage_measure():
    tasks = [(arm, f) for arm in list(ARM_FIELD) + NEVER for f in sorted(glob.glob(os.path.join(T1, "gens", arm, "*.png")))]
    print(f"[per] {len(tasks)} images", flush=True); t0 = time.time()
    with Pool(6, initializer=_init) as pool:
        res = pool.map(_task, tasks, chunksize=8)
    print(f"[per] measured in {(time.time() - t0) / 60:.1f} min", flush=True)
    rows = {}
    for (arm, _), s in zip(tasks, res):
        for k in FIELD_SET: rows.setdefault(f"{arm}|{k}", []).append(s[k])
    rows = {k: np.array(v) for k, v in rows.items()}
    np.savez_compressed(os.path.join(T1, "periodic_rows.npz"), **{k.replace("|", "__"): v for k, v in rows.items()})
    mat = json.load(open(os.path.join(T1, "periodic_materialise.json"))); vae = json.load(open(os.path.join(T1, "periodic_vae.json")))
    ds = json.load(open(os.path.join(T1, "wm_summary.json")))["cluster"]
    summ = {"fields": {}, "reference_DiffusionShield": {"lambda_wm_offset_corrected_pct": ds["lambda_wm_offset_corrected_pct"], "amp_gray_rms": 6.55}}
    for arm, k in ARM_FIELD.items():
        a = rows.get(f"{arm}|{k}")
        if a is None: continue
        never = np.array([rows[f"{n}|{k}"][:, 0].mean() for n in NEVER if f"{n}|{k}" in rows])
        off, off_se = float(never.mean()), float(never.std(ddof=1) / np.sqrt(len(never)))
        C, se = float(a[:, 0].mean()), float(a[:, 0].std(ddof=1) / np.sqrt(len(a)))
        dec = a[:, 2:].mean(0); rank = int(1 + (dec > a[:, 1].mean()).sum())
        never_ranks = [int(1 + (rows[f"{n}|{k}"][:, 2:].mean(0) > rows[f"{n}|{k}"][:, 1].mean()).sum()) for n in NEVER if f"{n}|{k}" in rows]
        R = mat[k]["R"]; ex = C - off; ex_se = float(np.sqrt(se ** 2 + off_se ** 2))
        summ["fields"][k] = {"arm": arm, "period_px": FIELD_SET[k][2], "R": R, "C": C, "C_se": se, "offset": off, "offset_se": off_se,
                             "excess": ex, "excess_se": ex_se, "t": ex / ex_se, "decoy_rank_true": rank, "never_injected_ranks": never_ranks,
                             "lambda_pct": 100 * ex / R, "lambda_lo2se_pct": 100 * (ex - 2 * ex_se) / R, "lambda_hi2se_pct": 100 * (ex + 2 * ex_se) / R,
                             "T_vae": vae["T_vae"][k]}
    json.dump(summ, open(os.path.join(T1, "periodic_summary.json"), "w"), indent=1)
    for k, v in summ["fields"].items(): print(k, {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()})

if __name__ == "__main__":
    {"fields": stage_fields, "materialise": stage_materialise, "vae": stage_vae, "measure": stage_measure}[sys.argv[1]]()
