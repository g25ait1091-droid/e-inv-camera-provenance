"""Entry 37 — the transfer function of personalization by spatial-frequency band.

    python t1_band.py fields       six band fields F_b and never-injected comparators F'_b
    python t1_band.py materialise  band{b}_a4 training crops: device A's T split + 4.0 gray RMS * F_b, rounded once
    python t1_band.py vae          A1: SD-3.5 autoencoder round trip of the stored crops, band statistic before/after
    python t1_band.py measure      A2: band statistic on the band arms' generations and the never-injected arms

Statistic s_b(Y) = NCC(P_b Y, F_b) - NCC(P_b Y, F'_b), P_b the same ideal radial band-pass; all six
bands on every image. No wavelet residual (it is itself high-pass). Outputs under out/t1/:
band_fields.json, band_materialise.json, band_vae.json, band_rows.npz, band_summary.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json, glob, time
import numpy as np
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, EINV.SRC)

V2 = EINV.V2; T1 = os.path.join(V2, "out", "t1"); FIELDS = os.path.join(T1, "fields"); FP = os.path.join(V2, "out", "fp")
MEAS = 1024; NB = 6; AMP = 4.0; SEED, SEEDP = 20260912, 20260913
EDGES = [(0.5 / 2 ** (b + 1), 0.5 / 2 ** b) for b in range(NB)]          # b0 = [0.25, 0.5] ... b5 = [0.0078, 0.0156)
BAND_ARMS = [f"band{b}_s0" for b in range(NB)]
NEVER = ["nomark_s0", "nomark_s1", "nomark_s2", "nomarkB_s0", "nomarkB_s1", "nomarkB_s2"]
HF_MODEL = "stabilityai/stable-diffusion-3.5-medium"

def masks():
    fy = np.fft.fftfreq(MEAS)[:, None]; fx = np.fft.rfftfreq(MEAS)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2); out = []
    for b, (lo, hi) in enumerate(EDGES):
        m = (r >= lo) & ((r <= hi) if b == 0 else (r < hi)); out.append(m.astype(np.float32))
    return out

def lum(a): return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def load_fields():
    return ([np.load(os.path.join(FIELDS, f"F_band{b}.npy")) for b in range(NB)],
            [np.load(os.path.join(FIELDS, f"Fp_band{b}.npy")) for b in range(NB)])

_G = {}
def _init():
    _G["M"] = masks(); _G["F"], _G["Fp"] = load_fields()

def band_stats(Y):
    X = np.fft.rfft2(Y - Y.mean()); out = np.empty(NB)
    for b in range(NB):
        yb = np.fft.irfft2(X * _G["M"][b], s=Y.shape)
        out[b] = ncc(yb, _G["F"][b]) - ncc(yb, _G["Fp"][b])
    return out

def _task(t):
    return band_stats(lum(np.asarray(Image.open(t[1]).convert("RGB"), np.float32)))

def band_energy(x, M):
    X = np.abs(np.fft.rfft2(x - x.mean())) ** 2; tot = X.sum()
    return [float((X * m).sum() / tot) for m in M]

# ------------------------------------------------------------------ stages
def stage_fields():
    os.makedirs(FIELDS, exist_ok=True); M = masks()
    for tag, seed in (("F", SEED), ("Fp", SEEDP)):
        g = np.random.default_rng(seed).standard_normal((MEAS, MEAS)).astype(np.float32); G = np.fft.rfft2(g)
        for b in range(NB):
            f = np.fft.irfft2(G * M[b], s=g.shape).astype(np.float32); f -= f.mean(); f /= f.std()
            np.save(os.path.join(FIELDS, f"{tag}_band{b}.npy"), f)
    F, Fp = load_fields()
    rep = {"edges_cycles_per_px": EDGES, "cross_ncc_F_Fp": [ncc(F[b], Fp[b]) for b in range(NB)],
           "energy_fraction_K_A_E2": band_energy(np.load(os.path.join(FP, "K_A_E2.npy")), M),
           "energy_fraction_DiffusionShield_lum": band_energy(np.load(os.path.join(FIELDS, "W_ds_lum.npy")), M)}
    json.dump(rep, open(os.path.join(T1, "band_fields.json"), "w"), indent=1); print(json.dumps(rep, indent=1))

def stage_materialise():
    import t1_ladder as L
    from fingerprints import dv, splits
    _init(); F = _G["F"]; T = splits(dv["Nikon_D200_1"])["T"]; assert len(T) == 50
    base = np.array([band_stats(lum(np.rint(L.load_rgb_crop(fp)))) for fp in T]); meta = {"uninjected": {"band_stats_mean": base.mean(0).tolist()}}
    for b in range(NB):
        d = os.path.join(T1, "train_png", f"band{b}_a4"); os.makedirs(d, exist_ok=True); st, clip, rms = [], [], []
        for i, fp in enumerate(T):
            rgb = L.load_rgb_crop(fp); Y = rgb + AMP * F[b][..., None]
            clip.append(float(((Y < 0) | (Y > 255)).mean()))
            inj = np.clip(np.rint(Y), 0, 255).astype(np.uint8)
            rms.append(float(np.sqrt(((inj.astype(np.float32) - np.rint(rgb)) ** 2).mean())))
            Image.fromarray(inj).save(os.path.join(d, f"{i:04d}.png"), compress_level=1)
            st.append(band_stats(lum(inj.astype(np.float32))))
        st = np.array(st)
        meta[f"band{b}"] = {"n": len(T), "clip_fraction": float(np.mean(clip)), "stored_change_rms_gray": float(np.mean(rms)),
                            "band_stats_mean": st.mean(0).tolist(), "band_stats_se": (st.std(0, ddof=1) / np.sqrt(len(T))).tolist(),
                            "R_b": float(st[:, b].mean() - base[:, b].mean())}
        print(f"band{b}", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in meta[f"band{b}"].items() if k in ("clip_fraction", "stored_change_rms_gray", "R_b")}, flush=True)
    json.dump(meta, open(os.path.join(T1, "band_materialise.json"), "w"), indent=1)

def stage_vae():
    import torch
    from diffusers import AutoencoderKL
    _init(); dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    vae = AutoencoderKL.from_pretrained(HF_MODEL, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
    sets = {"none": os.path.join(T1, "train_png", "none_a0")}
    sets.update({f"band{b}": os.path.join(T1, "train_png", f"band{b}_a4") for b in range(NB)})
    out = {}
    for name, d in sets.items():
        files = sorted(glob.glob(os.path.join(d, "*.png"))); assert len(files) == 50, (name, len(files))
        before, after = [], []
        for f in files:
            a = np.asarray(Image.open(f).convert("RGB"), np.float32)
            x = torch.from_numpy(a / 127.5 - 1.0).permute(2, 0, 1)[None].to(dev)
            with torch.no_grad():
                y = vae.decode(vae.encode(x).latent_dist.mode()).sample
            r = np.clip(np.rint((y[0].permute(1, 2, 0).float().cpu().numpy() + 1.0) * 127.5), 0, 255).astype(np.float32)
            before.append(band_stats(lum(a))); after.append(band_stats(lum(r)))
        before, after = np.array(before), np.array(after)
        out[name] = {"n": len(files), "before_mean": before.mean(0).tolist(), "after_mean": after.mean(0).tolist(),
                     "after_se": (after.std(0, ddof=1) / np.sqrt(len(files))).tolist()}
        print(name, "before", np.round(before.mean(0), 5), "after", np.round(after.mean(0), 5), flush=True)
    T = {}
    for b in range(NB):
        k = f"band{b}"
        num = out[k]["after_mean"][b] - out["none"]["after_mean"][b]; den = out[k]["before_mean"][b] - out["none"]["before_mean"][b]
        T[k] = num / den
    res = {"edges_cycles_per_px": EDGES, "sets": out, "T_vae": T}
    json.dump(res, open(os.path.join(T1, "band_vae.json"), "w"), indent=1); print("T_vae", {k: round(v, 4) for k, v in T.items()})

def stage_measure():
    tasks = [(arm, f) for arm in BAND_ARMS + NEVER for f in sorted(glob.glob(os.path.join(T1, "gens", arm, "*.png")))]
    print(f"[band] {len(tasks)} images", flush=True); t0 = time.time()
    with Pool(6, initializer=_init) as pool:
        res = pool.map(_task, tasks, chunksize=8)
    print(f"[band] measured in {(time.time() - t0) / 60:.1f} min", flush=True)
    rows = {}
    for (arm, _), s in zip(tasks, res): rows.setdefault(arm, []).append(s)
    rows = {k: np.array(v) for k, v in rows.items()}
    np.savez_compressed(os.path.join(T1, "band_rows.npz"), **rows)
    arms = {k: {"n": int(len(v)), "mean": v.mean(0).tolist(), "se": (v.std(0, ddof=1) / np.sqrt(len(v))).tolist()} for k, v in rows.items()}
    never = np.array([arms[a]["mean"] for a in NEVER if a in arms]); off = never.mean(0); off_se = never.std(0, ddof=1) / np.sqrt(len(never))
    mat = json.load(open(os.path.join(T1, "band_materialise.json"))); vae = json.load(open(os.path.join(T1, "band_vae.json")))
    curve = []
    for b in range(NB):
        a = arms[f"band{b}_s0"]; C, se = a["mean"][b], a["se"][b]; R = mat[f"band{b}"]["R_b"]
        excess = C - off[b]; se_c = float(np.sqrt(se ** 2 + off_se[b] ** 2)); Tf = excess / R; Tv = vae["T_vae"][f"band{b}"]
        curve.append({"band": b, "cycles_per_px": EDGES[b], "R_b": R, "C_b": C, "offset_b": float(off[b]), "excess": excess, "excess_se": se_c,
                      "t": excess / se_c, "detected_3se": bool(excess > 3 * se_c), "T_full": Tf, "T_full_se": se_c / R,
                      "T_vae": Tv, "T_full_over_T_vae": Tf / Tv if Tv else None})
    S = {"edges_cycles_per_px": EDGES, "amp_gray_rms": AMP, "offset": off.tolist(), "offset_se": off_se.tolist(), "arms": arms, "curve": curve}
    json.dump(S, open(os.path.join(T1, "band_summary.json"), "w"), indent=1)
    for c in curve: print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in c.items()})

if __name__ == "__main__":
    {"fields": stage_fields, "materialise": stage_materialise, "vae": stage_vae, "measure": stage_measure}[sys.argv[1]]()
