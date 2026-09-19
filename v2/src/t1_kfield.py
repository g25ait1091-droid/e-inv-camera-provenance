"""Entry 59, E1 and E2 — the fingerprint as a known pattern, and the strength/form of a spectrum-matched field.

    python t1_kfield.py fields | materialise | measure

E1: body B's E2 fingerprint estimate K (at its estimated magnitude) injected multiplicatively into body A's
    fifty training crops, Y (1 + alpha K), rounded once: kinj_a12 (three adapters), kinj_a48 (one), and
    kinjd_a3 with uniform dither in [-0.5, 0.5) added before the single rounding (one). Measured with the
    same K (known exactly) against a never-injected PRNU comparator of another camera (Kodak body 0, E2),
    multiplicative statistic rho(W, Y K) - rho(W, Y Kc).
E2: G = random field with the power spectrum of K (random phase), unit RMS; comparator G' the same with another
    seed. gkadd_a4 / gkadd_a1: Y + a G (a gray levels RMS, all channels); gkmul_a4: Y (1 + beta G) with beta
    set so that the RMS change matches the additive 4.0 field. Additive arms are read with the additive
    statistic rho(W, G) - rho(W, G'); the multiplicative arm with rho(W, Y G) - rho(W, Y G'); both are
    recorded for every arm.
Transmission per adapter = (contrast - never-injected offset) / stored contrast R, R measured on the crops
as written minus the same statistic on the uninjected crops (none_a0). Decoys: 30 circular rolls.
Outputs under out/t1/: kfield_fields.json, kfield_materialise.json, kfield_rows.npz, kfield_summary.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json, glob, time, math
import numpy as np
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, EINV.SRC)

V2 = EINV.V2; T1 = os.path.join(V2, "out", "t1"); FIELDS = os.path.join(T1, "fields"); FP = os.path.join(V2, "out", "fp")
MEAS = 1024; N_DECOY = 30; DECOY_SEED = 20260932; G_SEED, GP_SEED = 20260930, 20260931
ARMS = {"kinj_a12": ["kinj_a12_s0", "kinj_a12_s1", "kinj_a12_s2"], "kinj_a48": ["kinj_a48_s0"], "kinjd_a3": ["kinjd_a3_s0"],
        "gkadd_a4": ["gkadd_a4_s0", "gkadd_a4_s1"], "gkadd_a1": ["gkadd_a1_s0", "gkadd_a1_s1"], "gkmul_a4": ["gkmul_a4_s0", "gkmul_a4_s1"]}
STAT = {"kinj_a12": "k", "kinj_a48": "k", "kinjd_a3": "k", "gkadd_a4": "gadd", "gkadd_a1": "gadd", "gkmul_a4": "gmul"}
NEVER = ["nomark_s0", "nomark_s1", "nomark_s2", "nomarkB_s0", "nomarkB_s1", "nomarkB_s2"]
NEVER_A = NEVER[:3]   # K (body B's fingerprint) is naturally present in body-B arms, so the K offset uses body-A arms only

def lum(a): return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
def ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def load_K():
    K = np.load(os.path.join(FP, "K_B_E2.npy")).astype(np.float32)
    Kc = np.load(os.path.join(V2, "out", "fp_kodak", "K_D0_E2.npy")).astype(np.float32)
    assert K.shape == (MEAS, MEAS) and Kc.shape == (MEAS, MEAS), (K.shape, Kc.shape)
    return K, Kc

def stage_fields():
    K, _ = load_K(); mag = np.abs(np.fft.rfft2(K - K.mean()))
    def field(seed):
        ph = np.exp(1j * np.random.default_rng(seed).uniform(0, 2 * np.pi, mag.shape))
        f = np.fft.irfft2(mag * ph, s=K.shape).astype(np.float32); f -= f.mean(); return (f / f.std()).astype(np.float32)
    G, Gp = field(G_SEED), field(GP_SEED)
    np.save(os.path.join(FIELDS, "G_K.npy"), G); np.save(os.path.join(FIELDS, "G_Kp.npy"), Gp)
    import t1_band as B
    M = B.masks()
    rep = {"K_source": "out/fp/K_B_E2.npy", "comparator": "out/fp_kodak/K_D0_E2.npy", "K_std": float(K.std()),
           "ncc_K_Kc": ncc(K, np.load(os.path.join(V2, "out", "fp_kodak", "K_D0_E2.npy"))), "ncc_G_Gp": ncc(G, Gp), "ncc_G_K": ncc(G, K),
           "energy_by_band_K": B.band_energy(K, M), "energy_by_band_G": B.band_energy(G, M)}
    json.dump(rep, open(os.path.join(T1, "kfield_fields.json"), "w"), indent=1); print(json.dumps(rep, indent=1))

_G = {}
def _init():
    import torch as _t; _t.set_num_threads(1)
    from fingerprints import wavelet_residual
    K, Kc = load_K(); G = np.load(os.path.join(FIELDS, "G_K.npy")); Gp = np.load(os.path.join(FIELDS, "G_Kp.npy"))
    rng = np.random.default_rng(DECOY_SEED); sh = [(int(rng.integers(64, MEAS - 64)), int(rng.integers(64, MEAS - 64))) for _ in range(N_DECOY)]
    _G.update(wr=wavelet_residual, K=K, Kc=Kc, G=G, Gp=Gp, Kd=[np.roll(np.roll(K, a, 0), b, 1) for a, b in sh],
              Gd=[np.roll(np.roll(G, a, 0), b, 1) for a, b in sh])

def stats_Y(Y, full=True):
    g = _G; W = g["wr"](Y)
    k = ncc(W, Y * g["K"]); gm = ncc(W, Y * g["G"]); ga = ncc(W, g["G"])
    head = [k - ncc(W, Y * g["Kc"]), ga - ncc(W, g["Gp"]), gm - ncc(W, Y * g["Gp"]), k, ga, gm]
    if not full: return head
    return head + \
           [ncc(W, Y * D) for D in g["Kd"]] + [ncc(W, D) for D in g["Gd"]] + [ncc(W, Y * D) for D in g["Gd"]]
IDX = {"k": 0, "gadd": 1, "gmul": 2}; RAW = {"k": 3, "gadd": 4, "gmul": 5}; DEC0 = {"k": 6, "gadd": 6 + N_DECOY, "gmul": 6 + 2 * N_DECOY}

def _task(t): return stats_Y(lum(np.asarray(Image.open(t[1]).convert("RGB"), np.float32)))

def stage_materialise():
    import t1_ladder as L
    from fingerprints import dv, splits
    _init(); T = splits(dv["Nikon_D200_1"])["T"]; assert len(T) == 50
    K = _G["K"]; G = _G["G"]
    rgbs = [L.load_rgb_crop(fp) for fp in T]
    base = np.array([stats_Y(lum(np.rint(r)), False)[:3] for r in rgbs]); b0 = base.mean(0)
    beta = 4.0 / float(np.sqrt(np.mean([np.mean(r.astype(np.float64) ** 2) for r in rgbs])))
    rng = np.random.default_rng(20260933)
    SPEC = {"kinj_a12": ("k", 12.0, False), "kinj_a48": ("k", 48.0, False), "kinjd_a3": ("k", 3.0, True),
            "gkadd_a4": ("gadd", 4.0, False), "gkadd_a1": ("gadd", 1.0, False), "gkmul_a4": ("gmul", beta, False)}
    meta = {"uninjected_mean_stats": b0.tolist(), "beta_mul": beta}
    for key, (kind, amp, dither) in SPEC.items():
        d = os.path.join(T1, "train_png", key); os.makedirs(d, exist_ok=True); st, clip, rms = [], [], []
        for i, rgb in enumerate(rgbs):
            if kind == "k": Y = rgb * (1.0 + amp * K[..., None])
            elif kind == "gadd": Y = rgb + amp * G[..., None]
            else: Y = rgb * (1.0 + amp * G[..., None])
            if dither: Y = Y + rng.uniform(-0.5, 0.5, Y.shape)
            clip.append(float(((Y < 0) | (Y > 255)).mean())); inj = np.clip(np.rint(Y), 0, 255).astype(np.uint8)
            rms.append(float(np.sqrt(((inj.astype(np.float32) - np.rint(rgb)) ** 2).mean())))
            Image.fromarray(inj).save(os.path.join(d, f"{i:04d}.png"), compress_level=1)
            st.append(stats_Y(lum(inj.astype(np.float32)), False)[:3])
        st = np.array(st); j = IDX[kind]
        meta[key] = {"kind": kind, "amp": amp, "dither": dither, "clip_fraction": float(np.mean(clip)), "stored_change_rms_gray": float(np.mean(rms)),
                     "R": float(st[:, j].mean() - b0[j]), "R_se": float(st[:, j].std(ddof=1) / np.sqrt(len(st))),
                     "R_all_stats": (st.mean(0) - b0).tolist()}
        print(key, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in meta[key].items() if k != "R_all_stats"}, flush=True)
    json.dump(meta, open(os.path.join(T1, "kfield_materialise.json"), "w"), indent=1)

def predict(e, T, SE):
    e = list(e); e[0] += max(0.0, 1 - sum(e))
    return sum(x * y for x, y in zip(e, T)), math.sqrt(sum((x * s) ** 2 for x, s in zip(e, SE)))

def stage_measure():
    from scipy import stats as sps
    arms = [a for v in ARMS.values() for a in v] + NEVER
    rp = os.path.join(T1, "kfield_rows.npz"); rows = dict(np.load(rp)) if os.path.exists(rp) else {}
    files = {a: sorted(glob.glob(os.path.join(T1, "gens", a, "*.png"))) for a in arms}
    todo = [a for a in arms if files[a] and not (a in rows and len(rows[a]) == len(files[a]))]   # arms measured in an earlier pass are kept
    tasks = [(a, f) for a in todo for f in files[a]]
    print(f"[kf] {len(tasks)} images over {len(todo)} arms ({len(rows)} arms cached)", flush=True); t0 = time.time()
    if tasks:
        with Pool(int(os.environ.get("T1_W", "6")), initializer=_init) as pool:
            res = pool.map(_task, tasks, chunksize=8)
        new = {}
        for (a, _), s in zip(tasks, res): new.setdefault(a, []).append(s)
        rows.update({a: np.array(v) for a, v in new.items()})
        np.savez_compressed(rp, **rows)
    print(f"[kf] measured in {(time.time() - t0) / 60:.1f} min", flush=True)
    mat = json.load(open(os.path.join(T1, "kfield_materialise.json"))); fl = json.load(open(os.path.join(T1, "kfield_fields.json")))
    out = {"never_arms": {"k": [a for a in NEVER_A if a in rows], "g": [a for a in NEVER if a in rows]}, "fields": {}}
    for key, tags in ARMS.items():
        kind = STAT[key]; j = IDX[kind]; tags = [t for t in tags if t in rows]
        if not tags: continue
        nev = out["never_arms"]["k" if kind == "k" else "g"]
        nm = np.array([rows[a][:, j].mean() for a in nev]); off, off_se = float(nm.mean()), float(nm.std(ddof=1) / np.sqrt(len(nm)))
        R = mat[key]["R"]; per = []
        dec0 = DEC0[kind]; raw = RAW[kind]
        for t in tags:
            x = rows[t][:, j]; C, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
            rank = int(1 + (rows[t][:, dec0:dec0 + N_DECOY].mean(0) > rows[t][:, raw].mean()).sum())
            per.append({"arm": t, "C": C, "C_se": se, "T_pct": 100 * (C - off) / R, "T_image_se_pct": 100 * math.sqrt(se ** 2 + off_se ** 2) / R, "decoy_rank": rank})
        Tv = np.array([p["T_pct"] for p in per]); rec = {"statistic": kind, "R": R, "offset": off, "offset_se": off_se, "n_adapters": len(per), "adapters": per,
                                                       "T_mean_pct": float(Tv.mean())}
        if len(Tv) >= 2:
            se_ad = math.sqrt((Tv.std(ddof=1) / math.sqrt(len(Tv))) ** 2 + (100 * off_se / R) ** 2)
            rec.update(T_se_pct=se_ad, T_lower99_pct=float(Tv.mean() - sps.t.ppf(0.99, len(Tv) - 1) * se_ad), T_upper99_pct=float(Tv.mean() + sps.t.ppf(0.99, len(Tv) - 1) * se_ad))
        else:
            rec.update(T_se_pct=per[0]["T_image_se_pct"], se_kind="image (one adapter)")
        out["fields"][key] = rec; print(key, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items() if k != "adapters"}, flush=True)
    # prediction for a non-repeating pattern with K's spectrum (band response, bands 0-1 from three adapters)
    curve = json.load(open(os.path.join(T1, "band_summary.json")))["curve"]; b2 = json.load(open(os.path.join(T1, "band2_summary.json")))["bands"]
    T = [c["T_full"] for c in curve]; SE = [c["T_full_se"] for c in curve]
    for b in (0, 1): T[b] = b2[f"band{b}"]["T_mean_pct"] / 100; SE[b] = b2[f"band{b}"]["T_se_adapter_pct"] / 100
    p, ps = predict(fl["energy_by_band_K"], T, SE); out["prediction_K_spectrum_pct"] = 100 * p; out["prediction_se_pct"] = 100 * ps
    F = out["fields"]
    def ratio(a, b): return F[a]["T_mean_pct"] / F[b]["T_mean_pct"] if a in F and b in F and F[b]["T_mean_pct"] else None
    out["ratios"] = {"kinj48_over_kinj12": ratio("kinj_a48", "kinj_a12"), "gkadd1_over_gkadd4": ratio("gkadd_a1", "gkadd_a4"), "gkmul4_over_gkadd4": ratio("gkmul_a4", "gkadd_a4")}
    json.dump(out, open(os.path.join(T1, "kfield_summary.json"), "w"), indent=1)
    print("prediction %.4f%% (se %.4f); ratios %s" % (out["prediction_K_spectrum_pct"], out["prediction_se_pct"], out["ratios"]), flush=True)

if __name__ == "__main__":
    {"fields": stage_fields, "materialise": stage_materialise, "measure": stage_measure}[sys.argv[1]]()
