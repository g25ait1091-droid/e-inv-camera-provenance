"""H3 (RESULTS.md Entry 88) - is the shifted-template excess a property of generator residuals?

A circularly shifted fingerprint is the field's standard inert control, and Entry 58 (C5) found it elevated
at displacements that are multiples of eight on both axes, in every adapted arm and in base-model
generations that never saw the fingerprint. The autoencoder's 8-pixel grid was ruled out as an explanation
because the base model shows the same excess.

This tests a quantitative account instead of another control. The measured quantity is
rho(s) = NCC(W, shift_s(Y*K)). If W and Y*K each carry a component periodic on the 8-pixel grid - the
decoder's upsampling signature in the residual, and the same signature multiplied into Y*K - then their
cross-correlation is not zero at grid-aligned displacements, and the size of that peak is predicted by the
two periodic components alone, with nothing fitted.

For each image: split W and V = Y*K into their 8x8 phase-periodic part (the mean over each residue class
modulo 8, tiled back up) and the remainder. The prediction is the cross-correlation of the two periodic
parts; the observation is the cross-correlation of the full fields. Both are computed for every
displacement at once by FFT and read at the 769 displacements C5 used, so prediction and observation are
compared on identical points. CPU only. Writes out/h3_shift_model.json.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, glob, time
import numpy as np
from PIL import Image
from multiprocessing import Pool
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2
GENS = os.path.join(V2, "out", "t1", "gens")
FP = os.path.join(V2, "out", "fp")
C5 = os.path.join(V2, "out", "c5_shift_grid.json")
OUT = os.environ.get("H3_OUT", os.path.join(V2, "out", "h3_shift_model.json"))
N_IMG = int(os.environ.get("H3_IMG", "60"))
OFFSET = int(os.environ.get("H3_OFFSET", "0"))   # H3b runs on images disjoint from H3's
WORKERS = int(os.environ.get("H3_W", "5"))
ARMS = ["local_base", "local_A_raw_s0", "local_B_raw_s0",
        "nomark_s0", "nomark_s1", "nomark_s2", "nomarkB_s0", "nomarkB_s1", "nomarkB_s2"]
GRID = 8

_G = {}


def periodic_part(F, g=GRID):
    """The component of F that repeats on the g-pixel grid: the mean of each residue class, tiled back."""
    h, w = F.shape
    blocks = F.reshape(h // g, g, w // g, g).mean(axis=(0, 2))     # g x g phase means
    return np.tile(blocks, (h // g, w // g)), blocks


def xcorr_map(A, B):
    """NCC(A, shift_s(B)) for every displacement s, via FFT. Both inputs are used as given."""
    a = A - A.mean(); b = B - B.mean()
    num = np.fft.irfft2(np.conj(np.fft.rfft2(a)) * np.fft.rfft2(b), s=a.shape)
    return num / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def _init():
    import torch; torch.set_num_threads(1)
    from fingerprints import wavelet_residual, MEAS
    K = np.load(os.path.join(FP, "K_A_E2.npy"))
    disp = np.array(json.load(open(C5))["design"]["displacements_dy_dx"], dtype=int)
    _G.update(wr=wavelet_residual, MEAS=MEAS, K=K, disp=disp)


def _measure(task):
    arm, path = task
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    if a.shape[0] != _G["MEAS"]: return (arm, None)
    Y = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
    W = _G["wr"](Y)
    V = Y * _G["K"]
    Wp, _ = periodic_part(W)
    Vp, _ = periodic_part(V)
    obs = xcorr_map(W, V)
    # predicted from the grid-periodic components alone, normalised by the FULL fields so that the
    # prediction is on the same scale as the observation and nothing is fitted
    aw, av = W - W.mean(), V - V.mean()
    num = np.fft.irfft2(np.conj(np.fft.rfft2(Wp - Wp.mean())) * np.fft.rfft2(Vp - Vp.mean()), s=W.shape)
    pred = num / (np.linalg.norm(aw) * np.linalg.norm(av) + 1e-12)
    d = _G["disp"]
    return (arm, (obs[d[:, 0], d[:, 1]].astype(np.float64), pred[d[:, 0], d[:, 1]].astype(np.float64),
                  float((Wp.std() / W.std()) ** 2), float((Vp.std() / V.std()) ** 2)))


def main():
    t0 = time.time()
    c5 = json.load(open(C5))
    disp = np.array(c5["design"]["displacements_dy_dx"], dtype=int)
    on8 = (disp[:, 0] % GRID == 0) & (disp[:, 1] % GRID == 0)
    print(f"[h3] {len(disp)} displacements, {on8.sum()} on the {GRID}-pixel grid", flush=True)

    tasks = []
    for arm in ARMS:
        fs = sorted(glob.glob(os.path.join(GENS, arm, "*.png")))[OFFSET:OFFSET + N_IMG]
        assert fs, f"{arm}: no generations"
        tasks += [(arm, f) for f in fs]
    print(f"[h3] {len(tasks)} images over {len(ARMS)} arms, {WORKERS} workers", flush=True)

    acc = {}
    with Pool(WORKERS, initializer=_init) as pool:
        for i, (arm, v) in enumerate(pool.imap_unordered(_measure, tasks, chunksize=2), 1):
            if v is None: continue
            a = acc.setdefault(arm, {"obs": [], "pred": [], "fW": [], "fV": []})
            a["obs"].append(v[0]); a["pred"].append(v[1]); a["fW"].append(v[2]); a["fV"].append(v[3])
            if i % 60 == 0: print(f"[h3] {i}/{len(tasks)} ({(time.time()-t0)/60:.1f} min)", flush=True)

    arms_out, rho_all, rank_all = {}, [], []
    for arm in ARMS:
        if arm not in acc: continue
        obs = np.mean(acc[arm]["obs"], axis=0)
        pred = np.mean(acc[arm]["pred"], axis=0)
        n = len(acc[arm]["obs"])
        rank = float(stats.spearmanr(obs, pred).statistic)
        pear = float(np.corrcoef(obs, pred)[0, 1])
        oex = float(obs[on8].mean() - obs[~on8].mean())
        pex = float(pred[on8].mean() - pred[~on8].mean())
        se = float(np.std(acc[arm]["obs"], axis=0)[on8].mean() / np.sqrt(n))
        arms_out[arm] = {"n_images": n,
                         "observed_on8": float(obs[on8].mean()), "observed_off": float(obs[~on8].mean()),
                         "observed_excess": oex, "observed_excess_se": se,
                         "predicted_on8": float(pred[on8].mean()), "predicted_off": float(pred[~on8].mean()),
                         "predicted_excess": pex,
                         "predicted_over_observed": pex / oex if oex else None,
                         "within_2se": bool(abs(pex - oex) <= 2 * se),
                         "spearman_pred_obs": rank, "pearson_pred_obs": pear,
                         "grid_variance_share_W": float(np.mean(acc[arm]["fW"])),
                         "grid_variance_share_YK": float(np.mean(acc[arm]["fV"]))}
        rho_all.append(pear); rank_all.append(rank)
        print(f"[h3] {arm:16s} excess obs {oex:+.3e} pred {pex:+.3e} "
              f"({arms_out[arm]['predicted_over_observed']:.2f}x) rank {rank:+.2f} "
              f"{'within 2 SE' if arms_out[arm]['within_2se'] else 'OUTSIDE 2 SE'}", flush=True)

    # pooled on-grid agreement across arms: the H3b criterion of Entry 91
    oex = np.array([a["observed_excess"] for a in arms_out.values()])
    pex = np.array([a["predicted_excess"] for a in arms_out.values()])
    ose = np.array([a["observed_excess_se"] for a in arms_out.values()])
    pooled_obs = float(oex.mean()); pooled_pred = float(pex.mean())
    pooled_se = float(np.sqrt((ose ** 2).sum()) / len(ose))
    ratios = pex / oex
    h3b_met = bool(abs(pooled_pred - pooled_obs) <= 2 * pooled_se and
                   0.67 <= float(np.median(ratios)) <= 1.5)
    pooled = {"observed_excess": pooled_obs, "predicted_excess": pooled_pred, "se": pooled_se,
              "difference_in_se": float((pooled_pred - pooled_obs) / pooled_se),
              "median_ratio": float(np.median(ratios)), "ratios": ratios.tolist(),
              "criterion_met": h3b_met,
              "reading": ("the on-grid excess is quantitatively accounted for by the grid-periodic "
                          "components of the residual and of Y*K, with nothing fitted" if h3b_met else
                          "the grid-periodic components do not account for the on-grid excess")}
    print(f"[h3] pooled on-grid excess: observed {pooled_obs:+.3e}, predicted {pooled_pred:+.3e}, "
          f"{pooled['difference_in_se']:+.2f} SE, median ratio {pooled['median_ratio']:.2f} "
          f"-> {'MET' if h3b_met else 'not met'}", flush=True)
    rank_med = float(np.median(rank_all))
    n_within = sum(1 for a in arms_out.values() if a["within_2se"])
    if rank_med >= 0.8 and n_within >= len(arms_out) - 1:
        reading = ("the shifted-template excess is explained by residual autocorrelation: the grid-periodic "
                   "components of the residual and of Y*K predict it with nothing fitted")
    else:
        reading = ("the prediction does not account for the excess; the artifact remains open and the "
                   "prediction's failure is reported")
    res = {"entry": "RESULTS.md Entry 88 (H3)", "grid_px": GRID, "n_displacements": int(len(disp)),
           "n_on_grid": int(on8.sum()), "images_per_arm": N_IMG, "fingerprint": c5["fingerprint"],
           "arms": arms_out, "median_spearman": rank_med, "median_pearson": float(np.median(rho_all)),
           "arms_within_2se": n_within, "n_arms": len(arms_out), "reading": reading,
           "pooled_on_grid": pooled, "image_offset": OFFSET,
           "runtime_min": (time.time() - t0) / 60}
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"[h3] median rank correlation {rank_med:+.2f}; {n_within}/{len(arms_out)} arms within 2 SE",
          flush=True)
    print(f"[h3] READING: {reading}", flush=True)


if __name__ == "__main__":
    main()
