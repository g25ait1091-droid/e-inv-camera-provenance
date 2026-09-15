"""Entry 51 (review follow-up) — replicates of the two finest octave bands (band0: periods 2-4 px, where
58 % of the fingerprint's energy lies; band1: 4-8 px). Adapters band{0,1}_s{0,1,2}; the s0 arms are the
Entry 37 adapters, re-measured with the same statistic (t1_band.band_stats: band-pass the luminance,
NCC with the band field minus NCC with its never-injected comparator). Per adapter: transmission
T_i = (C_i - offset) / R_b, offset from the never-injected arms; then the mean over adapters, the
adapter-level SE, and a one-sided 99 % t upper limit over adapters.
Writes out/t1/band2_rows.npz and out/t1/band2_summary.json (Entry 37's band_summary.json is untouched)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json, glob, time
import numpy as np
from multiprocessing import Pool
from scipy import stats
sys.path.insert(0, EINV.SRC)
import t1_band as B

BANDS = (0, 1); ARMS = [f"band{b}_s{s}" for b in BANDS for s in range(3)]

def main():
    arms = [a for a in ARMS + B.NEVER if os.path.isdir(os.path.join(B.T1, "gens", a))]
    tasks = [(arm, f) for arm in arms for f in sorted(glob.glob(os.path.join(B.T1, "gens", arm, "*.png")))]
    print(f"[band2] {len(tasks)} images over {len(arms)} arms", flush=True); t0 = time.time()
    with Pool(6, initializer=B._init) as pool:
        res = pool.map(B._task, tasks, chunksize=8)
    print(f"[band2] measured in {(time.time() - t0) / 60:.1f} min", flush=True)
    rows = {}
    for (arm, _), s in zip(tasks, res): rows.setdefault(arm, []).append(s)
    rows = {k: np.array(v) for k, v in rows.items()}
    np.savez_compressed(os.path.join(B.T1, "band2_rows.npz"), **rows)
    never = np.array([rows[a].mean(0) for a in B.NEVER if a in rows]); off = never.mean(0); off_se = never.std(0, ddof=1) / np.sqrt(len(never))
    mat = json.load(open(os.path.join(B.T1, "band_materialise.json")))
    out = {"offset": off.tolist(), "offset_se": off_se.tolist(), "bands": {}}
    for b in BANDS:
        R = mat[f"band{b}"]["R_b"]; per = []
        for s in range(3):
            a = f"band{b}_s{s}"
            if a not in rows: continue
            x = rows[a][:, b]; C, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
            per.append({"arm": a, "C": C, "C_se": se, "T_pct": 100 * (C - off[b]) / R, "T_image_se_pct": 100 * float(np.sqrt(se ** 2 + off_se[b] ** 2)) / R})
        T = np.array([p["T_pct"] for p in per]); rec = {"R_b": R, "n_adapters": len(per), "adapters": per, "T_mean_pct": float(T.mean())}
        if len(T) >= 2:
            se_ad = float(np.sqrt((T.std(ddof=1) / np.sqrt(len(T))) ** 2 + (100 * off_se[b] / R) ** 2))
            rec.update(T_se_adapter_pct=se_ad, T_upper99_pct=float(T.mean() + stats.t.ppf(0.99, len(T) - 1) * se_ad))
        out["bands"][f"band{b}"] = rec; print(f"band{b}", {k: v for k, v in rec.items() if k != "adapters"}, flush=True)
    json.dump(out, open(os.path.join(B.T1, "band2_summary.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
