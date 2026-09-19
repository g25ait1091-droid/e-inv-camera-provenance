"""Tier 1 — measurement and statistics, exactly as registered in RESULTS.md Entry 01.

For every image in every mark arm (out/t1/gens/<arm>) and in the never-injected arms
(data/gens/{base, A_raw_s0_r16, B_raw_s0_r16}): rho(gen -> M_rand), rho(gen -> M_lowmid),
rho(gen -> M'), rho against 30 circular rolls of M_rand (decoys), and rho against K_A_E2 / K_B_E2.
Statistics: per-arm mean contrast rho(M) - rho(M'); decoy rank of the true M; cluster-level t
over the three alpha=3 seeds; lambda_mark = contrast / R_mark (R_mark from materialise.json);
natural paired contrast rho(K_A) - rho(K_B). Outputs out/t1/measure_rows.csv, out/t1/summary.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json, glob, csv, time
import numpy as np
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, EINV.SRC)

V2 = EINV.V2; T1 = os.path.join(V2, "out", "t1"); FP = os.path.join(V2, "out", "fp")
ROWS = os.path.join(T1, "measure_rows.csv"); SUMM = os.path.join(T1, "summary.json")
MARK_ARMS = ["mark_rand_a1_s0", "mark_rand_a3_s0", "mark_rand_a3_s1", "mark_rand_a3_s2", "mark_rand_a12_s0", "mark_lowmid_a12_s0"]
NEVER = ["base", "A_raw_s0_r16", "B_raw_s0_r16"]
N_DECOY = 30; DECOY_SEED = 20260910; WORKERS = int(os.environ.get("T1_W", "6"))
if os.environ.get("T1_ARMSET") == "nomark":      # F4 control arms (Entry 20): same measurement, separate files
    MARK_ARMS = ["nomark_s0", "nomark_s1", "nomark_s2"]; NEVER = []
    ROWS = os.path.join(T1, "measure_rows_nomark.csv"); SUMM = os.path.join(T1, "summary_nomark.json")
if os.environ.get("T1_ARMSET") == "rcrop":       # E3 random-crop arms (Entry 27)
    MARK_ARMS = ["rcrop_s0", "rcrop_s1", "rcrop_s2"]; NEVER = []
    ROWS = os.path.join(T1, "measure_rows_rcrop.csv"); SUMM = os.path.join(T1, "summary_rcrop.json")
if os.environ.get("T1_ARMSET") == "dose8k":      # Entry 39 dose arms
    MARK_ARMS = ["dose8k_A_s0", "dose8k_A_s1", "dose8k_B_s0", "dose8k_B_s1"]; NEVER = []
    ROWS = os.path.join(T1, "measure_rows_dose8k.csv"); SUMM = os.path.join(T1, "summary_dose8k.json")
if os.environ.get("T1_ARMSET") == "dose16k":
    MARK_ARMS = ["dose16k_A_s0", "dose16k_B_s0"]; NEVER = []
    ROWS = os.path.join(T1, "measure_rows_dose16k.csv"); SUMM = os.path.join(T1, "summary_dose16k.json")
if os.environ.get("T1_ARMSET") == "dose16krep":  # Entry 48 F9 replication arms
    MARK_ARMS = ["dose16k_A_s1", "dose16k_B_s1", "dose16k_A_s2", "dose16k_B_s2"]; NEVER = []
    ROWS = os.path.join(T1, "measure_rows_dose16krep.csv"); SUMM = os.path.join(T1, "summary_dose16krep.json")
if os.environ.get("T1_ARMSET") == "dose16krep2":  # Entry 55 second 16000-step replication (seeds 3-5)
    MARK_ARMS = [f"dose16k_{b}_s{i}" for i in (3, 4, 5) for b in ("A", "B")]; NEVER = []
    ROWS = os.path.join(T1, "measure_rows_dose16krep2.csv"); SUMM = os.path.join(T1, "summary_dose16krep2.json")
if os.environ.get("T1_ARMSET") == "cm":          # Entry 59 E3 content-matched arms
    MARK_ARMS = [f"cm_{b}_s{s}" for s in range(3) for b in ("A", "B")]; NEVER = []
    ROWS = os.path.join(T1, "measure_rows_cm.csv"); SUMM = os.path.join(T1, "summary_cm.json")
if os.environ.get("T1_ARMSET") == "nomarkB":     # F7 unmarked B-body arms (Entry 35)
    MARK_ARMS = ["nomarkB_s0", "nomarkB_s1", "nomarkB_s2"]; NEVER = []
    ROWS = os.path.join(T1, "measure_rows_nomarkB.csv"); SUMM = os.path.join(T1, "summary_nomarkB.json")
if os.environ.get("T1_ARMSET") == "colab":       # F6 decode-only replication (Entry 31)
    MARK_ARMS = ["colab_s0", "colab_s1", "colab_s2"]; NEVER = []
    ROWS = os.path.join(T1, "measure_rows_colab.csv"); SUMM = os.path.join(T1, "summary_colab.json")
if os.environ.get("T1_ARMSET") == "f5":          # F5 local-environment arms (Entry 25)
    MARK_ARMS = ["local_base", "local_A_raw_s0", "local_B_raw_s0"]; NEVER = []
    ROWS = os.path.join(T1, "measure_rows_f5.csv"); SUMM = os.path.join(T1, "summary_f5.json")

_G = {}
def _init():
    import torch; torch.set_num_threads(1)
    from fingerprints import wavelet_residual, MEAS
    rng = np.random.default_rng(DECOY_SEED)
    M = np.load(os.path.join(T1, "fields", "M_rand.npy"))
    shifts = [(int(rng.integers(64, MEAS-64)), int(rng.integers(64, MEAS-64))) for _ in range(N_DECOY)]
    _G.update(wr=wavelet_residual, MEAS=MEAS, M=M, ML=np.load(os.path.join(T1, "fields", "M_lowmid.npy")),
              Mp=np.load(os.path.join(T1, "fields", "M_prime.npy")),
              KA=np.load(os.path.join(FP, "K_A_E2.npy")), KB=np.load(os.path.join(FP, "K_B_E2.npy")),
              decoys=[np.roll(np.roll(M, dr, 0), dc, 1) for dr, dc in shifts], shifts=shifts)

def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

def _measure(task):
    arm, path = task; g = _G
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    if a.shape[0] != g["MEAS"]: return None
    Y = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(np.float32); W = g["wr"](Y)
    row = [arm, os.path.basename(path), _ncc(W, Y*g["M"]), _ncc(W, Y*g["ML"]), _ncc(W, Y*g["Mp"]), _ncc(W, Y*g["KA"]), _ncc(W, Y*g["KB"])]
    row += [_ncc(W, Y*D) for D in g["decoys"]]
    return row

def main():
    hdr = ["arm", "image", "rho_M", "rho_ML", "rho_Mp", "rho_KA", "rho_KB"] + [f"decoy{i}" for i in range(N_DECOY)]
    done = set()
    if os.path.exists(ROWS):
        for r in csv.DictReader(open(ROWS)): done.add((r["arm"], r["image"]))
    tasks = []
    for arm in MARK_ARMS:
        for f in sorted(glob.glob(os.path.join(T1, "gens", arm, "*.png"))):
            if (arm, os.path.basename(f)) not in done: tasks.append((arm, f))
    for arm in NEVER:
        for f in sorted(glob.glob(os.path.join(V2, "data", "gens", arm, "*.png"))):
            if (arm, os.path.basename(f)) not in done: tasks.append((arm, f))
    print(f"[t1m] {len(done)} present, {len(tasks)} to measure", flush=True)
    new = not os.path.exists(ROWS); fh = open(ROWS, "a", newline=""); wr = csv.writer(fh)
    if new: wr.writerow(hdr)
    t0 = time.time(); n = 0
    with Pool(WORKERS, initializer=_init) as pool:
        for row in pool.imap_unordered(_measure, tasks, chunksize=4):
            if row is None: continue
            wr.writerow(row); n += 1
            if n % 200 == 0: fh.flush(); print(f"[t1m] {n}/{len(tasks)} {(time.time()-t0)/60:.1f} min", flush=True)
    fh.close()
    # ---- statistics ----
    import pandas as pd
    from scipy import stats as sps
    d = pd.read_csv(ROWS); mat = json.load(open(os.path.join(T1, "train_png", "materialise.json")))
    S = {"arms": {}}
    for arm in MARK_ARMS + NEVER:
        g = d[d.arm == arm]
        if not len(g): continue
        contrast = g.rho_M - g.rho_Mp; contrast_L = g.rho_ML - g.rho_Mp
        dec = g[[f"decoy{i}" for i in range(N_DECOY)]].mean().values
        rank = int(1 + (dec > g.rho_M.mean()).sum())            # 1 = true M beats every decoy
        e = {"n": int(len(g)), "rho_M": float(g.rho_M.mean()), "rho_ML": float(g.rho_ML.mean()), "rho_Mp": float(g.rho_Mp.mean()),
             "contrast_M": float(contrast.mean()), "contrast_M_se": float(contrast.std(ddof=1)/np.sqrt(len(g))),
             "contrast_ML": float(contrast_L.mean()), "contrast_ML_se": float(contrast_L.std(ddof=1)/np.sqrt(len(g))),
             "decoy_rank_of_true_M": rank, "decoy_mean": float(dec.mean()), "decoy_max": float(dec.max()),
             "natural_paired_KA_minus_KB": float((g.rho_KA - g.rho_KB).mean())}
        key = {"mark_rand_a1_s0": "rand_a1", "mark_rand_a3_s0": "rand_a3", "mark_rand_a3_s1": "rand_a3", "mark_rand_a3_s2": "rand_a3",
               "mark_rand_a12_s0": "rand_a12", "mark_lowmid_a12_s0": "lowmid_a12"}.get(arm)
        if key:
            R = mat[key]["R_mark"]; e["R_mark"] = R
            c = e["contrast_ML"] if key.startswith("lowmid") else e["contrast_M"]
            e["lambda_mark_pct"] = 100 * c / R if R > 1e-3 else None
        S["arms"][arm] = e
    # cluster-level at alpha = 3 (three seeds)
    a3 = [S["arms"][t]["contrast_M"] for t in ("mark_rand_a3_s0", "mark_rand_a3_s1", "mark_rand_a3_s2") if t in S["arms"]]
    if len(a3) == 3:
        m, sd = float(np.mean(a3)), float(np.std(a3, ddof=1)); t = m / (sd/np.sqrt(3)); tcrit = float(sps.t.ppf(0.995, 2))
        R = mat["rand_a3"]["R_mark"]
        S["alpha3_cluster"] = {"per_seed": a3, "mean": m, "sd": sd, "t": t, "t_crit_0995_df2": tcrit,
                               "one_sided_p": float(1 - sps.t.cdf(t, 2)), "U_plugin": m + tcrit*sd/np.sqrt(3),
                               "lambda_mark_pct": 100*m/R, "lambda_mark_U_pct": 100*(m + tcrit*sd/np.sqrt(3))/R}
    json.dump(S, open(SUMM, "w"), indent=1); print(json.dumps(S, indent=1))

if __name__ == "__main__":
    main()
