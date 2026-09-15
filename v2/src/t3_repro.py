"""Tier 3 — reproduction check: re-measure the fetched Kodak / P20 generations locally with the
v1 core against the archive's own E2 fingerprints and compare, row by row, with the archived
per-row CSVs (c3_measure_raw.csv, d5_measure.csv). Reports max |delta rho| and correlation.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, glob, json, numpy as np, pandas as pd
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, EINV.SRC)
V2 = EINV.V2; OUT = os.path.join(V2, "out", "t3_repro.json")

_G = {}
def _init(group):
    import torch; torch.set_num_threads(1)
    from fingerprints import wavelet_residual
    if group == "kodak":
        K = {d: np.load(os.path.join(V2, "out", "fp_kodak", f"K_{d}_E2.npy")) for d in ("D0","D1","D2","D3","D4")}
    else:
        K = {d: np.load(os.path.join(V2, "out", "fp_p20", f"KR_{d}_E2.npy")) for d in ("1101","1102","1103","1104","1105")}
    _G.update(wr=wavelet_residual, K=K)

def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

def _m(task):
    tag, path = task
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    Y = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(np.float32); W = _G["wr"](Y)
    return [(tag, int(os.path.basename(path)[:5]), d, _ncc(W, Y*K)) for d, K in _G["K"].items()]

def run(group, gens_dir, archive_csv, rho_col):
    tasks = [(os.path.basename(os.path.dirname(f)), f) for f in sorted(glob.glob(os.path.join(gens_dir, "*", "*.png")))]
    rows = []
    with Pool(6, initializer=_init, initargs=(group,)) as pool:
        for r in pool.imap_unordered(_m, tasks, chunksize=4): rows += r
    loc = pd.DataFrame(rows, columns=["tag", "gen_idx", "K", "rho_local"])
    arc = pd.read_csv(archive_csv, dtype={"tag": str, "K": str}).rename(columns={rho_col: "rho_archive"})
    m = loc.merge(arc[["tag", "gen_idx", "K", "rho_archive"]], on=["tag", "gen_idx", "K"])
    dlt = (m.rho_local - m.rho_archive).abs()
    return {"n_rows": int(len(m)), "n_images": int(len(tasks)), "max_abs_delta": float(dlt.max()), "median_abs_delta": float(dlt.median()),
            "pearson": float(np.corrcoef(m.rho_local, m.rho_archive)[0, 1]), "archive_rho_sd": float(m.rho_archive.std())}

if __name__ == "__main__":
    R = {}
    if glob.glob(os.path.join(V2, "data", "gens_kodak", "*", "*.png")):
        R["kodak"] = run("kodak", os.path.join(V2, "data", "gens_kodak"), os.path.join(V2, "data", "csv_kodak", "c3_measure_raw.csv"), "rho")
        print("kodak:", R["kodak"], flush=True)
    if glob.glob(os.path.join(V2, "data", "gens_p20", "*", "*.png")):
        R["p20"] = run("p20", os.path.join(V2, "data", "gens_p20"), os.path.join(V2, "data", "csv_p20", "d5_measure.csv"), "rho_K")
        print("p20:", R["p20"], flush=True)
    json.dump(R, open(OUT, "w"), indent=1); print("written", OUT)
