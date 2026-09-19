"""C7 (RESULTS.md Entry 58) — Noiseprint arm, run in the TF 1.15 env (conda env a4tf1):
    <conda>/envs/a4tf1/python.exe src/c7_noiseprint_helper.py [N]

Rebuilds exactly the images of src/c7_transplant.py (same residual stack, read from the same scratch
directory — C7_SCRATCH, else EINV_TMP, else <EINV_V2>/tmp, as c7_transplant.py writes it; same pairing
i -> i % 40, same arithmetic Z_s = round(clip(Z + s*N, 0, 255))) and scores each with Noiseprint as
src/t2_noiseprint.py does: luminance / 255 -> a4_noiseprint.extract(., qf=101 for PNG) -> zero-mean ->
zero-lag NCC against the study's body fingerprints out/np_fingerprint_{A,B}.npy (mean Noiseprint over E2).
The image checksum is logged so c7_transplant.py can verify both processes scored identical images.
TF session limited to 2 intra-op threads, CPU only. Rows: logs/c7_noiseprint_rows.log (resumable).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # "" deletes the variable on Windows; -1 hides every GPU
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[_v] = "2"
import sys, glob, csv, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
sys.path.insert(0, EINV.SRC)
import a4_noiseprint as A
A._np_mod.configSess.intra_op_parallelism_threads = 2
A._np_mod.configSess.inter_op_parallelism_threads = 1
A._np_mod.configSess.device_count["GPU"] = 0

V2 = EINV.V2; GENS = os.path.join(V2, "out", "t1", "gens", "local_base")
ROWS = os.path.join(V2, "logs", "c7_noiseprint_rows.log")
SCR = os.environ.get("C7_SCRATCH", EINV.TMP)    # must match c7_transplant.py, which writes the stack
RES = os.path.join(SCR, "c7_residuals_A_H.npy")
S_GRID = [0.0, 0.1, 0.25, 0.5, 1.0]; N_RES = 40

def transplant(Z, N, s):                       # identical to c7_transplant.transplant
    if s == 0.0: return Z.copy()
    F = Z.astype(np.float32) + np.float32(s) * N
    return np.clip(F, 0, 255).round().astype(np.uint8)

def ncc(a, b):                                  # as t2_noiseprint.ncc
    a = a - a.mean(); b = b - b.mean(); return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    FA = np.load(os.path.join(V2, "out", "np_fingerprint_A.npy")); FB = np.load(os.path.join(V2, "out", "np_fingerprint_B.npy"))
    Nst = np.load(RES, mmap_mode="r")
    files = sorted(glob.glob(os.path.join(GENS, "*.png")))[:n]
    done = set()
    if os.path.exists(ROWS):
        for r in csv.DictReader(open(ROWS)): done.add((int(r["idx"]), float(r["s"])))
    new = not os.path.exists(ROWS); fh = open(ROWS, "a", newline=""); wr = csv.writer(fh)
    if new: wr.writerow(["idx", "image", "res_idx", "s", "np_A", "np_B", "checksum"]); fh.flush()
    t0 = time.time(); k = 0
    for i, f in enumerate(files):
        todo = [s for s in S_GRID if (i, s) not in done]
        if not todo: continue
        Z = np.asarray(Image.open(f).convert("RGB"), np.uint8); N = np.asarray(Nst[i % N_RES], np.float32)
        for s in todo:
            Zs = transplant(Z, N, s); a = Zs.astype(np.float32)
            lum = ((0.299*a[..., 0] + 0.587*a[..., 1] + 0.114*a[..., 2]) / 255.0).astype(np.float32)
            r = A.extract(lum, 101).astype(np.float32); r = r - r.mean()
            wr.writerow([i, os.path.basename(f), i % N_RES, s, ncc(r, FA), ncc(r, FB), int(Zs.sum(dtype=np.int64))])
        fh.flush(); k += 1
        if k % 5 == 0 or k <= 2:
            el = time.time() - t0; print("[c7np] %d images %.1f min (%.1f s/noiseprint)" % (k, el/60, el/(k*len(S_GRID))), flush=True)
    fh.close(); print("[c7np] done %d images in %.1f min" % (k, (time.time()-t0)/60), flush=True)

if __name__ == "__main__":
    main()
