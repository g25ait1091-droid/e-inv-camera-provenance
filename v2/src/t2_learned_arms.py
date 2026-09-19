"""Reusable scorer: the saved learned detector (out/t2_learned_net.pt) applied to A-trained and B-trained
arm folders, with the Entry 17 statistics (t2_learned_ext.py).

Per image (exactly as t2_learned_ext.py): luminance 0.299R + 0.587G + 0.114B of the full PNG, the study's
wavelet residual (fingerprints.wavelet_residual), 16 patches of 256 x 256 at positions drawn from
np.random.default_rng(0) afresh for every image (same positions for every image), score = mean over
patches of logit(A) - logit(B). Nothing is cached (t2_learned.resid would write into out/resid_cache;
this module never imports t2_learned, whose import alone creates that directory); Net and patches are
copied verbatim from t2_learned.py.

Statistics over adapters (mA, mB = per-arm mean scores of the A- and B-trained arms):
  theta_A = mean(mA), theta_B = mean(-mB), theta_sym = (theta_A + theta_B)/2 = (mean mA - mean mB)/2,
  additive part (mean mA + mean mB)/2; adapter-level SE = 0.5 sqrt(var(mA)/nA + var(mB)/nB), Welch df,
  t = theta_sym/SE, one-sided p; exact label permutation of mean_A - mean_B over all C(nA+nB, nA)
  relabellings (one-sided, >= obs - 1e-15); IU sign-flip per arm on additive-corrected contrasts (exact).
  Note: with 3 + 3 adapters the permutation has 20 relabellings, so its smallest attainable p is 0.05.

Usage
  python t2_learned_arms.py --A <dirA1> <dirA2> ... --B <dirB1> ... --n 250 --out <json>
         [--workers 3] [--threads 2] [--block-shuffle] [--content-labels out/t6_weapon_clip.json]
Runs on CPU only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, glob, time, zlib, argparse, itertools, math
import numpy as np

V2 = EINV.V2
NET_PATH = os.path.join(V2, "out", "t2_learned_net.pt")
EXT_JSON = os.path.join(V2, "out", "t2_learned_ext.json")
P, NP, SEED, MEAS, BLK = 256, 16, 0, 1024, 16
SHUF_SEED = 58                                   # Entry 58 (C8); block permutation seed base

_net = None


def _torch():
    import torch
    return torch


def make_net():
    torch = _torch(); nn = torch.nn

    class Net(nn.Module):                        # verbatim from t2_learned.py
        def __init__(s):
            super().__init__()
            c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
            s.f = nn.Sequential(c(1, 16), c(16, 32), c(32, 64), c(64, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2))
        def forward(s, x): return s.f(x)
    return Net()


def init_worker(threads=2, net_path=NET_PATH):
    global _net
    torch = _torch(); torch.set_num_threads(threads)
    sys.path.insert(0, os.path.join(V2, "src"))
    _net = make_net(); _net.load_state_dict(torch.load(net_path, map_location="cpu")); _net.eval()


def lum_of(path):
    from PIL import Image
    if path.lower().endswith(".png"):
        a = np.asarray(Image.open(path).convert("RGB"), np.float32)
        return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
    from fingerprints import load_lum_crop
    return load_lum_crop(path)


def residual(path):
    from fingerprints import wavelet_residual
    return wavelet_residual(lum_of(path))


def patches(W, rng):                             # verbatim from t2_learned.py
    ys = rng.integers(0, MEAS - P, NP); xs = rng.integers(0, MEAS - P, NP)
    return np.stack([W[y:y+P, x:x+P] for y, x in zip(ys, xs)]).astype(np.float32)


def block_shuffle(X, rng):
    """X: (NP, P, P). Each patch is cut into (P/BLK)^2 blocks of BLK x BLK and the blocks are permuted
    by an independent random permutation per patch."""
    n, g = X.shape[0], P // BLK
    B = X.reshape(n, g, BLK, g, BLK).transpose(0, 1, 3, 2, 4).reshape(n, g * g, BLK, BLK)
    out = np.empty_like(B)
    for i in range(n):
        out[i] = B[i, rng.permutation(g * g)]
    return out.reshape(n, g, g, BLK, BLK).transpose(0, 1, 3, 2, 4).reshape(n, P, P)


def shuffle_seed(tag, idx):
    return [SHUF_SEED, zlib.crc32(tag.encode()) & 0x7FFFFFFF, int(idx)]


def _logit_diff(X):
    torch = _torch()
    with torch.no_grad():
        lg = _net(torch.from_numpy(np.ascontiguousarray(X, np.float32))[:, None])
    return float((lg[:, 0] - lg[:, 1]).mean())


def score_task(task):
    """task = (path, tag, idx, variants). variants subset of {'orig', 'orig_fp16', 'shuffle'}.
    'orig'      float32 residual (t2_learned.resid on a cache miss)
    'orig_fp16' residual rounded to float16 (t2_learned.resid on a cache hit)
    'shuffle'   float32 residual, each patch block-shuffled (16 x 16 blocks, seeded per image)"""
    path, tag, idx, variants = task
    W = residual(path)
    out = {}
    if "orig" in variants or "shuffle" in variants:
        X = patches(W, np.random.default_rng(SEED))
        if "orig" in variants: out["orig"] = _logit_diff(X)
        if "shuffle" in variants: out["shuffle"] = _logit_diff(block_shuffle(X, np.random.default_rng(shuffle_seed(tag, idx))))
    if "orig_fp16" in variants:
        out["orig_fp16"] = _logit_diff(patches(W.astype(np.float16), np.random.default_rng(SEED)))
    return out


# ---------------------------------------------------------------- statistics (t2_learned_ext.py)
def signflip_p(c):
    # relative tolerance: an absolute 1e-15 is below float rounding for values of order 10
    c = np.asarray(c, float); n = len(c); obs = c.mean(); tol = 1e-9 * max(1.0, np.abs(c).mean()); cnt = 0
    for signs in itertools.product((1, -1), repeat=n):
        if (c * np.array(signs)).mean() >= obs - tol: cnt += 1
    return cnt / 2 ** n


def label_perm_p(mA, mB, chunk=200000):
    """Exact one-sided label permutation of mean_A - mean_B. The statistic is increasing in the A-sum, so
    relabellings are compared on the A-sum; the observed A-sum is computed by the same operation as every
    relabelled sum (so the observed labelling always counts itself), with a relative tolerance for ties."""
    allv = np.concatenate([mA, mB]).astype(float); nA, n = len(mA), len(mA) + len(mB)
    obs_s = allv[np.arange(nA)[None, :]].sum(1)[0]; tol = 1e-9 * max(1.0, np.abs(allv).sum()); cnt = ncomb = 0
    it = itertools.combinations(range(n), nA)
    while True:
        blk = list(itertools.islice(it, chunk))
        if not blk: break
        sA = allv[np.array(blk)].sum(1)
        cnt += int((sA >= obs_s - tol).sum()); ncomb += len(blk)
    return cnt / ncomb, ncomb


def interaction_stats(mA, mB, R=None):
    from scipy import stats
    mA, mB = np.asarray(mA, float), np.asarray(mB, float); nA, nB = len(mA), len(mB)
    cA, cB = mA, -mB
    thA, thB = cA.mean(), cB.mean(); th = 0.5 * (thA + thB); add = 0.5 * (mA.mean() + mB.mean())
    vA, vB = cA.var(ddof=1) / nA, cB.var(ddof=1) / nB
    se = 0.5 * math.sqrt(vA + vB); df = (vA + vB) ** 2 / (vA ** 2 / (nA - 1) + vB ** 2 / (nB - 1))
    t = th / se; p_t = float(1 - stats.t.cdf(t, df)); UL = th + stats.t.ppf(0.99, df) * se
    p_perm, ncomb = label_perm_p(mA, mB)
    r = {"n_A_adapters": nA, "n_B_adapters": nB, "A_means": mA.tolist(), "B_means": mB.tolist(),
         "theta_A": float(thA), "theta_B": float(thB), "theta_sym": float(th), "additive_part": float(add),
         "adapter_level_SE": float(se), "welch_df": float(df), "t": float(t), "p_t_one_sided": p_t,
         "upper_99_one_sided": float(UL),
         "label_permutation_p_one_sided": p_perm, "label_permutation_combinations": ncomb,
         "label_permutation_min_attainable_p": 1.0 / ncomb,
         "IU_signflip_p_A": signflip_p(cA - add), "IU_signflip_p_B": signflip_p(cB + add)}
    if R:
        r["R_used"] = float(R); r["lambda_sym_pct"] = float(100 * th / R); r["lambda_U_plugin_pct"] = float(100 * UL / R)
    return r


# ---------------------------------------------------------------- driver
def list_files(d, n):
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    if len(files) < n: raise SystemExit(f"{d}: only {len(files)} images, need {n}")
    return files[:n]


def score_arms(arm_dirs, n, variants, workers, threads, log=print):
    """arm_dirs: dict tag -> dir. Returns dict tag -> {variant: np.array of per-image scores}, file lists."""
    import multiprocessing as mp
    tasks, index = [], []
    files_of = {}
    for tag, d in arm_dirs.items():
        fs = list_files(d, n); files_of[tag] = fs
        for i, f in enumerate(fs):
            tasks.append((f, tag, i, tuple(variants))); index.append((tag, i))
    res = {tag: {v: np.full(n, np.nan) for v in variants} for tag in arm_dirs}
    t0 = time.time(); done = 0
    with mp.get_context("spawn").Pool(workers, initializer=init_worker, initargs=(threads,)) as pool:
        for (tag, i), out in zip(index, pool.imap(score_task, tasks, chunksize=4)):
            for v, s in out.items(): res[tag][v][i] = s
            done += 1
            if done % n == 0:
                el = time.time() - t0
                log(f"[arms] {tag} done ({done}/{len(tasks)} images, {el/60:.1f} min, ETA {el/done*(len(tasks)-done)/60:.1f} min)"
                    + "  " + "  ".join(f"{v} {np.nanmean(res[tag][v]):+.4f}" for v in variants))
    return res, files_of


def content_split(scores, files_of, A_tags, B_tags, label_json, log=print):
    lab = json.load(open(label_json))
    by_path = {}
    for r in lab["per_image"]:
        p = lab["folders"][r["folder"]]["path"]
        by_path[os.path.normcase(os.path.join(p, r["file"]))] = (bool(r["firearm"]), float(r["p_firearm"]))
    out = {"label_source": label_json, "arms": {}}
    for tag in A_tags + B_tags:
        keys = [os.path.normcase(f) for f in files_of[tag]]
        if not all(k in by_path for k in keys):
            out["arms"][tag] = {"covered": sum(k in by_path for k in keys), "n": len(keys)}; continue
        fa = np.array([by_path[k][0] for k in keys]); s = scores[tag]
        out["arms"][tag] = {"covered": len(keys), "n": len(keys), "n_firearm": int(fa.sum()),
                            "mean_firearm": float(s[fa].mean()) if fa.any() else None,
                            "mean_nonfirearm": float(s[~fa].mean()) if (~fa).any() else None,
                            "corr_score_pfirearm": float(np.corrcoef(s, [by_path[k][1] for k in keys])[0, 1])}
    ok = all(out["arms"][t].get("mean_firearm") is not None and out["arms"][t].get("mean_nonfirearm") is not None
             for t in A_tags + B_tags)
    out["all_covered_both_strata"] = ok
    if ok:
        for key, name in (("mean_firearm", "firearm_only"), ("mean_nonfirearm", "nonfirearm_only")):
            mA = [out["arms"][t][key] for t in A_tags]; mB = [out["arms"][t][key] for t in B_tags]
            out[name] = {"theta_sym": 0.5 * (np.mean(mA) - np.mean(mB)), "A_means": mA, "B_means": mB}
        nfa = [out["arms"][t]["n_firearm"] for t in A_tags]; nfb = [out["arms"][t]["n_firearm"] for t in B_tags]
        out["firearm_fraction_A"] = float(np.sum(nfa) / sum(out["arms"][t]["n"] for t in A_tags))
        out["firearm_fraction_B"] = float(np.sum(nfb) / sum(out["arms"][t]["n"] for t in B_tags))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", nargs="+", required=True); ap.add_argument("--B", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=250); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--block-shuffle", action="store_true", help="also score with block-shuffled patches")
    ap.add_argument("--content-labels", default=None, help="t6_weapon_clip.json: descriptive firearm split where covered")
    a = ap.parse_args()
    if a.workers > 3: raise SystemExit("at most 3 workers")
    t0 = time.time(); log = lambda m: print(m, flush=True)
    def tag_of(d, pre):
        return os.path.basename(os.path.normpath(d))
    A_tags = [tag_of(d, "A") for d in a.A]; B_tags = [tag_of(d, "B") for d in a.B]
    if len(set(A_tags + B_tags)) != len(A_tags + B_tags): raise SystemExit("arm folder names must be distinct")
    dirs = dict(zip(A_tags, a.A)); dirs.update(zip(B_tags, a.B))
    variants = ["orig"] + (["shuffle"] if a.block_shuffle else [])
    scores, files_of = score_arms(dirs, a.n, variants, a.workers, a.threads, log)
    R = json.load(open(EXT_JSON))["real_paired_contrast_R"]
    res = {"script": os.path.abspath(__file__), "command": " ".join(sys.argv), "model": NET_PATH, "device": "cpu",
           "n_per_arm": a.n, "A_dirs": a.A, "B_dirs": a.B,
           "score": "per image: mean over 16 fixed 256x256 patches (rng seed 0) of logit(A)-logit(B); float32 residual, not cached",
           "R_source": f"{EXT_JSON} real_paired_contrast_R (held-out real H, Entry 17)",
           "arms": {t: {"dir": dirs[t], "role": "A" if t in A_tags else "B",
                        **{f"mean_{v}": float(scores[t][v].mean()) for v in variants},
                        **{f"se_{v}": float(scores[t][v].std(ddof=1) / math.sqrt(a.n)) for v in variants}} for t in dirs},
           "per_image": {t: {v: [round(float(x), 5) for x in scores[t][v]] for v in variants} for t in dirs}}
    for v in variants:
        res[f"stats_{v}"] = interaction_stats([scores[t][v].mean() for t in A_tags], [scores[t][v].mean() for t in B_tags], R)
    if a.content_labels:
        res["content_split_orig"] = content_split({t: scores[t]["orig"] for t in dirs}, files_of, A_tags, B_tags, a.content_labels, log)
    res["runtime_min"] = (time.time() - t0) / 60
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for v in variants:
        s = res[f"stats_{v}"]
        log(f"[arms] {v}: theta_sym {s['theta_sym']:+.4f}  SE {s['adapter_level_SE']:.4f}  t {s['t']:.3f} (df {s['welch_df']:.1f})"
            f"  p_t {s['p_t_one_sided']:.4g}  perm p {s['label_permutation_p_one_sided']:.4g} (of {s['label_permutation_combinations']})")
    log(f"[arms] written {a.out} ({res['runtime_min']:.1f} min)")


if __name__ == "__main__":
    main()
