"""Pre-specified descriptive analysis (RESULTS.md Entry 116, item 5): the closed-set expectation.

CPU only; about ten minutes on five worker processes. Writes out/fv_closedset_expect.json and, beside it, the
per-replicate margins out/fv_closedset_expect_margins.npz, from which the expectation at any other transfer level
can be read exactly without rerunning. Refuses to overwrite either file (pass --out / --out-npz for a new name).

QUESTION (Entry 116, item 5; review section 4 item 9, hostile-r2-15). What closed-set accuracy would Entry 06's
registered experiment have been expected to show if transfer sat at a stated limit? The experiment: five bodies,
two adapters per body (ten adapters), the first 250 generations of each as one block, the main-effect-corrected
argmax with leave-the-candidate-out main effects. Observed (Entry 07, out/t3_attrib.json): Kodak M1063 3/10,
Huawei P20 with residualised fingerprints 3/10 (and 4/10 with the low/mid representation, which is not in the
pre-specification and is reported here only as a labelled supplement).

DATA AND SCORER (as pre-specified)
  Kodak: raw K rows data/csv_kodak/c3_measure_raw.csv (byte-identical Drive mirror asserted).
  P20:   residualised K rows $EINV_MYDRIVE/inv_channel/E_DAXING/csv/d5_measure.csv, SHA-256 recorded (a byte-identical
         copy is on D: at data/csv_p20/d5_measure.csv, the file t3_attrib.py read; asserted).
  Scorer: t3_attrib.attribute. src/t3_attrib.py is a script whose module body re-reads the CSVs and rewrites
         out/t3_attrib.json, an existing result file, so `import t3_attrib` is not used: the file is parsed and only
         its import statements, its four constants (V2, OUT, G_LIST, N_USE) and its function definitions are
         executed, unchanged, from the same file (SHA-256 of the file and of attribute's source recorded). Nothing is
         re-implemented for the pre-specified numbers. The script first reproduces Entry 07 in full (every G, raw and
         corrected, main effects) against out/t3_attrib.json; the corrected accuracies at G = 250 are 3/10 and 3/10.
  In the bootstrap the scorer's module constant G_LIST is set to [250] (only the G = 250 block enters the
  expectation); on the observed data its G = 250 output is asserted identical to the full-G_LIST run.

PLANTED LEVELS. A shift delta is added to every image's score against its own body's fingerprint (all 500 images
for the group statistic, the first 250 for the closed set). The group's transfer statistic is recomputed exactly as
the file that produced its device-level limit defines it (Kodak: cells/C4_FINAL_cell.py -> E_MULTIDEV/
C4_multidev.json; P20: notebooks/13_daxing_smartphone.ipynb D6 -> E_DAXING/D6_results.json): per adapter
theta = mean over its generations of rho(own K) - mean of rho(the other four K); device means over the two seeds;
grand mean; U = grand mean + t_{0.995,4} sd(device means)/sqrt(5); lambda_U = U / R_real_lower99. The script asserts
that the unplanted rows recompute the files' U and lambda_U and the ledger's 1.74 % and 1.1443 % (1.14 %), and that
planting raises the grand mean (and U) by exactly delta.
  (a) the group's device-level limit: lambda_U (full precision from the group file) x R_real_lower99, i.e. delta = U.
  (b) the primary limits applied to the group's own (point) R_real: nominal 0.15072 % and calibrated 0.17524 %
      (out/h6_calibrated_limit.json), plus the item 1 limit when item 1's rule replaced 0.1752 % (read from
      out/fv_seedbank_calib.json; see D3).
  (c) zero.

EXPECTATION
  Primary (empirical noise; pre-specified): 2,000 two-stage bootstrap replicates. Adapters resampled with
  replacement within each body; images resampled with replacement within each adapter's first 250, independently
  per adapter; the full imported scorer (main effects included) rerun in each replicate at each pre-specified level
  (the sensitivity levels are counted from the margins, asserted equal to the scorer at every scored level).
  Bootstrap chunks are checkpointed to a temporary directory keyed by settings, inputs and script, so an
  interrupted run resumes.
  Beside it (parametric; pre-specified): each adapter's corrected five-score vector drawn independently from
  N(delta e_own, sigma_mu^2 I + Sigma_img/250 + Sigma_b); Sigma_img the pooled within-adapter per-image covariance
  of the five scores (first 250 images, ddof 1, mean of the ten adapters); Sigma_b the covariance of the
  leave-the-candidate-out main-effect vector over the 2,000 primary bootstrap replicates (the pre-specification
  does not fix Sigma_b's estimator; the analytic value is reported beside); sigma_mu = 0 or 4.108e-05 (the D200
  paired-contrast exact 95 % upper limit, out/fv_sigma.json, transported to the five-score vector as sigma_mu^2 I);
  100,000 simulated experiments.
  Reported: E[accuracy] with its Monte Carlo SE, the distribution of correct adapters out of 10, P(X >= 6) (above
  the central 95 % binomial interval of chance; 0.0064 under chance), P(X >= 5) (0.033 under chance) and
  P(X <= observed). No reading attaches.

DEVIATIONS FROM THE WORDING OF ENTRY 116 (each stated in the output):
  D1. P20 level (a) uses R_real_lower99 as the denominator. The pre-specification applies the lower limit to Kodak
      only, but the ledger's P20 1.1443 % is also U / R_real_lower99 (D6_results.json: lambda_U = 0.0114428 =
      U/R_real_lower99; lambda_U_point = 0.0071358 = U/R_real), as is stated in the supplement's Fig. S6 caption;
      the pre-specified assertion that the script "recomputes the ledger's limit" requires it. The literal wording
      (1.1443 % x point R_real, delta = 6.55e-04) is reported beside as a_device_level_limit_x_R_point.
  D2. The scorer is executed from its file by AST extraction rather than `import` (reason above).
  D3. Item 1's limit is read from out/fv_seedbank_calib.json (the name item 1 was run under; Entry 116 names
      out/fv_seedbank.json) and enters level (b) only if that file's rule branch replaced 0.1752 %; --item1-pct
      overrides.

NOT PRE-SPECIFIED, LABELLED AS SUCH IN THE OUTPUT (sensitivities; they change no pre-specified number):
  S1. A seed-paired two-stage bootstrap: the same resampled image indices for every adapter of a replicate. All
      adapters generate from one seed bank (770000 + j), the seed effect is about half of the per-image variance of
      every score here, and it cancels exactly in the corrected score when the bank is shared; independent
      per-adapter image resampling (the pre-specified scheme) adds it back as noise.
  S2. A structural parametric simulation: per-adapter mean score vectors drawn with the seed-removed residual
      covariance, main effects computed from the simulated adapters exactly as the scorer does.
  S3. Proportional planting: delta_d = delta x R_d / R_real (R_d each body's own real-image contrast), which raises
      the group statistic by the same delta.
  S4. Level (b) with the lower-99 % R_real; a lambda grid and the transfer at which E[accuracy] = 0.5 and
      P(X >= 6) = 0.5 / 0.8 / 0.95.
  S5. The P20 low/mid representation (observed 4/10) at its own device-level limit and at zero.
  S6. Lean removed first: the pre-specified planting raises the group statistic by the target on top of the lean
      already in the data (Kodak +1.26e-05, P20 +4.59e-05 on all images); planting delta - lean instead makes the
      own-lean equal the target (lean from the closed-set images and from all images, both reported).
  The imported scorer's counts are asserted equal, replicate by replicate and level by level, to the count of
  adapters whose corrected-argmax margin is below delta; the margins then give every other level exactly.

Run:  python src/fv/fv_closedset_expect.py [--workers 5] [--item1-pct X --out ... --out-npz ...]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import argparse
import ast
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy
from scipy import stats

V2 = EINV.V2
OUT = V2 + "/out"
DRIVE = (EINV.MYDRIVE + "/inv_channel")
REPO = EINV.REPO
SCORER = EINV.SRC + "/t3_attrib.py"
LEDGER = REPO + "/analysis/FINAL_LEDGER.json"
H6 = OUT + "/h6_calibrated_limit.json"
SIGMA = OUT + "/fv_sigma.json"
T3 = OUT + "/t3_attrib.json"
SEEDBANK = OUT + "/fv_seedbank_calib.json"           # item 1's output as written (Entry 116 named fv_seedbank.json)
SEEDBANK_PRESPEC_NAME = OUT + "/fv_seedbank.json"
DST = OUT + "/fv_closedset_expect.json"
DST_NPZ = OUT + "/fv_closedset_expect_margins.npz"

N_USE = 250                                          # first 250 generations per adapter (Entry 06)
N_BOOT = 2000
N_PARAM = 100000
BOOT_SEED = 116005                                   # Entry 116, item 5
PARAM_SEED = 116051
STRUCT_SEED = 116052
N_WORKERS = 5
CHUNK = 50
CKPT_ROOT = os.path.join(tempfile.gettempdir(), "fv_closedset_expect_ckpt")   # resumable bootstrap chunks
CHANCE = 0.2
N_ADAPTERS = 10
LAMBDA_GRID_PCT = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6,
                   0.8, 1.0, 1.25, 1.5, 2.0]

GROUPS = {
    "kodak": {
        "label": "Kodak M1063 (Dresden), five bodies x two seeds, natural-image fingerprint estimates (raw K)",
        "rows": V2 + "/data/csv_kodak/c3_measure_raw.csv",
        "rows_mirror": DRIVE + "/E_MULTIDEV/csv/c3_measure_raw.csv",
        "rho": "rho",
        "devices": ["D0", "D1", "D2", "D3", "D4"],
        "t3_path": ["kodak", "raw_fingerprints"],
        "group_file": DRIVE + "/E_MULTIDEV/C4_multidev.json",
        "group_file_producer": "cells/C4_FINAL_cell.py (repository) -> E_MULTIDEV/C4_multidev.json",
        "real_rows": V2 + "/data/csv_kodak/c0_positive_control.csv",
        "real_rows_mirror": DRIVE + "/E_MULTIDEV/csv/c0_positive_control.csv",
        "real_fp_col": "K_dev",
        "prespecified": True,
    },
    "p20": {
        "label": "Huawei P20 (Daxing), five bodies x two seeds, residualised fingerprint estimates (K)",
        "rows": DRIVE + "/E_DAXING/csv/d5_measure.csv",
        "rows_mirror": V2 + "/data/csv_p20/d5_measure.csv",
        "rho": "rho_K",
        "devices": ["1101", "1102", "1103", "1104", "1105"],
        "t3_path": ["p20", "K_residualised"],
        "group_file": DRIVE + "/E_DAXING/D6_results.json",
        "group_file_producer": "notebooks/13_daxing_smartphone.ipynb cell D6 -> E_DAXING/D6_results.json (reps.K)",
        "real_rows": DRIVE + "/E_DAXING/csv/d1_positive_K.csv",
        "real_rows_mirror": V2 + "/data/csv_p20/d1_positive_K.csv",
        "real_fp_col": "F_dev",
        "prespecified": True,
    },
    "p20_lowmid": {
        "label": "Huawei P20, low/mid representation (L) -- SUPPLEMENTARY, not in the pre-specification",
        "rows": DRIVE + "/E_DAXING/csv/d5_measure.csv",
        "rows_mirror": V2 + "/data/csv_p20/d5_measure.csv",
        "rho": "rho_L",
        "devices": ["1101", "1102", "1103", "1104", "1105"],
        "t3_path": ["p20", "L_lowmid"],
        "group_file": DRIVE + "/E_DAXING/D6_results.json",
        "group_file_producer": "notebooks/13_daxing_smartphone.ipynb cell D6 -> E_DAXING/D6_results.json (reps.L)",
        "real_rows": DRIVE + "/E_DAXING/csv/d1_positive_L.csv",
        "real_rows_mirror": V2 + "/data/csv_p20/d1_positive_L.csv",
        "real_fp_col": "F_dev",
        "prespecified": False,
    },
}


# ============================================================================ helpers
def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def py(o):
    """numpy -> plain Python for json (full precision floats)."""
    if isinstance(o, dict):
        return {str(k): py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [py(v) for v in o]
    if isinstance(o, np.ndarray):
        return py(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def mismatches(a, b, rtol=1e-12, path=""):
    """Recursive comparison of two json-like structures; returns a list of differing paths."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{path}: keys {sorted(set(a) ^ set(b))}")
        for k in set(a) & set(b):
            out += mismatches(a[k], b[k], rtol, f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            out += mismatches(x, y, rtol, f"{path}[{i}]")
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            out.append(f"{path}: {a} vs {b}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (a == b or abs(a - b) <= rtol * max(abs(a), abs(b))):
            out.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


# ============================================================================ the imported scorer
def load_scorer():
    """Execute only the imports, the constants V2/OUT/G_LIST/N_USE and the function definitions of src/t3_attrib.py
    (the module body would rewrite out/t3_attrib.json). Returns the namespace and a record of what was executed."""
    src = open(SCORER, encoding="utf-8").read()
    tree = ast.parse(src)
    keep, kept, skipped = [], [], []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(node); kept.append(f"line {node.lineno}: import")
        elif isinstance(node, ast.FunctionDef):
            keep.append(node); kept.append(f"line {node.lineno}: def {node.name}")
        elif (isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets)
              and {t.id for t in node.targets} <= {"V2", "OUT", "G_LIST", "N_USE"}):
            keep.append(node); kept.append(f"line {node.lineno}: {', '.join(t.id for t in node.targets)} =")
        else:
            skipped.append(node.lineno)
    ns = {"__name__": "t3_attrib_definitions_only", "__file__": SCORER}
    exec(compile(ast.Module(body=keep, type_ignores=[]), SCORER, "exec"), ns)
    fdef = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "attribute")
    fsrc = ast.get_source_segment(src, fdef)
    info = {"file": SCORER, "file_sha256": sha256(SCORER),
            "attribute_source_sha256": hashlib.sha256(fsrc.encode("utf-8")).hexdigest(),
            "attribute_lines": [fdef.lineno, fdef.end_lineno],
            "executed_top_level_statements": kept,
            "not_executed_top_level_statement_lines": skipped,
            "G_LIST_in_file": list(ns["G_LIST"]), "N_USE_in_file": ns["N_USE"],
            "how": ("parsed with ast; only imports, the constants V2/OUT/G_LIST/N_USE and the function definitions "
                    "are executed from the file, unchanged. `import t3_attrib` would run the module body, which "
                    "re-reads the CSVs and rewrites out/t3_attrib.json (an existing result file).")}
    return ns, info


def margins_of(X, own):
    """Main-effect-corrected argmax on one block of all images per adapter, in margin form.
    X: adapters x images x candidates; own: each adapter's body index. b_e is the mean of candidate e's score over
    every image of the adapters not trained on e (the scorer's leave-the-candidate-out main effect). Returns
    margin_a = max over e != own of (m_ae - b_e) minus (m_a,own - b_own): adapter a is attributed to its own body
    at a planted own-score shift delta if and only if margin_a < delta. Also returns b."""
    A, _, E = X.shape
    m = X.mean(axis=1)
    b = np.array([X[own != e, :, e].mean() for e in range(E)])
    c = m - b
    ia = np.arange(A)
    oth = c.copy()
    oth[ia, own] = -np.inf
    return oth.max(axis=1) - c[ia, own], b


def frame(X, tags, devices, index_cache=None):
    """The scorer's wide table: (tag, gen_idx) x candidate."""
    A, J, E = X.shape
    if index_cache is None:
        index_cache = pd.MultiIndex.from_arrays([np.repeat(np.asarray(tags, dtype=object), J),
                                                 np.tile(np.arange(J), A)], names=["tag", "gen_idx"])
    return pd.DataFrame(X.reshape(A * J, E), index=index_cache, columns=list(devices)), index_cache


def draws(gidx, r, members):
    """Replicate r of group gidx: adapters with replacement within each body (two per body), then image indices
    for the independent scheme (one row per drawn adapter), then one shared index vector (seed-paired scheme)."""
    rng = np.random.default_rng([BOOT_SEED, gidx, r])
    ad = np.concatenate([rng.choice(mem, size=len(mem), replace=True) for mem in members])
    ii_ind = rng.integers(0, N_USE, size=(len(ad), N_USE))
    ii_pair = rng.integers(0, N_USE, size=N_USE)
    return ad, ii_ind, ii_pair


# ============================================================================ bootstrap workers
_W = {}


def _init(payload):
    warnings.simplefilter("ignore")
    ns, _ = load_scorer()
    ns["G_LIST"] = [N_USE]
    _W.clear()
    _W.update(payload)
    _W["ns"] = ns


def _boot_chunk(args):
    gname, r0, r1 = args
    P = _W["groups"][gname]
    attribute = _W["ns"]["attribute"]
    X, own, tags, devices, deltas = P["X"], P["own"], P["tags"], P["devices"], P["deltas"]
    A, _, E = X.shape
    members = [np.flatnonzero(own == e) for e in range(E)]
    onehot = np.eye(E)
    res = {k: [] for k in ("ad", "m_ind", "b_ind", "x_ind", "bs_ind", "m_pair", "b_pair", "x_pair", "bs_pair")}
    for r in range(r0, r1):
        ad, ii_ind, ii_pair = draws(P["gidx"], r, members)
        own_s = own[ad]
        tags_s = [f"{tags[a]}#{k}" for k, a in enumerate(ad)]
        dev_of = {t: devices[o] for t, o in zip(tags_s, own_s)}
        res["ad"].append(ad)
        for scheme, Xs in (("ind", X[ad[:, None], ii_ind]), ("pair", X[ad][:, ii_pair])):
            mg, b = margins_of(Xs, own_s)
            xs, bsc, idx = [], None, None
            for d in deltas:
                Xp = Xs + d * onehot[own_s][:, None, :]
                W, idx = frame(Xp, tags_s, devices, idx)
                out = attribute(W, devices, dev_of, use_first=N_USE)
                cg = out["corrected"]["G"][N_USE]
                if cg["n_blocks"] != A:
                    raise RuntimeError(f"{gname} r{r}: {cg['n_blocks']} blocks")
                xs.append(cg["adapters_majority_correct"])
                bv = np.array([out["main_effects_b"][dv] for dv in devices])
                if bsc is None:
                    bsc = bv
                elif not np.array_equal(bv, bsc):
                    raise RuntimeError(f"{gname} r{r}: main effects changed with the planted level")
            res["m_" + scheme].append(mg)
            res["b_" + scheme].append(b)
            res["x_" + scheme].append(xs)
            res["bs_" + scheme].append(bsc)
    return gname, r0, {k: np.array(v) for k, v in res.items()}


# ============================================================================ group statistics
def transfer_statistic(df, rho, devices, delta=0.0, seeds=(0, 1)):
    """The group's transfer statistic as the file that produced its device-level limit defines it (seed-major order
    of all_arms(); every generation of the adapter), with delta added to every own-fingerprint score first."""
    rows = []
    for s in seeds:
        for dv in devices:
            tag = f"{dv}_s{s}"
            g = df[df.tag == tag]
            own = g[g.K == dv].sort_values("gen_idx")[rho].to_numpy(float) + delta
            oth = [g[g.K == o].sort_values("gen_idx")[rho].to_numpy(float) for o in devices if o != dv]
            n = min([len(own)] + [len(x) for x in oth])
            th = own[:n] - np.mean([x[:n] for x in oth], axis=0)
            rows.append({"tag": tag, "dev": dv, "seed": s, "theta": float(th.mean()), "n_images": int(n)})
    R = pd.DataFrame(rows)
    dm = R.groupby("dev")["theta"].mean()
    gm = float(R.theta.mean())
    k = len(dm)
    tcrit = float(stats.t.ppf(1 - 0.01 / 2, df=k - 1))
    U = float(gm + tcrit * dm.std(ddof=1) / np.sqrt(k))
    return {"per_adapter": rows, "device_means": {str(k_): float(v) for k_, v in dm.items()}, "grand_mean": gm,
            "t_crit": tcrit, "U": U, "sd_device_means": float(dm.std(ddof=1)), "n_devices": int(k)}


def real_contrast(df, fp_col, devices):
    """R_real exactly as C4_FINAL_cell.py (Kodak) and D1 (P20) compute it: per body, own minus the mean of the other
    four on its held-out real photographs; mean over bodies; lower 99 % limit R - z_0.99 SE."""
    per = []
    for d in devices:
        g = df[df.img_dev == d]
        own = g[g[fp_col] == d].sort_values("idx")["rho"].to_numpy(float)
        oth = np.mean([g[g[fp_col] == o].sort_values("idx")["rho"].to_numpy(float) for o in devices if o != d],
                      axis=0)
        n = min(len(own), np.size(oth))
        per.append(float((own[:n] - oth[:n]).mean()))
    per = np.array(per)
    R = float(per.mean())
    se = float(per.std(ddof=1) / np.sqrt(len(per)))
    return {"per_device": per.tolist(), "R_real": R, "se": se, "R_real_lower99": float(R - stats.norm.ppf(0.99) * se)}


# ============================================================================ summaries
def summarize(counts, obs, n_adapters=N_ADAPTERS):
    x = np.asarray(counts, dtype=int)
    n = len(x)
    dist = np.bincount(x, minlength=n_adapters + 1)[: n_adapters + 1] / n
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    return {"n": int(n), "E_accuracy": float(x.mean() / n_adapters), "mc_se_E_accuracy": sd / np.sqrt(n) / n_adapters,
            "E_correct": float(x.mean()), "sd_correct": sd,
            "P_X_eq_k": [float(v) for v in dist],
            "P_X_ge_6": float((x >= 6).mean()), "P_X_ge_5": float((x >= 5).mean()),
            "observed_correct": int(obs), "P_X_le_observed": float((x <= obs).mean()),
            "P_X_ge_observed": float((x >= obs).mean()),
            "quantiles_correct_2.5_50_97.5": [float(v) for v in np.percentile(x, [2.5, 50, 97.5])]}


def counts_at(M, delta):
    """Correct adapters per replicate at planted shift delta (scalar, or one value per adapter column)."""
    return (M < np.asarray(delta)).sum(axis=1)


def crossings(M, R_point, lam_nom_pct):
    """Exact transfer levels (as % of the group's point R_real and as multiples of the nominal limit) at which the
    expectation first reaches 0.5 and P(X >= 6) first reaches 0.5, 0.8 and 0.95 (order statistics of the margins)."""
    n_rep, A = M.shape
    out = {}
    flat = np.sort(M.ravel())
    d = float(flat[int(np.ceil(0.5 * flat.size)) - 1])
    out["E_accuracy_0.5"] = d
    s6 = np.sort(M, axis=1)[:, 5]
    for q in (0.5, 0.8, 0.95):
        out[f"P_X_ge_6_{q}"] = float(np.sort(s6)[int(np.ceil(q * n_rep)) - 1])
    return {k: {"delta": v, "lambda_pct_of_R_point": 100 * v / R_point,
                "multiple_of_nominal_limit": 100 * v / R_point / lam_nom_pct,
                "note": "smallest delta at or above which the quantity reaches the target (just above this margin)"}
            for k, v in out.items()}


def grid(M, R_point, obs):
    rows = []
    for lam in LAMBDA_GRID_PCT:
        d = lam / 100 * R_point
        c = counts_at(M, d)
        rows.append({"lambda_pct_of_R_point": lam, "delta": d, "E_accuracy": float(c.mean() / N_ADAPTERS),
                     "P_X_ge_6": float((c >= 6).mean()), "P_X_ge_5": float((c >= 5).mean()),
                     "P_X_le_observed": float((c <= obs).mean())})
    return rows


def parametric_margins(Z, V, own):
    """Z: n x A x E standard normals; V: covariance of each adapter's corrected five-score vector. Independent across
    adapters, mean zero (delta enters through the margin threshold)."""
    Y = Z @ np.linalg.cholesky(V).T
    ia = np.arange(len(own))
    ownv = Y[:, ia, own]
    oth = Y.copy()
    oth[:, ia, own] = -np.inf
    return oth.max(axis=2) - ownv


def structural_margins(Z, V_adapter, own):
    """Per-adapter mean score deviations drawn with covariance V_adapter (seed term removed: it cancels exactly when
    all adapters share one seed bank); main effects computed from the simulated adapters as the scorer does."""
    Y = Z @ np.linalg.cholesky(V_adapter).T                    # n x A x E
    E = Y.shape[2]
    b = np.stack([Y[:, own != e, e].mean(axis=1) for e in range(E)], axis=1)   # n x E
    C = Y - b[:, None, :]
    ia = np.arange(len(own))
    ownv = C[:, ia, own]
    oth = C.copy()
    oth[:, ia, own] = -np.inf
    return oth.max(axis=2) - ownv


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-param", type=int, default=N_PARAM)
    ap.add_argument("--item1-pct", type=float, default=None,
                    help="item 1 bound of record in % (only if item 1's rule replaced 0.1752 %)")
    ap.add_argument("--item1-source", type=str, default=None)
    ap.add_argument("--out", type=str, default=DST)
    ap.add_argument("--out-npz", type=str, default=DST_NPZ)
    ap.add_argument("--ckpt-dir", type=str, default=CKPT_ROOT)
    args = ap.parse_args()
    for p in (args.out, args.out_npz):
        if os.path.exists(p):
            sys.exit(f"[fv_closedset_expect] {p} exists; results are never overwritten (use --out/--out-npz)")
    t0 = time.time()
    warnings.simplefilter("ignore")
    ns, scorer_info = load_scorer()
    ledger, h6, sig, t3 = load(LEDGER), load(H6), load(SIGMA), load(T3)
    res = {"entry": "RESULTS.md Entry 116, item 5 (closed-set expectation; review section 4 item 9, hostile-r2-15)",
           "status": "pre-specified in Entry 116 as descriptive; no reading attaches",
           "script": "src/fv/fv_closedset_expect.py",
           "run": {"date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "python": sys.version.split()[0],
                   "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                   "platform": platform.platform(), "workers": args.workers},
           "scorer": scorer_info}

    # ---------------------------------------------------------------- limits of the primary design
    pa = np.array(ledger["primary"]["per_adapter_A"]); pb = np.array(ledger["primary"]["per_adapter_B"])
    tc11 = stats.t.ppf(0.995, 11)
    U_led = max(pa.mean() + tc11 * pa.std(ddof=1) / np.sqrt(12), pb.mean() + tc11 * pb.std(ddof=1) / np.sqrt(12))
    lam_nom = float(h6["primary_d200"]["lambda_U_nominal_pct"])
    lam_cal = float(h6["primary_d200"]["lambda_U_calibrated_pct"])
    assert abs(100 * U_led / ledger["denominators"]["R_real"] - lam_nom) < 1e-9 * lam_nom, (U_led, lam_nom)
    assert round(lam_nom, 4) == 0.1507 and round(lam_cal, 4) == 0.1752
    sigma_mu_hi = float(sig["primary_crossed_model"]["sigma_u_interval_exact"]["two_sided_95"][1])
    assert abs(sigma_mu_hi - 4.1e-05) < 5e-7
    # item 1: its limit enters level (b) only if its rule replaced 0.1752 % (Entry 116, item 5 (b))
    item1 = {"file": SEEDBANK, "present": os.path.exists(SEEDBANK),
             "file_named_in_entry_116": SEEDBANK_PRESPEC_NAME,
             "present_under_entry_116_name": os.path.exists(SEEDBANK_PRESPEC_NAME)}
    item1_pct = args.item1_pct
    if item1_pct is not None:
        item1.update({"included": True, "pct": item1_pct, "source": args.item1_source or "--item1-pct"})
    elif item1["present"]:
        sb = load(SEEDBANK)
        item1.update({"sha256": sha256(SEEDBANK), "rule": sb["rule"], "summary_reading": sb["summary"]["reading"]})
        if sb["rule"]["branch"] == "changes":
            item1_pct = float(sb["limits"]["bound_of_record"]["lambda_U_pct_of_R_real"])
            assert item1_pct == sb["summary"]["limit_pct_of_R_real"]["bound_of_record"]
            assert abs(sb["limits"]["H6_calibrated_c_1.25"]["lambda_U_pct_of_R_real"] - lam_cal) < 1e-12
            item1.update({"included": True, "pct": item1_pct, "c_star": sb["rule"]["c_star"],
                          "source": "limits.bound_of_record.lambda_U_pct_of_R_real"})
        else:
            item1.update({"included": False, "reason": "item 1's rule kept 0.1752 % as the bound of record"})
    else:
        item1.update({"included": False,
                      "note": ("item 1 had not written its result when this ran; if its rule replaces 0.1752 %, rerun "
                               "with --item1-pct and new --out/--out-npz names, or read the expectation at that level "
                               "exactly from the stored margins (counts = margins < delta)")})
    res["primary_limits"] = {
        "nominal_pct": lam_nom, "calibrated_pct": lam_cal, "source": "out/h6_calibrated_limit.json primary_d200",
        "nominal_recomputed_from_ledger_per_adapter_pct": 100 * U_led / ledger["denominators"]["R_real"],
        "item1": item1}

    # ---------------------------------------------------------------- inputs
    inputs = {}
    for gname, G in GROUPS.items():
        for key in ("rows", "rows_mirror", "real_rows", "real_rows_mirror", "group_file"):
            if G[key] not in inputs:
                inputs[G[key]] = sha256(G[key])
        assert inputs[G["rows"]] == inputs[G["rows_mirror"]], f"{gname}: rows differ from their mirror"
        assert inputs[G["real_rows"]] == inputs[G["real_rows_mirror"]], f"{gname}: real rows differ from mirror"
    for p in (LEDGER, H6, SIGMA, T3, SCORER):
        inputs[p] = sha256(p)
    res["inputs_sha256"] = inputs
    res["input_notes"] = [
        "P20 rows are read from the Drive path named in Entry 116; the pre-specification says they are not on D:, "
        "but a byte-identical copy is at $EINV_V2/data/csv_p20/d5_measure.csv (the file t3_attrib.py read). "
        "Kodak rows are read from D: and asserted byte-identical to the Drive mirror.",
        "sigma_mu for the parametric model: 0 and %.15g (out/fv_sigma.json primary_crossed_model."
        "sigma_u_interval_exact.two_sided_95[1], the D200 paired-contrast exact 95 %% upper limit; transported to the "
        "five-candidate score vector as sigma_mu^2 I, as pre-specified)" % sigma_mu_hi]

    # ---------------------------------------------------------------- per group: reproduction, statistic, levels
    payload = {"groups": {}}
    groups_out = {}
    frames = {}
    for gidx, (gname, G) in enumerate(GROUPS.items()):
        devs = G["devices"]
        df = pd.read_csv(G["rows"], dtype={"tag": str, "K": str})
        frames[gname] = df
        gout = {"label": G["label"], "prespecified": G["prespecified"], "rows_file": G["rows"],
                "score_column": G["rho"], "devices": devs}
        # Entry 07 reproduction with the imported scorer (full G_LIST) against out/t3_attrib.json
        W = ns["wide"](df, rho=G["rho"])
        dev_of = {t: t.split("_")[0] for t in df.tag.unique()}
        ns["G_LIST"] = list(scorer_info["G_LIST_in_file"])
        full = ns["attribute"](W, devs, dev_of)
        stored = t3
        for k in G["t3_path"]:
            stored = stored[k]
        mm = mismatches(json.loads(json.dumps(py(full))), stored)
        assert not mm, f"{gname}: Entry 07 not reproduced: {mm[:5]}"
        ns["G_LIST"] = [N_USE]
        only250 = ns["attribute"](W, devs, dev_of)
        assert not mismatches(json.loads(json.dumps(py(only250["corrected"]["G"][N_USE]))),
                              stored["corrected"]["G"][str(N_USE)])
        obs = int(full["corrected"]["G"][N_USE]["adapters_majority_correct"])
        gout["entry07_reproduction"] = {
            "compared_with": "out/t3_attrib.json " + ".".join(G["t3_path"]),
            "identical": True, "rtol": 1e-12,
            "observed_corrected_correct_G250": obs, "observed_raw_correct_G250":
                int(full["raw"]["G"][N_USE]["adapters_majority_correct"]),
            "main_effects_b": full["main_effects_b"],
            "G250_only_run_identical": True}
        # the closed-set array exactly as the scorer sees it
        W250 = W.groupby(level=0, group_keys=False).apply(lambda g: g.iloc[:N_USE])
        tags = sorted(W250.index.get_level_values(0).unique())
        for t in tags:
            assert list(W250.loc[t].index) == list(range(N_USE)), t
        X = np.stack([W250.loc[t][devs].to_numpy(float) for t in tags])
        own = np.array([devs.index(dev_of[t]) for t in tags])
        assert X.shape == (N_ADAPTERS, N_USE, 5) and np.bincount(own).tolist() == [2] * 5
        mg_obs, b_obs = margins_of(X, own)
        assert int((mg_obs < 0).sum()) == obs
        assert np.allclose(b_obs, [full["main_effects_b"][d] for d in devs], rtol=1e-12, atol=0)

        # the group's transfer statistic and R_real, against the file that produced the device-level limit
        gf = load(G["group_file"])
        if gname == "kodak":
            gv = gf["variants"]["raw"]
            file_vals = {"U": gv["U_device_level"], "lambda_U": gv["lambda_U"], "grand_mean": gv["grand_mean"],
                         "R_real": gf["R_real"], "R_real_lower99": gf["R_real_lower99"],
                         "per_adapter": [r_["theta"] for r_ in gv["per_adapter"]],
                         "per_device_real": gf["per_device_real"]}
        else:
            gv = gf["reps"]["K" if gname == "p20" else "L"]
            file_vals = {"U": gv["U"], "lambda_U": gv["lambda_U"], "lambda_U_point": gv["lambda_U_point"],
                         "grand_mean": gv["grand_mean"], "R_real": gv["R_real"], "R_real_lower99": gv["R_real_lower99"],
                         "per_adapter": [r_["theta"] for r_ in gv["per_adapter"]]}
        st0 = transfer_statistic(df, G["rho"], devs)
        rr = real_contrast(pd.read_csv(G["real_rows"], dtype={"img_dev": str, G["real_fp_col"]: str}),
                           G["real_fp_col"], devs)
        R_pt, R_low = rr["R_real"], rr["R_real_lower99"]
        lamU = st0["U"] / R_low
        close = lambda a, b: abs(a - b) <= 1e-12 * max(abs(a), abs(b))
        assert all(close(a, b) for a, b in zip([r_["theta"] for r_ in st0["per_adapter"]], file_vals["per_adapter"]))
        assert close(st0["grand_mean"], file_vals["grand_mean"]) and close(st0["U"], file_vals["U"])
        assert close(R_pt, file_vals["R_real"]) and close(R_low, file_vals["R_real_lower99"])
        assert close(lamU, file_vals["lambda_U"]), (lamU, file_vals["lambda_U"])
        if gname == "kodak":
            assert all(close(a, b) for a, b in zip(rr["per_device"], file_vals["per_device_real"]))
            led = next(r_ for r_ in ledger["replications"] if r_["name"] == "Kodak M1063, CCD")["lambda_U_pct"]
            assert round(100 * lamU, 2) == led == 1.74
            ledger_check = {"FINAL_LEDGER replications['Kodak M1063, CCD'].lambda_U_pct": led}
        elif gname == "p20":
            led = next(r_ for r_ in ledger["replications"] if r_["name"] == "Huawei P20, smartphone")["lambda_U_pct"]
            led4 = ledger["smartphone"]["lambda_U_pct"]
            assert round(100 * lamU, 2) == led == 1.14 and round(100 * lamU, 4) == led4 == 1.1443
            assert close(st0["U"] / R_pt, file_vals["lambda_U_point"])
            ledger_check = {"FINAL_LEDGER replications['Huawei P20, smartphone'].lambda_U_pct": led,
                            "FINAL_LEDGER smartphone.lambda_U_pct": led4,
                            "U_over_point_R_real_is_lambda_U_point_in_D6": st0["U"] / R_pt}
        else:
            ledger_check = {"note": "no ledger row for the low/mid representation; D6_results.json reps.L reproduced"}
        gout["group_statistic"] = {
            "producer": G["group_file_producer"], "file": G["group_file"],
            "definition": ("per adapter theta = mean over all its generations of rho(own K) - mean of rho(other four K);"
                           " device means over two seeds; grand mean; U = grand mean + t_{0.995,4} sd(device means)/"
                           "sqrt(5); lambda_U = U / R_real_lower99"),
            "recomputed": st0, "R_real": rr, "lambda_U": lamU, "lambda_U_pct": 100 * lamU,
            "lambda_U_point_R_pct": 100 * st0["U"] / R_pt, "lambda_hat_pct": 100 * st0["grand_mean"] / R_pt,
            "reproduces_file": True, "ledger": ledger_check}

        # planted levels
        lv = [("a_device_level_limit", 100 * lamU, "R_real_lower99", R_low, "pre-specified (a)"),
              ("b_nominal", lam_nom, "R_real", R_pt, "pre-specified (b)"),
              ("b_calibrated", lam_cal, "R_real", R_pt, "pre-specified (b)"),
              ("c_zero", 0.0, None, None, "pre-specified (c)"),
              ("a_device_level_limit_x_R_point", 100 * lamU, "R_real", R_pt,
               "sensitivity; for P20 this is the pre-specification's literal wording (see deviation D1)"),
              ("b_nominal_x_R_lower99", lam_nom, "R_real_lower99", R_low, "sensitivity (not pre-specified)"),
              ("b_calibrated_x_R_lower99", lam_cal, "R_real_lower99", R_low, "sensitivity (not pre-specified)")]
        if item1_pct is not None and G["prespecified"]:
            lv.insert(3, ("b_item1_bound_of_record", item1_pct, "R_real", R_pt,
                          "pre-specified (b): the item 1 limit, which replaced 0.1752 %"))
            lv.append(("b_item1_x_R_lower99", item1_pct, "R_real_lower99", R_low, "sensitivity (not pre-specified)"))
        if not G["prespecified"]:
            lv = [x for x in lv if x[0] in ("a_device_level_limit", "c_zero", "a_device_level_limit_x_R_point")]
        levels = {}
        for name, lam, dname, den, status in lv:
            delta = 0.0 if den is None else lam / 100 * den
            st = transfer_statistic(df, G["rho"], devs, delta=delta)
            rise = st["grand_mean"] - st0["grand_mean"]
            assert abs(rise - delta) <= 1e-15, (gname, name, rise, delta)
            assert abs((st["U"] - st0["U"]) - delta) <= 1e-15
            levels[name] = {"lambda_pct": lam, "denominator": dname, "denominator_value": den, "delta": delta,
                            "status": status, "group_statistic_rise": rise, "U_rise": st["U"] - st0["U"],
                            "rise_equals_delta": True,
                            "lambda_U_after_planting_pct": 100 * st["U"] / R_low}
        if "a_device_level_limit" in levels:
            assert abs(levels["a_device_level_limit"]["delta"] - st0["U"]) <= 1e-18
        gout["levels"] = levels

        # observed data (no resampling): margins and the plug-in accuracy at each level with the imported scorer
        idx = None
        tags_arr = np.asarray(tags, dtype=object)
        plug = {}
        for name, L in levels.items():
            Xp = X + L["delta"] * np.eye(5)[own][:, None, :]
            Wp, idx = frame(Xp, tags_arr, devs, idx)
            o = ns["attribute"](Wp, devs, {t: dev_of[t] for t in tags}, use_first=N_USE)
            k = int(o["corrected"]["G"][N_USE]["adapters_majority_correct"])
            assert k == int((mg_obs < L["delta"]).sum())
            plug[name] = k
        gout["observed_data"] = {
            "adapters": tags, "own_index": own.tolist(),
            "margin_delta_needed_to_attribute": dict(zip(tags, mg_obs.tolist())),
            "margin_as_pct_of_R_point": dict(zip(tags, (100 * mg_obs / R_pt).tolist())),
            "plugin_correct_at_level": plug,
            "note": ("margin_a = best other corrected score minus own corrected score on the observed 250 images; an "
                     "adapter is attributed to its own body when the planted shift exceeds its margin. No resampling.")}

        # payload for the bootstrap. The imported scorer is rerun in every replicate at the pre-specified levels
        # ((a), (b), (c); for the supplementary low/mid group its (a) and (c)); the sensitivity levels are counted
        # from the margins, which are asserted equal to the scorer at every scored level and replicate.
        scored = [n for n in levels if levels[n]["status"].startswith("pre-specified")] if G["prespecified"] \
            else ["a_device_level_limit", "c_zero"]
        for n in levels:
            levels[n]["bootstrap_counts_from"] = ("imported scorer, every replicate" if n in scored else
                                                  "margins (exact; equal to the imported scorer at every scored level)")
        payload["groups"][gname] = {"gidx": gidx, "X": X, "own": own, "tags": tags, "devices": devs,
                                    "deltas": [levels[n]["delta"] for n in scored], "scored_levels": scored,
                                    "level_names": list(levels)}
        gout["_R"] = (R_pt, R_low, obs)
        groups_out[gname] = gout

    # ---------------------------------------------------------------- bootstrap (both schemes), parallel, resumable
    key_src = json.dumps({"n_boot": args.n_boot, "chunk": CHUNK, "seed": BOOT_SEED, "script": sha256(__file__),
                          "scorer": scorer_info["file_sha256"],
                          "groups": {g: {"X": hashlib.sha256(P_["X"].tobytes()).hexdigest(), "deltas": P_["deltas"],
                                         "scored": P_["scored_levels"]} for g, P_ in payload["groups"].items()}},
                         sort_keys=True)
    ckdir = os.path.join(args.ckpt_dir, hashlib.sha256(key_src.encode()).hexdigest()[:16])
    os.makedirs(ckdir, exist_ok=True)
    tasks = [(g, r0, min(r0 + CHUNK, args.n_boot)) for g in GROUPS for r0 in range(0, args.n_boot, CHUNK)]
    ck = lambda g, r0, r1: os.path.join(ckdir, f"{g}_{r0:05d}_{r1:05d}.npz")
    parts = {g: [] for g in GROUPS}
    todo, n_loaded = [], 0
    for g, r0, r1 in tasks:
        if os.path.exists(ck(g, r0, r1)):
            with np.load(ck(g, r0, r1)) as z:
                parts[g].append((r0, {k: z[k] for k in z.files}))
            n_loaded += 1
        else:
            todo.append((g, r0, r1))
    print(f"[fv_closedset_expect] bootstrap: {len(tasks)} chunks, {n_loaded} from checkpoint {ckdir}", flush=True)
    tb = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(payload,)) as ex:
            futs = [ex.submit(_boot_chunk, t) for t in todo]
            for i, f in enumerate(as_completed(futs)):
                g, r0, part = f.result()
                r1 = r0 + len(part["ad"])
                tmp = ck(g, r0, r1)[:-4] + ".tmp.npz"
                np.savez(tmp, **part)
                os.replace(tmp, ck(g, r0, r1))
                parts[g].append((r0, part))
                if (i + 1) % 10 == 0 or i + 1 == len(todo):
                    print(f"[fv_closedset_expect] bootstrap {i + 1}/{len(todo)} chunks ({time.time() - tb:.0f} s)",
                          flush=True)
    res["checkpoint"] = {"dir": ckdir, "chunks": len(tasks), "chunks_loaded_from_checkpoint": n_loaded,
                         "chunk_size": CHUNK, "note": "temporary working files, keyed by settings, inputs and script"}
    boot = {}
    for g in GROUPS:
        parts[g].sort(key=lambda x: x[0])
        boot[g] = {k: np.concatenate([p[k] for _, p in parts[g]]) for k in parts[g][0][1]}
        assert boot[g]["ad"].shape == (args.n_boot, N_ADAPTERS)

    # ---------------------------------------------------------------- per group: summaries
    npz = {}
    chance = stats.binom(N_ADAPTERS, CHANCE)
    for gidx, gname in enumerate(GROUPS):
        gout = groups_out[gname]
        P = payload["groups"][gname]
        R_pt, R_low, obs = gout.pop("_R")
        levels = gout["levels"]
        lnames = P["level_names"]
        scored = P["scored_levels"]
        B = boot[gname]
        counts = {}
        for scheme in ("ind", "pair"):
            # the imported scorer's counts equal the margin counts, replicate by replicate and level by level
            for j, n in enumerate(scored):
                assert np.array_equal(B["x_" + scheme][:, j], counts_at(B["m_" + scheme], levels[n]["delta"])), \
                    (gname, scheme, n)
            assert np.allclose(B["b_" + scheme], B["bs_" + scheme], rtol=1e-12, atol=1e-20)
            counts[scheme] = {n: (B["x_" + scheme][:, scored.index(n)] if n in scored else
                                  counts_at(B["m_" + scheme], levels[n]["delta"])) for n in lnames}
        own = P["own"]
        X = P["X"]
        wts = np.array(gout["group_statistic"]["R_real"]["per_device"])[own] / R_pt   # proportional planting weights
        assert (wts > 0).all()
        # ------------- bootstrap summaries
        bs = {"method": ("two-stage bootstrap: adapters with replacement within each body, images with replacement "
                         "within each adapter's first 250 independently per adapter; the imported scorer (main "
                         "effects included) rerun in every replicate at the pre-specified levels"),
              "status": "pre-specified (primary)", "n_replicates": args.n_boot, "seed": [BOOT_SEED, gidx, "r"],
              "levels": {n: summarize(counts["ind"][n], obs) for n in lnames}}
        sp = {"method": ("seed-paired two-stage bootstrap: the same adapter draws; one resampled image-index vector "
                         "shared by every adapter of the replicate (all adapters generate from one seed bank, whose "
                         "effect cancels in the corrected score); imported scorer rerun at the pre-specified levels"),
              "status": "sensitivity S1 (not pre-specified)", "n_replicates": args.n_boot,
              "levels": {n: summarize(counts["pair"][n], obs) for n in lnames}}
        # ------------- parametric (pre-specified) and structural (sensitivity)
        S_img = np.mean([np.cov(X[a].T, ddof=1) for a in range(N_ADAPTERS)], axis=0)
        S_b = np.cov(B["b_ind"].T, ddof=1)
        V0 = S_img / N_USE
        n_not = lambda e, f: int(((own != e) & (own != f)).sum())
        S_b_analytic = np.array([[n_not(e, f) / 64 * V0[e, f] for f in range(5)] for e in range(5)])
        gm_ = X.mean(axis=(0, 1)); am = X.mean(axis=1); jm = X.mean(axis=0)
        Rres = X - am[:, None, :] - jm[None, :, :] + gm_
        S_r = np.einsum("aje,ajf->ef", Rres, Rres) / ((N_ADAPTERS - 1) * (N_USE - 1))
        seed_share = [float(1 - S_r[e, e] / S_img[e, e]) for e in range(5)]
        rng = np.random.default_rng([PARAM_SEED, gidx])
        Z = rng.standard_normal((args.n_param, N_ADAPTERS, 5))
        rng2 = np.random.default_rng([STRUCT_SEED, gidx])
        Z2 = rng2.standard_normal((args.n_param, N_ADAPTERS, 5))
        par, struc, par_M, struc_M = {}, {}, {}, {}
        for smu_name, smu in (("sigma_mu_0", 0.0), ("sigma_mu_%.3e" % sigma_mu_hi, sigma_mu_hi)):
            V = smu ** 2 * np.eye(5) + V0 + S_b
            Mp = parametric_margins(Z, V, own)
            par_M[smu_name] = Mp
            par[smu_name] = {"sigma_mu": smu,
                             "levels": {n: summarize(counts_at(Mp, levels[n]["delta"]), obs) for n in lnames}}
            Ms = structural_margins(Z2, smu ** 2 * np.eye(5) + S_r / N_USE, own)
            struc_M[smu_name] = Ms
            struc[smu_name] = {"sigma_mu": smu,
                               "levels": {n: summarize(counts_at(Ms, levels[n]["delta"]), obs) for n in lnames}}
        pm = {"method": ("each adapter's corrected five-score vector drawn independently from N(delta e_own, "
                         "sigma_mu^2 I + Sigma_img/250 + Sigma_b); argmax; 10 adapters (two per body) per experiment"),
              "status": "pre-specified (beside the bootstrap)", "n_experiments": args.n_param,
              "seed": [PARAM_SEED, gidx],
              "inputs": {"Sigma_img": S_img, "Sigma_img_over_250": V0, "Sigma_b": S_b,
                         "Sigma_b_estimator": ("covariance over the primary bootstrap replicates of the scorer's "
                                               "leave-the-candidate-out main-effect vector (the pre-specification does "
                                               "not fix the estimator)"),
                         "Sigma_b_analytic_independent_adapters": S_b_analytic,
                         "sd_img_per_score": np.sqrt(np.diag(S_img)),
                         "corr_img": S_img / np.sqrt(np.outer(np.diag(S_img), np.diag(S_img)))},
              "by_sigma_mu": par}
        st = {"method": ("per-adapter mean score vectors m_a ~ N(delta e_own, sigma_mu^2 I + Sigma_resid/250), "
                         "Sigma_resid the adapter-by-seed residual covariance of the ten adapters' first 250 images "
                         "(seed main effect removed: it is common to every adapter and cancels in the corrected score); "
                         "main effects b_e computed from the simulated adapters not trained on e, as the scorer does"),
              "status": "sensitivity S2 (not pre-specified)", "n_experiments": args.n_param,
              "seed": [STRUCT_SEED, gidx],
              "inputs": {"Sigma_resid": S_r, "sd_resid_per_score": np.sqrt(np.diag(S_r)),
                         "seed_share_of_per_image_variance": seed_share},
              "by_sigma_mu": struc}
        # ------------- noise-structure diagnostics (not pre-specified): which yardstick fits these data
        noise_var = np.diag(S_r) / N_USE                        # variance of one adapter's 250-image mean, seed removed
        pairs = np.array([am[own == e][0] - am[own == e][1] for e in range(5)])   # body x candidate, seed 0 - seed 1
        s2u = (pairs ** 2).mean(axis=0) / 2 - noise_var
        T = np.array([am[own == e].mean(axis=0) for e in range(5)])               # body x candidate means
        off = [(d_, e_) for d_ in range(5) for e_ in range(5) if d_ != e_]
        D = np.zeros((len(off), 10))
        for i_, (d_, e_) in enumerate(off):
            D[i_, d_] = 1.0
            D[i_, 5 + e_] = 1.0
        y_ = np.array([T[d_, e_] for d_, e_ in off])
        coef, *_ = np.linalg.lstsq(D, y_, rcond=None)
        rss = float(((y_ - D @ coef) ** 2).sum())
        df_off = len(off) - np.linalg.matrix_rank(D)
        cellvar = (max(float(s2u.mean()), 0.0) + float(noise_var.mean())) / 2
        diagn = {"status": "descriptive diagnostics (not pre-specified)",
                 "adapter_mean_noise_sd_by_candidate": np.sqrt(noise_var),
                 "within_body_pair_variance_over_noise_variance_by_candidate": (pairs ** 2).mean(axis=0) / 2 / noise_var,
                 "adapter_persistent_variance_by_candidate_mom": s2u,
                 "adapter_persistent_variance_pooled_mom": float(s2u.mean()),
                 "body_x_candidate_interaction_off_diagonal": {
                     "residual_mean_square": rss / df_off, "df": int(df_off),
                     "expected_from_image_noise_and_adapter_term": cellvar,
                     "ratio": rss / df_off / cellvar},
                 "note": ("adapter term: half the mean squared difference of the two seeds' 250-image means of one body "
                          "on one candidate, minus the image-noise variance (seed effect removed). Interaction: body and "
                          "candidate effects fitted to the 20 off-diagonal cells of the body x candidate table of means "
                          "(own cells excluded); a ratio above 1 is persistent body-by-candidate structure, which the "
                          "parametric models leave out and the bootstrap keeps as observed.")}
        # ------------- proportional planting (sensitivity S3)
        prop = {"status": "sensitivity S3 (not pre-specified)",
                "method": ("delta_d = delta x R_d / R_real for body d (R_d its own real-image contrast), which raises "
                           "the group statistic by the same delta; evaluated through the margins"),
                "weights_by_adapter": dict(zip(P["tags"], wts.tolist())), "levels": {}}
        for n in lnames:
            d = levels[n]["delta"] * wts                     # original adapter order (observed data)
            d_boot = levels[n]["delta"] * wts[B["ad"]]       # each replicate's drawn adapters
            prop["levels"][n] = {
                "bootstrap_two_stage": summarize(counts_at(B["m_ind"], d_boot), obs),
                "bootstrap_seed_paired": summarize(counts_at(B["m_pair"], d_boot), obs),
                "observed_data_plugin": int((np.array(list(gout["observed_data"]["margin_delta_needed_to_attribute"]
                                                          .values())) < d).sum())}
        # ------------- lean removed first (sensitivity S6): plant delta - lean, so that after planting the own-lean
        # equals the target instead of rising by it (the pre-specified planting stacks the target on the observed lean)
        mg_obs_arr = np.array(list(gout["observed_data"]["margin_delta_needed_to_attribute"].values()))
        lean = {"closed_set_images_first_250": float(np.mean(
                    [am[a, own[a]] - np.mean([am[a, e] for e in range(5) if e != own[a]]) for a in range(N_ADAPTERS)])),
                "group_statistic_all_images": float(gout["group_statistic"]["recomputed"]["grand_mean"])}
        lean_rm = {"status": "sensitivity S6 (not pre-specified)",
                   "method": ("delta_planted = delta - lean, lean = the mean over the ten adapters of own-minus-mean-of-"
                              "the-other-four (equal with or without main-effect correction), on the closed-set images "
                              "or on all images as the group statistic defines it; evaluated through the margins"),
                   "lean": lean, "lean_pct_of_R_point": {k: 100 * v / R_pt for k, v in lean.items()}, "levels": {}}
        for n in lnames:
            lean_rm["levels"][n] = {}
            for lk, lvv in lean.items():
                d = levels[n]["delta"] - lvv
                lean_rm["levels"][n][lk] = {
                    "delta_planted": d,
                    "bootstrap_two_stage": summarize(counts_at(B["m_ind"], d), obs),
                    "bootstrap_seed_paired": summarize(counts_at(B["m_pair"], d), obs),
                    "observed_data_plugin": int((mg_obs_arr < d).sum())}
        # ------------- grid and crossings (sensitivity S4)
        methods_M = {"bootstrap_two_stage": B["m_ind"], "bootstrap_seed_paired": B["m_pair"]}
        for k_, v_ in par_M.items():
            methods_M["parametric_" + k_] = v_
        for k_, v_ in struc_M.items():
            methods_M["structural_" + k_] = v_
        gr = {"status": "sensitivity S4 (not pre-specified); lambda as % of the group's point R_real",
              "R_real_point": R_pt,
              "grid": {k_: grid(v_, R_pt, obs) for k_, v_ in methods_M.items()},
              "crossings": {k_: crossings(v_, R_pt, lam_nom) for k_, v_ in methods_M.items()}}
        gout.update({"observed_correct": obs, "bootstrap_two_stage": bs, "bootstrap_seed_paired": sp,
                     "parametric": pm, "structural_parametric": st, "noise_structure_diagnostics": diagn,
                     "proportional_planting": prop, "lean_removed_first": lean_rm, "lambda_grid": gr,
                     "validation": {"imported_scorer_equals_margin_counts_every_replicate_at_scored_levels": True,
                                    "scored_levels": scored,
                                    "main_effects_invariant_to_planting": True,
                                    "fast_main_effects_equal_scorer_rtol": 1e-12}})
        npz[f"{gname}__adapter_draws"] = B["ad"]
        npz[f"{gname}__margins_two_stage"] = B["m_ind"]
        npz[f"{gname}__margins_seed_paired"] = B["m_pair"]
        npz[f"{gname}__main_effects_two_stage"] = B["b_ind"]
        npz[f"{gname}__main_effects_seed_paired"] = B["b_pair"]
        npz[f"{gname}__scorer_counts_two_stage"] = B["x_ind"]
        npz[f"{gname}__scorer_counts_seed_paired"] = B["x_pair"]
        npz[f"{gname}__scored_level_names"] = np.array(scored)
        npz[f"{gname}__scored_level_deltas"] = np.array([levels[n]["delta"] for n in scored])
        npz[f"{gname}__adapter_tags"] = np.array(P["tags"])
        npz[f"{gname}__R_real_point"] = np.array(R_pt)
        npz[f"{gname}__R_real_lower99"] = np.array(R_low)

    res["settings"] = {
        "images_per_adapter": N_USE, "block": "one block of all 250 images per adapter (G = 250)",
        "n_bootstrap": args.n_boot, "bootstrap_seed": BOOT_SEED, "bootstrap_rng": "np.random.default_rng([seed, group index, replicate])",
        "n_parametric": args.n_param, "parametric_seed": PARAM_SEED, "structural_seed": STRUCT_SEED,
        "sigma_mu_values": [0.0, sigma_mu_hi], "group_index": {g: i for i, g in enumerate(GROUPS)},
        "lambda_grid_pct_of_R_point": LAMBDA_GRID_PCT,
        "scorer_G_LIST_in_bootstrap": [N_USE]}
    res["chance_reference"] = {
        "binomial_n": N_ADAPTERS, "p": CHANCE, "E_accuracy": CHANCE,
        "P_X_ge_6": float(chance.sf(5)), "P_X_ge_5": float(chance.sf(4)), "P_X_le_3": float(chance.cdf(3)),
        "P_X_le_4": float(chance.cdf(4)), "P_X_eq_k": [float(chance.pmf(k)) for k in range(N_ADAPTERS + 1)],
        "central_95_interval_correct": [int(chance.ppf(0.025)), int(chance.ppf(0.975))]}
    res["deviations"] = [
        ("D1. P20 level (a) uses R_real_lower99 as its denominator (delta = U = %.6e). Entry 116 applies the lower "
         "99 %% limit to Kodak only, but the ledger's P20 1.1443 %% is also U / R_real_lower99 (D6_results.json: "
         "lambda_U = U/R_real_lower99, lambda_U_point = U/R_real = 0.71 %%; supplement Fig. S6 caption: both "
         "device-level limits divide by the lower 99 %% bound), and the pre-specified assertion that the script "
         "recomputes the ledger's limit needs it. The literal wording (1.1443 %% x point R_real, delta = %.6e) is "
         "reported as level a_device_level_limit_x_R_point."
         % (groups_out["p20"]["levels"]["a_device_level_limit"]["delta"],
            groups_out["p20"]["levels"]["a_device_level_limit_x_R_point"]["delta"])),
        ("D2. The scorer is executed from src/t3_attrib.py by AST extraction (imports, constants, function definitions "
         "only) instead of `import t3_attrib`, whose module body would rewrite out/t3_attrib.json. Its code is "
         "unchanged; Entry 07 is reproduced in full first. In the bootstrap its G_LIST constant is [250]."),
        ("D3. Item 1's result is read from %s, the name item 1 was run under (Entry 116 names out/fv_seedbank.json; "
         "present under that name: %s). %s" % (
             SEEDBANK, item1["present_under_entry_116_name"],
             ("Its rule replaced 0.1752 %%, so its bound of record, %.17g %%, is level b_item1_bound_of_record (as "
              "recorded in that file; item 1 had not been logged in RESULTS.md when this ran)." % item1_pct)
             if item1.get("included") else "No item 1 level is included: %s." % item1.get("reason", item1.get("note")))),
        ("D4. Level (b) multiplies the primary limits by each group's point R_real (the primary limits are plug-in "
         "limits; Entry 116 says 'the group's own R_real'); the lower-99 % variants are reported as sensitivities."),
        ("D5. Sigma_b (parametric model) is the covariance of the main-effect vector over the 2,000 primary bootstrap "
         "replicates; Entry 116 names the quantity but not its estimator. The analytic value for independent "
         "adapters is reported beside it.")]
    res["groups"] = groups_out
    # ------------- compact table
    table = []
    for gname, gout in groups_out.items():
        for n, L in gout["levels"].items():
            row = {"group": gname, "level": n, "lambda_pct": L["lambda_pct"], "denominator": L["denominator"],
                   "delta": L["delta"], "observed_correct": gout["observed_correct"],
                   "observed_data_plugin_correct": gout["observed_data"]["plugin_correct_at_level"][n]}
            for key, blk in (("bootstrap_two_stage", gout["bootstrap_two_stage"]["levels"][n]),
                             ("bootstrap_seed_paired", gout["bootstrap_seed_paired"]["levels"][n])):
                row[key] = {k: blk[k] for k in ("E_accuracy", "mc_se_E_accuracy", "P_X_ge_6", "P_X_ge_5",
                                                "P_X_le_observed", "quantiles_correct_2.5_50_97.5")}
            for smu_name, blk in gout["parametric"]["by_sigma_mu"].items():
                row["parametric_" + smu_name] = {k: blk["levels"][n][k] for k in ("E_accuracy", "P_X_ge_6",
                                                                                   "P_X_ge_5", "P_X_le_observed")}
            for smu_name, blk in gout["structural_parametric"]["by_sigma_mu"].items():
                row["structural_" + smu_name] = {k: blk["levels"][n][k] for k in ("E_accuracy", "P_X_ge_6",
                                                                                   "P_X_ge_5", "P_X_le_observed")}
            lr = gout["lean_removed_first"]["levels"][n]["closed_set_images_first_250"]
            row["lean_removed_first250"] = {"delta_planted": lr["delta_planted"],
                                            "observed_data_plugin": lr["observed_data_plugin"]}
            for key in ("bootstrap_two_stage", "bootstrap_seed_paired"):
                row["lean_removed_first250"][key] = {k: lr[key][k] for k in ("E_accuracy", "P_X_ge_6",
                                                                             "P_X_le_observed")}
            table.append(row)
    res["summary_table"] = table
    res["reading"] = "none (Entry 116: item 5 is descriptive; no reading attaches)"
    res["runtime_s"] = time.time() - t0
    res = py(res)
    np.savez_compressed(args.out_npz, **npz)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for r_ in table:
        b1, b2 = r_["bootstrap_two_stage"], r_["bootstrap_seed_paired"]
        p0 = [v for k, v in r_.items() if k.startswith("parametric_sigma_mu_0")][0]
        print(f"[fv_closedset_expect] {r_['group']:10s} {r_['level']:32s} lambda {r_['lambda_pct']:.4f}% "
              f"delta {r_['delta']:.3e} | obs {r_['observed_correct']}/10 | boot E {b1['E_accuracy']:.3f} "
              f"P>=6 {b1['P_X_ge_6']:.3f} P<=obs {b1['P_X_le_observed']:.3f} | paired E {b2['E_accuracy']:.3f} | "
              f"param0 E {p0['E_accuracy']:.3f}")
    print(f"[fv_closedset_expect] written {args.out} and {args.out_npz} ({res['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
