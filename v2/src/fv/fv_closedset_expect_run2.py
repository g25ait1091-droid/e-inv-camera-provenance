"""Closed-set expectation, second run (RESULTS.md Entry 116, item 5; review section 4 item 9, hostile-r2-15).

Descriptive and pre-specified in Entry 116; no reading attaches. CPU only.

WHY A SECOND SCRIPT. Entry 116 names src/fv/fv_closedset_expect.py -> out/fv_closedset_expect.json. Both exist from
a first attempt (30 Sep 2026, 13:32-13:46 UTC). Results are never overwritten and files are never deleted, and the
first script is the provenance of the first output, so this run is written under new names:
    src/fv/fv_closedset_expect_run2.py -> out/fv_closedset_expect_run2.json
                                          out/fv_closedset_expect_run2_boot_rows.csv (per-replicate rows; appended,
                                          resumable; full-precision margins give the expectation at any planted level)
It is a separate implementation written for this run. Its random-number protocol is set equal to the first
attempt's (documented below), so the two runs can be compared replicate by replicate; the comparison is in the output.

QUESTION. What closed-set accuracy would Entry 06's registered experiment have been expected to show if transfer sat
at a stated limit? Five bodies, two adapters per body (ten adapters), the first 250 generations of each as one block,
the main-effect-corrected argmax with leave-the-candidate-out main effects. Observed (Entry 07): Kodak M1063 3/10,
Huawei P20 (residualised K) 3/10; the P20 low/mid representation 4/10 is not in the pre-specification and is carried
as a labelled supplement.

DATA AND SCORER (Entry 116, item 5).
  Kodak rows: data/csv_kodak/c3_measure_raw.csv (raw K).
  P20 rows:   $EINV_MYDRIVE/inv_channel/E_DAXING/csv/d5_measure.csv (residualised K, column rho_K), SHA-256 recorded.
  Scorer:     t3_attrib.attribute. `import t3_attrib` would execute the module body, which re-reads the CSVs and
              rewrites out/t3_attrib.json (an existing result file). The file is therefore parsed with ast and only
              its import statements, the constants V2/OUT/G_LIST/N_USE and its function definitions are executed,
              unchanged; the SHA-256 of the file and of attribute's source are recorded. Entry 07 is reproduced in
              full against out/t3_attrib.json before anything else (corrected G = 250: 3/10 and 3/10).

PLANTED LEVELS. delta is added to every image's score against its own body's fingerprint. The group's transfer
statistic is recomputed exactly as the file that produced its device-level limit defines it (Kodak:
cells/C4_FINAL_cell.py -> E_MULTIDEV/C4_multidev.json, variant raw; P20: notebooks/13_daxing_smartphone.ipynb cell D6
-> E_DAXING/D6_results.json, reps K): per adapter theta = mean over all its generations of rho(own) - mean of
rho(other four); device means over the two seeds; grand mean (the statistic); U = grand mean + t_{0.995,4} sd(device
means)/sqrt(5); lambda_U = U / R_real_lower99. The script asserts that the unplanted rows recompute the file and the
ledger (Kodak 1.74 %, P20 1.1443 %) and that planting raises the statistic (and U) by exactly delta.
  (a) the group's device-level limit x the denominator the ledger used (R_real_lower99 for both groups; see D2);
  (b) the primary limits x the group's point R_real: nominal 0.1507 %, calibrated 0.1752 %, and item 1's limit
      because item 1's rule replaced 0.1752 % (read from item 1's output; see D4);
  (c) zero.

EXPECTATION.
  Primary (pre-specified): 2,000 two-stage bootstrap replicates. Adapters with replacement within each body (two from
  two), then 250 image indices with replacement within each drawn adapter's first 250, independently per adapter; the
  imported scorer, main effects included, is rerun on every replicate at every level (its G_LIST constant set to
  [250]: only the one-block G = 250 decision enters; asserted identical to the full-G_LIST run on the observed data).
  Parametric (pre-specified, beside it): each adapter's corrected five-score vector drawn independently from
  N(delta e_own, sigma_mu^2 I + Sigma_img/250 + Sigma_b); Sigma_img the pooled within-adapter per-image covariance
  (first 250 images, ddof 1, mean over the ten adapters); Sigma_b the covariance of the scorer's leave-the-candidate-
  out main-effect vector over the 2,000 primary bootstrap replicates (D5); sigma_mu = 0 and 4.108229517051979e-05
  (out/fv_sigma.json, the D200 paired-contrast exact 95 % upper limit, transported; D6); 100,000 experiments.
  Reported for every group, level and method: E[accuracy] (Monte Carlo SE), P(X = k) for k = 0..10, P(X >= 6)
  (0.0064 under chance), P(X >= 5) (0.033 under chance), P(X <= observed).

NOT PRE-SPECIFIED, LABELLED SO IN THE OUTPUT (they change no pre-specified number):
  S1 seed-paired bootstrap (one resampled image-index vector shared by all adapters of a replicate: every adapter
     generates from one seed bank, whose effect cancels in the corrected score; independent resampling adds it back);
  S2 structural parametric model (adapter means drawn with the seed-removed residual covariance; main effects
     computed from the simulated adapters exactly as the scorer computes them; chance-centred);
  S3 level (a) at the point R_real (for P20 the literal wording of Entry 116) and level (b) at R_real_lower99;
  S4 lean removed first (plant delta - lean, so that the statistic equals the target instead of rising by it);
  S5 a lambda grid and the transfer at which E[accuracy] and P(X >= 6) reach stated values;
  S6 the P20 low/mid representation at its own levels;
  S7 a body x candidate interaction diagnostic against the S2 noise model.

RANDOM-NUMBER PROTOCOL (equal to the first attempt's): replicate r of group g (kodak 0, p20 1, p20_lowmid 2) uses
np.random.default_rng([116005, g, r]): for body e = 0..4 in order, rng.choice(adapters of e, 2, replace=True); then
rng.integers(0, 250, (10, 250)) (independent scheme); then rng.integers(0, 250, 250) (seed-paired scheme).
Parametric draws: default_rng([116051, g]).standard_normal((100000, 10, 5)); structural: default_rng([116052, g]).

Run:  python src/fv/fv_closedset_expect_run2.py [--workers 4]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import argparse
import ast
import csv
import glob
import hashlib
import io
import json
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy
from scipy import stats

# ------------------------------------------------------------------------------------------------ paths
V2 = EINV.V2
OUT = V2 + "/out"
DRIVE = (EINV.MYDRIVE + "/inv_channel")
REPO = EINV.REPO
SCORER = EINV.SRC + "/t3_attrib.py"
LEDGER = REPO + "/analysis/FINAL_LEDGER.json"
H6 = OUT + "/h6_calibrated_limit.json"
SIGMA = OUT + "/fv_sigma.json"
T3 = OUT + "/t3_attrib.json"
ITEM1_PRESPEC = OUT + "/fv_seedbank.json"            # the name Entry 116 gives item 1's output
ITEM1_FIRST = OUT + "/fv_seedbank_calib.json"        # the name item 1 was first run under
PREV_JSON = OUT + "/fv_closedset_expect.json"        # first attempt of this item
PREV_NPZ = OUT + "/fv_closedset_expect_margins.npz"
DST = OUT + "/fv_closedset_expect_run2.json"
ROWS = OUT + "/fv_closedset_expect_run2_boot_rows.csv"

# ------------------------------------------------------------------------------------------------ settings
N_USE = 250
N_AD = 10
N_CAND = 5
N_BOOT = 2000
N_PARAM = 100000
BOOT_SEED = 116005
PARAM_SEED = 116051
STRUCT_SEED = 116052
CHUNK = 25
CHANCE = 0.2
LAMBDA_GRID_PCT = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,
                   1.25, 1.5, 2.0]

GROUPS = {
    "kodak": {"gidx": 0, "prespecified": True,
              "label": "Kodak M1063 (Dresden): five bodies x two seeds, natural-image E2 fingerprints (raw K)",
              "rows": V2 + "/data/csv_kodak/c3_measure_raw.csv",
              "rows_copies": [DRIVE + "/E_MULTIDEV/csv/c3_measure_raw.csv"],
              "rho": "rho", "devices": ["D0", "D1", "D2", "D3", "D4"], "t3": ("kodak", "raw_fingerprints"),
              "group_file": DRIVE + "/E_MULTIDEV/C4_multidev.json",
              "group_producer": "cells/C4_FINAL_cell.py (repository) -> E_MULTIDEV/C4_multidev.json, variant raw",
              "real_rows": V2 + "/data/csv_kodak/c0_positive_control.csv",
              "real_copies": [DRIVE + "/E_MULTIDEV/csv/c0_positive_control.csv"], "real_fp": "K_dev"},
    "p20": {"gidx": 1, "prespecified": True,
            "label": "Huawei P20 (Daxing): five bodies x two seeds, residualised E2 fingerprints (K)",
            "rows": DRIVE + "/E_DAXING/csv/d5_measure.csv",
            "rows_copies": [V2 + "/data/csv_p20/d5_measure.csv"],
            "rho": "rho_K", "devices": ["1101", "1102", "1103", "1104", "1105"], "t3": ("p20", "K_residualised"),
            "group_file": DRIVE + "/E_DAXING/D6_results.json",
            "group_producer": "notebooks/13_daxing_smartphone.ipynb cell D6 -> E_DAXING/D6_results.json, reps K",
            "real_rows": DRIVE + "/E_DAXING/csv/d1_positive_K.csv",
            "real_copies": [V2 + "/data/csv_p20/d1_positive_K.csv"], "real_fp": "F_dev"},
    "p20_lowmid": {"gidx": 2, "prespecified": False,
                   "label": ("Huawei P20, low/mid representation (L): SUPPLEMENT S6, not in the pre-specification "
                             "(observed 4/10)"),
                   "rows": DRIVE + "/E_DAXING/csv/d5_measure.csv",
                   "rows_copies": [V2 + "/data/csv_p20/d5_measure.csv"],
                   "rho": "rho_L", "devices": ["1101", "1102", "1103", "1104", "1105"], "t3": ("p20", "L_lowmid"),
                   "group_file": DRIVE + "/E_DAXING/D6_results.json",
                   "group_producer": "notebooks/13_daxing_smartphone.ipynb cell D6 -> E_DAXING/D6_results.json, reps L",
                   "real_rows": DRIVE + "/E_DAXING/csv/d1_positive_L.csv",
                   "real_copies": [V2 + "/data/csv_p20/d1_positive_L.csv"], "real_fp": "F_dev"},
}
LEVEL_ORDER = ["a_device_limit", "b_nominal", "b_calibrated", "b_item1", "c_zero",
               "S3_a_device_limit_x_R_point", "S3_b_nominal_x_R_low", "S3_b_calibrated_x_R_low",
               "S3_b_item1_x_R_low"]


# ------------------------------------------------------------------------------------------------ helpers
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def plain(o):
    """numpy -> plain Python, floats at full precision."""
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return plain(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def same(a, b, rtol=1e-12, path=""):
    """List of differences between two json-like structures (numbers compared with relative tolerance)."""
    d = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(map(str, a)) != set(map(str, b)):
            d.append(f"{path}: keys differ {sorted(set(map(str, a)) ^ set(map(str, b)))}")
        bb = {str(k): v for k, v in b.items()}
        for k, v in a.items():
            if str(k) in bb:
                d += same(v, bb[str(k)], rtol, f"{path}/{k}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            d.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            d += same(x, y, rtol, f"{path}[{i}]")
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            d.append(f"{path}: {a} vs {b}")
    elif isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
        if not (a == b or abs(a - b) <= rtol * max(abs(a), abs(b))):
            d.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        d.append(f"{path}: {a!r} vs {b!r}")
    return d


# ------------------------------------------------------------------------------------------------ imported scorer
def load_scorer():
    src = open(SCORER, encoding="utf-8").read()
    tree = ast.parse(src)
    keep, executed, skipped = [], [], []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(node)
            executed.append(f"line {node.lineno}: import")
        elif isinstance(node, ast.FunctionDef):
            keep.append(node)
            executed.append(f"line {node.lineno}: def {node.name}")
        elif (isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets)
              and {t.id for t in node.targets} <= {"V2", "OUT", "G_LIST", "N_USE"}):
            keep.append(node)
            executed.append(f"line {node.lineno}: assign {', '.join(t.id for t in node.targets)}")
        else:
            skipped.append(node.lineno)
    ns = {"__name__": "t3_attrib__definitions_only", "__file__": SCORER}
    exec(compile(ast.Module(body=keep, type_ignores=[]), SCORER, "exec"), ns)
    fnode = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "attribute")
    fsrc = ast.get_source_segment(src, fnode)
    info = {"file": SCORER, "file_sha256": sha256(SCORER),
            "attribute_source_sha256": hashlib.sha256(fsrc.encode("utf-8")).hexdigest(),
            "attribute_lines": [fnode.lineno, fnode.end_lineno],
            "executed_top_level_statements": executed, "skipped_top_level_statement_lines": skipped,
            "G_LIST_in_file": list(ns["G_LIST"]), "N_USE_in_file": int(ns["N_USE"]),
            "how": ("parsed with ast; the import statements, the constants V2/OUT/G_LIST/N_USE and the function "
                    "definitions are executed unchanged from the file; the module body (which rewrites "
                    "out/t3_attrib.json) is not executed")}
    return ns, info


def scorer_frame(X, tags, devices):
    """The scorer's wide table: index (tag, gen_idx), one column per candidate fingerprint."""
    A, J, E = X.shape
    idx = pd.MultiIndex.from_arrays([np.repeat(np.asarray(tags, dtype=object), J), np.tile(np.arange(J), A)],
                                    names=["tag", "gen_idx"])
    return pd.DataFrame(X.reshape(A * J, E), index=idx, columns=list(devices))


def corrected_margins(X, own):
    """Margins of the main-effect-corrected one-block argmax, from adapter means.
    b_e = mean of candidate e over every image of the adapters not trained on e (the scorer's leave-the-candidate-out
    main effect; with equal image counts this is the mean of those adapters' means). Adapter a is attributed to its
    own body at a planted own-score shift delta if and only if margin_a < delta, where margin_a is the best other
    corrected score minus the own corrected score."""
    m = X.mean(axis=1)                                            # A x E
    E = m.shape[1]
    b = np.array([m[own != e, e].mean() for e in range(E)])
    c = m - b
    ia = np.arange(len(own))
    oth = c.copy()
    oth[ia, own] = -np.inf
    return oth.max(axis=1) - c[ia, own], b


def boot_draw(gidx, r, own):
    rng = np.random.default_rng([BOOT_SEED, gidx, r])
    members = [np.flatnonzero(own == e) for e in range(N_CAND)]
    ad = np.concatenate([rng.choice(mem, size=len(mem), replace=True) for mem in members])
    ii_ind = rng.integers(0, N_USE, size=(len(ad), N_USE))
    ii_pair = rng.integers(0, N_USE, size=N_USE)
    return ad, ii_ind, ii_pair


# ------------------------------------------------------------------------------------------------ workers
_WK = {}


def _winit(payload):
    warnings.simplefilter("ignore")
    ns, _ = load_scorer()
    ns["G_LIST"] = [N_USE]
    _WK.clear()
    _WK.update(payload)
    _WK["attribute"] = ns["attribute"]


def _wtask(args):
    """Bootstrap replicates r0..r1-1 of one group, both schemes; the imported scorer at every level."""
    gname, r0, r1 = args
    P = _WK["groups"][gname]
    attribute = _WK["attribute"]
    X, own, tags, devs = P["X"], P["own"], P["tags"], P["devices"]
    names, deltas = P["level_names"], P["level_deltas"]
    eye = np.eye(N_CAND)
    rows = []
    for r in range(r0, r1):
        ad, ii_ind, ii_pair = boot_draw(P["gidx"], r, own)
        own_s = own[ad]
        tags_s = [f"{tags[a]}#{k}" for k, a in enumerate(ad)]
        dev_of = {t: devs[o] for t, o in zip(tags_s, own_s)}
        plant = eye[own_s][:, None, :]                            # A x 1 x E, one on the own column
        for scheme, Xs in (("two_stage", X[ad[:, None], ii_ind]), ("seed_paired", X[ad][:, ii_pair])):
            mg, b = corrected_margins(Xs, own_s)
            counts = []
            for nm, d in zip(names, deltas):
                out = attribute(scorer_frame(Xs + d * plant, tags_s, devs), devs, dev_of, use_first=N_USE)
                g250 = out["corrected"]["G"][N_USE]
                k = int(g250["adapters_majority_correct"])
                if g250["n_blocks"] != N_AD or abs(g250["block_accuracy"] * N_AD - k) > 1e-9:
                    raise RuntimeError(f"{gname} r{r} {scheme} {nm}: unexpected block structure {g250}")
                if k != int((mg < d).sum()):
                    raise RuntimeError(f"{gname} r{r} {scheme} {nm}: scorer {k} != margin count {(mg < d).sum()}")
                bs = np.array([out["main_effects_b"][dv] for dv in devs])
                if not np.allclose(bs, b, rtol=1e-10, atol=1e-17):
                    raise RuntimeError(f"{gname} r{r} {scheme} {nm}: main effects differ")
                counts.append(k)
            rows.append([gname, scheme, r] + ad.tolist() + mg.tolist() + b.tolist() + counts)
    return gname, r0, r1, rows


# ------------------------------------------------------------------------------------------------ group statistic
def group_statistic(df, rho, devices, delta=0.0, n_first=None):
    """Exactly as C4_FINAL_cell.c4 (Kodak) and notebook 13 d6 (P20): arms in seed-major order (all_arms()),
    theta per adapter over every generation (or the first n_first), delta added to the own-fingerprint score first."""
    rows = []
    for s in (0, 1):
        for dv in devices:
            tag = f"{dv}_s{s}"
            g = df[df.tag == tag]
            own = g[g.K == dv].sort_values("gen_idx")[rho].to_numpy(float) + delta
            oth = [g[g.K == o].sort_values("gen_idx")[rho].to_numpy(float) for o in devices if o != dv]
            n = min([len(own)] + [len(x) for x in oth])
            if n_first is not None:
                n = min(n, n_first)
            th = own[:n] - np.mean([x[:n] for x in oth], axis=0)
            rows.append({"tag": tag, "dev": dv, "seed": s, "theta": float(th.mean()), "n_images": int(n)})
    R = pd.DataFrame(rows)
    dm = R.groupby("dev")["theta"].mean()
    gm = float(R.theta.mean())
    k = len(dm)
    tc = float(stats.t.ppf(1 - 0.01 / 2, df=k - 1))
    U = float(gm + tc * dm.std(ddof=1) / np.sqrt(k))
    return {"per_adapter": rows, "device_means": {str(a): float(b) for a, b in dm.items()}, "grand_mean": gm,
            "t_crit": tc, "sd_device_means": float(dm.std(ddof=1)), "U": U, "n_devices": int(k)}


def real_contrast(df, fp_col, devices):
    """R_real as C4_FINAL_cell.c4 (Kodak) and notebook 13 D1 gates (P20): per body, own minus the mean of the other
    four over its held-out real photographs; mean over bodies; lower limit R - z_0.99 SE."""
    per = []
    for d in devices:
        g = df[df.img_dev == d]
        own = g[g[fp_col] == d].sort_values("idx")["rho"].to_numpy(float)
        oth = np.mean([g[g[fp_col] == o].sort_values("idx")["rho"].to_numpy(float) for o in devices if o != d], axis=0)
        n = min(len(own), np.size(oth))
        per.append(float((own[:n] - oth[:n]).mean()))
    per = np.array(per)
    R = float(per.mean())
    se = float(per.std(ddof=1) / np.sqrt(len(per)))
    return {"per_body": per.tolist(), "R_real": R, "se": se, "R_real_lower99": float(R - stats.norm.ppf(0.99) * se)}


# ------------------------------------------------------------------------------------------------ summaries
CHANCE_DIST = stats.binom(N_AD, CHANCE)


def summarize(x, obs):
    x = np.asarray(x, dtype=int)
    n = x.size
    pk = np.bincount(x, minlength=N_AD + 1)[: N_AD + 1] / n
    sd = float(x.std(ddof=1))
    p6, p5 = float((x >= 6).mean()), float((x >= 5).mean())
    ple, pge = float((x <= obs).mean()), float((x >= obs).mean())
    mc = lambda p: float(np.sqrt(p * (1 - p) / n))
    return {"n": int(n), "E_accuracy": float(x.mean()) / N_AD, "mc_se_E_accuracy": sd / np.sqrt(n) / N_AD,
            "E_correct": float(x.mean()), "sd_correct": sd, "P_X_eq_k": pk.tolist(),
            "P_X_ge_6": p6, "mc_se_P_X_ge_6": mc(p6), "P_X_ge_5": p5, "mc_se_P_X_ge_5": mc(p5),
            "observed_correct": int(obs), "P_X_le_observed": ple, "mc_se_P_X_le_observed": mc(ple),
            "P_X_ge_observed": pge,
            "quantiles_correct_2.5_50_97.5": [float(v) for v in np.percentile(x, [2.5, 50, 97.5])]}


def count(M, delta):
    return (M < np.asarray(delta)).sum(axis=1)


def parametric_margins(Z, V, own):
    """Each adapter's corrected five-score vector N(0, V) independently (delta enters through the margin)."""
    Y = Z @ np.linalg.cholesky(V).T
    ia = np.arange(len(own))
    oth = Y.copy()
    oth[:, ia, own] = -np.inf
    return oth.max(axis=2) - Y[:, ia, own]


def structural_margins(Z, V_ad, own):
    """Adapter mean deviations N(0, V_ad) independently; main effects from the simulated adapters as the scorer does."""
    Y = Z @ np.linalg.cholesky(V_ad).T
    b = np.stack([Y[:, own != e, e].mean(axis=1) for e in range(N_CAND)], axis=1)
    C = Y - b[:, None, :]
    ia = np.arange(len(own))
    oth = C.copy()
    oth[:, ia, own] = -np.inf
    return oth.max(axis=2) - C[:, ia, own], Y


def crossings(M, R_point, lam_nom_pct):
    """Smallest planted shift at which E[accuracy] reaches 0.5 and P(X >= 6) reaches 0.5 / 0.8 / 0.95, exactly from the
    order statistics of the margins (the target is reached for any delta strictly above the returned margin)."""
    n_rep = M.shape[0]
    flat = np.sort(M.ravel())
    out = {"E_accuracy_0.5": float(flat[int(np.ceil(0.5 * flat.size)) - 1])}
    sixth = np.sort(np.sort(M, axis=1)[:, 5])
    for q in (0.5, 0.8, 0.95):
        out[f"P_X_ge_6_{q}"] = float(sixth[int(np.ceil(q * n_rep)) - 1])
    return {k: {"delta": v, "lambda_pct_of_R_point": 100 * v / R_point,
                "multiple_of_nominal_limit": 100 * v / R_point / lam_nom_pct} for k, v in out.items()}


def lambda_grid(M, R_point, obs):
    rows = []
    for lam in LAMBDA_GRID_PCT:
        d = lam / 100 * R_point
        c = count(M, d)
        rows.append({"lambda_pct_of_R_point": lam, "delta": d, "E_accuracy": float(c.mean()) / N_AD,
                     "P_X_ge_6": float((c >= 6).mean()), "P_X_ge_5": float((c >= 5).mean()),
                     "P_X_le_observed": float((c <= obs).mean())})
    return rows


# ------------------------------------------------------------------------------------------------ item 1
def item1_limit(lam_cal):
    """Item 1's bound of record, if its rule replaced 0.1752 %. Prefers the file name Entry 116 gives; otherwise the
    newest item-1 output in out/ that carries the rule. All candidates are recorded."""
    cands = []
    for p in sorted(set(glob.glob(OUT + "/fv_seedbank*.json"))):
        if "progress" in os.path.basename(p):
            continue
        try:
            d = jload(p)
        except Exception as e:                                   # noqa: BLE001
            cands.append({"file": p, "readable": False, "error": repr(e)})
            continue
        rec = {"file": p.replace("\\", "/"), "sha256": sha256(p), "mtime_utc":
               time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(os.path.getmtime(p)))}
        rule = d.get("rule", {})
        rec["branch"] = rule.get("branch")
        rec["c_star"] = rule.get("c_star")
        rec["rule_value"] = rule.get("value")
        try:
            rec["bound_of_record_pct"] = float(d["limits"]["bound_of_record"]["lambda_U_pct_of_R_real"])
            rec["H6_c_1.25_pct"] = float(d["limits"]["H6_calibrated_c_1.25"]["lambda_U_pct_of_R_real"])
        except Exception:                                         # noqa: BLE001
            rec["bound_of_record_pct"] = None
        cands.append(rec)
    usable = [c for c in cands if c.get("branch") is not None
              and (c.get("branch") != "changes" or c.get("bound_of_record_pct") is not None)]
    chosen = None
    for c in usable:
        if c["file"] == ITEM1_PRESPEC:
            chosen = c
    if chosen is None and usable:
        chosen = max(usable, key=lambda c: c["mtime_utc"])
    res = {"candidates": cands, "chosen": chosen}
    if chosen is None:
        res.update({"included": False, "reason": "no item 1 output found"})
    elif chosen.get("branch") == "changes":
        assert chosen["bound_of_record_pct"] is not None
        assert abs(chosen["H6_c_1.25_pct"] - lam_cal) < 1e-12, "item 1's H6 value differs from out/h6_calibrated_limit.json"
        res.update({"included": True, "pct": chosen["bound_of_record_pct"], "c_star": chosen["c_star"]})
    else:
        res.update({"included": False, "reason": f"item 1's rule branch is {chosen.get('branch')!r}: 0.1752 % stays"})
    return res


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-param", type=int, default=N_PARAM)
    ap.add_argument("--out", default=DST)
    ap.add_argument("--rows", default=ROWS)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; results are never overwritten")
    for p in (args.out, args.rows):
        assert os.path.basename(p) not in ("fv_closedset_expect.json", "fv_closedset_expect_margins.npz")
    t0 = time.time()
    warnings.simplefilter("ignore")
    ns, scorer_info = load_scorer()
    ledger, h6, sig, t3 = jload(LEDGER), jload(H6), jload(SIGMA), jload(T3)

    res = {"entry": "RESULTS.md Entry 116, item 5: closed-set expectation (review section 4 item 9, hostile-r2-15)",
           "status": "pre-specified in Entry 116 as descriptive; no reading attaches",
           "script": "src/fv/fv_closedset_expect_run2.py", "script_sha256": sha256(os.path.abspath(__file__)),
           "run": {"started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                   "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                   "scipy": scipy.__version__, "platform": platform.platform(), "workers": args.workers},
           "scorer": scorer_info}

    # ------------------------------------------------------------ primary limits (D200)
    pa = np.array(ledger["primary"]["per_adapter_A"], float)
    pb = np.array(ledger["primary"]["per_adapter_B"], float)
    t11 = stats.t.ppf(0.995, 11)
    U_nom = max(pa.mean() + t11 * pa.std(ddof=1) / np.sqrt(12), pb.mean() + t11 * pb.std(ddof=1) / np.sqrt(12))
    R_d200 = float(ledger["denominators"]["R_real"])
    lam_nom = float(h6["primary_d200"]["lambda_U_nominal_pct"])
    lam_cal = float(h6["primary_d200"]["lambda_U_calibrated_pct"])
    assert abs(100 * U_nom / R_d200 - lam_nom) <= 1e-12 * lam_nom
    assert round(lam_nom, 4) == 0.1507 and round(lam_cal, 4) == 0.1752
    assert abs(100 * float(h6["primary_d200"]["U_device"]) / R_d200 - lam_cal) <= 1e-12 * lam_cal
    sig_hi = float(sig["primary_crossed_model"]["sigma_u_interval_exact"]["two_sided_95"][1])
    assert abs(sig_hi - 4.1e-05) < 5e-7
    it1 = item1_limit(lam_cal)
    res["primary_limits"] = {"nominal_pct": lam_nom, "nominal_recomputed_from_ledger_pct": 100 * U_nom / R_d200,
                             "calibrated_pct": lam_cal, "source": "out/h6_calibrated_limit.json primary_d200",
                             "D200_R_real": R_d200, "item1": it1}

    # ------------------------------------------------------------ inputs
    inputs = {}
    for G in GROUPS.values():
        for p in [G["rows"], G["real_rows"], G["group_file"]] + G["rows_copies"] + G["real_copies"]:
            inputs.setdefault(p, sha256(p))
        for c in G["rows_copies"]:
            assert inputs[c] == inputs[G["rows"]], f"{c} differs from {G['rows']}"
        for c in G["real_copies"]:
            assert inputs[c] == inputs[G["real_rows"]], f"{c} differs from {G['real_rows']}"
    for p in (LEDGER, H6, SIGMA, T3, SCORER):
        inputs[p] = sha256(p)
    res["inputs_sha256"] = inputs
    res["input_notes"] = [
        "P20 rows are read from the Drive path Entry 116 names; Entry 116 says they are not on D:, but a byte-identical "
        "copy is at $EINV_V2/data/csv_p20/d5_measure.csv (the file t3_attrib.py read in Entry 07); asserted.",
        "Kodak rows are read from D: (data/csv_kodak/c3_measure_raw.csv) and asserted byte-identical to the Drive copy.",
        "sigma_mu = %r: out/fv_sigma.json primary_crossed_model.sigma_u_interval_exact.two_sided_95[1], the D200 "
        "paired-contrast exact 95 %% upper limit that Entry 116 rounds to 4.1e-05; transported to the five-score "
        "vector as sigma_mu^2 I, as pre-specified (a margin between two candidates then carries 2 sigma_mu^2)." % sig_hi]

    # ------------------------------------------------------------ per group: Entry 07, statistic, levels, observed
    payload = {"groups": {}}
    gout_all = {}
    for gname, G in GROUPS.items():
        devs = G["devices"]
        df = pd.read_csv(G["rows"], dtype={"tag": str, "K": str})
        assert len(df) == 10 * 500 * 5 and df.groupby("tag").size().eq(2500).all()
        gout = {"label": G["label"], "prespecified": G["prespecified"], "rows_file": G["rows"],
                "score_column": G["rho"], "devices": devs}
        W = ns["wide"](df, rho=G["rho"])
        dev_of = {t: t.split("_")[0] for t in df.tag.unique()}
        # Entry 07, in full, with the scorer's own constants
        ns["G_LIST"] = list(scorer_info["G_LIST_in_file"])
        full = ns["attribute"](W, devs, dev_of)
        stored = t3[G["t3"][0]][G["t3"][1]]
        diffs = same(json.loads(json.dumps(plain(full))), stored)
        assert not diffs, f"{gname}: Entry 07 not reproduced: {diffs[:5]}"
        ns["G_LIST"] = [N_USE]
        only = ns["attribute"](W, devs, dev_of)
        assert not same(json.loads(json.dumps(plain(only["corrected"]["G"][N_USE]))), stored["corrected"]["G"]["250"])
        assert not same(json.loads(json.dumps(plain(only["raw"]["G"][N_USE]))), stored["raw"]["G"]["250"])
        obs = int(full["corrected"]["G"][N_USE]["adapters_majority_correct"])
        gout["entry07_reproduction"] = {
            "compared_with": "out/t3_attrib.json " + ".".join(G["t3"]), "identical_within_rtol": 1e-12,
            "corrected_correct_G250": obs, "raw_correct_G250": int(full["raw"]["G"][N_USE]["adapters_majority_correct"]),
            "corrected_confusion_G250": full["corrected"]["confusion_G250"], "main_effects_b": full["main_effects_b"],
            "G250_only_scorer_identical": True}
        if gname in ("kodak", "p20"):
            assert obs == 3
        else:
            assert obs == 4
        # the closed-set array as the scorer sees it (first 250 per adapter, tags sorted)
        W250 = W.groupby(level=0, group_keys=False).apply(lambda g: g.iloc[:N_USE])
        tags = sorted(W250.index.get_level_values(0).unique())
        for t in tags:
            assert list(W250.loc[t].index) == list(range(N_USE))
        X = np.stack([W250.loc[t][devs].to_numpy(float) for t in tags])
        own = np.array([devs.index(dev_of[t]) for t in tags])
        assert X.shape == (N_AD, N_USE, N_CAND) and np.bincount(own).tolist() == [2] * N_CAND
        mg_obs, b_obs = corrected_margins(X, own)
        assert int((mg_obs < 0).sum()) == obs
        assert np.allclose(b_obs, [full["main_effects_b"][d] for d in devs], rtol=1e-10, atol=1e-17)

        # group statistic and R_real against the producing file and the ledger
        gf = jload(G["group_file"])
        if gname == "kodak":
            v = gf["variants"]["raw"]
            fv = {"U": v["U_device_level"], "lambda_U": v["lambda_U"], "grand_mean": v["grand_mean"],
                  "R_real": gf["R_real"], "R_real_lower99": gf["R_real_lower99"],
                  "theta": [x["theta"] for x in v["per_adapter"]], "tags": [x["tag"] for x in v["per_adapter"]],
                  "per_body_real": gf["per_device_real"]}
        else:
            v = gf["reps"]["K" if gname == "p20" else "L"]
            fv = {"U": v["U"], "lambda_U": v["lambda_U"], "lambda_U_point": v["lambda_U_point"],
                  "grand_mean": v["grand_mean"], "R_real": v["R_real"], "R_real_lower99": v["R_real_lower99"],
                  "theta": [x["theta"] for x in v["per_adapter"]],
                  "tags": [f"{x['dev']}_s{x['seed']}" for x in v["per_adapter"]]}
        st0 = group_statistic(df, G["rho"], devs)
        rr = real_contrast(pd.read_csv(G["real_rows"], dtype={"img_dev": str, G["real_fp"]: str}), G["real_fp"], devs)
        R_pt, R_low = rr["R_real"], rr["R_real_lower99"]
        close = lambda a, b: abs(a - b) <= 1e-12 * max(abs(a), abs(b))
        assert [x["tag"] for x in st0["per_adapter"]] == fv["tags"]
        assert all(close(x["theta"], y) for x, y in zip(st0["per_adapter"], fv["theta"]))
        assert close(st0["grand_mean"], fv["grand_mean"]) and close(st0["U"], fv["U"])
        assert close(R_pt, fv["R_real"]) and close(R_low, fv["R_real_lower99"])
        lamU = st0["U"] / R_low
        assert close(lamU, fv["lambda_U"])
        if gname == "kodak":
            assert all(close(a, b) for a, b in zip(rr["per_body"], fv["per_body_real"]))
            led = next(x for x in ledger["replications"] if x["name"] == "Kodak M1063, CCD")["lambda_U_pct"]
            assert round(100 * lamU, 2) == led == 1.74
            ledger_check = {"FINAL_LEDGER.replications['Kodak M1063, CCD'].lambda_U_pct": led,
                            "recomputed_pct": 100 * lamU}
        elif gname == "p20":
            led = next(x for x in ledger["replications"] if x["name"] == "Huawei P20, smartphone")["lambda_U_pct"]
            led4 = ledger["smartphone"]["lambda_U_pct"]
            assert round(100 * lamU, 2) == led == 1.14 and round(100 * lamU, 4) == led4 == 1.1443
            assert close(st0["U"] / R_pt, fv["lambda_U_point"])
            ledger_check = {"FINAL_LEDGER.replications['Huawei P20, smartphone'].lambda_U_pct": led,
                            "FINAL_LEDGER.smartphone.lambda_U_pct": led4, "recomputed_pct": 100 * lamU,
                            "D6_lambda_U_point_pct (U / point R_real)": 100 * fv["lambda_U_point"]}
        else:
            ledger_check = {"note": "no ledger row for the low/mid representation; D6_results.json reps.L reproduced",
                            "recomputed_pct": 100 * lamU}
        st250 = group_statistic(df, G["rho"], devs, n_first=N_USE)
        gout["group_statistic"] = {
            "producer": G["group_producer"], "file": G["group_file"],
            "definition": ("per adapter theta = mean over all its generations of rho(own) - mean of rho(other four); "
                           "device means over the two seeds; the grand mean is the transfer statistic; "
                           "U = grand mean + t_{0.995,4} sd(device means)/sqrt(5); lambda_U = U / R_real_lower99"),
            "recomputed_all_images": st0, "R_real": rr, "lambda_U": lamU, "lambda_U_pct": 100 * lamU,
            "lambda_U_point_R_pct": 100 * st0["U"] / R_pt, "lambda_hat_pct": 100 * st0["grand_mean"] / R_pt,
            "reproduces_group_file_rtol": 1e-12, "ledger": ledger_check,
            "same_statistic_first_250_images (descriptive)": {"grand_mean": st250["grand_mean"],
                                                               "lambda_hat_pct": 100 * st250["grand_mean"] / R_pt}}

        # planted levels
        lv = [("a_device_limit", 100 * lamU, "R_real_lower99", R_low,
               "pre-specified (a): the group's device-level limit x the denominator the ledger used"),
              ("b_nominal", lam_nom, "R_real", R_pt, "pre-specified (b)"),
              ("b_calibrated", lam_cal, "R_real", R_pt, "pre-specified (b)"),
              ("b_item1", it1.get("pct"), "R_real", R_pt,
               "pre-specified (b): item 1's limit, which replaced 0.1752 % under item 1's rule"),
              ("c_zero", 0.0, None, 0.0, "pre-specified (c)"),
              ("S3_a_device_limit_x_R_point", 100 * lamU, "R_real", R_pt,
               "S3 (not pre-specified); for P20 the literal wording of Entry 116 (see D2)"),
              ("S3_b_nominal_x_R_low", lam_nom, "R_real_lower99", R_low, "S3 (not pre-specified)"),
              ("S3_b_calibrated_x_R_low", lam_cal, "R_real_lower99", R_low, "S3 (not pre-specified)"),
              ("S3_b_item1_x_R_low", it1.get("pct"), "R_real_lower99", R_low, "S3 (not pre-specified)")]
        if not it1["included"]:
            lv = [x for x in lv if "item1" not in x[0]]
        if not G["prespecified"]:
            lv = [(n, l, dn, dv, "S6 supplement (not pre-specified); level defined as in " + s) for n, l, dn, dv, s in lv]
        levels = {}
        for nm, lam, dname, den, status in lv:
            delta = lam / 100 * den
            st = group_statistic(df, G["rho"], devs, delta=delta)
            rise, urise = st["grand_mean"] - st0["grand_mean"], st["U"] - st0["U"]
            assert abs(rise - delta) <= 1e-15 and abs(urise - delta) <= 1e-15, (gname, nm, rise, urise, delta)
            levels[nm] = {"lambda_pct": lam, "denominator": dname, "denominator_value": den, "delta": delta,
                          "status": status, "statistic_rise": rise, "U_rise": urise,
                          "rise_equals_delta_within": 1e-15, "lambda_U_after_planting_pct": 100 * st["U"] / R_low}
        assert abs(levels["a_device_limit"]["delta"] - st0["U"]) <= 1e-18
        gout["levels"] = levels

        # observed data: margins and the plug-in count at every level with the imported scorer
        plug = {}
        for nm, L in levels.items():
            o = ns["attribute"](scorer_frame(X + L["delta"] * np.eye(N_CAND)[own][:, None, :], tags, devs), devs,
                                {t: dev_of[t] for t in tags}, use_first=N_USE)
            k = int(o["corrected"]["G"][N_USE]["adapters_majority_correct"])
            assert k == int((mg_obs < L["delta"]).sum())
            plug[nm] = k
        am = X.mean(axis=1)
        gout["observed_data"] = {
            "adapters": tags, "own_index": own.tolist(),
            "corrected_scores_G250": dict(zip(tags, (am - b_obs).tolist())),
            "margin": dict(zip(tags, mg_obs.tolist())),
            "margin_pct_of_R_point": dict(zip(tags, (100 * mg_obs / R_pt).tolist())),
            "plugin_correct_at_level": plug,
            "note": ("margin = best other corrected score minus own corrected score on the observed first 250 "
                     "images; an adapter is attributed to its own body once the planted shift exceeds its margin "
                     "(negative margin: already attributed). No resampling.")}
        names = list(levels)
        payload["groups"][gname] = {"gidx": G["gidx"], "X": X, "own": own, "tags": tags, "devices": devs,
                                    "level_names": names, "level_deltas": [levels[n]["delta"] for n in names]}
        gout["_keep"] = {"obs": obs, "R_pt": R_pt, "R_low": R_low, "lean_all": st0["grand_mean"],
                         "lean_250": st250["grand_mean"]}
        gout_all[gname] = gout
        lv_txt = ", ".join("%s=%.4e" % (n, levels[n]["delta"]) for n in names)
        print(f"[run2] {gname}: Entry 07 reproduced ({obs}/10); lambda_U {100 * lamU:.6f} %; R_real {R_pt:.6e}, "
              f"lower99 {R_low:.6e}; levels {lv_txt}", flush=True)

    # ------------------------------------------------------------ bootstrap: per-replicate rows, appended, resumable
    max_levels = len(LEVEL_ORDER)
    header = (["group", "scheme", "replicate"] + [f"adapter_{k}" for k in range(N_AD)] +
              [f"margin_{k}" for k in range(N_AD)] + [f"b_{e}" for e in range(N_CAND)] +
              [f"scorer_count_{n}" for n in LEVEL_ORDER])
    done = {}
    if os.path.exists(args.rows):
        with open(args.rows, "r", newline="", encoding="utf-8") as f:
            text = f.read()
        lines = text.splitlines()
        if lines and lines[0].split(",") != header:
            sys.exit(f"{args.rows} has a different header; refusing to resume into it")
        for ln in lines[1:]:
            parts = ln.split(",")
            if len(parts) != len(header):
                continue                                          # a torn last line is ignored
            done.setdefault((parts[0], int(parts[2])), set()).add(parts[1])
        if text and not text.endswith("\n"):
            with open(args.rows, "a", newline="", encoding="utf-8") as f:
                f.write("\n")
    else:
        with open(args.rows, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
    complete = lambda g, r: done.get((g, r), set()) >= {"two_stage", "seed_paired"}
    tasks = []
    for gname in GROUPS:
        for r0 in range(0, args.n_boot, CHUNK):
            r1 = min(r0 + CHUNK, args.n_boot)
            if not all(complete(gname, r) for r in range(r0, r1)):
                tasks.append((gname, r0, r1))
    n_skip = len(GROUPS) * int(np.ceil(args.n_boot / CHUNK)) - len(tasks)
    print(f"[run2] bootstrap: {len(tasks)} chunks to run, {n_skip} complete on disk", flush=True)
    tb = time.time()
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_winit, initargs=(payload,)) as ex:
            futs = [ex.submit(_wtask, t) for t in tasks]
            for i, fu in enumerate(as_completed(futs)):
                gname, r0, r1, rows = fu.result()
                buf = io.StringIO()
                w = csv.writer(buf, lineterminator="\n")
                for row in rows:
                    g_, sch, r = row[:3]
                    if sch in done.get((g_, r), set()):
                        continue
                    vals = row[3:3 + N_AD] + [repr(float(x)) for x in row[3 + N_AD:3 + 2 * N_AD + N_CAND]]
                    cnt = row[3 + 2 * N_AD + N_CAND:]
                    names = payload["groups"][gname]["level_names"]
                    cmap = dict(zip(names, cnt))
                    w.writerow([g_, sch, r] + vals + [cmap.get(n, "") for n in LEVEL_ORDER])
                    done.setdefault((g_, r), set()).add(sch)
                with open(args.rows, "a", newline="", encoding="utf-8") as f:
                    f.write(buf.getvalue())
                    f.flush()
                    os.fsync(f.fileno())
                if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                    el = time.time() - tb
                    print(f"[run2] bootstrap {i + 1}/{len(tasks)} chunks, {el:.0f} s "
                          f"(eta {el / (i + 1) * (len(tasks) - i - 1):.0f} s)", flush=True)

    # ------------------------------------------------------------ read the rows back (the analysis uses the file)
    B = pd.read_csv(args.rows, dtype={"group": str, "scheme": str}, float_precision="round_trip")
    B = B.drop_duplicates(subset=["group", "scheme", "replicate"], keep="first")
    boot = {}
    for gname in GROUPS:
        boot[gname] = {}
        for sch in ("two_stage", "seed_paired"):
            s = B[(B.group == gname) & (B.scheme == sch)].sort_values("replicate")
            assert s.replicate.tolist() == list(range(args.n_boot)), (gname, sch)
            boot[gname][sch] = {"ad": s[[f"adapter_{k}" for k in range(N_AD)]].to_numpy(int),
                                "M": s[[f"margin_{k}" for k in range(N_AD)]].to_numpy(float),
                                "b": s[[f"b_{e}" for e in range(N_CAND)]].to_numpy(float),
                                "counts": {n: s[f"scorer_count_{n}"].to_numpy(int)
                                           for n in payload["groups"][gname]["level_names"]}}
        # the two schemes share the adapter draw; re-derive the draws and check the stored ones
        for r in (0, 1, args.n_boot // 2, args.n_boot - 1):
            ad, _, _ = boot_draw(GROUPS[gname]["gidx"], r, payload["groups"][gname]["own"])
            assert (boot[gname]["two_stage"]["ad"][r] == ad).all() and (boot[gname]["seed_paired"]["ad"][r] == ad).all()

    # ------------------------------------------------------------ summaries
    for gname, G in GROUPS.items():
        gout = gout_all[gname]
        keep = gout.pop("_keep")
        obs, R_pt, R_low = keep["obs"], keep["R_pt"], keep["R_low"]
        P = payload["groups"][gname]
        X, own = P["X"], P["own"]
        levels = gout["levels"]
        names = P["level_names"]
        bt = boot[gname]
        for sch in ("two_stage", "seed_paired"):
            for n in names:                                       # scorer counts equal margin counts, every replicate
                assert np.array_equal(bt[sch]["counts"][n], count(bt[sch]["M"], levels[n]["delta"])), (gname, sch, n)
        gout["observed_correct"] = obs
        gout["bootstrap_two_stage"] = {
            "status": "pre-specified primary (empirical noise)",
            "method": ("two-stage bootstrap: adapters with replacement within each body (two from two), then 250 image "
                       "indices with replacement within each drawn adapter's first 250, independently per adapter; the "
                       "imported scorer (main effects included) rerun on every replicate at every level"),
            "n_replicates": args.n_boot, "rng": f"default_rng([{BOOT_SEED}, {G['gidx']}, r])",
            "counts_from": "imported scorer (asserted equal to the margin counts in every replicate)",
            "levels": {n: summarize(bt["two_stage"]["counts"][n], obs) for n in names}}
        gout["bootstrap_seed_paired"] = {
            "status": "S1 sensitivity (not pre-specified)",
            "method": ("as the two-stage bootstrap with the same adapter draws, but one image-index vector shared by "
                       "all ten drawn adapters (all adapters generate from one seed bank; the seed effect then cancels "
                       "in the corrected score as it does in the real experiment)"),
            "n_replicates": args.n_boot, "counts_from": "imported scorer",
            "levels": {n: summarize(bt["seed_paired"]["counts"][n], obs) for n in names}}

        # parametric (pre-specified)
        S_img = np.mean([np.cov(X[a].T, ddof=1) for a in range(N_AD)], axis=0)
        V0 = S_img / N_USE
        S_b = np.cov(bt["two_stage"]["b"].T, ddof=1)
        n_both = np.array([[int(((own != e) & (own != f)).sum()) for f in range(N_CAND)] for e in range(N_CAND)])
        S_b_analytic = n_both / 64.0 * V0
        Z = np.random.default_rng([PARAM_SEED, G["gidx"]]).standard_normal((args.n_param, N_AD, N_CAND))
        par, parM = {}, {}
        for tag_s, s_mu in (("sigma_mu_0", 0.0), ("sigma_mu_4.108e-05", sig_hi)):
            Mp = parametric_margins(Z, s_mu ** 2 * np.eye(N_CAND) + V0 + S_b, own)
            parM[tag_s] = Mp
            par[tag_s] = {"sigma_mu": s_mu, "levels": {n: summarize(count(Mp, levels[n]["delta"]), obs) for n in names}}
        # Sigma_b estimator sensitivity: the analytic image-noise value instead of the bootstrap covariance
        Mp_an = parametric_margins(Z, V0 + S_b_analytic, own)
        gout["parametric"] = {
            "status": "pre-specified (beside the bootstrap)",
            "method": ("each adapter's corrected five-score vector drawn independently from N(delta e_own, sigma_mu^2 I "
                       "+ Sigma_img/250 + Sigma_b); argmax; ten adapters (two per body) per simulated experiment"),
            "n_experiments": args.n_param, "rng": f"default_rng([{PARAM_SEED}, {G['gidx']}])",
            "inputs": {"Sigma_img": S_img, "Sigma_img_over_250": V0, "Sigma_b": S_b,
                       "Sigma_b_estimator": ("covariance (ddof 1) over the 2,000 two-stage bootstrap replicates of the "
                                             "scorer's main-effect vector; Entry 116 names the quantity, not its "
                                             "estimator (D5)"),
                       "Sigma_b_analytic_image_noise_only": S_b_analytic,
                       "sd_per_image_score": np.sqrt(np.diag(S_img)), "sd_adapter_mean_image_noise": np.sqrt(np.diag(V0)),
                       "sd_main_effect_bootstrap": np.sqrt(np.diag(S_b)),
                       "sd_main_effect_analytic": np.sqrt(np.diag(S_b_analytic)),
                       "corr_per_image_scores": S_img / np.sqrt(np.outer(np.diag(S_img), np.diag(S_img)))},
            "by_sigma_mu": par,
            "Sigma_b_analytic_variant_sigma_mu_0 (D5 sensitivity)": {
                n: summarize(count(Mp_an, levels[n]["delta"]), obs) for n in names}}

        # structural (S2) with the seed effect removed
        gm_ = X.mean(axis=(0, 1))
        am = X.mean(axis=1)
        jm = X.mean(axis=0)
        Rres = X - am[:, None, :] - jm[None, :, :] + gm_
        S_r = np.einsum("aje,ajf->ef", Rres, Rres) / ((N_AD - 1) * (N_USE - 1))
        seed_share = 1 - np.diag(S_r) / np.diag(S_img)
        Z2 = np.random.default_rng([STRUCT_SEED, G["gidx"]]).standard_normal((args.n_param, N_AD, N_CAND))
        stru, struM, Ysim0 = {}, {}, None
        for tag_s, s_mu in (("sigma_mu_0", 0.0), ("sigma_mu_4.108e-05", sig_hi)):
            Ms, Ysim = structural_margins(Z2, s_mu ** 2 * np.eye(N_CAND) + S_r / N_USE, own)
            struM[tag_s] = Ms
            if s_mu == 0.0:
                Ysim0 = Ysim
            stru[tag_s] = {"sigma_mu": s_mu, "levels": {n: summarize(count(Ms, levels[n]["delta"]), obs) for n in names}}
        gout["structural_parametric"] = {
            "status": "S2 sensitivity (not pre-specified)",
            "method": ("adapter mean vectors drawn independently from N(delta e_own, sigma_mu^2 I + Sigma_resid/250), "
                       "Sigma_resid the adapter-by-image residual covariance of the ten adapters' first 250 images "
                       "(adapter and image-index main effects removed: the seed effect is common to every adapter and "
                       "cancels in the corrected score); main effects computed from the simulated adapters exactly "
                       "as the scorer computes them; chance-centred"),
            "n_experiments": args.n_param, "rng": f"default_rng([{STRUCT_SEED}, {G['gidx']}])",
            "inputs": {"Sigma_resid": S_r, "sd_adapter_mean_seed_removed": np.sqrt(np.diag(S_r) / N_USE),
                       "seed_share_of_per_image_variance": seed_share},
            "by_sigma_mu": stru}

        # S7: body x candidate interaction (off-diagonal cells) against the S2 noise model
        def inter_rss(amat):
            T = np.stack([amat[..., own == d, :].mean(axis=-2) for d in range(N_CAND)], axis=-2)   # ... x body x cand
            off = [(d, e) for d in range(N_CAND) for e in range(N_CAND) if d != e]
            D = np.zeros((len(off), 2 * N_CAND))
            for i_, (d, e) in enumerate(off):
                D[i_, d] = 1.0
                D[i_, N_CAND + e] = 1.0
            H = D @ np.linalg.pinv(D)
            y = np.stack([T[..., d, e] for d, e in off], axis=-1)
            res_ = y - y @ H.T
            return (res_ ** 2).sum(axis=-1), len(off) - np.linalg.matrix_rank(D)
        rss_obs, df_int = inter_rss(am)
        rss_sim, _ = inter_rss(Ysim0)
        gout["S7_body_by_candidate_interaction"] = {
            "status": "S7 diagnostic (not pre-specified)",
            "method": ("body x candidate table of adapter-mean scores (first 250 images); body and candidate effects "
                       "fitted by least squares to the 20 off-diagonal cells (own cells excluded, so transfer does not "
                       "enter); residual sum of squares against its distribution under the S2 noise model with "
                       "sigma_mu = 0 (no persistent adapter or body-by-candidate term), 100,000 simulated experiments"),
            "rss_observed": float(rss_obs), "df": int(df_int), "rss_sim_mean": float(rss_sim.mean()),
            "ratio_observed_to_sim_mean": float(rss_obs / rss_sim.mean()),
            "p_sim_rss_ge_observed": float((rss_sim >= rss_obs).mean()),
            "note": ("a ratio well above 1 is persistent body-by-candidate structure that both parametric models "
                     "leave out and the bootstrap keeps fixed at its observed value")}

        # S4: lean removed first (through the margins; the scorer equals the margin count, asserted above)
        lean = {"all_images (the group statistic as defined)": keep["lean_all"],
                "first_250_images (the closed-set images)": keep["lean_250"]}
        lr = {"status": "S4 sensitivity (not pre-specified)",
              "method": ("plant delta - lean instead of delta, so that after planting the group statistic equals the "
                         "target instead of rising by it (Entry 116 plants on top of the lean already in the data)"),
              "lean": lean, "lean_pct_of_R_point": {k: 100 * v / R_pt for k, v in lean.items()}, "levels": {}}
        for n in names:
            lr["levels"][n] = {}
            for lk, lval in lean.items():
                d = levels[n]["delta"] - lval
                lr["levels"][n][lk] = {
                    "delta_planted": d,
                    "observed_data_plugin": int((mg_obs_of(gout) < d).sum()),
                    "bootstrap_two_stage": summarize(count(bt["two_stage"]["M"], d), obs),
                    "bootstrap_seed_paired": summarize(count(bt["seed_paired"]["M"], d), obs)}
        gout["S4_lean_removed_first"] = lr

        # S5: lambda grid and crossings
        methods = {"bootstrap_two_stage": bt["two_stage"]["M"], "bootstrap_seed_paired": bt["seed_paired"]["M"]}
        methods.update({"parametric_" + k: v for k, v in parM.items()})
        methods.update({"structural_" + k: v for k, v in struM.items()})
        gout["S5_lambda_grid"] = {
            "status": "S5 sensitivity (not pre-specified); lambda as % of the group's point R_real",
            "R_real_point": R_pt, "grid": {k: lambda_grid(v, R_pt, obs) for k, v in methods.items()},
            "crossings": {k: crossings(v, R_pt, lam_nom) for k, v in methods.items()}}

    # ------------------------------------------------------------ comparison with the first attempt
    cmp_ = {"first_attempt_files": {PREV_JSON: os.path.exists(PREV_JSON), PREV_NPZ: os.path.exists(PREV_NPZ)}}
    if os.path.exists(PREV_JSON) and os.path.exists(PREV_NPZ):
        cmp_["first_attempt_sha256"] = {PREV_JSON: sha256(PREV_JSON), PREV_NPZ: sha256(PREV_NPZ)}
        prev = jload(PREV_JSON)
        z = np.load(PREV_NPZ)
        per = {}
        prev_level_map = {"a_device_level_limit": "a_device_limit", "b_nominal": "b_nominal",
                          "b_calibrated": "b_calibrated", "b_item1_bound_of_record": "b_item1", "c_zero": "c_zero",
                          "a_device_level_limit_x_R_point": "S3_a_device_limit_x_R_point",
                          "b_nominal_x_R_lower99": "S3_b_nominal_x_R_low",
                          "b_calibrated_x_R_lower99": "S3_b_calibrated_x_R_low",
                          "b_item1_x_R_lower99": "S3_b_item1_x_R_low"}
        for gname in GROUPS:
            pg = prev["groups"][gname]
            mine = gout_all[gname]
            c = {}
            nb = args.n_boot
            c["replicates_compared"] = nb
            c["adapter_draws_identical"] = bool(np.array_equal(z[f"{gname}__adapter_draws"][:nb], boot[gname]["two_stage"]["ad"]))
            c["max_abs_diff_margins_two_stage"] = float(np.abs(z[f"{gname}__margins_two_stage"][:nb] - boot[gname]["two_stage"]["M"]).max())
            c["max_abs_diff_margins_seed_paired"] = float(np.abs(z[f"{gname}__margins_seed_paired"][:nb] - boot[gname]["seed_paired"]["M"]).max())
            c["max_abs_diff_main_effects_two_stage"] = float(np.abs(z[f"{gname}__main_effects_two_stage"][:nb] - boot[gname]["two_stage"]["b"]).max())
            c["max_abs_diff_main_effects_seed_paired"] = float(np.abs(z[f"{gname}__main_effects_seed_paired"][:nb] - boot[gname]["seed_paired"]["b"]).max())
            prev_names = [str(x) for x in z[f"{gname}__scored_level_names"]]
            prev_counts = z[f"{gname}__scorer_counts_two_stage"][:nb]
            c["scorer_counts_two_stage_identical_at_first_attempt_scored_levels"] = {
                pn: bool(np.array_equal(prev_counts[:, j], boot[gname]["two_stage"]["counts"][prev_level_map[pn]]))
                for j, pn in enumerate(prev_names) if prev_level_map.get(pn) in boot[gname]["two_stage"]["counts"]}
            c["observed_margins_max_abs_diff"] = float(max(abs(pg["observed_data"]["margin_delta_needed_to_attribute"][t] - mine["observed_data"]["margin"][t]) for t in mine["observed_data"]["adapters"]))
            lvl = {}
            for pn, mn in prev_level_map.items():
                if pn not in pg["levels"] or mn not in mine["levels"]:
                    continue
                e = {"delta_prev": pg["levels"][pn]["delta"], "delta_run2": mine["levels"][mn]["delta"]}
                e["delta_equal"] = e["delta_prev"] == e["delta_run2"]
                for key_p, key_m in (("bootstrap_two_stage", "bootstrap_two_stage"),
                                     ("bootstrap_seed_paired", "bootstrap_seed_paired")):
                    a_, b_ = pg[key_p]["levels"][pn], mine[key_m]["levels"][mn]
                    e[key_m] = {"E_accuracy_prev": a_["E_accuracy"], "E_accuracy_run2": b_["E_accuracy"],
                                "P_X_eq_k_identical": a_["P_X_eq_k"] == b_["P_X_eq_k"]}
                for s_k in ("sigma_mu_0", "sigma_mu_4.108e-05"):
                    a_ = pg["parametric"]["by_sigma_mu"][s_k]["levels"][pn]
                    b_ = mine["parametric"]["by_sigma_mu"][s_k]["levels"][mn]
                    e["parametric_" + s_k] = {"E_accuracy_prev": a_["E_accuracy"], "E_accuracy_run2": b_["E_accuracy"],
                                              "P_X_eq_k_max_abs_diff": float(np.abs(np.array(a_["P_X_eq_k"]) - np.array(b_["P_X_eq_k"])).max())}
                lvl[mn] = e
            c["levels"] = lvl
            per[gname] = c
        cmp_["groups"] = per
    res["comparison_with_first_attempt"] = cmp_

    # ------------------------------------------------------------ settings, chance, deviations, table
    res["settings"] = {"images_per_adapter": N_USE, "decision": "one block of all 250 images per adapter (G = 250)",
                       "n_bootstrap": args.n_boot, "bootstrap_seed": BOOT_SEED, "chunk": CHUNK,
                       "n_parametric": args.n_param, "parametric_seed": PARAM_SEED, "structural_seed": STRUCT_SEED,
                       "sigma_mu_values": [0.0, sig_hi], "group_index": {g: G["gidx"] for g, G in GROUPS.items()},
                       "scorer_G_LIST_in_bootstrap": [N_USE], "lambda_grid_pct_of_R_point": LAMBDA_GRID_PCT,
                       "rows_file": args.rows.replace("\\", "/"), "rows_file_sha256": sha256(args.rows)}
    res["chance_reference"] = {"binomial_n": N_AD, "p": CHANCE, "E_accuracy": CHANCE,
                               "P_X_eq_k": [float(CHANCE_DIST.pmf(k)) for k in range(N_AD + 1)],
                               "P_X_ge_6": float(CHANCE_DIST.sf(5)), "P_X_ge_5": float(CHANCE_DIST.sf(4)),
                               "P_X_le_3": float(CHANCE_DIST.cdf(3)), "P_X_le_4": float(CHANCE_DIST.cdf(4)),
                               "central_95_interval": [int(CHANCE_DIST.ppf(0.025)), int(CHANCE_DIST.ppf(0.975))]}
    res["deviations"] = [
        "D1. File names: Entry 116 names src/fv/fv_closedset_expect.py -> out/fv_closedset_expect.json; both exist from a "
        "first attempt (30 Sep 2026). Results are never overwritten and the first script is the provenance of the first "
        "output, so this run is src/fv/fv_closedset_expect_run2.py -> out/fv_closedset_expect_run2.json (+ the "
        "per-replicate rows file).",
        "D2. P20 level (a) multiplies the device-level limit by R_real_lower99, as the ledger does (D6_results.json: "
        "lambda_U = U / R_real_lower99 = 1.1443 %%; U / point R_real = %.4f %%), so delta = U. Entry 116 states the lower "
        "limit for Kodak only, but its own requirement that the script recompute the ledger's limit fixes the "
        "denominator. The literal wording (1.1443 %% x point R_real, delta = %.6e) is S3_a_device_limit_x_R_point."
        % (100 * gout_all["p20"]["group_statistic"]["recomputed_all_images"]["U"] / gout_all["p20"]["group_statistic"]["R_real"]["R_real"],
           gout_all["p20"]["levels"]["S3_a_device_limit_x_R_point"]["delta"]),
        "D3. The scorer is executed from src/t3_attrib.py by ast extraction (imports, constants, function definitions), "
        "not by `import t3_attrib`, whose module body rewrites out/t3_attrib.json. In the bootstrap its G_LIST constant "
        "is [250]; on the observed data the G = 250 output is asserted identical to the full-G_LIST run.",
        "D4. Item 1's limit is read from %s (Entry 116 names out/fv_seedbank.json, present: %s); %s The last "
        "RESULTS.md entry when this ran was Entry %s, so item 1's result had not been logged there."
        % (it1["chosen"]["file"] if it1.get("chosen") else None, os.path.exists(ITEM1_PRESPEC),
           ("its rule replaced 0.1752 %%, so its bound of record (%.17g %%, c* = %s) is level b_item1."
            % (it1["pct"], it1.get("c_star"))) if it1["included"]
           else "no item 1 level is included: " + it1.get("reason", "") + ".",
           last_results_entry()),
        "D5. Sigma_b (parametric) is the covariance of the scorer's main-effect vector over the 2,000 two-stage bootstrap "
        "replicates; Entry 116 names the quantity but not its estimator. The image-noise-only analytic value and the "
        "expectation it gives are reported beside.",
        "D6. sigma_mu is the full-precision file value 4.108229517051979e-05 (Entry 116 prints 4.1e-05).",
        "D7. Level (b) multiplies the primary limits by the group's point R_real (Entry 116: 'the group's own R_real'); "
        "the R_real_lower99 versions are S3."]
    table = []
    for gname, gout in gout_all.items():
        for n, L in gout["levels"].items():
            row = {"group": gname, "level": n, "status": L["status"], "lambda_pct": L["lambda_pct"],
                   "denominator": L["denominator"], "delta": L["delta"], "observed_correct": gout["observed_correct"],
                   "observed_data_plugin_correct": gout["observed_data"]["plugin_correct_at_level"][n]}
            for key, blk in (("bootstrap_two_stage", gout["bootstrap_two_stage"]["levels"][n]),
                             ("bootstrap_seed_paired", gout["bootstrap_seed_paired"]["levels"][n])):
                row[key] = {k: blk[k] for k in ("E_accuracy", "mc_se_E_accuracy", "P_X_eq_k", "P_X_ge_6", "P_X_ge_5",
                                                "P_X_le_observed")}
            for s_k, blk in gout["parametric"]["by_sigma_mu"].items():
                row["parametric_" + s_k] = {k: blk["levels"][n][k] for k in ("E_accuracy", "mc_se_E_accuracy",
                                                                             "P_X_eq_k", "P_X_ge_6", "P_X_ge_5",
                                                                             "P_X_le_observed")}
            for s_k, blk in gout["structural_parametric"]["by_sigma_mu"].items():
                row["structural_" + s_k] = {k: blk["levels"][n][k] for k in ("E_accuracy", "P_X_ge_6", "P_X_ge_5",
                                                                             "P_X_le_observed")}
            table.append(row)
    res["summary_table"] = table
    res["groups"] = gout_all
    res["reading"] = "none: Entry 116 item 5 is descriptive and no reading attaches"
    res["runtime_s"] = time.time() - t0
    res = plain(res)
    if os.path.exists(args.out):
        sys.exit(f"{args.out} appeared during the run; not overwritten")
    with open(args.out, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for r_ in table:
        b1, b2 = r_["bootstrap_two_stage"], r_["bootstrap_seed_paired"]
        p0, p1 = r_["parametric_sigma_mu_0"], r_["parametric_sigma_mu_4.108e-05"]
        print(f"[run2] {r_['group']:10s} {r_['level']:30s} lam {r_['lambda_pct']:.5f}% delta {r_['delta']:.4e} "
              f"obs {r_['observed_correct']} plug {r_['observed_data_plugin_correct']} | boot E {b1['E_accuracy']:.4f} "
              f"P6 {b1['P_X_ge_6']:.4f} P5 {b1['P_X_ge_5']:.4f} P<=obs {b1['P_X_le_observed']:.4f} | paired E "
              f"{b2['E_accuracy']:.4f} | par0 E {p0['E_accuracy']:.4f} P6 {p0['P_X_ge_6']:.4f} | par4.1 E "
              f"{p1['E_accuracy']:.4f} P6 {p1['P_X_ge_6']:.4f}", flush=True)
    print(f"[run2] written {args.out} ({res['runtime_s']:.0f} s)")


def mg_obs_of(gout):
    return np.array([gout["observed_data"]["margin"][t] for t in gout["observed_data"]["adapters"]])


def last_results_entry():
    """Number of the last '## Entry NNN' heading in RESULTS.md (read only)."""
    nums = []
    with open(V2 + "/RESULTS.md", encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("## Entry "):
                tok = ln[len("## Entry "):].split()[0]
                if tok.isdigit():
                    nums.append(int(tok))
    return max(nums) if nums else None


if __name__ == "__main__":
    main()
