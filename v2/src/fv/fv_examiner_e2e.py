"""End-to-end examiner test, pre-specified in RESULTS.md Entry 116, item 2 (review report section 4, item 4).

QUESTION. Does a transfer planted into real generated images at a known size come out of the paper's own scoring
and its two-candidate examiner at the rate that eq. (power) predicts?

IMAGES (the archived primary generations, on local disk; asserted present at run time)
  $EINV_DATA/gens      adapters seeds 0-2, 500 images each (00000-00499.png)
  $EINV_DATA/gens_ext  adapters seeds 3-11, images 0-249 (00000-00249.png)
  * calibration of the planting amplitude: adapters s0-s2 of each body, images 250-499 (750 per body); never examined;
  * examined (held-out) adapters: s3-s11 of each body, nine per body, 18 in all, images 0-249;
  * main effect known to the examiner: m_X = mean unplanted paired contrast of the OTHER eleven adapters of arm X over
    images 0-249 (the examined adapter left out).

PLANTING. Y' = Y (1 + a_X K_X^E1), Y the float32 luminance of the PNG exactly as the scorer computes it; the product is
formed in float64 and stored in float32 (the scorer's precision), with no re-quantisation to 8 bits. K_X^E1 =
out/fp/K_{A,B}_E1.npy (80 photographs). SCORING: the c6_estimator_swap / t1_measure scorer, imported: W =
fingerprints.wavelet_residual(Y'), rho = c6_estimator_swap._ncc(W, Y' K) for K_A^E2, K_B^E2 (140 photographs,
disjoint from E1; the E1 templates are scored too, as a diagnostic). Paired contrast d_A = rho(K_A^E2) - rho(K_B^E2)
on arm-A images, d_B = rho(K_B^E2) - rho(K_A^E2) on arm-B images. The unplanted rows are those of
out/c6_estimator_swap.json, reused after 50 randomly chosen images are recomputed (by c6_estimator_swap._measure
itself and by this script's planting path at a = 0) to within 1e-9.

AMPLITUDE. a_X is set on the calibration images so that the mean planted-minus-unplanted paired contrast equals
T = 1 x and 10 x U_device (5.3761e-05, 5.3761e-04): secant iteration from two starting amplitudes, stopped at 1 %
relative error, at most 8 iterations. Starting amplitudes (not fixed by the registration; chosen here before any
planted image was scored): T_1x / R_real and T_10x / R_real, shared by both targets and both bodies. The shift achieved
on the examined images is reported, not tuned.

EXAMINER. For examined adapter a of arm X and G in {10, 20, 50, 100}: theta_hat_G = mean of d_X over G distinct
images of that adapter - m_X; 1,000 random subsets per adapter and G (numpy default_rng(20260930 + 100 g + i), g the
index of G, i the index of the adapter, A s3..s11 then B s3..s11), the same subsets in every condition (null, 1 x,
10 x). Threshold: the 99th percentile (numpy linear quantile) of the null theta_hat_G pooled over the 18 adapters,
M = 2. Empirical TPR: fraction of planted trials strictly above it. Uncertainty: cluster bootstrap over adapters,
2,000 replicates (default_rng(20260930 + 1000)), stratified by body (nine drawn with replacement per body), the
threshold recomputed in every replicate; percentile intervals simultaneous over the eight (G, target) cells at
1 - 0.05/8.

MODEL COMPARATOR. Eq. (power) with lambda R_real = T, M = 2, at sigma_mu = 0 and 4.1e-05, with the variance this design
has under the crossed model: sigma_G^2 = sigma_mu^2 (1 + 1/11) + [sigma_vX^2 (N - G)/(N - 1) + sigma_eX^2]/G +
sigma_eX^2/(11 x 250), N = 250, sigma_v and sigma_e per arm from out/fv_sigma.json; per-arm TPRs averaged over the
arms. The unadjusted eq. (power) curves (SE_500 = 7.035e-05; verify_v2.tpr() imported) are printed beside.
AGREEMENT: band [TPR at sigma_mu = 4.1e-05, TPR at sigma_mu = 0] per cell; readings as registered (see READINGS).

OUTPUT. out/fv_examiner_e2e.json (the registration named out/fv_examiner.json; the file name follows the task that
ran it) and per-image planted rows out/fv_examiner_e2e_rows.csv (appended; the run resumes from it).

PROVENANCE OF THE ROWS. The first measurement run (--measure-only, script sha256 664580d8..., log
logs/fv_examiner_e2e_measure.log, 30 Sep 2026 13:14-14:17) wrote 16,450 of the 16,500 rows and ended before the last
25 examined images (B_raw_s11_r16, 00225-00249, both amplitudes). A later run resumed from the rows file (log
logs/fv_examiner_e2e_resume.log); every run logs its own script sha256 at start. With --measure-version and
--resume-version (byte copies of the two measuring versions, sha256 asserted), the analysis embeds the unified diff
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
from each to this script in the output, so the claim that only reporting code changed can be checked from the output
itself. Before anything is measured, a sample of already-stored planted rows is recomputed through this script's
planting path and must agree exactly.

Run:  python src/fv/fv_examiner_e2e.py               (about 16,600 residual computations on 6 workers)
      python src/fv/fv_examiner_e2e.py --test --rows <scratch.csv> --out <scratch.json>   (a tiny smoke test)
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["EINV_V2"] = EINV.V2                      # einv_paths (imported by c6_estimator_swap)
os.environ["EINV_DATA"] = EINV.DATA         # -> GENS, GENS_EXT: the local primary generations
SRC = EINV.SRC
for _p in (os.path.join(SRC, "fv"), SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import csv
import hashlib
import json
import math
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy import stats

import c6_estimator_swap as C6                                  # the scorer (its _init, _ncc, _measure, TEMPL)

V2 = EINV.V2
OUT = V2 + "/out"
FP = OUT + "/fp"
GENS = (EINV.DATA + "/gens")
GENS_EXT = (EINV.DATA + "/gens_ext")
C6_JSON = OUT + "/c6_estimator_swap.json"
SIGMA_JSON = OUT + "/fv_sigma.json"
POWER_JSON = OUT + "/t3_power_v4.json"
LEDGER = (EINV.REPO + "/analysis/FINAL_LEDGER.json")
ROWS_DEFAULT = OUT + "/fv_examiner_e2e_rows.csv"
OUT_DEFAULT = OUT + "/fv_examiner_e2e.json"
SCRIPT = os.path.abspath(__file__)
MEASURE_VERSION_SHA256 = "664580d83db96f20c314d7b3b3fc29555cbdc2c4d60f1d2e8e840b1c598da06c"   # rows 1-16,450
RESUME_VERSION_SHA256 = "b7eac1d39a712b255e465d7b177ae77606f3060d2b5f6b350f92e604ef570914"    # rows 16,451-16,500

R_REAL = 0.0356703416571125            # c6_estimator_swap.R_REAL (asserted equal)
U_DEVICE = 5.3761e-05                  # FINAL_LEDGER primary.U_device (asserted equal)
TARGETS = {"1x": 1.0 * U_DEVICE, "10x": 10.0 * U_DEVICE}
G_LIST = (10, 20, 50, 100)
M_LIST = (2, 5, 50)
N_SUB = 1000
N_IMG = 250                            # examined images per adapter (0-249); also N of the model
K_OTHER = 11                           # adapters in the leave-one-out main effect
SEED = 20260930
SEED_BOOT = SEED + 1000
SEED_VERIFY = SEED + 2000
N_BOOT = 2000
FAMILY_ALPHA = 0.05
N_CELLS = 8
SIGMA_MU_HIGH = 4.1e-05                # Entry 114 exact 95 % upper limit, as registered (file value 4.108e-05)
SIGMA_MU_ARCHIVE = 5.23e-05            # the archive value; unadjusted curve only, descriptive
SE500 = 7.035e-05                      # t3_power_v4.json inputs.SE_img_500 (asserted equal)
TOL = 0.01
MAX_IT = 8
N_VERIFY = 50
VERIFY_TOL = 1e-9
N_REPLANT = 12                         # stored planted rows recomputed before anything is measured (exact agreement)
SEED_REPLANT = SEED + 3000
CAL_SEEDS = (0, 1, 2)
CAL_IMAGES = tuple(range(250, 500))
EXAM_SEEDS = tuple(range(3, 12))
EXAM_IMAGES = tuple(range(0, 250))
BODIES = ("A", "B")
SGN = {"A": 1.0, "B": -1.0}
HDR = ["stage", "label", "body", "tag", "image", "amplitude", "rho_KA_E2", "rho_KB_E2", "rho_KA_E1", "rho_KB_E1"]
READINGS = {
    "agree": ("eq. (power) agrees end-to-end: planted transfer at the nominal limit and at ten times it is detected "
              "at the rates the model gives"),
    "below": "the paper's scoring detects less than eq. (power) predicts",
    "above": "eq. (power) understates the examiner's power",
    "both": "does not agree",
}


def log(msg):
    print(f"[e2e {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def tag_of(body, s):
    return f"{body}_raw_s{s}_r16"


def adapter_dir(body, s):
    return os.path.join(GENS if s <= 2 else GENS_EXT, tag_of(body, s)).replace("\\", "/")


def img_name(j):
    return f"{j:05d}.png"


# ============================================================================ workers
_P = {}


def _winit():
    C6._init()                                                   # wavelet_residual, MEAS, the four templates
    _P.update({b: C6._G["K"][f"K_{b}_E1"].astype(np.float64) for b in BODIES})


def _luminance(path):
    """Exactly the scorer's luminance (c6_estimator_swap._measure / t1_measure._measure)."""
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    if a.shape[0] != C6._G["MEAS"]:
        raise ValueError(f"{path}: size {a.shape}")
    return (0.299*a[..., 0] + 0.587*a[..., 1] + 0.114*a[..., 2]).astype(np.float32)


def _scores(Y, body, amp):
    """Plant Y' = Y (1 + amp K_body^E1) (float64 product, stored float32; amp = 0 leaves Y bit-identical) and score
    it with the imported scorer: W = wavelet_residual(Y'), rho = NCC(W, Y' K) for the four templates of C6.TEMPL."""
    g = C6._G
    if amp == 0.0:
        Yp = Y
    else:
        Yp = (Y.astype(np.float64) * (1.0 + amp * _P[body])).astype(np.float32)
    W = g["wr"](Yp)
    return [C6._ncc(W, Yp * g["K"][t]) for t in C6.TEMPL]


def _task(t):
    stage, body, tag, path, amps = t
    Y = _luminance(path)
    return stage, body, tag, os.path.basename(path), [(lab, amp, _scores(Y, body, amp)) for lab, amp in amps]


def _verify(t):
    tag, path = t
    ref = C6._measure((tag, path))                               # the imported measurement itself
    mine = _scores(_luminance(path), tag[0], 0.0)                # this script's path at a = 0
    return tag, os.path.basename(path), ref[2], mine


def _replant(t):
    label, body, tag, path, amp = t
    return label, tag, os.path.basename(path), _scores(_luminance(path), body, amp)


# ============================================================================ rows file (append-only, resumable)
def read_rows(path):
    rows, bad = {}, 0
    if not os.path.exists(path):
        return rows, bad
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        hdr = next(rd, None)
        assert hdr == HDR, f"{path}: unexpected header {hdr}"
        for r in rd:
            try:
                assert len(r) == len(HDR)
                amp = float(r[5])
                rho = [float(x) for x in r[6:]]
            except (AssertionError, ValueError):
                bad += 1                                         # a line cut by an interrupted run: recomputed
                continue
            key = (r[1], r[3], r[4])
            if key in rows:
                assert rows[key]["amp_repr"] == r[5] and rows[key]["rho"] == rho, f"conflicting duplicate row {key}"
            rows[key] = {"stage": r[0], "body": r[2], "amp": amp, "amp_repr": r[5], "rho": rho}
    return rows, bad


def open_rows_for_append(path):
    new = not os.path.exists(path)
    if not new:
        with open(path, "rb") as f:
            f.seek(0, 2)
            if f.tell() > 0:
                f.seek(-1, 2)
                if f.read(1) != b"\n":                           # terminate a cut line so it stays malformed alone
                    with open(path, "ab") as g:
                        g.write(b"\n")
    fh = open(path, "a", newline="", encoding="utf-8")
    wr = csv.writer(fh)
    if new:
        wr.writerow(HDR)
        fh.flush()
    return fh, wr


def run_tasks(pool, tasks, rows, fh, wr, what):
    if not tasks:
        log(f"{what}: nothing to do (all rows present)")
        return
    t0 = time.time()
    n = 0
    log(f"{what}: {len(tasks)} images, {sum(len(t[4]) for t in tasks)} residuals")
    for stage, body, tag, name, out in pool.imap_unordered(_task, tasks, chunksize=2):
        for lab, amp, rho in out:
            rec = [stage, lab, body, tag, name, repr(float(amp))] + [repr(float(x)) for x in rho]
            wr.writerow(rec)
            rows[(lab, tag, name)] = {"stage": stage, "body": body, "amp": float(amp), "amp_repr": repr(float(amp)),
                                      "rho": [float(x) for x in rho]}
        n += 1
        if n % 25 == 0:
            fh.flush()
            os.fsync(fh.fileno())
        if n % 250 == 0 or n == len(tasks):
            el = time.time() - t0
            log(f"{what}: {n}/{len(tasks)} images, {el/60:.1f} min, eta {el/n*(len(tasks)-n)/60:.1f} min")
    fh.flush()
    os.fsync(fh.fileno())


# ============================================================================ calibration
def d_e2(rho, body):
    return SGN[body] * (rho[0] - rho[1])


def d_e1(rho, body):
    return SGN[body] * (rho[2] - rho[3])


def cal_list(body, test):
    imgs = CAL_IMAGES[:6] if test else CAL_IMAGES
    return [(tag_of(body, s), img_name(j), adapter_dir(body, s) + "/" + img_name(j)) for s in CAL_SEEDS for j in imgs]


def label_amp(rows, label, items):
    reps = {rows[(label, t, n)]["amp_repr"] for t, n, _ in items if (label, t, n) in rows}
    assert len(reps) <= 1, f"{label}: several amplitudes in the rows file {reps}"
    return reps.pop() if reps else None


def shift_of(rows, body, label, items, which=d_e2):
    base = np.array([which(rows[("cal_a0", t, n)]["rho"], body) for t, n, _ in items])
    pl = np.array([which(rows[(label, t, n)]["rho"], body) for t, n, _ in items])
    return float(np.mean(pl - base)), pl - base


def calibrate(pool, rows, fh, wr, test):
    items = {b: cal_list(b, test) for b in BODIES}
    starts = {"cal_s0": TARGETS["1x"] / R_REAL, "cal_s1": TARGETS["10x"] / R_REAL}
    # phase A: unplanted and the two starting amplitudes
    tasks = []
    for b in BODIES:
        for lab, amp in (("cal_a0", 0.0),) + tuple(starts.items()):
            rep = label_amp(rows, lab, items[b])
            assert rep is None or rep == repr(float(amp)), (b, lab, rep, amp)
        for t, n, p in items[b]:
            amps = [(lab, amp) for lab, amp in (("cal_a0", 0.0),) + tuple(starts.items()) if (lab, t, n) not in rows]
            if amps:
                tasks.append(("cal", b, t, p, amps))
    run_tasks(pool, tasks, rows, fh, wr, "calibration, unplanted + two starting amplitudes")
    state = {}
    for b in BODIES:
        for tk, T in TARGETS.items():
            pts = [{"label": lab, "amplitude": amp, "shift": shift_of(rows, b, lab, items[b])[0]}
                   for lab, amp in starts.items()]
            for p in pts:
                p["rel_err"] = (p["shift"] - T) / T
            acc = next((p for p in pts if abs(p["rel_err"]) <= TOL), None)
            state[(b, tk)] = {"points": pts, "accepted": acc, "iterations": 0}
    it = 0
    while any(s["accepted"] is None for s in state.values()) and it < MAX_IT:
        it += 1
        plan = {}
        for (b, tk), s in state.items():
            if s["accepted"] is not None:
                continue
            p0, p1 = s["points"][-2], s["points"][-1]
            T = TARGETS[tk]
            a_new = p1["amplitude"] - (p1["shift"] - T) * (p1["amplitude"] - p0["amplitude"]) / (p1["shift"] - p0["shift"])
            lab = f"cal_{tk}_i{it}"
            rep = label_amp(rows, lab, items[b])
            assert rep is None or rep == repr(float(a_new)), f"{b} {lab}: rows file holds {rep}, recomputed {a_new!r}"
            plan.setdefault(b, []).append((lab, float(a_new), tk))
        tasks = []
        for b, labs in plan.items():
            for t, n, p in items[b]:
                amps = [(lab, a) for lab, a, _ in labs if (lab, t, n) not in rows]
                if amps:
                    tasks.append(("cal", b, t, p, amps))
        run_tasks(pool, tasks, rows, fh, wr, f"calibration, secant iteration {it}")
        for b, labs in plan.items():
            for lab, a, tk in labs:
                T = TARGETS[tk]
                sh = shift_of(rows, b, lab, items[b])[0]
                p = {"label": lab, "amplitude": a, "shift": sh, "rel_err": (sh - T) / T}
                s = state[(b, tk)]
                s["points"].append(p)
                s["iterations"] = it
                if abs(p["rel_err"]) <= TOL:
                    s["accepted"] = p
    out = {}
    for (b, tk), s in state.items():
        conv = s["accepted"] is not None
        acc = s["accepted"] if conv else min(s["points"], key=lambda p: abs(p["rel_err"]))
        e1 = shift_of(rows, b, acc["label"], items[b], which=d_e1)
        e2 = shift_of(rows, b, acc["label"], items[b])
        out.setdefault(b, {})[tk] = {
            "target": TARGETS[tk], "amplitude": acc["amplitude"], "amplitude_repr": repr(float(acc["amplitude"])),
            "accepted_label": acc["label"], "achieved_shift_calibration": acc["shift"],
            "rel_err_calibration": acc["rel_err"], "converged_within_1pct": conv,
            "secant_iterations": s["iterations"], "evaluations": s["points"],
            "per_image_shift_sd_calibration": float(np.std(e2[1], ddof=1)),
            "E1_template_shift_calibration": e1[0],
            "E2_over_E1_template_shift": e2[0] / e1[0] if e1[0] != 0 else None,
            "n_images": len(items[b])}
        log(f"calibration {b} {tk}: a = {acc['amplitude']:.6e}, shift {acc['shift']:.6e} "
            f"(rel err {acc['rel_err']:+.4f}, {'converged' if conv else 'NOT converged'}, {s['iterations']} it)")
    return out


# ============================================================================ examined adapters
def exam_items(test):
    seeds = EXAM_SEEDS[:2] if test else EXAM_SEEDS
    imgs = EXAM_IMAGES[:3] if test else EXAM_IMAGES
    return [(b, s, tag_of(b, s), img_name(j), adapter_dir(b, s) + "/" + img_name(j))
            for b in BODIES for s in seeds for j in imgs]


def plant_examined(pool, rows, fh, wr, cal, test):
    items = exam_items(test)
    amps = {b: [(f"exam_{tk}", cal[b][tk]["amplitude"]) for tk in TARGETS] for b in BODIES}
    for b in BODIES:
        for lab, a in amps[b]:
            rep = label_amp(rows, lab, [(t, n, p) for bb, s, t, n, p in items if bb == b])
            assert rep is None or rep == repr(float(a)), f"{b} {lab}: rows file holds {rep}, calibrated {a!r}"
    tasks = []
    for b, s, t, n, p in items:
        todo = [(lab, a) for lab, a in amps[b] if (lab, t, n) not in rows]
        if todo:
            tasks.append(("exam", b, t, p, todo))
    run_tasks(pool, tasks, rows, fh, wr, "examined adapters, 1 x and 10 x")


# ============================================================================ checks
def verify_unplanted(pool, c6rows):
    order = [tag_of(b, s) for b in BODIES for s in range(12)]
    rng = np.random.default_rng(SEED_VERIFY)
    pick = rng.choice(len(order) * N_IMG, N_VERIFY, replace=False)
    tasks = []
    for k in sorted(int(x) for x in pick):
        tag, j = order[k // N_IMG], k % N_IMG
        tasks.append((tag, adapter_dir(tag[0], int(tag.split("_s")[1].split("_")[0])) + "/" + img_name(j)))
    res, worst_c6, worst_mine = [], 0.0, 0.0
    for tag, name, ref, mine in pool.imap_unordered(_verify, tasks):
        stored = c6rows[tag][name]
        d_ref = max(abs(a - b) for a, b in zip(ref, stored))
        d_mine = max(abs(a - b) for a, b in zip(mine, stored))
        worst_c6, worst_mine = max(worst_c6, d_ref), max(worst_mine, d_mine)
        res.append({"tag": tag, "image": name, "max_abs_diff_c6_measure": d_ref, "max_abs_diff_planting_path_a0": d_mine})
    res.sort(key=lambda r: (r["tag"], r["image"]))
    ok = worst_c6 <= VERIFY_TOL and worst_mine <= VERIFY_TOL
    log(f"unplanted recompute on {N_VERIFY} images: max |diff| c6 _measure {worst_c6:.3e}, planting path a=0 "
        f"{worst_mine:.3e} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("the stored unplanted rows do not recompute to within 1e-9; stopping as registered")
    return {"n": N_VERIFY, "seed": SEED_VERIFY, "tolerance": VERIFY_TOL, "passed": ok,
            "max_abs_diff_c6_measure": worst_c6, "max_abs_diff_planting_path_a0": worst_mine, "images": res,
            "what": ("50 (adapter, image) pairs drawn from the 24 x 250 stored rows; recomputed by the imported "
                     "c6_estimator_swap._measure and by this script's planting path at a = 0; all four templates")}


def verify_planted(pool, rows):
    """Recompute N_REPLANT stored planted rows (amplitude > 0) through this invocation's planting path; they must agree
    exactly with what an earlier invocation wrote (same code, same environment). Guards a resumed rows file."""
    keys = sorted(k for k, v in rows.items() if v["amp"] != 0.0)
    if not keys:
        return {"n": 0, "note": "no planted rows stored yet"}
    rng = np.random.default_rng(SEED_REPLANT)
    pick = [keys[int(i)] for i in rng.choice(len(keys), min(N_REPLANT, len(keys)), replace=False)]
    tasks = []
    for lab, tag, name in pick:
        v = rows[(lab, tag, name)]
        s = int(tag.split("_s")[1].split("_")[0])
        tasks.append((lab, v["body"], tag, adapter_dir(tag[0], s) + "/" + name, v["amp"]))
    res, worst = [], 0.0
    for lab, tag, name, rho in pool.imap_unordered(_replant, tasks):
        d = max(abs(a - b) for a, b in zip(rho, rows[(lab, tag, name)]["rho"]))
        worst = max(worst, d)
        res.append({"label": lab, "tag": tag, "image": name, "max_abs_diff": d})
    res.sort(key=lambda r: (r["label"], r["tag"], r["image"]))
    ok = worst == 0.0
    log(f"stored planted rows recomputed on {len(res)} images: max |diff| {worst:.3e} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("stored planted rows do not recompute exactly; the rows file and this script disagree")
    return {"n": len(res), "seed": SEED_REPLANT, "tolerance": 0.0, "passed": ok, "max_abs_diff": worst, "rows": res,
            "what": ("planted rows drawn at random from the rows file as it stood when this invocation started, "
                     "recomputed through this script's planting and scoring path; exact equality required")}


def archive_agreement(c6d, cal_a0):
    """Per-image agreement of this instrument (local scorer, local K^E2) with the archive rows (the instrument of the
    paper's primary numbers) on the same images."""
    import fv_sigma
    XA, XB, _ = fv_sigma.primary_rows()
    arch = {"A": XA, "B": XB}
    led = load(LEDGER)["primary"]
    ledger_diff = max(float(np.max(np.abs(XA.mean(1) - np.array(led["per_adapter_A"])))),
                      float(np.max(np.abs(XB.mean(1) - np.array(led["per_adapter_B"])))))
    sig = load(SIGMA_JSON)["data"]["rows"]
    sha = {k: {"file": v["file"], "sha256": sha256(v["file"]), "sha256_in_fv_sigma": v["sha256"]} for k, v in sig.items()}
    for v in sha.values():
        v["match"] = v["sha256"] == v["sha256_in_fv_sigma"]

    def comp(x, y):
        x, y = np.ravel(x), np.ravel(y)
        return {"n_images": int(x.size), "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                "mean_diff_local_minus_archive": float(np.mean(x - y)), "sd_diff": float(np.std(x - y, ddof=1)),
                "sd_local": float(np.std(x, ddof=1)), "sd_archive": float(np.std(y, ddof=1)),
                "max_abs_diff": float(np.max(np.abs(x - y)))}
    out = {"archive_rows": sha, "archive_adapter_means_minus_ledger_max_abs": ledger_diff,
           "images_0_249_all_24_adapters": {}, "images_250_499_calibration_adapters": {},
           "what": ("paired contrast d per image: local = c6 scorer with out/fp K^E2; archive = rho_mult rows of "
                    "s5/b3/sx2_measure.csv (archive templates); same generated image")}
    both_l, both_a = [], []
    for b in BODIES:
        out["images_0_249_all_24_adapters"][b] = comp(c6d[b], arch[b][:, :N_IMG])
        both_l.append(c6d[b].ravel()); both_a.append(arch[b][:, :N_IMG].ravel())
        per = (c6d[b].mean(1) - arch[b][:, :N_IMG].mean(1))
        out["images_0_249_all_24_adapters"][b]["adapter_mean_diff"] = [float(x) for x in per]
        out["images_0_249_all_24_adapters"][b]["adapter_mean_r"] = float(np.corrcoef(c6d[b].mean(1), arch[b][:, :N_IMG].mean(1))[0, 1])
        if cal_a0 is not None:
            out["images_250_499_calibration_adapters"][b] = comp(cal_a0[b], arch[b][:3, 250:500])
    out["images_0_249_all_24_adapters"]["pooled"] = comp(np.concatenate(both_l), np.concatenate(both_a))
    return out


# ============================================================================ analysis
def zq(M):
    return float(stats.norm.ppf(1 - 0.01 / (M - 1)))


def model_blocks(sv, se):
    def sigma_adj(b, G, smu):
        return math.sqrt(smu ** 2 * (1 + 1 / K_OTHER) + (sv[b] ** 2 * (N_IMG - G) / (N_IMG - 1) + se[b] ** 2) / G
                         + se[b] ** 2 / (K_OTHER * N_IMG))

    def tpr(T, s, M):
        return float(1 - stats.norm.cdf(zq(M) - T / s))
    return sigma_adj, tpr


def import_verifier_tpr():
    import fv_sigma
    vv, status, summary = fv_sigma.import_verifier()
    tprf, _ = fv_sigma.make_power(vv)
    return tprf, status, summary


def analyse(rows, c6rows, cal, meta):
    t0 = time.time()
    names = [img_name(j) for j in range(N_IMG)]

    def c6mat(tag):
        r = c6rows[tag]
        assert sorted(r) == names, tag
        return np.array([r[n] for n in names], float)
    M4 = {b: np.array([c6mat(tag_of(b, s)) for s in range(12)]) for b in BODIES}          # 12 x 250 x 4
    D0 = {b: SGN[b] * (M4[b][:, :, 0] - M4[b][:, :, 1]) for b in BODIES}                   # 12 x 250
    D0e1 = {b: SGN[b] * (M4[b][:, :, 2] - M4[b][:, :, 3]) for b in BODIES}
    exam = [(b, s) for b in BODIES for s in EXAM_SEEDS]
    idx_body = {b: [i for i, (bb, _) in enumerate(exam) if bb == b] for b in BODIES}
    null = np.array([D0[b][s] for b, s in exam])                                           # 18 x 250
    pl, pl_e1 = {}, {}
    for tk in TARGETS:
        pl[tk] = np.array([[d_e2(rows[(f"exam_{tk}", tag_of(b, s), n)]["rho"], b) for n in names] for b, s in exam])
        pl_e1[tk] = np.array([[d_e1(rows[(f"exam_{tk}", tag_of(b, s), n)]["rho"], b) for n in names] for b, s in exam])
        for bb in BODIES:
            amps = {rows[(f"exam_{tk}", tag_of(b, s), n)]["amp_repr"] for b, s in exam for n in names if b == bb}
            assert amps == {cal[bb][tk]["amplitude_repr"]}, (bb, tk, amps)
    null_e1 = np.array([D0e1[b][s] for b, s in exam])
    m = np.array([np.delete(D0[b].mean(1), s).mean() for b, s in exam])                    # leave-one-out main effect

    # ---------------------------------------------------------------- achieved shift on the examined images
    shift = {}
    for tk, T in TARGETS.items():
        dlt = pl[tk] - null
        dlt1 = pl_e1[tk] - null_e1
        blk = {}
        for b in BODIES:
            x = dlt[idx_body[b]]
            blk[b] = {"mean_shift": float(x.mean()), "rel_dev_from_target": float(x.mean() / T - 1),
                      "per_image_sd": float(x.std(ddof=1)), "per_adapter_mean_shift": [float(v) for v in x.mean(1)],
                      "sd_of_adapter_mean_shift": float(x.mean(1).std(ddof=1)),
                      "E1_template_mean_shift": float(dlt1[idx_body[b]].mean()),
                      "E2_over_E1_template_shift": float(x.mean() / dlt1[idx_body[b]].mean())}
        blk["pooled"] = {"mean_shift": float(dlt.mean()), "rel_dev_from_target": float(dlt.mean() / T - 1),
                         "per_image_sd": float(dlt.std(ddof=1))}
        blk["target"] = T
        blk["differs_by_more_than_10pct"] = bool(any(abs(blk[b]["rel_dev_from_target"]) > 0.10 for b in BODIES)
                                                 or abs(blk["pooled"]["rel_dev_from_target"]) > 0.10)
        shift[tk] = blk

    # ---------------------------------------------------------------- examiner trials
    conds = ("null",) + tuple(TARGETS)
    Dc = {"null": null, **pl}
    TH = {c: {G: np.empty((len(exam), N_SUB)) for G in G_LIST} for c in conds}
    subset_seeds = {}
    for i, (b, s) in enumerate(exam):
        for gi, G in enumerate(G_LIST):
            sd = SEED + 100 * gi + i
            subset_seeds[f"{tag_of(b, s)}_G{G}"] = sd
            rng = np.random.default_rng(sd)
            S = np.stack([rng.choice(N_IMG, G, replace=False) for _ in range(N_SUB)])
            assert all(len(set(r)) == G for r in S[:5])
            for c in conds:
                TH[c][G][i] = Dc[c][i][S].mean(1) - m[i]

    def q_of(M):
        return 1 - 0.01 / (M - 1)
    thr = {G: {M: float(np.quantile(TH["null"][G].ravel(), q_of(M))) for M in M_LIST} for G in G_LIST}
    emp = {}
    for G in G_LIST:
        e = {}
        for M in M_LIST:
            e[f"M{M}"] = {"threshold": thr[G][M], "quantile": q_of(M),
                          **{c: float((TH[c][G] > thr[G][M]).mean()) for c in conds},
                          "per_body": {b: {c: float((TH[c][G][idx_body[b]] > thr[G][M]).mean()) for c in conds}
                                       for b in BODIES}}
        e["null_sd_pooled"] = float(TH["null"][G].std(ddof=1))
        e["null_sd_per_body"] = {b: float(TH["null"][G][idx_body[b]].std(ddof=1)) for b in BODIES}
        e["null_mean_per_body"] = {b: float(TH["null"][G][idx_body[b]].mean()) for b in BODIES}
        e["per_adapter"] = {tag_of(b, s): {
            "m_X": float(m[i]), "null_mean": float(TH["null"][G][i].mean()), "null_sd": float(TH["null"][G][i].std(ddof=1)),
            "FPR_at_pooled_threshold": float((TH["null"][G][i] > thr[G][2]).mean()),
            **{f"TPR_{tk}": float((TH[tk][G][i] > thr[G][2]).mean()) for tk in TARGETS}}
            for i, (b, s) in enumerate(exam)}
        emp[f"G{G}"] = e

    # ---------------------------------------------------------------- model
    sig = load(SIGMA_JSON)["primary_crossed_model"]["arms"]
    sv = {b: float(sig[b]["sigma_v"]) for b in BODIES}
    se = {b: float(sig[b]["sigma_e"]) for b in BODIES}
    sigma_adj, tprm = model_blocks(sv, se)
    smus = {"sigma_mu_0": 0.0, "sigma_mu_4.1e-05": SIGMA_MU_HIGH}
    G_GRID = (5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 250)
    adj = {}
    for sk, smu in smus.items():
        adj[sk] = {}
        for tk, T in TARGETS.items():
            adj[sk][tk] = {}
            for G in sorted(set(G_LIST) | set(G_GRID)):
                cell = {}
                for M in M_LIST:
                    per = {b: tprm(T, sigma_adj(b, G, smu), M) for b in BODIES}
                    cell[f"M{M}"] = {**per, "mean": float(np.mean([per[b] for b in BODIES]))}
                cell["sigma_G"] = {b: sigma_adj(b, G, smu) for b in BODIES}
                adj[sk][tk][f"G{G}"] = cell
    tprv, vstatus, vsummary = import_verifier_tpr()
    unadj, max_dev_vs_verifier = {}, 0.0
    for sk, smu in (("sigma_mu_0", 0.0), ("sigma_mu_4.1e-05", SIGMA_MU_HIGH), ("sigma_mu_5.23e-05_archive", SIGMA_MU_ARCHIVE)):
        unadj[sk] = {}
        for tk, T in TARGETS.items():
            unadj[sk][tk] = {}
            for G in sorted(set(G_LIST) | set(G_GRID)):
                s = math.sqrt(smu ** 2 + SE500 ** 2 * 500 / G)
                cell = {}
                for M in M_LIST:
                    own = tprm(T, s, M)
                    ver = tprv(T, G, smu, SE500, M)
                    max_dev_vs_verifier = max(max_dev_vs_verifier, abs(own - ver))
                    cell[f"M{M}"] = ver
                cell["sigma_G"] = s
                unadj[sk][tk][f"G{G}"] = cell
    assert max_dev_vs_verifier < 1e-12, max_dev_vs_verifier

    # ---------------------------------------------------------------- bootstrap
    rngb = np.random.default_rng(SEED_BOOT)
    nA, nB = len(idx_body["A"]), len(idx_body["B"])
    mod_thr = {G: np.array([zq(2) * sigma_adj(b, G, 0.0) for b, _ in exam])[:, None] for G in G_LIST}
    BT = {G: {M: {c: np.empty(N_BOOT) for c in conds} for M in M_LIST} for G in G_LIST}
    BTb = {G: {b: {c: np.empty(N_BOOT) for c in conds} for b in BODIES} for G in G_LIST}
    BTmod = {G: {c: np.empty(N_BOOT) for c in conds} for G in G_LIST}
    for r in range(N_BOOT):
        sa = rngb.integers(0, nA, nA)
        sb = rngb.integers(0, nB, nB)
        sel = np.concatenate([np.array(idx_body["A"])[sa], np.array(idx_body["B"])[sb]])
        for G in G_LIST:
            nb = TH["null"][G][sel].ravel()
            for M in M_LIST:
                t = np.quantile(nb, q_of(M))
                for c in conds:
                    x = TH[c][G][sel] > t
                    BT[G][M][c][r] = x.mean()
                    if M == 2:
                        BTb[G]["A"][c][r] = x[:nA].mean()
                        BTb[G]["B"][c][r] = x[nA:].mean()
            for c in conds:
                BTmod[G][c][r] = (TH[c][G][sel] > mod_thr[G][sel]).mean()
    q_sim = FAMILY_ALPHA / N_CELLS / 2
    boot = {"n_boot": N_BOOT, "seed": SEED_BOOT, "stratified_by_body": True, "threshold_recomputed_each_replicate": True,
            "simultaneous_percentiles": [q_sim, 1 - q_sim], "cells": {}}
    for G in G_LIST:
        for M in M_LIST:
            for c in conds:
                v = BT[G][M][c]
                boot["cells"][f"G{G}_M{M}_{c}"] = {
                    "percentile_95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
                    "simultaneous_1_minus_0.05_over_8": [float(np.quantile(v, q_sim)), float(np.quantile(v, 1 - q_sim))],
                    "boot_mean": float(v.mean()), "boot_sd": float(v.std(ddof=1))}
        for b in BODIES:
            for c in conds:
                v = BTb[G][b][c]
                boot["cells"][f"G{G}_M2_{c}_body{b}"] = {
                    "percentile_95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
                    "boot_mean": float(v.mean())}
        for c in conds:
            v = BTmod[G][c]
            boot["cells"][f"G{G}_modelthreshold_{c}"] = {
                "percentile_95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))], "boot_mean": float(v.mean())}

    # ---------------------------------------------------------------- agreement (registered criterion)
    agreement, below, above = [], [], []
    for tk in TARGETS:
        for G in G_LIST:
            e = emp[f"G{G}"]["M2"][tk]
            lo, hi = boot["cells"][f"G{G}_M2_{tk}"]["simultaneous_1_minus_0.05_over_8"]
            b_hi = adj["sigma_mu_0"][tk][f"G{G}"]["M2"]["mean"]
            b_lo = adj["sigma_mu_4.1e-05"][tk][f"G{G}"]["M2"]["mean"]
            rel = "below" if hi < b_lo else ("above" if lo > b_hi else "meets")
            if rel == "below":
                below.append(f"{tk} G={G}")
            if rel == "above":
                above.append(f"{tk} G={G}")
            # resolution of the check (descriptive; added after a dry run of this analysis; not part of the reading):
            # how wide the band and the interval are, the point difference from the sigma_mu = 0 rate, and the share
            # of the model variance that a persistent term of 4.1e-05 would contribute at this G
            s2_0 = np.mean([sigma_adj(b, G, 0.0) ** 2 for b in BODIES])
            s2_h = np.mean([sigma_adj(b, G, SIGMA_MU_HIGH) ** 2 for b in BODIES])
            agreement.append({"target": tk, "G": G, "empirical_TPR": e, "simultaneous_interval": [lo, hi],
                              "band": [b_lo, b_hi], "relation_to_band": rel,
                              "model_unadjusted_sigma_mu_0": unadj["sigma_mu_0"][tk][f"G{G}"]["M2"],
                              "model_unadjusted_sigma_mu_4.1e-05": unadj["sigma_mu_4.1e-05"][tk][f"G{G}"]["M2"],
                              "resolution_descriptive": {
                                  "band_width": b_hi - b_lo, "interval_width": hi - lo,
                                  "empirical_minus_model_sigma_mu_0": e - b_hi,
                                  "share_of_sigma_G2_from_sigma_mu_4.1e-05": float((s2_h - s2_0) / s2_h)}})
    if not below and not above:
        reading, rkey = READINGS["agree"], "agree"
    elif below and not above:
        reading, rkey = READINGS["below"] + " (cells: " + ", ".join(below) + ")", "below"
    elif above and not below:
        reading, rkey = READINGS["above"] + " (cells: " + ", ".join(above) + ")", "above"
    else:
        reading, rkey = (READINGS["both"] + " (below the band: " + ", ".join(below) + "; above the band: "
                         + ", ".join(above) + ")"), "both"
    # band at the file value of the upper limit, to show the literal 4.1e-05 decides nothing
    smu_file = float(load(SIGMA_JSON)["primary_crossed_model"]["sigma_u_interval_exact"]["two_sided_95"][1])
    band_file = []
    for a in agreement:
        T = TARGETS[a["target"]]
        lo_f = float(np.mean([tprm(T, sigma_adj(b, a["G"], smu_file), 2) for b in BODIES]))
        lo, hi = a["simultaneous_interval"]
        band_file.append({"target": a["target"], "G": a["G"], "band_low_at_file_value": lo_f,
                          "relation": "below" if hi < lo_f else ("above" if lo > a["band"][1] else "meets")})

    # ---------------------------------------------------------------- descriptive
    desc = {}
    # (a) the model-threshold examiner (z_0.99 sigma_G, adjusted, sigma_mu = 0), per arm
    mt = {}
    for G in G_LIST:
        row = {}
        for c in conds:
            x = TH[c][G] > mod_thr[G]
            row[c] = {"pooled": float(x.mean()), **{b: float(x[idx_body[b]].mean()) for b in BODIES},
                      "percentile_95": boot["cells"][f"G{G}_modelthreshold_{c}"]["percentile_95"]}
        row["thresholds"] = {b: float(zq(2) * sigma_adj(b, G, 0.0)) for b in BODIES}
        row["empirical_pooled_threshold_M2"] = thr[G][2]
        mt[f"G{G}"] = row
    desc["model_threshold_examiner"] = {"what": "threshold z_0.99 sigma_G(adjusted, sigma_mu = 0) per arm; FPR = the null row",
                                        "cells": mt}
    # (b) five and fifty candidates
    desc["five_and_fifty_candidates"] = {
        f"M{M}": {f"G{G}": {c: {"empirical": emp[f"G{G}"][f"M{M}"][c],
                                "percentile_95": boot["cells"][f"G{G}_M{M}_{c}"]["percentile_95"],
                                **({"model_adjusted_sigma_mu_0": adj["sigma_mu_0"][c][f"G{G}"][f"M{M}"]["mean"],
                                    "model_adjusted_sigma_mu_4.1e-05": adj["sigma_mu_4.1e-05"][c][f"G{G}"][f"M{M}"]["mean"],
                                    "model_unadjusted_sigma_mu_0": unadj["sigma_mu_0"][c][f"G{G}"][f"M{M}"]}
                                   if c != "null" else {})}
                            for c in conds} for G in G_LIST}
        for M in (5, 50)}
    for M in (5, 50):
        for G in G_LIST:
            desc["five_and_fifty_candidates"][f"M{M}"][f"G{G}"]["threshold"] = thr[G][M]
            desc["five_and_fifty_candidates"][f"M{M}"][f"G{G}"]["null_trials_above"] = int(
                (TH["null"][G] > thr[G][M]).sum())
    # (c) per body
    desc["per_body"] = {b: {f"G{G}": {c: {"empirical": emp[f"G{G}"]["M2"]["per_body"][b][c],
                                          "percentile_95": boot["cells"][f"G{G}_M2_{c}_body{b}"]["percentile_95"],
                                          **({"model_adjusted_sigma_mu_0": adj["sigma_mu_0"][c][f"G{G}"]["M2"][b],
                                              "model_adjusted_sigma_mu_4.1e-05": adj["sigma_mu_4.1e-05"][c][f"G{G}"]["M2"][b]}
                                             if c != "null" else {})}
                                      for c in conds} for G in G_LIST} for b in BODIES}
    # (d) this instrument's crossed-model components on images 0-249 (12 x 250 per arm), and the band recomputed with them
    import fv_sigma
    comp = {}
    for b in BODIES:
        t = fv_sigma.two_way(D0[b])
        msa, mss, mse = t["ss_a"] / t["df_a"], t["ss_s"] / t["df_s"], t["ss_e"] / t["df_e"]
        comp[b] = {"sigma_u2_mom": (msa - mse) / N_IMG, "sigma_v": math.sqrt(max((mss - mse) / 12, 0.0)),
                   "sigma_e": math.sqrt(mse), "F_adapter": msa / mse,
                   "F_p_upper": float(stats.f.sf(msa / mse, t["df_a"], t["df_e"])),
                   "adapter_means": [float(x) for x in D0[b].mean(1)],
                   "fv_sigma_sigma_v": sv[b], "fv_sigma_sigma_e": se[b]}
    sa2, _ = model_blocks({b: comp[b]["sigma_v"] for b in BODIES}, {b: comp[b]["sigma_e"] for b in BODIES})
    su_pool = math.sqrt(max(np.mean([comp[b]["sigma_u2_mom"] for b in BODIES]), 0.0))
    inst_band = {}
    for tk, T in TARGETS.items():
        for G in G_LIST:
            inst_band[f"{tk}_G{G}"] = {
                "sigma_mu_0": float(np.mean([tprm(T, sa2(b, G, 0.0), 2) for b in BODIES])),
                "sigma_mu_4.1e-05": float(np.mean([tprm(T, sa2(b, G, SIGMA_MU_HIGH), 2) for b in BODIES])),
                "sigma_mu_at_this_instruments_sigma_u": float(np.mean([tprm(T, sa2(b, G, su_pool), 2) for b in BODIES])),
                "sigma_G_this_instrument_sigma_mu_0": {b: sa2(b, G, 0.0) for b in BODIES},
                "null_sd_empirical": emp[f"G{G}"]["null_sd_per_body"]}
    desc["this_instrument_components_images_0_249"] = {
        "arms": comp, "sigma_u_pooled_mom_clipped": su_pool, "model_TPR_with_these_components": inst_band,
        "what": ("two-way ANOVA (fv_sigma.two_way, imported) of the 12 x 250 unplanted paired contrasts per arm, this "
                 "instrument; the registered criterion uses fv_sigma.json (archive rows, 500 images) instead")}
    # (e) diagnostic curves at the achieved shift (registered when the achieved shift differs from T by > 10 %)
    diag = {}
    for tk in TARGETS:
        diag[tk] = {"triggered_by_10pct_rule": shift[tk]["differs_by_more_than_10pct"]}
        for sk, smu in smus.items():
            diag[tk][sk] = {f"G{G}": float(np.mean([tprm(shift[tk][b]["mean_shift"], sigma_adj(b, G, smu), 2)
                                                    for b in BODIES])) for G in G_LIST}
        rel = {}
        for G in G_LIST:
            lo, hi = boot["cells"][f"G{G}_M2_{tk}"]["simultaneous_1_minus_0.05_over_8"]
            b_lo, b_hi = diag[tk]["sigma_mu_4.1e-05"][f"G{G}"], diag[tk]["sigma_mu_0"][f"G{G}"]
            rel[f"G{G}"] = "below" if hi < b_lo else ("above" if lo > b_hi else "meets")
        diag[tk]["relation_of_simultaneous_interval_to_band_at_achieved_shift"] = rel
    desc["model_at_achieved_shift"] = diag
    # (f) the null distribution against the model: empirical SD and tail
    desc["null_vs_model"] = {f"G{G}": {
        "empirical_sd_pooled": emp[f"G{G}"]["null_sd_pooled"], "empirical_sd_per_body": emp[f"G{G}"]["null_sd_per_body"],
        "model_sigma_G_sigma_mu_0": {b: sigma_adj(b, G, 0.0) for b in BODIES},
        "empirical_threshold_M2": thr[G][2],
        "model_threshold_M2_sigma_mu_0": {b: zq(2) * sigma_adj(b, G, 0.0) for b in BODIES},
        "empirical_q99_over_sd": thr[G][2] / emp[f"G{G}"]["null_sd_pooled"],
        "null_skewness": float(stats.skew(TH["null"][G].ravel())),
        "null_excess_kurtosis": float(stats.kurtosis(TH["null"][G].ravel()))} for G in G_LIST}

    # (g) post hoc decomposition, NOT registered: where a gap between the model and the empirical rate comes from.
    #     model at T (normal null, constant shift) -> empirical null + constant T (the null's shape and spread)
    #     -> empirical null + each adapter's own mean achieved shift (adapter-level shift heterogeneity)
    #     -> the empirical rate (within-adapter, per-image variation of the shift over the drawn subsets)
    ach = {tk: (pl[tk] - null).mean(1) for tk in TARGETS}                                   # 18 adapter mean shifts
    dec = {}
    for tk, T in TARGETS.items():
        for G in G_LIST:
            dec[f"{tk}_G{G}"] = {
                "model_adjusted_sigma_mu_0_at_T": adj["sigma_mu_0"][tk][f"G{G}"]["M2"]["mean"],
                "model_adjusted_sigma_mu_0_at_each_adapters_shift": float(np.mean(
                    [tprm(ach[tk][i], sigma_adj(b, G, 0.0), 2) for i, (b, _) in enumerate(exam)])),
                "empirical_null_plus_constant_T": float(((TH["null"][G] + T) > thr[G][2]).mean()),
                "empirical_null_plus_each_adapters_mean_shift": float(((TH["null"][G] + ach[tk][:, None]) > thr[G][2]).mean()),
                "empirical": emp[f"G{G}"]["M2"][tk]}
    desc["decomposition_post_hoc"] = {
        "status": "post hoc, descriptive, not registered; computed from the same trials, threshold and subsets",
        "adapter_mean_shift_over_T": {tk: [float(x / TARGETS[tk]) for x in ach[tk]] for tk in TARGETS},
        "cells": dec}

    # ---------------------------------------------------------------- tables for reading
    table = []
    for G in G_LIST:
        for c in conds:
            row = {"G": G, "condition": c, "empirical_rate": emp[f"G{G}"]["M2"][c],
                   "simultaneous_interval": boot["cells"][f"G{G}_M2_{c}"]["simultaneous_1_minus_0.05_over_8"],
                   "percentile_95": boot["cells"][f"G{G}_M2_{c}"]["percentile_95"]}
            if c != "null":
                row.update({"model_adjusted_sigma_mu_0": adj["sigma_mu_0"][c][f"G{G}"]["M2"]["mean"],
                            "model_adjusted_sigma_mu_4.1e-05": adj["sigma_mu_4.1e-05"][c][f"G{G}"]["M2"]["mean"],
                            "model_unadjusted_sigma_mu_0": unadj["sigma_mu_0"][c][f"G{G}"]["M2"],
                            "model_unadjusted_sigma_mu_4.1e-05": unadj["sigma_mu_4.1e-05"][c][f"G{G}"]["M2"]})
            else:
                row["note"] = "FPR at the pooled threshold, 0.01 by construction"
            table.append(row)
    return {
        "reading": reading, "reading_key": rkey, "cells_below_band": below, "cells_above_band": above,
        "agreement": agreement, "band_low_end_at_file_value_4.108e-05": {"value": smu_file, "cells": band_file},
        "table_M2": table,
        "planted_shift_examined": shift,
        "examiner_empirical": emp,
        "bootstrap": boot,
        "model": {"adjusted": adj, "unadjusted_eq_power": unadj,
                  "inputs": {"sigma_v": sv, "sigma_e": se, "N": N_IMG, "K_other": K_OTHER, "SE_500": SE500,
                             "sigma_mu_values": {"0": 0.0, "high (registered)": SIGMA_MU_HIGH,
                                                 "archive (unadjusted, descriptive)": SIGMA_MU_ARCHIVE},
                             "source": "out/fv_sigma.json primary_crossed_model.arms.{A,B}.sigma_v / sigma_e"},
                  "adjusted_formula": ("sigma_G^2 = sigma_mu^2 (1 + 1/11) + [sigma_vX^2 (N - G)/(N - 1) + sigma_eX^2]/G "
                                       "+ sigma_eX^2/(11 x 250); TPR_X = 1 - Phi(z_{1-0.01/(M-1)} - T/sigma_G); mean "
                                       "over arms"),
                  "unadjusted_formula": "eq. (power): sigma_G^2 = sigma_mu^2 + SE_500^2 500/G, via verify_v2.tpr()",
                  "verifier_import": {"status": vstatus, "summary": vsummary,
                                      "max_abs_dev_own_formula_vs_verifier": max_dev_vs_verifier}},
        "subset_seeds": subset_seeds,
        "descriptive": desc,
        "analysis_runtime_s": time.time() - t0,
    }


# ============================================================================ main
def inputs_block(c6):
    ks = {t: {"file": f"{FP}/{t}.npy", "sha256": sha256(f"{FP}/{t}.npy")} for t in C6.TEMPL}
    counts = {}
    for b in BODIES:
        for s in range(12):
            d = adapter_dir(b, s)
            counts[tag_of(b, s)] = len([f for f in os.listdir(d) if f.endswith(".png")])
    return {"images": {"gens": GENS, "gens_ext": GENS_EXT, "png_count_per_adapter": counts,
                       "calibration": "adapters s0-s2 of each body, images 250-499 (750 per body)",
                       "examined": "adapters s3-s11 of each body (9 per body, 18), images 0-249",
                       "main_effect": "mean unplanted d of the other eleven adapters of the arm, images 0-249"},
            "templates": ks, "planting_fields": {"A": ks["K_A_E1"], "B": ks["K_B_E1"]},
            "scoring_templates": {"A": ks["K_A_E2"], "B": ks["K_B_E2"]},
            "c6_rows": {"file": C6_JSON, "sha256": sha256(C6_JSON), "spec": c6.get("_spec"), "rows_cols": c6["rows_cols"]},
            "fv_sigma": {"file": SIGMA_JSON, "sha256": sha256(SIGMA_JSON)},
            "t3_power_v4": {"file": POWER_JSON, "sha256": sha256(POWER_JSON)},
            "ledger": {"file": LEDGER, "sha256": sha256(LEDGER)},
            "script": {"file": SCRIPT, "sha256": sha256(SCRIPT)},
            "script_version_that_measured_the_rows": {
                "first_run_sha256": MEASURE_VERSION_SHA256, "resume_run_sha256": RESUME_VERSION_SHA256,
                "note": ("the first --measure-only run (log logs/fv_examiner_e2e_measure.log) wrote rows 1-16,450 and "
                         "ended before the last 25 examined images of B_raw_s11_r16 (both amplitudes); the resume run "
                         "(--measure-only, log logs/fv_examiner_e2e_resume.log, which prints its script sha256) wrote "
                         "rows 16,451-16,500. The planting, scoring and calibration functions are the same in every "
                         "version; the diffs from both measuring versions to this script are under "
                         "checks.code_diff_from_measuring_versions")}}


MEASURE_LOGS = (V2 + "/logs/fv_examiner_e2e_measure.log", V2 + "/logs/fv_examiner_e2e_resume.log")


def measurement_log_summary():
    import re
    out = []
    for path in MEASURE_LOGS:
        if not os.path.exists(path):
            out.append({"file": path, "exists": False})
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip("\n") for l in f if l.startswith("[e2e ")]
        keep = []
        for l in lines:
            mm = re.search(r": (\d+)/(\d+) images,", l)
            if mm is None or mm.group(1) == mm.group(2):              # drop intermediate progress lines
                keep.append(l)
        out.append({"file": path, "exists": True, "first": lines[0] if lines else None,
                    "last": lines[-1] if lines else None, "milestones": keep[:60]})
    return out


def code_diff(path, want_sha, which):
    """Unified diff from an earlier measuring version of this script (a byte copy, sha256 asserted) to this script."""
    import difflib
    sha = sha256(path)
    assert sha == want_sha, f"{path}: sha256 {sha} is not the {which}'s {want_sha}"
    with open(path, encoding="utf-8") as f:
        a = f.read().splitlines(keepends=True)
    with open(SCRIPT, encoding="utf-8") as f:
        b = f.read().splitlines(keepends=True)
    diff = "".join(difflib.unified_diff(a, b, f"fv_examiner_e2e.py ({which})", "fv_examiner_e2e.py (this run)"))
    return {"copy_compared": path, "measuring_version_sha256": sha, "this_script_sha256": sha256(SCRIPT),
            "measuring_run": which, "unified_diff": diff,
            "functions_used_to_measure": ["_winit", "_luminance", "_scores", "_task", "_verify", "read_rows",
                                          "open_rows_for_append", "run_tasks", "d_e2", "d_e1", "cal_list", "label_amp",
                                          "shift_of", "calibrate", "exam_items", "plant_examined", "verify_unplanted"],
            "what": "the diff shows which lines changed; none of the functions listed is touched by it"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=ROWS_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--test", action="store_true", help="tiny smoke test; rows/out must be outside out/")
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--measure-version", default=None,
                    help="byte copy of the first measurement run's script; its diff to this script goes in the output")
    ap.add_argument("--resume-version", default=None,
                    help="byte copy of the resuming run's script; its diff to this script goes in the output")
    args = ap.parse_args()
    rows_path, out_path = os.path.abspath(args.rows), os.path.abspath(args.out)
    if args.test:
        assert os.path.normcase(os.path.abspath(OUT)) not in (os.path.normcase(os.path.dirname(rows_path)),
                                                              os.path.normcase(os.path.dirname(out_path))), \
            "--test must write outside out/"
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists; result files are never overwritten")
    t0 = time.time()
    # ---------------------------------------------------------------- inputs and their consistency
    led = load(LEDGER)["primary"]
    assert led["U_device"] == U_DEVICE, led["U_device"]
    assert C6.R_REAL == R_REAL, C6.R_REAL
    assert load(POWER_JSON)["inputs"]["SE_img_500"] == SE500
    assert C6.TEMPL == ["K_A_E2", "K_B_E2", "K_A_E1", "K_B_E1"], C6.TEMPL
    for b in BODIES:
        for s in range(12):
            assert os.path.normcase(os.path.abspath(C6.adapter_dir(b, s))) == os.path.normcase(os.path.abspath(adapter_dir(b, s)))
            n_expected = 500 if s <= 2 else 250
            names = sorted(f for f in os.listdir(adapter_dir(b, s)) if f.endswith(".png"))
            assert names == [img_name(j) for j in range(n_expected)], (b, s, len(names))
    c6 = load(C6_JSON)
    assert c6["rows_cols"] == ["rho_KA_E2", "rho_KB_E2", "rho_KA_E1", "rho_KB_E1"]
    c6rows = c6["rows"]
    inputs = inputs_block(c6)
    log(f"script {SCRIPT} sha256 {inputs['script']['sha256']}")
    diff_block = {
        "rows_1_to_16450_first_run": (code_diff(os.path.abspath(args.measure_version), MEASURE_VERSION_SHA256,
                                                "first measurement run") if args.measure_version else None),
        "rows_16451_to_16500_resume_run": (code_diff(os.path.abspath(args.resume_version), RESUME_VERSION_SHA256,
                                                     "resume run") if args.resume_version else None)}
    log(f"inputs checked; rows file {rows_path}")
    rows, bad = read_rows(rows_path)
    n_rows_at_start = len(rows)
    log(f"rows file: {len(rows)} rows present, {bad} malformed lines skipped")
    fh, wr = open_rows_for_append(rows_path)
    with Pool(args.workers, initializer=_winit) as pool:
        verification = verify_unplanted(pool, c6rows)
        verification_planted = verify_planted(pool, rows)
        cal = calibrate(pool, rows, fh, wr, args.test)
        plant_examined(pool, rows, fh, wr, cal, args.test)
    fh.close()
    t_measure = time.time() - t0
    log(f"measurement finished in {t_measure/60:.1f} min")
    if args.test or args.measure_only:
        log("test / measure-only run: no analysis written")
        return
    rows, bad2 = read_rows(rows_path)                               # analyse exactly what is on disk
    items_cal = {b: cal_list(b, False) for b in BODIES}
    cal_a0 = {b: np.array([[d_e2(rows[("cal_a0", tag_of(b, s), img_name(j))]["rho"], b) for j in CAL_IMAGES]
                           for s in CAL_SEEDS]) for b in BODIES}
    for b in BODIES:                                                  # re-derive the calibration from disk
        for tk in TARGETS:
            sh = shift_of(rows, b, cal[b][tk]["accepted_label"], items_cal[b])[0]
            assert sh == cal[b][tk]["achieved_shift_calibration"], (b, tk, sh)
    c6d = {b: np.array([[d_e2(c6rows[tag_of(b, s)][img_name(j)], b) for j in range(N_IMG)] for s in range(12)])
           for b in BODIES}
    agreement_archive = archive_agreement(c6d, cal_a0)
    res = analyse(rows, c6rows, cal, None)
    out = {
        "entry": "RESULTS.md Entry 116, item 2 (end-to-end examiner test; review report section 4 item 4)",
        "status": ("registered model check with a fixed agreement criterion (Entry 116); computed "
                   + time.strftime("%Y-%m-%d") + "; not yet logged in RESULTS.md"),
        "script": "src/fv/fv_examiner_e2e.py",
        "question": ("does a transfer planted into real generated images at a known size come out of the paper's own "
                     "scoring and its two-candidate examiner at the rate eq. (power) predicts?"),
        "reading": res["reading"], "reading_key": res["reading_key"],
        "deviations_from_registration": [
            "file names: the registration named src/fv/fv_examiner.py -> out/fv_examiner.json and "
            "out/fv_examiner_rows.csv; this run was commissioned as fv_examiner_e2e.py -> out/fv_examiner_e2e.json "
            "and writes its rows to out/fv_examiner_e2e_rows.csv",
            "6 worker processes instead of 7 (the run's CPU allocation)",
            "starting amplitudes of the secant were not fixed by the registration: T_1x/R_real and T_10x/R_real, "
            "shared by both targets and both bodies, chosen before any planted image was scored",
            "the planted luminance is formed in float64 and stored in float32 before scoring (forming 1 + a K in "
            "float32 would quantise the modulation, about 1.5e-06 at 1 x, in steps of 1.2e-07)",
            "the measurement ran in two invocations: the first (--measure-only) stopped after 16,450 of 16,500 rows "
            "when its process ended; the second resumed from the rows file (as the registration provides) after "
            "recomputing 50 stored unplanted rows and 12 stored planted rows exactly; see checks and "
            "measurement_run_log",
            "two checks beyond the registration: stored planted rows are recomputed exactly before resuming, and the "
            "post hoc decomposition and the band at the achieved shift are reported (labelled descriptive)",
            "the analysis was first run once to a scratch file as a dry run of untested reporting code; afterwards "
            "only descriptive fields were added (agreement[*].resolution_descriptive: band and interval widths, the "
            "empirical-minus-model difference, the share of sigma_G^2 from sigma_mu = 4.1e-05). Every registered "
            "quantity is seeded and identical between the two runs; nothing that enters the reading changed",
        ],
        "settings": {"targets": TARGETS, "U_device": U_DEVICE, "R_real": R_REAL, "G": list(G_LIST), "M": list(M_LIST),
                     "subsets_per_adapter_per_G": N_SUB, "subset_seed_rule": "default_rng(20260930 + 100*g + i)",
                     "threshold": "numpy.quantile (linear) of the pooled null theta_hat_G at 1 - 0.01/(M-1)",
                     "tpr_rule": "planted theta_hat_G strictly above the threshold",
                     "n_boot": N_BOOT, "boot_seed": SEED_BOOT, "family_alpha": FAMILY_ALPHA, "n_cells": N_CELLS,
                     "secant_tolerance": TOL, "secant_max_iterations": MAX_IT, "workers": args.workers,
                     "sigma_mu_high": SIGMA_MU_HIGH, "SE_500": SE500, "N": N_IMG, "K_other": K_OTHER,
                     "verify": {"n": N_VERIFY, "tolerance": VERIFY_TOL, "seed": SEED_VERIFY},
                     "verify_planted": {"n": N_REPLANT, "tolerance": 0.0, "seed": SEED_REPLANT}},
        "inputs": inputs,
        "checks": {"unplanted_rows_recompute": verification,
                   "stored_planted_rows_recompute": verification_planted,
                   "rows_file": {"file": rows_path, "rows": len(rows), "rows_present_when_this_invocation_started":
                                 n_rows_at_start, "malformed_lines_skipped": bad2,
                                 "sha256_after_measurement": sha256(rows_path)},
                   "code_diff_from_measuring_versions": diff_block,
                   "archive_agreement": agreement_archive},
        "calibration": cal,
        **{k: v for k, v in res.items() if k not in ("reading", "reading_key")},
        "runtime_s": {"total": time.time() - t0, "measurement": t_measure,
                      "note": ("this invocation only; the rows were measured by the invocations logged under "
                               "measurement_run_log")},
        "measurement_run_log": measurement_log_summary(),
    }
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} appeared during the run; not overwritten")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log(f"reading: {res['reading']}")
    for a in res["agreement"]:
        log(f"  {a['target']:>3} G={a['G']:>3}: TPR {a['empirical_TPR']:.4f} [{a['simultaneous_interval'][0]:.4f}, "
            f"{a['simultaneous_interval'][1]:.4f}] band [{a['band'][0]:.4f}, {a['band'][1]:.4f}] {a['relation_to_band']}")
    log(f"written {out_path} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
