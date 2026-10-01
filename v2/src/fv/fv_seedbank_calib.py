"""Seed-bank calibration of the headline limit - RESULTS.md Entry 116, item 1 (review item hostile-r2-13).

CPU only, a few minutes. Reads the archive per-image rows from the Google Drive for desktop mirror (G:).
Writes out/fv_seedbank_calib.json and, while it runs, the resumable progress file
out/fv_seedbank_calib_progress.json. Refuses to overwrite an existing output file.

WHAT ENTRY 116 REGISTERS (summarised; the entry is the authority)
  Why. All 24 primary adapters generate from one 500-seed bank (770000 + j). A fresh bank would move every adapter
  of an arm by that arm's bank mean; the within-arm SD and the Welch SE do not see this term, and H6 (Entry 107)
  did not include it.
  Data. The per-image rows of the 24 primary adapters exactly as Entry 114 read them (fv_sigma.primary_rows(),
  imported), the three Drive CSVs' SHA-256 checked against out/fv_sigma.json, the adapter means asserted equal to
  FINAL_LEDGER.json primary.per_adapter_A/B. d_A = rho(K_A) - rho(K_B) on arm-A images, d_B = rho(K_B) - rho(K_A)
  on arm-B images, 12 x 500 per arm, seed j shared by all 24.
  Estimation. Entry 114's crossed model per arm, d_X[a,j] = m_X + u_Xa + v_Xj + e_Xaj, with the seed effect made
  bivariate across arms, (v_Aj, v_Bj) ~ (0, Sigma_v):
    sigma_vX^2    = (MS_seed - MS_resid) / 12, Entry 114's ANOVA estimate per arm (fv_sigma.components, imported);
    cov(v_A, v_B) = the covariance over the 500 seeds of the two arms' seed means (no image-noise correction: the
                    two arms share no adapter);
    sigma_vs^2    = var_j(s_j) - (sigma_eA^2 + sigma_eB^2) / 48, s_j = (mean_a d_A[a,j] + mean_a d_B[a,j]) / 2, the
                    symmetric seed variance (the part that enters theta_sym); its one-sided 95 % upper limit from
                    20,000 bootstrap resamples of the 500 seeds with sigma_e^2 held at its full-data value.
  Bank term of a replication: (b_A, b_B) ~ N(0, Sigma_v / 500), added to every adapter of each arm; it enters
  theta_sym as (b_A + b_B)/2 and the additive part as (b_A - b_B)/2.
  Simulation. H6 unchanged: 27 cells, 4,000 replications, master seed 106061 and the draw order of
  h6_calibrated_limit.simulate() with h4_coverage2.draw(); estimation SD 1.224e-05 and training-set SD 2.201e-05
  read from out/h6_calibrated_limit.json. The only addition is a bank shift per replication, drawn from an
  independent generator (seed 116001), so the runs with and without the term share their random numbers. Before
  the term is added the script asserts that with it set to zero H6's worst-cell coverage is reproduced exactly at
  c = 1 (0.9795) and c = 1.25 (0.9900).
  Target. The minimum over the 27 cells of max-arm coverage >= 0.99, with c on H6's grid 1.00-2.50 (step 0.01),
  extended in the same steps if 0.99 is not reached.
  Rule (point estimate of Sigma_v, the 4,000-replication common-random-number run):
    worst-cell coverage at c = 1.25 >= 0.99 -> "the seed-bank term does not change the calibration"; 0.175 %
      (c = 1.25) stays the bound of record.
    worst-cell coverage at c = 1.25 <  0.99 -> the smallest c* reaching 0.99 is applied to the ledger values exactly
      as H6 applied c (U_X = mean_X + c* t_{0.995,11} s_X / sqrt(12), U = max(U_A, U_B), / R_real); that limit becomes
      the bound of record, calibrated for adapter, fingerprint-estimation, training-set and seed-bank variation;
      H6's 0.175 % and the nominal 0.1507 % are printed beside it; the rates derived at the calibrated limit
      (Entry 112 N-A1, the calibrated rows of Entry 114) are recomputed at the new limit as derived quantities.
  Beside it, descriptive (the rule does not use them): (i) the probe's form, the symmetric part only, added as a
  shift shared by both arms; (ii) the symmetric variance at its 95 % upper limit, sigma_vX^2 kept and the covariance
  raised to match; (iii) the symmetric construction's worst-cell coverage with the term; (iv) a 20,000-replication
  rerun at c* (fresh master seed 116002) with its Monte Carlo SE.

CHOICES THE ENTRY LEAVES OPEN, MADE HERE (also written to the output under "settings")
  * Bank draws: for each of the 27 cells, in H6's cell order, z = standard normal (4000 x 2) from
    default_rng(116001); (b_A, b_B) = z L^T with L the Cholesky factor of Sigma_v / 500. The same z serve every
    variant, so the variants share random numbers too. The probe's form uses b_A = b_B = sqrt(sigma_vs^2 / 500) z_1.
  * Bootstrap seed 116003 (the entry fixes the count, 20,000, not the seed); percentiles by linear interpolation.
  * Interval ends: the registered one is the one-sided 95 % upper limit (95th percentile). The run was also asked for
    the lower end: the 5th percentile (one-sided 95 % lower limit) is used, so the two ends form a 90 % two-sided
    percentile interval; the 2.5th and 97.5th percentiles are added. At every end sigma_vX^2 is kept and the
    covariance moved to match, as the entry does for (ii).
  * Rerun (iv): master seed 116002 in H6's structure (a per-cell generator from master.integers(2**62)), 20,000
    replications per cell; the bank normals are drawn from each cell's generator after its H6 draws. Coverage at c*
    per cell with its binomial SE; the c the rerun itself would select, with and without the term, is added.
  * Not in the entry, added as descriptive: the spread of the selected c over 20 independent 4,000-replication
    runs (master seeds 116100-116119; with and without the term on the same draws); the branch the rule would take
    with H6's draws held and the bank shift redrawn from 200 other generator seeds (116200-116399); the scratch
    probe's own numbers (the symmetric term folded into the estimation SD, as seedbank_probe.py did); a
    normal-theory interval for sigma_vs^2.
  * File names: Entry 116 names src/fv/fv_seedbank.py -> out/fv_seedbank.json; this run was asked to write
    src/fv/fv_seedbank_calib.py -> out/fv_seedbank_calib.json. The method is Entry 116 item 1 unchanged.

Run:  python src/fv/fv_seedbank_calib.py   [--out PATH] [--ckpt PATH]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import os
import sys

# einv_paths (imported by the H4/H6 modules) defaults to <repo>/v2/workspace; this is the working tree, and the
# H6 replication count is fixed at the registered 4,000.
os.environ["EINV_V2"] = EINV.V2
os.environ["H4_REP"] = "4000"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
import math
import platform
import time

import numpy as np
import scipy
from scipy import stats

V2 = EINV.V2
SRC = EINV.SRC + ""
for _p in (SRC, SRC + "/fv"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import h4_coverage2 as H4            # noqa: E402  (draw(), m_obs, tc_max, K, LEDGER)
import h6_calibrated_limit as H6     # noqa: E402  (simulate(), coverage_by_c(), GRID, SEED)
import fv_sigma as FS                # noqa: E402  (primary_rows(), components(), sha256(), power helpers)

OUT = V2 + "/out"
DST_DEFAULT = OUT + "/fv_seedbank_calib.json"
CKPT_DEFAULT = OUT + "/fv_seedbank_calib_progress.json"
H6_JSON = OUT + "/h6_calibrated_limit.json"
SIGMA_JSON = OUT + "/fv_sigma.json"
DERIVED_JSON = OUT + "/fv_derived.json"
LEDGER = H4.LEDGER
SCRIPT = os.path.abspath(__file__)

J = 500                       # seeds in the bank (images per primary adapter)
NREP = 4000                   # H6's replications per cell
H6_SEED = 106061              # H6's master seed
BANK_SEED = 116001            # the independent generator of the bank shift (Entry 116)
RERUN_SEED = 116002           # (iv) fresh master seed (Entry 116)
RERUN_NREP = 20000            # (iv) replications per cell (Entry 116)
BOOT_N = 20000                # bootstrap resamples of the 500 seeds (Entry 116)
BOOT_SEED = 116003            # chosen here
SPREAD_SEEDS = list(range(116100, 116120))   # chosen here; descriptive only
ALT_BANK_SEEDS = list(range(116200, 116400)) # chosen here; descriptive only (H6's draws held, bank redrawn)
TARGET = 0.99
C_H6 = 1.25
C_STEP = 0.01
C_EXT_MAX = 10.0
H6_EXPECT = {"worst_at_1": 0.9795, "worst_at_1.25": 0.99}   # Entry 116's assertion values


# ============================================================================ small helpers
def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def need_count(n):
    """Smallest count k with k / n >= 0.99, in integers (0.99 n exactly: n is a multiple of 100 here)."""
    return -(-99 * n // 100)


class Progress:
    """Resumable progress: each finished unit is written to disk at once; a rerun with the same fingerprint (script,
    inputs, settings) reuses the finished units. A file with another fingerprint stops the run (nothing is
    deleted)."""

    def __init__(self, path, fingerprint):
        self.path, self.fp, self.units, self.resumed = path, fingerprint, {}, []
        if os.path.exists(path):
            d = load(path)
            if d.get("fingerprint") != fingerprint:
                raise SystemExit(f"[seedbank] {path} was written by another script version or other inputs; "
                                 f"move it aside or pass --ckpt")
            self.units = d.get("units", {})
            self.resumed = list(self.units)
            print(f"[seedbank] resuming: {len(self.resumed)} finished units in {path}", flush=True)

    def unit(self, name, fn):
        if name in self.units:
            return self.units[name]
        t0 = time.time()
        r = jsonable(fn())
        self.units[name] = r
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": self.fp, "script": SCRIPT, "units": self.units}, f, indent=1)
        os.replace(tmp, self.path)
        print(f"[seedbank] unit {name} done ({time.time() - t0:.1f} s)", flush=True)
        return r


# ============================================================================ simulation
H6_GRID = np.round(np.arange(1.00, 2.501, 0.01), 2)      # H6's grid, built exactly as H6 builds it


def cell_labels():
    out = []
    for sp in ("normal", "t3", "resample"):
        for mlab, m in (("observed", H4.m_obs), ("5e-5", 5e-5), ("1e-4", 1e-4)):
            for i in (0.0, 1e-5, 3e-5):
                out.append({"spread": sp, "m": mlab, "m_value": float(m), "i": i})
    return out


def simulate_general(master_seed, nrep, sd_est, sd_train, bank_normals):
    """h6_calibrated_limit.simulate() line for line, with the master seed and the replication count as arguments.
    With bank_normals, each cell's generator then draws the (nrep x 2) standard normals of the bank shift after its
    H6 draws (used for the fresh-seed reruns only; the registered run takes them from the independent generator)."""
    master = np.random.default_rng(master_seed)
    cells, Z = [], []
    for sp in ("normal", "t3", "resample"):
        for mlab, m in (("observed", H4.m_obs), ("5e-5", 5e-5), ("1e-4", 1e-4)):
            for i in (0.0, 1e-5, 3e-5):
                rng = np.random.default_rng(master.integers(2 ** 62))
                eA, eB = H4.draw(sp, rng, nrep)
                A = i + m + eA; B = i - m + eB
                if sd_train:
                    A = A + rng.normal(0, sd_train, (nrep, 1)); B = B + rng.normal(0, sd_train, (nrep, 1))
                if sd_est:
                    sh = rng.normal(0, sd_est, (nrep, 1)); A = A + sh; B = B + sh
                cells.append({"spread": sp, "m": mlab, "i": i, "A": A, "B": B})
                if bank_normals:
                    Z.append(rng.standard_normal((nrep, 2)))
    return cells, Z


def bank_from(Z, S=None, sym_var=None):
    """Bank shifts per cell: (b_A, b_B) = z L^T, L = chol(S / 500); or, for the probe's form, the symmetric part
    only, b_A = b_B = sqrt(sym_var / 500) z_1."""
    if S is not None:
        L = np.linalg.cholesky(np.asarray(S, float) / J)
        return [z @ L.T for z in Z]
    s = math.sqrt(sym_var / J)
    return [np.repeat(s * z[:, 0:1], 2, axis=1) for z in Z]


def counts_over_grid(cells, bank, grid):
    """Max-arm coverage counts, H6's construction: U = max_X (mean_X + c t_{0.995,11} s_X / sqrt 12)."""
    k, tc = H4.K, H4.tc_max
    out = np.zeros((len(cells), len(grid)), dtype=np.int64)
    for j, c in enumerate(cells):
        A, B = c["A"], c["B"]
        if bank is not None:
            A = A + bank[j][:, 0:1]; B = B + bank[j][:, 1:2]
        mA, mB = A.mean(1), B.mean(1)
        hA = tc * A.std(1, ddof=1) / np.sqrt(k); hB = tc * B.std(1, ddof=1) / np.sqrt(k)
        for g, mult in enumerate(grid):
            U = np.maximum(mA + mult * hA, mB + mult * hB)
            out[j, g] = int(np.count_nonzero(U >= c["i"]))
    return out


def search_c(cells, bank, n):
    """Smallest c on H6's grid (extended in steps of 0.01 if needed) with worst-cell coverage >= 0.99."""
    need = need_count(n)
    grid = H6_GRID.copy()
    cnt = counts_over_grid(cells, bank, grid)
    lo = 2.51
    while not np.any(cnt.min(0) >= need):
        if lo > C_EXT_MAX:
            raise RuntimeError("0.99 not reached by c = %.2f" % C_EXT_MAX)
        ext = np.round(np.arange(lo, lo + 0.495, 0.01), 2)
        cnt = np.concatenate([cnt, counts_over_grid(cells, bank, ext)], axis=1)
        grid = np.concatenate([grid, ext])
        lo = float(np.round(ext[-1] + 0.01, 2))
    g = int(np.where(cnt.min(0) >= need)[0][0])
    return grid, cnt, float(grid[g]), g


def gidx(grid, c):
    w = np.where(np.isclose(grid, c, rtol=0, atol=1e-9))[0]
    assert len(w) == 1, c
    return int(w[0])


def describe(grid, cnt, n, c_star, labels):
    worst = cnt.min(0)
    need = need_count(n)
    i1, i125, ic = gidx(grid, 1.0), gidx(grid, C_H6), gidx(grid, c_star)

    def worst_cells(g):
        return [labels[j] for j in np.where(cnt[:, g] == worst[g])[0]]

    per_cell = []
    for j, lab in enumerate(labels):
        row = dict(lab)
        for key, g in (("at_1", i1), ("at_1.25", i125), ("at_c_star", ic)):
            p = cnt[j, g] / n
            row[f"coverage_{key}"] = p
            row[f"count_{key}"] = int(cnt[j, g])
        per_cell.append(row)
    return {"c_star": c_star, "n_rep": n, "count_needed": need,
            "worst_coverage_at_1": worst[i1] / n, "worst_coverage_at_1.25": worst[i125] / n,
            "worst_coverage_at_c_star": worst[ic] / n,
            "worst_cells_at_1.25": worst_cells(i125), "worst_cells_at_c_star": worst_cells(ic),
            "worst_coverage_curve": {"c": grid.tolist(), "worst_coverage": (worst / n).tolist()},
            "per_cell": per_cell}


def apply_c(c, led):
    """The ledger application, exactly as h6_calibrated_limit.main() does it."""
    A = np.array(led["primary"]["per_adapter_A"]); B = np.array(led["primary"]["per_adapter_B"])
    R = led["denominators"]["R_real"]
    k = len(A); tc = stats.t.ppf(0.995, k - 1)
    UA = A.mean() + c * tc * A.std(ddof=1) / np.sqrt(k); UB = B.mean() + c * tc * B.std(ddof=1) / np.sqrt(k)
    U = max(UA, UB)
    return {"c": c, "U_A": float(UA), "U_B": float(UB), "U_device": float(U), "binding_arm": "A" if UA >= UB else "B",
            "U_A_pct_of_R_real": float(100 * UA / R), "U_B_pct_of_R_real": float(100 * UB / R),
            "lambda_U_pct_of_R_real": float(100 * U / R), "R_real": R, "t_0995_11": float(tc)}


def sym_coverage(cells, bank):
    """(iii) coverage of the symmetric construction (h4_coverage2.limits: theta_sym + t_{0.99, Welch df} SE); the
    max-arm limit at c = 1 from the same function is returned as a cross-check."""
    cov_s, cov_m = [], []
    for j, c in enumerate(cells):
        A, B = c["A"], c["B"]
        if bank is not None:
            A = A + bank[j][:, 0:1]; B = B + bank[j][:, 1:2]
        U, L = H4.limits(A, B)
        cov_s.append(float(np.mean(L >= c["i"]))); cov_m.append(float(np.mean(U >= c["i"])))
    return np.array(cov_s), np.array(cov_m)


# ============================================================================ estimation of the bank term
def estimate(XA, XB, fvs):
    arms, _ = FS.components({"A": XA, "B": XB}, J)
    vA, vB = arms["A"]["sigma_v2_mom"], arms["B"]["sigma_v2_mom"]
    eA, eB = arms["A"]["sigma_e2"], arms["B"]["sigma_e2"]
    pc = fvs["primary_crossed_model"]["arms"]
    for x, key in (("A", "sigma_v2_mom"), ("B", "sigma_v2_mom"), ("A", "sigma_e2"), ("B", "sigma_e2")):
        assert abs(arms[x][key] - pc[x][key]) <= 1e-12 * abs(pc[x][key]), (x, key)
    csA, csB = XA.mean(0), XB.mean(0)
    cov = float(np.cov(csA, csB, ddof=1)[0, 1])
    r_means = float(np.corrcoef(csA, csB)[0, 1])
    assert abs(r_means - fvs["seed_effect_and_image_noise"]["corr_seed_means_A_vs_B"]) < 1e-12
    s = 0.5 * (csA + csB)
    noise = (eA + eB) / 48.0
    vs = float(s.var(ddof=1) - noise)
    vs_implied = (vA + vB + 2.0 * cov) / 4.0
    assert abs(vs - vs_implied) <= 1e-9 * abs(vs), (vs, vs_implied)
    vd = (vA + vB - 2.0 * cov) / 4.0

    rng = np.random.default_rng(BOOT_SEED)
    boot = np.empty(BOOT_N)
    for b in range(BOOT_N):
        js = rng.integers(0, J, J)
        boot[b] = s[js].var(ddof=1) - noise          # sigma_e^2 held at its full-data value
    pct = {str(q): float(np.percentile(boot, q)) for q in (2.5, 5.0, 50.0, 95.0, 97.5)}

    # normal-theory check (descriptive): var(s) ~ (sigma_vs^2 + noise) chi2_499 / 499
    v = float(s.var(ddof=1))
    nt = {"one_sided_95_lower": v * (J - 1) / stats.chi2.ppf(0.95, J - 1) - noise,
          "one_sided_95_upper": v * (J - 1) / stats.chi2.ppf(0.05, J - 1) - noise,
          "two_sided_95": [v * (J - 1) / stats.chi2.ppf(0.975, J - 1) - noise,
                           v * (J - 1) / stats.chi2.ppf(0.025, J - 1) - noise]}

    def at_sym(vs_t):
        c_t = 2.0 * vs_t - 0.5 * (vA + vB)
        S = np.array([[vA, c_t], [c_t, vB]])
        ev = np.linalg.eigvalsh(S)
        return {"sigma_vs2": vs_t, "cov_vA_vB": c_t, "corr_vA_vB": c_t / math.sqrt(vA * vB),
                "Sigma_v": S.tolist(), "psd": bool(ev.min() > 0), "min_eigenvalue": float(ev.min()),
                "per_replication_sd": {"b_A": math.sqrt(vA / J), "b_B": math.sqrt(vB / J),
                                       "symmetric_(b_A+b_B)/2": math.sqrt(max(vs_t, 0.0) / J),
                                       "additive_(b_A-b_B)/2": math.sqrt((vA + vB - 2 * c_t) / 4 / J)}}

    point = at_sym(vs)
    assert abs(point["cov_vA_vB"] - cov) <= 1e-9 * abs(cov)
    point["cov_vA_vB"] = cov                       # the estimate itself (identical to rounding)
    point["Sigma_v"] = [[vA, cov], [cov, vB]]
    variants = {"point": point,
                "upper_one_sided_95 (registered, descriptive ii)": at_sym(pct["95.0"]),
                "lower_one_sided_95 (5th percentile; requested with the run)": at_sym(pct["5.0"]),
                "two_sided_95_lower (2.5th percentile; added)": at_sym(pct["2.5"]),
                "two_sided_95_upper (97.5th percentile; added)": at_sym(pct["97.5"])}
    return {
        "per_arm": {"A": {"sigma_v2": vA, "sigma_e2": eA, "MS_seed": arms["A"]["MS_seed"],
                          "MS_resid": arms["A"]["MS_resid"], "seed_mean_variance": float(csA.var(ddof=1))},
                    "B": {"sigma_v2": vB, "sigma_e2": eB, "MS_seed": arms["B"]["MS_seed"],
                          "MS_resid": arms["B"]["MS_resid"], "seed_mean_variance": float(csB.var(ddof=1))}},
        "cov_seed_means_A_B": cov, "corr_seed_means_A_B": r_means,
        "corr_v_A_v_B_implied": cov / math.sqrt(vA * vB),
        "symmetric": {"var_s": v, "image_noise_(eA+eB)/48": noise, "sigma_vs2": vs,
                      "sigma_vs2_from_Sigma_v": vs_implied, "sigma_vs": math.sqrt(vs),
                      "noise_share_of_var_s": noise / v},
        "additive_part_seed_variance_sigma_vd2": vd,
        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "resampled": "the 500 seeds (both arms together)",
                      "sigma_e2": "held at the full-data per-arm MS_resid",
                      "percentiles_sigma_vs2": pct, "mean": float(boot.mean()), "sd": float(boot.std(ddof=1)),
                      "P_le_0": float(np.mean(boot <= 0)),
                      "one_sided_95_upper_registered": pct["95.0"],
                      "one_sided_95_lower_added": pct["5.0"]},
        "normal_theory_check_sigma_vs2": nt,
        "s_moments": {"skew": float(stats.skew(s)), "excess_kurtosis": float(stats.kurtosis(s))},
        "variants": variants}


# ============================================================================ derived quantities at a new limit
def derived_rates(U_new, fvs):
    """Entry 112 N-A1 and the calibrated rows of Entry 114, recomputed at transfer U_new with fv_sigma's own power
    helpers (verify_v2.tpr() imported from the repository). The same function evaluated at H6's U_calibrated must
    reproduce the filed values; that check is returned beside the new values."""
    vv, status, summary = FS.import_verifier()
    tpr, g_needed = FS.make_power(vv)
    pw = fvs["power"]
    inp = pw["inputs"]
    sig, se500 = inp["sigma_cases"], inp["SE_img_500"]
    se_sm = fvs["seed_effect_and_image_noise"]["SE_500_seed_matched_symmetric"]["rms_over_144_pairs"]
    se24 = fvs["seed_effect_and_image_noise"]["SE_500_single_adapter"]["rms_over_24"]
    sm = pw["seed_matched_reference"]
    s_t = pw["training_set_component_persistent"]["sigma"]
    zA2 = FS.z_alpha(2)
    Jn = float(J)

    def mult50(sigma, s500, U_ref):
        return zA2 * math.sqrt(sigma ** 2 + s500 ** 2 * 500.0 / 50.0) / U_ref

    def tpr_sep(U, G, s0, s1):
        return float(stats.norm.sf((zA2 * s0 * math.sqrt(Jn / G) - U) / (s1 * math.sqrt(Jn / G))))

    def block(U):
        d = {"N_A1_two_candidates_archive_sigma_mu": {
                 k: FS.power_block(tpr, g_needed, U, sig["archive_sigma_mu"], se500)["M2"][k]
                 for k in ("TPR_G500", "TPR_G5000", "TPR_G_inf")},
             "calibrated_rows": {k: FS.power_block(tpr, g_needed, U, s, se500) for k, s in sig.items()},
             "sigma_critical": {f"M2_TPRinf_{int(t * 100)}": U / (zA2 + float(stats.norm.ppf(t))) for t in (0.5, 0.9)},
             "seed_matched_reference": FS.power_block(tpr, g_needed, U, sm["persistent_sigma"], se_sm)["M2"],
             "seed_matched_reference_at_paired_high_95":
                 FS.power_block(tpr, g_needed, U, sm["persistent_sigma_at_paired_high_95"], se_sm)["M2"],
             "training_set_component_persistent": FS.power_block(tpr, g_needed, U, s_t, se500),
             "transfer_for_TPR50_with_50_images_x": {
                 "fresh_seeds_sigma_zero": mult50(0.0, se500, U),
                 "fresh_seeds_archive_sigma_mu": mult50(sig["archive_sigma_mu"], se500, U),
                 "seed_matched_sigma_zero": mult50(0.0, se_sm, U)},
             "sensitivity_SE_500_all_24_sigma_zero_M2": FS.power_block(tpr, g_needed, U, 0.0, se24)["M2"]}
        rows = {}
        for ref, r in sm["null_specific"]["per_orientation"].items():
            rows[ref] = {"G_for_TPR50": Jn * (zA2 * r["SE_null_500"] / U) ** 2,
                         "TPR_G500": tpr_sep(U, 500.0, r["SE_null_500"], se_sm)}
        d["seed_matched_null_specific"] = {
            "per_orientation": rows,
            "mean_of_orientations": {k: float(np.mean([rows[x][k] for x in rows])) for k in ("G_for_TPR50", "TPR_G500")}}
        return d

    # reproduction at H6's calibrated U against the filed values
    U_old = inp["U_calibrated"]
    old = block(U_old)
    ref = {"N_A1": load(DERIVED_JSON)["N_A1_tpr_calibrated"]["tpr_calibrated"]}
    diffs = []

    def cmp(a, b, where):
        if a is None or b is None:
            assert a is None and b is None, where
            return
        diffs.append((abs(a - b) / max(abs(b), 1e-300), where))

    for k, kk in (("TPR_G500", "G500"), ("TPR_G5000", "G5000"), ("TPR_G_inf", "G_inf")):
        cmp(old["N_A1_two_candidates_archive_sigma_mu"][k], ref["N_A1"][kk], "N_A1." + k)
    for s_key, blk in old["calibrated_rows"].items():
        for M, r in blk.items():
            for k, v in r.items():
                cmp(v, pw["calibrated"][s_key][M][k], f"calibrated.{s_key}.{M}.{k}")
    for t in (50, 90):
        cmp(old["sigma_critical"][f"M2_TPRinf_{t}"], pw["sigma_critical"][f"calibrated_M2_TPRinf_{t}"], f"sigma_crit_{t}")
    for k, v in old["seed_matched_reference"].items():
        cmp(v, sm["calibrated"][k], "seed_matched." + k)
    for k, v in old["seed_matched_reference_at_paired_high_95"].items():
        cmp(v, sm["calibrated_at_paired_high_95"][k], "seed_matched_high." + k)
    for M, r in old["training_set_component_persistent"].items():
        for k, v in r.items():
            cmp(v, pw["training_set_component_persistent"]["calibrated"][M][k], f"training.{M}.{k}")
    tx = pw["transfer_for_TPR50_with_50_images"]
    for k, v in old["transfer_for_TPR50_with_50_images_x"].items():
        cmp(v, tx[k]["x_calibrated"], "x50." + k)
    for k, v in old["sensitivity_SE_500_all_24_sigma_zero_M2"].items():
        cmp(v, pw["sensitivity_SE_500_all_24"]["calibrated_sigma_zero_M2"][k], "se24." + k)
    for ref_o, r in old["seed_matched_null_specific"]["per_orientation"].items():
        f = sm["null_specific"]["per_orientation"][ref_o]
        cmp(r["G_for_TPR50"], f["G_for_TPR50_calibrated"], f"null.{ref_o}.G50")
        cmp(r["TPR_G500"], f["TPR_G500_calibrated"], f"null.{ref_o}.TPR500")
    mo = sm["null_specific"]["mean_of_orientations"]
    cmp(old["seed_matched_null_specific"]["mean_of_orientations"]["G_for_TPR50"], mo["G_for_TPR50_calibrated"], "null.mean.G50")
    cmp(old["seed_matched_null_specific"]["mean_of_orientations"]["TPR_G500"], mo["TPR_G500_calibrated"], "null.mean.TPR500")
    worst = max(diffs)
    assert worst[0] < 1e-9, worst
    new = block(U_new)
    return {"U_new": U_new, "U_H6_calibrated": U_old, "verifier_import": status, "verifier_summary": summary,
            "reproduction_at_H6_calibrated": {"n_values_compared": len(diffs), "max_rel_diff": worst[0],
                                              "at": worst[1]},
            "at_new_limit": new,
            "inputs": {"SE_img_500": se500, "sigma_cases": sig, "SE_500_seed_matched": se_sm,
                       "SE_500_all_24": se24, "training_sd_persistent": s_t,
                       "seed_matched_persistent_sigma": sm["persistent_sigma"],
                       "seed_matched_persistent_sigma_at_paired_high_95": sm["persistent_sigma_at_paired_high_95"],
                       "sources": ["out/fv_sigma.json power, seed_effect_and_image_noise",
                                   "out/fv_derived.json N_A1_tpr_calibrated (reproduction only)"]},
            "what": ("rates derived at the calibrated limit (Entry 112 N-A1; the calibrated rows of Entry 114 and its "
                     "addendum) recomputed at the new limit with the same functions and inputs; derived quantities")}


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DST_DEFAULT)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"[seedbank] {a.out} exists; result files are never overwritten")
    t0 = time.time()

    # ---------------------------------------------------------------- inputs
    assert H4.V2.replace("\\", "/").rstrip("/") == V2, H4.V2
    assert H4.NREP == NREP and H6.SEED == H6_SEED, (H4.NREP, H6.SEED)
    assert np.array_equal(H6.GRID, H6_GRID)
    h6 = load(H6_JSON)
    fvs = load(SIGMA_JSON)
    led = load(LEDGER)
    sd_est, sd_train = h6["estimation_sd"], h6["training_sd"]
    assert h6["n_rep"] == NREP and h6["seed"] == H6_SEED
    rows_sha = {k: v["sha256"] for k, v in fvs["data"]["rows"].items()}
    settings = {"J_seeds": J, "n_rep": NREP, "h6_master_seed": H6_SEED, "bank_seed": BANK_SEED,
                "rerun_master_seed": RERUN_SEED, "rerun_n_rep": RERUN_NREP, "bootstrap_n": BOOT_N,
                "bootstrap_seed": BOOT_SEED, "spread_master_seeds": SPREAD_SEEDS,
                "alternative_bank_seeds": [ALT_BANK_SEEDS[0], ALT_BANK_SEEDS[-1]], "target": TARGET,
                "c_grid": [1.00, 2.50, C_STEP], "c_extension_max": C_EXT_MAX, "c_H6": C_H6}
    fingerprint = hashlib.sha256(json.dumps(
        {"script": sha256_file(SCRIPT), "settings": settings, "sd_est": sd_est, "sd_train": sd_train,
         "rows_sha": rows_sha, "ledger_A": led["primary"]["per_adapter_A"],
         "ledger_B": led["primary"]["per_adapter_B"]}, sort_keys=True).encode()).hexdigest()
    prog = Progress(a.ckpt, fingerprint)
    labels = cell_labels()

    # ---------------------------------------------------------------- H6 reproduction (must give c = 1.25)
    def h6_repro():
        res = {}
        h4_est, h4_train = H4.estimation_sd()[0], H4.training_sd()[0]
        res["sd_from_h6_json"] = {"estimation_sd": sd_est, "training_sd": sd_train}
        res["sd_recomputed_by_h4_coverage2"] = {"estimation_sd": h4_est, "training_sd": h4_train,
                                                "equal": bool(h4_est == sd_est and h4_train == sd_train)}
        scen = {"both components (headline)": (sd_est, sd_train), "estimation only": (sd_est, 0.0),
                "neither (C6 reproduction)": (0.0, 0.0)}
        for name, (se_, st_) in scen.items():
            cov = H6.coverage_by_c(H6.simulate(se_, st_))
            worst = cov.min(0)
            ok = np.where(worst >= 0.99)[0]
            c = float(H6.GRID[ok[0]]) if len(ok) else None
            got = {"c": c, "worst_coverage_at_c": float(worst[ok[0]]) if len(ok) else None,
                   "worst_coverage_at_1": float(worst[0])}
            filed = h6["scenarios"][name]
            assert got == filed, (name, got, filed)
            res[name] = {"reproduced": got, "filed": filed, "equal": True}
        p = h6["primary_d200"]
        mine = apply_c(1.25, led)
        nom = apply_c(1.0, led)
        for k in ("U_A", "U_B", "U_device"):
            assert mine[k] == p[k], (k, mine[k], p[k])
        assert mine["lambda_U_pct_of_R_real"] == p["lambda_U_calibrated_pct"]
        assert nom["lambda_U_pct_of_R_real"] == p["lambda_U_nominal_pct"]
        res["primary_d200"] = {"reproduced_calibrated": mine, "reproduced_nominal": nom, "filed": p, "equal": True}
        return res

    rep = prog.unit("h6_reproduction", h6_repro)
    print("[seedbank] H6 reproduced exactly: c = 1.25, worst coverage 0.9795 at c = 1, 0.9900 at c = 1.25", flush=True)

    # H6's draws (both components), regenerated every run: identical arrays from H6's own function and ours
    cells = H6.simulate(sd_est, sd_train)
    mine, _ = simulate_general(H6_SEED, NREP, sd_est, sd_train, bank_normals=False)
    assert all(np.array_equal(x["A"], y["A"]) and np.array_equal(x["B"], y["B"]) and x["i"] == y["i"]
               for x, y in zip(cells, mine))
    cnt0 = counts_over_grid(cells, None, H6_GRID)
    assert np.array_equal(cnt0 / NREP, H6.coverage_by_c(cells))
    w0 = cnt0.min(0) / NREP
    # the registered assertion, with the bank term set to zero
    assert w0[gidx(H6_GRID, 1.0)] == H6_EXPECT["worst_at_1"] and w0[gidx(H6_GRID, 1.25)] == H6_EXPECT["worst_at_1.25"]
    del mine

    # ---------------------------------------------------------------- data and estimation of Sigma_v
    def data_check():
        got = {k: FS.sha256(p) for k, p in FS.ROWS.items()}
        assert got == rows_sha, (got, rows_sha)
        return {"files": {k: {"file": FS.ROWS[k], "sha256": got[k], "matches_fv_sigma_json": True} for k in got}}

    data = data_check()           # always re-checked (cheap), never taken from the progress file
    XA, XB, _ = FS.primary_rows()
    assert XA.shape == (12, J) and XB.shape == (12, J)
    devA = np.abs(XA.mean(1) - np.array(led["primary"]["per_adapter_A"])).max()
    devB = np.abs(XB.mean(1) - np.array(led["primary"]["per_adapter_B"])).max()
    assert max(devA, devB) < 1e-15, (devA, devB)
    data["ledger_reproduction_max_abs_diff"] = float(max(devA, devB))
    data["adapters"] = {"A": [f"A_raw_s{s}_r16" for s in range(12)], "B": [f"B_raw_s{s}_r16" for s in range(12)]}
    data["paired_contrast"] = "arm A: rho_mult(K_A) - rho_mult(K_B); arm B: rho_mult(K_B) - rho_mult(K_A)"
    data["seed_bank"] = "770000 + j, j = 0..499, shared by all 24 adapters"
    est = prog.unit("estimation", lambda: estimate(XA, XB, fvs))
    del XA, XB
    V = est["variants"]
    print(f"[seedbank] Sigma_v: sigma_vA2 {est['per_arm']['A']['sigma_v2']:.4e}, sigma_vB2 "
          f"{est['per_arm']['B']['sigma_v2']:.4e}, cov {est['cov_seed_means_A_B']:.4e}; sigma_vs2 "
          f"{est['symmetric']['sigma_vs2']:.4e} (bootstrap 5-95 %: {est['bootstrap']['percentiles_sigma_vs2']['5.0']:.4e}"
          f" - {est['bootstrap']['percentiles_sigma_vs2']['95.0']:.4e})", flush=True)

    # ---------------------------------------------------------------- the common-random-number run (4,000)
    bank_rng = np.random.default_rng(BANK_SEED)
    Z = [bank_rng.standard_normal((NREP, 2)) for _ in cells]

    def crn_variant(S=None, sym_var=None):
        bank = bank_from(Z, S=S, sym_var=sym_var)
        grid, cnt, c_star, _ = search_c(cells, bank, NREP)
        d = describe(grid, cnt, NREP, c_star, labels)
        d["limit_at_c_star"] = apply_c(c_star, led)
        return d

    def zero_unit():
        grid, cnt, c, _ = search_c(cells, None, NREP)
        return describe(grid, cnt, NREP, c, labels)
    zero = prog.unit("crn_zero_term", zero_unit)
    assert zero["c_star"] == C_H6, zero["c_star"]
    variants = {}
    for name, v in V.items():
        if not v["psd"]:
            variants[name] = {"skipped": "Sigma_v not positive definite at this end", "Sigma_v": v["Sigma_v"]}
            continue
        variants[name] = prog.unit("crn_" + name, lambda v=v: crn_variant(S=v["Sigma_v"]))
    probe_form = prog.unit("crn_probe_form_symmetric_only",
                           lambda: crn_variant(sym_var=est["symmetric"]["sigma_vs2"]))
    point = variants["point"]
    print(f"[seedbank] point Sigma_v: worst coverage {point['worst_coverage_at_1']:.4f} at c = 1, "
          f"{point['worst_coverage_at_1.25']:.4f} at c = 1.25; c* = {point['c_star']:.2f}", flush=True)

    # ---------------------------------------------------------------- the rule
    rule_value = point["worst_coverage_at_1.25"]
    changes = not (rule_value >= TARGET)
    c_star = point["c_star"]
    lim_star = apply_c(c_star, led)
    lim_h6 = apply_c(C_H6, led)
    lim_nom = apply_c(1.0, led)
    if changes:
        reading = (f"worst-cell coverage at c = 1.25 is {rule_value:.4f} < 0.99, so the seed-bank term changes the "
                   f"calibration: c* = {c_star:.2f}, and the limit {lim_star['lambda_U_pct_of_R_real']:.4f} % of R_real "
                   f"becomes the bound of record, calibrated for adapter, fingerprint-estimation, training-set and "
                   f"seed-bank variation (H6's {lim_h6['lambda_U_pct_of_R_real']:.4f} % at c = 1.25 and the nominal "
                   f"{lim_nom['lambda_U_pct_of_R_real']:.4f} % beside it)")
    else:
        reading = ("the seed-bank term does not change the calibration: worst-cell coverage at c = 1.25 is "
                   f"{rule_value:.4f} >= 0.99; 0.175 % (c = 1.25) stays the bound of record")

    # ---------------------------------------------------------------- (iii) symmetric construction
    def sym_unit():
        s0, m0 = sym_coverage(cells, None)
        s1, m1 = sym_coverage(cells, bank_from(Z, S=V["point"]["Sigma_v"]))
        i1 = gidx(H6_GRID, 1.0)
        assert np.array_equal(m0, cnt0[:, i1] / NREP)
        per = [dict(lab, coverage_without_term=float(a_), coverage_with_term=float(b_))
               for lab, a_, b_ in zip(labels, s0, s1)]
        return {"construction": "theta_sym + t_{0.99, Welch df} * Welch SE (h4_coverage2.limits), one-sided 99 %",
                "worst_without_term": float(s0.min()), "worst_with_term": float(s1.min()),
                "worst_cells_with_term": [labels[j] for j in np.where(s1 == s1.min())[0]],
                "per_cell": per, "max_arm_c1_crosscheck_equal": True}
    symc = prog.unit("symmetric_construction", sym_unit)

    # ---------------------------------------------------------------- (iv) 20,000-replication rerun
    def rerun_unit():
        c20, Z20 = simulate_general(RERUN_SEED, RERUN_NREP, sd_est, sd_train, bank_normals=True)
        b20 = bank_from(Z20, S=V["point"]["Sigma_v"])
        grid, cnt, c_sel, _ = search_c(c20, b20, RERUN_NREP)
        grid0, cnt_n, c_sel0, _ = search_c(c20, None, RERUN_NREP)
        pts = np.array([c_star, C_H6, 1.0])
        cp = counts_over_grid(c20, b20, pts)            # the three multipliers reported, whatever grid was searched
        per = []
        for j, lab in enumerate(labels):
            row = dict(lab)
            for g, key in enumerate(("at_c_star", "at_1.25", "at_1")):
                p = cp[j, g] / RERUN_NREP
                row[f"coverage_{key}"] = p
                row[f"mc_se_{key}"] = math.sqrt(p * (1 - p) / RERUN_NREP)
            per.append(row)
        wc = cp[:, 0] / RERUN_NREP
        j = int(np.argmin(wc))
        w125 = cp[:, 1] / RERUN_NREP
        j125 = int(np.argmin(w125))
        out = {"master_seed": RERUN_SEED, "n_rep": RERUN_NREP, "c_star_from_rule_run": c_star,
               "worst_coverage_at_c_star": float(wc[j]), "worst_cell_at_c_star": labels[j],
               "mc_se_worst_cell_at_c_star": math.sqrt(wc[j] * (1 - wc[j]) / RERUN_NREP),
               "worst_coverage_at_1.25": float(w125[j125]), "worst_cell_at_1.25": labels[j125],
               "mc_se_worst_cell_at_1.25": math.sqrt(w125[j125] * (1 - w125[j125]) / RERUN_NREP),
               "c_selected_by_rerun_with_term": c_sel, "c_selected_by_rerun_without_term": c_sel0,
               "worst_coverage_curve_with_term": {"c": grid.tolist(), "worst": (cnt.min(0) / RERUN_NREP).tolist()},
               "worst_coverage_curve_without_term": {"c": grid0.tolist(),
                                                     "worst": (cnt_n.min(0) / RERUN_NREP).tolist()},
               "per_cell": per,
               "note": ("the minimum over 27 cells is biased low by selection; each cell's own MC SE is given; the bank "
                        "normals come from each cell's generator after its H6 draws")}
        return out
    rerun = prog.unit("rerun_20000", rerun_unit)
    print(f"[seedbank] rerun 20000: worst {rerun['worst_coverage_at_c_star']:.4f} at c* (SE "
          f"{rerun['mc_se_worst_cell_at_c_star']:.4f}); rerun selects c {rerun['c_selected_by_rerun_with_term']:.2f} "
          f"with the term, {rerun['c_selected_by_rerun_without_term']:.2f} without", flush=True)

    # ---------------------------------------------------------------- added: spread of c over independent runs
    spread_runs = []
    for sd in SPREAD_SEEDS:
        def one(sd=sd):
            cs, Zs = simulate_general(sd, NREP, sd_est, sd_train, bank_normals=True)
            _, cw, c_with, _ = search_c(cs, bank_from(Zs, S=V["point"]["Sigma_v"]), NREP)
            _, cn, c_without, _ = search_c(cs, None, NREP)
            return {"master_seed": sd, "c_with_term": c_with, "c_without_term": c_without,
                    "worst_at_1.25_with_term": float(cw[:, gidx(H6_GRID, C_H6)].min() / NREP),
                    "worst_at_1.25_without_term": float(cn[:, gidx(H6_GRID, C_H6)].min() / NREP)}
        spread_runs.append(prog.unit(f"spread_{sd}", one))
    cw = np.array([r["c_with_term"] for r in spread_runs]); cn = np.array([r["c_without_term"] for r in spread_runs])
    spread = {"runs": spread_runs, "n_runs": len(spread_runs), "n_rep_each": NREP,
              "c_with_term": {"min": float(cw.min()), "median": float(np.median(cw)), "mean": float(cw.mean()),
                              "max": float(cw.max()), "sd": float(cw.std(ddof=1))},
              "c_without_term": {"min": float(cn.min()), "median": float(np.median(cn)), "mean": float(cn.mean()),
                                 "max": float(cn.max()), "sd": float(cn.std(ddof=1))},
              "difference_with_minus_without": {"mean": float((cw - cn).mean()), "min": float((cw - cn).min()),
                                                "max": float((cw - cn).max())},
              "share_of_runs_with_term_needing_more_than_1.25": float(np.mean(cw > C_H6)),
              "share_of_runs_without_term_needing_more_than_1.25": float(np.mean(cn > C_H6)),
              "status": "added descriptive (not in Entry 116): Monte Carlo spread of the selected multiplier"}

    # ---------------------------------------------------------------- added: the branch under other bank draws
    alt_rows = []
    for k0 in range(0, len(ALT_BANK_SEEDS), 50):
        chunk = ALT_BANK_SEEDS[k0:k0 + 50]

        def alt_chunk(chunk=chunk):
            rows = []
            for bs in chunk:
                r_ = np.random.default_rng(bs)
                Zk = [r_.standard_normal((NREP, 2)) for _ in cells]
                grid, cnt, c_k, _ = search_c(cells, bank_from(Zk, S=V["point"]["Sigma_v"]), NREP)
                rows.append({"bank_seed": bs, "worst_at_1.25": cnt[:, gidx(grid, C_H6)].min() / NREP, "c_star": c_k})
            return rows
        alt_rows += prog.unit(f"alt_bank_{chunk[0]}_{chunk[-1]}", alt_chunk)
    ca = np.array([r["c_star"] for r in alt_rows]); wa = np.array([r["worst_at_1.25"] for r in alt_rows])
    alt = {"rows": alt_rows, "n": len(alt_rows),
           "share_worst_at_1.25_below_0.99": float(np.mean(wa < TARGET)),
           "worst_at_1.25": {"min": float(wa.min()), "median": float(np.median(wa)), "max": float(wa.max())},
           "c_star": {"min": float(ca.min()), "median": float(np.median(ca)), "mean": float(ca.mean()),
                      "max": float(ca.max()), "share_equal_or_above_registered": float(np.mean(ca >= c_star))},
           "status": ("added descriptive (not in Entry 116): H6's draws held fixed (worst coverage exactly 0.9900 at "
                      "c = 1.25 without the term) and the bank shift redrawn from other generator seeds; how often the "
                      "registered rule would have read 'changes', and the c* it would have applied")}

    # ---------------------------------------------------------------- added: the scratch probe's own numbers
    def probe_unit():
        fv = fvs["seed_effect_and_image_noise"]
        eA, eB = est["per_arm"]["A"]["sigma_e2"], est["per_arm"]["B"]["sigma_e2"]
        vA, vB = est["per_arm"]["A"]["sigma_v2"], est["per_arm"]["B"]["sigma_v2"]
        se_m = fv["SE_500_seed_matched_symmetric"]["rms_over_144_pairs"]
        r = fv["corr_seed_means_A_vs_B"]
        sds = {"seed-matched pairs (probe 'matched')": math.sqrt((se_m ** 2 * J - (eA + eB) / 4) / J),
               "seed-mean correlation, not disattenuated (probe 'naive')":
                   math.sqrt((vA + vB + 2 * r * math.sqrt(vA * vB)) / 4 / J)}
        out = {}
        for name, x in sds.items():
            cov = H6.coverage_by_c(H6.simulate(math.sqrt(sd_est ** 2 + x ** 2), sd_train))
            worst = cov.min(0)
            ok = np.where(worst >= 0.99)[0]
            c = float(H6.GRID[ok[0]]) if len(ok) else None
            out[name] = {"symmetric_sd_per_replication": x, "worst_coverage_at_1": float(worst[0]),
                         "c": c, "limit": apply_c(c, led) if c is not None else None}
        out["how"] = ("as seedbank_probe.py did it: the symmetric term folded into H6's estimation shift "
                      "(SD sqrt(1.224e-05^2 + x^2)), shared by both arms, on H6's own draws; no additive part")
        return out
    probe = prog.unit("probe_reproduction", probe_unit)

    # ---------------------------------------------------------------- derived quantities at the new limit
    derived = None
    if changes:
        derived = prog.unit("derived_at_new_limit", lambda: derived_rates(lim_star["U_device"], fvs))

    # ---------------------------------------------------------------- interval ends: c and limit
    ends = {}
    for name, v in variants.items():
        if "skipped" in v:
            ends[name] = v
            continue
        ends[name] = {"sigma_vs2": V[name]["sigma_vs2"], "cov_vA_vB": V[name]["cov_vA_vB"],
                      "symmetric_sd_per_replication": V[name]["per_replication_sd"]["symmetric_(b_A+b_B)/2"],
                      "c": v["c_star"], "worst_coverage_at_1.25": v["worst_coverage_at_1.25"],
                      "lambda_U_pct_of_R_real": v["limit_at_c_star"]["lambda_U_pct_of_R_real"]}

    res = {
        "entry": "RESULTS.md Entry 116, item 1 (seed-bank calibration of the headline limit; hostile-r2-13)",
        "status": ("pre-specified construction rule (Entry 116); the point-estimate 4,000-replication run decides; "
                   "(i)-(iv) descriptive as registered; items marked 'added' are not in the entry"),
        "script": "src/fv/fv_seedbank_calib.py", "script_sha256": sha256_file(SCRIPT),
        "file_name_note": ("Entry 116 names src/fv/fv_seedbank.py -> out/fv_seedbank.json; this run was asked to write "
                           "src/fv/fv_seedbank_calib.py -> out/fv_seedbank_calib.json; the method is unchanged"),
        "inputs": {"h6_json": H6_JSON, "estimation_sd": sd_est, "training_sd": sd_train,
                   "fv_sigma_json": SIGMA_JSON, "ledger": LEDGER,
                   "ledger_per_adapter_A": led["primary"]["per_adapter_A"],
                   "ledger_per_adapter_B": led["primary"]["per_adapter_B"],
                   "R_real": led["denominators"]["R_real"], "m_obs": float(H4.m_obs),
                   "adapter_sd": {"A": float(H4.sdA), "B": float(H4.sdB)}, "t_0995_11": float(H4.tc_max),
                   "data": data},
        "settings": dict(settings, **{
            "bank_draws": ("per cell, in H6's order, z ~ N(0, I_2) (4000 x 2) from default_rng(116001); "
                           "(b_A, b_B) = z chol(Sigma_v / 500)^T; the same z for every variant"),
            "probe_form_draws": "b_A = b_B = sqrt(sigma_vs2 / 500) z_1, same z",
            "interval_ends": ("registered: one-sided 95 % upper (95th percentile); requested with the run: the lower "
                              "end, taken as the 5th percentile (one-sided 95 % lower; the two form a 90 % interval); "
                              "added: 2.5th and 97.5th; at each end sigma_vX^2 kept and the covariance moved to match"),
            "rerun_draws": ("master default_rng(116002), a per-cell generator from master.integers(2**62) as in H6; "
                            "the bank normals from that cell generator after its H6 draws"),
            "coverage_threshold": "count >= 0.99 n (3960 of 4000; 19800 of 20000)",
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}),
        "h6_reproduction": rep,
        "zero_term_check": {"worst_coverage_at_1": w0[gidx(H6_GRID, 1.0)], "worst_coverage_at_1.25":
                            w0[gidx(H6_GRID, 1.25)], "expected": H6_EXPECT, "equal": True,
                            "same_arrays_as_h6_simulate": True},
        "zero_term_run": zero,
        "seed_bank_estimate": est,
        "crn_run_4000": {"variants": variants},
        "rule": {"statistic": "worst-cell max-arm coverage at c = 1.25, point Sigma_v, 4,000-replication CRN run",
                 "value": rule_value, "threshold": TARGET, "branch": "changes" if changes else "does not change",
                 "c_star": c_star if changes else C_H6, "reading": reading},
        "limits": {"bound_of_record": lim_star if changes else lim_h6,
                   "H6_calibrated_c_1.25": lim_h6, "nominal_c_1": lim_nom,
                   "at_interval_ends": ends},
        "descriptive_registered": {
            "i_probe_form_symmetric_only": probe_form,
            "ii_symmetric_variance_at_95_upper": variants["upper_one_sided_95 (registered, descriptive ii)"],
            "iii_symmetric_construction": symc,
            "iv_rerun_20000": rerun},
        "added_descriptive": {"spread_of_c_over_independent_runs": spread,
                              "branch_under_alternative_bank_draws": alt,
                              "probe_reproduction": probe,
                              "normal_theory_interval_sigma_vs2": est["normal_theory_check_sigma_vs2"]},
        "derived_at_new_limit": derived,
        "scope": ("per-image rows analysed for the primary D200 pair only; every other design also shares one bank "
                  "and keeps its nominal limit, stated as such. The term concerns re-running the design with a fresh "
                  "bank; an examiner who generates with fresh seeds already carries the seed effect as image noise "
                  "in SE_500, so eq. (power) is unaffected (Entry 116)"),
        "progress_file": a.ckpt, "resumed_units": prog.resumed,
        "runtime_s": time.time() - t0}
    lo_key, up_key = ("lower_one_sided_95 (5th percentile; requested with the run)",
                      "upper_one_sided_95 (registered, descriptive ii)")
    res["summary"] = {
        "reading": reading,
        "rule_worst_coverage_at_1.25": rule_value, "branch": res["rule"]["branch"], "c_star": c_star,
        "limit_pct_of_R_real": {"bound_of_record": res["limits"]["bound_of_record"]["lambda_U_pct_of_R_real"],
                                "H6_c_1.25": lim_h6["lambda_U_pct_of_R_real"],
                                "nominal_c_1": lim_nom["lambda_U_pct_of_R_real"]},
        "sigma_vs2": {"point": est["symmetric"]["sigma_vs2"],
                      "bootstrap_5th": est["bootstrap"]["percentiles_sigma_vs2"]["5.0"],
                      "bootstrap_95th_registered_upper": est["bootstrap"]["percentiles_sigma_vs2"]["95.0"]},
        "c_at_interval_ends": {"lower_5th": ends[lo_key].get("c"), "upper_95th": ends[up_key].get("c")},
        "limit_at_interval_ends_pct": {"lower_5th": ends[lo_key].get("lambda_U_pct_of_R_real"),
                                       "upper_95th": ends[up_key].get("lambda_U_pct_of_R_real")},
        "i_probe_form_c": probe_form["c_star"],
        "i_probe_form_limit_pct": probe_form["limit_at_c_star"]["lambda_U_pct_of_R_real"],
        "iii_symmetric_worst_coverage": {"without_term": symc["worst_without_term"], "with_term": symc["worst_with_term"]},
        "iv_rerun_worst_at_c_star": rerun["worst_coverage_at_c_star"], "iv_rerun_mc_se": rerun["mc_se_worst_cell_at_c_star"],
        "iv_rerun_worst_at_1.25": rerun["worst_coverage_at_1.25"],
        "iv_rerun_c_with_term": rerun["c_selected_by_rerun_with_term"],
        "iv_rerun_c_without_term": rerun["c_selected_by_rerun_without_term"],
        "added_spread_c_with_term_median": spread["c_with_term"]["median"],
        "added_spread_c_without_term_median": spread["c_without_term"]["median"],
        "added_alt_bank_share_changes": alt["share_worst_at_1.25_below_0.99"]}
    res = jsonable(res)
    if os.path.exists(a.out):
        raise SystemExit(f"[seedbank] {a.out} appeared during the run; not overwritten")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"[seedbank] READING: {reading}", flush=True)
    print(f"[seedbank] written {a.out} ({res['runtime_s']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
