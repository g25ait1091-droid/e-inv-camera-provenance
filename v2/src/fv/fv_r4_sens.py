"""RESULTS.md Entry 118 (post hoc, descriptive): two sensitivities from the fourth review round.

A. The calibration multiplier c if the training-set component is larger than measured (hostile-r4-02).
   Entry 116 item 1's simulation, run as src/fv/fv_seedbank_calib.py runs it (functions imported): H6's 27 cells,
   4,000 replications, master seed 106061; the bank shift from default_rng(116001), per cell in H6's order, with the
   point Sigma_v. The training-set SD is multiplied by f in {1, 2, f95}, f95 = sqrt(2 / chi2.ppf(0.05, 2)); each f
   is run without and with the bank term on otherwise identical draws. f = 1 must reproduce c = 1.25 (no term) and
   c* = 1.31 (term) exactly.
B. One body's transfer under the max-arm limit (hostile-r4-01). eq. (power) (two candidates, FPR 0.01,
   sigma_mu = 0) at theta = U and theta = 2U, for the nominal U_device and the bound of record's U_device.
   theta = U must reproduce the printed rates.

Writes out/fv_r4_sens.json (never overwritten). CPU only; about a minute.
Run:  python src/fv/fv_r4_sens.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import os
import sys

os.environ["EINV_V2"] = EINV.V2
os.environ["H4_REP"] = "4000"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import hashlib
import json
import math
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fv_seedbank_calib as SB   # noqa: E402  (simulate_general, bank_from, search_c, counts_over_grid, apply_c)

V2 = EINV.V2
OUT = V2 + "/out/fv_r4_sens.json"
SCRIPT = os.path.abspath(__file__)


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def part_a():
    h6 = SB.load(SB.H6_JSON)
    seed = SB.load(V2 + "/out/fv_seedbank_calib.json")
    led = SB.load(SB.LEDGER)
    sd_est, sd_train = h6["estimation_sd"], h6["training_sd"]
    Sigma_v = seed["seed_bank_estimate"]["variants"]["point"]["Sigma_v"]
    f95 = math.sqrt(2.0 / stats.chi2.ppf(0.05, 2))
    factors = [("x1 (measured)", 1.0), ("x2", 2.0), ("x f95 (chi2(2) one-sided 95 % upper limit)", f95)]
    need = SB.need_count(SB.NREP)
    res = {"training_sd_measured": sd_train, "estimation_sd": sd_est, "f95": f95, "Sigma_v_point": Sigma_v,
           "n_rep": SB.NREP, "h6_master_seed": SB.H6_SEED, "bank_seed": SB.BANK_SEED, "count_needed": need,
           "c_search": "smallest c on 1.00, 1.01, ... (extended to 10.00) with worst-cell coverage >= 0.99",
           "rows": {}}
    for label, f in factors:
        cells, _ = SB.simulate_general(SB.H6_SEED, SB.NREP, sd_est, f * sd_train, bank_normals=False)
        bank_rng = np.random.default_rng(SB.BANK_SEED)
        Z = [bank_rng.standard_normal((SB.NREP, 2)) for _ in cells]
        row = {"factor": f, "training_sd": f * sd_train}
        for term, bank in (("without_seed_bank_term", None), ("with_seed_bank_term", SB.bank_from(Z, S=Sigma_v))):
            fixed = np.array([1.0, 1.25, 1.31])
            cnt = SB.counts_over_grid(cells, bank, fixed)
            w = cnt.min(0) / SB.NREP
            r = {"worst_coverage_at_1": float(w[0]), "worst_coverage_at_1.25": float(w[1]),
                 "worst_coverage_at_1.31": float(w[2])}
            try:
                grid, cg, c_star, g = SB.search_c(cells, bank, SB.NREP)
                r["c_star"] = c_star
                r["worst_coverage_at_c_star"] = float(cg.min(0)[g] / SB.NREP)
                r["limit_at_c_star"] = SB.apply_c(c_star, led)
            except RuntimeError as e:
                big = np.array([4.0, 6.0, 8.0, 10.0])
                cb = SB.counts_over_grid(cells, bank, big).min(0) / SB.NREP
                r["c_star"] = None
                r["not_reached"] = str(e)
                r["worst_coverage_at"] = {str(c): float(v) for c, v in zip(big, cb)}
            row[term] = r
            print(f"[r4] {label:44s} {term:24s} c* {r.get('c_star')}  "
                  f"lim {r.get('limit_at_c_star', {}).get('lambda_U_pct_of_R_real')}", flush=True)
        res["rows"][label] = row
    one = res["rows"]["x1 (measured)"]
    assert one["without_seed_bank_term"]["c_star"] == 1.25, one
    assert one["with_seed_bank_term"]["c_star"] == 1.31, one
    assert abs(one["without_seed_bank_term"]["worst_coverage_at_1"]
               - h6["scenarios"]["both components (headline)"]["worst_coverage_at_1"]) < 1e-12
    pt = seed["crn_run_4000"]["variants"]["point"]
    assert abs(one["with_seed_bank_term"]["worst_coverage_at_1.25"] - pt["worst_coverage_at_1.25"]) < 1e-12
    assert abs(one["with_seed_bank_term"]["limit_at_c_star"]["lambda_U_pct_of_R_real"]
               - seed["limits"]["bound_of_record"]["lambda_U_pct_of_R_real"]) < 1e-12
    res["reproduction"] = "f = 1 reproduces H6 (c = 1.25) and Entry 117 item 1 (c* = 1.31, 0.1811 %) exactly"
    return res


def part_b():
    pw = SB.load(V2 + "/out/t3_power_v4.json")["inputs"]
    seed = SB.load(V2 + "/out/fv_seedbank_calib.json")
    led = SB.load(SB.LEDGER)
    se5, R = pw["SE_img_500"], led["denominators"]["R_real"]
    z = stats.norm.ppf(0.99)

    def tpr(theta, G):
        return float(1 - stats.norm.cdf(z - theta / (se5 * math.sqrt(500.0 / G))))

    def g50(theta):
        return float(500 * se5 ** 2 / (theta / z) ** 2)   # sigma_mu = 0, target 0.5 (z_0.5 = 0)

    out = {"model": "eq. (power), two candidates, FPR 0.01, sigma_mu = 0 (t3_power_v4 inputs; exact G)",
           "SE_500": se5, "R_real": R, "limits": {}}
    for name, U in (("nominal", pw["U_device"]), ("bound_of_record", seed["limits"]["bound_of_record"]["U_device"])):
        d = {"U_device": U, "U_pct_of_R_real": 100 * U / R}
        for lab, th in (("at_U", U), ("at_2U", 2 * U)):
            d[lab] = {"theta": th, "theta_pct_of_R_real": 100 * th / R, "TPR_G500": tpr(th, 500),
                      "TPR_G5000": tpr(th, 5000), "G_for_TPR50": g50(th)}
        d["two_body_average_TPR_G500_one_at_2U_other_0"] = 0.5 * (d["at_2U"]["TPR_G500"] + 0.01)
        out["limits"][name] = d
        print(f"[r4] {name}: U {d['at_U']['TPR_G500']:.4f} / {d['at_U']['G_for_TPR50']:.0f};  "
              f"2U {d['at_2U']['TPR_G500']:.4f} / {d['at_2U']['G_for_TPR50']:.0f}", flush=True)
    n = out["limits"]["nominal"]["at_U"]
    assert abs(n["TPR_G500"] - 0.0591) < 5e-4 and abs(n["G_for_TPR50"] - 4633) < 2, n
    r = out["limits"]["bound_of_record"]["at_U"]
    an = seed["derived_at_new_limit"]["at_new_limit"]["calibrated_rows"]["zero"]["M2"]
    assert abs(r["TPR_G500"] - an["TPR_G500"]) < 1e-9 and abs(r["G_for_TPR50"] - an["G_for_TPR50"]) < 1e-3, (r, an)
    out["reproduction"] = "theta = U reproduces nAttTprZeroTwoFiveHundred / nAttZeroGFiftyExact and the Rec values"
    out["why_2U"] = ("E theta_A = (b_A - b_B) + t_A, E theta_B = -(b_A - b_B) + t_B; U >= both arm expectations, so "
                     "t_A + t_B <= 2U and, with the other body's transfer non-negative, one body's t_x <= 2U")
    return out


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists; result files are never overwritten")
    t0 = time.time()
    res = {"entry": "RESULTS.md Entry 118 (post hoc, descriptive; registered before computing)",
           "script": SCRIPT.replace("\\", "/"), "script_sha256": sha(SCRIPT),
           "inputs_sha256": {p: sha(p) for p in (SB.H6_JSON, V2 + "/out/fv_seedbank_calib.json",
                                                 V2 + "/out/t3_power_v4.json", SB.LEDGER)},
           "A_training_sd_sensitivity": part_a(), "B_one_body_worst_case": part_b()}
    res["runtime_s"] = time.time() - t0
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"[r4] wrote {OUT} ({res['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
