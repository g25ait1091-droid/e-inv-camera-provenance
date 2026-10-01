"""Number macros, part n3: groups Lim (primary limit, constructions, coverage, calibration, pooling,
estimator sensitivity) and Main (the shared fingerprint main effect).

Every value is read from a result file at run time (FINAL_LEDGER.json, out/*.json) or computed from one
here; the few quantities whose only source is RESULTS.md text carry check='text-only'.
Percentages marked "of R_real" use the D200 E2 denominator 0.0356703 unless the check says otherwise
(each device pair, estimator and bootstrap replicate is expressed on its own real-image contrast).
Run alone:  python src/fv/num_n3.py   (prints the table); built by src/fv/fv_numbers.py.
"""
import itertools
import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_numlib import (LEDGER, OUT, R_REAL, V2, dec, get, integer, load, macro, sig, sci)  # noqa: E402

RES = "RESULTS.md"


def _p(x, n=2):
    """p-value, 2 significant figures (3 decimals max unless the value is below 0.001)."""
    return sig(x, n)


def macros():
    M = []

    def add(name, value, text, source, key, entry, check, meaning):
        d = macro(name, value, text, source, key, entry, check)
        d["meaning"] = meaning
        M.append(d)

    # ------------------------------------------------------------------ primary ledger (D200, k = 12)
    L = load(LEDGER)
    P = L["primary"]
    den = L["denominators"]
    R = den["R_real"]
    assert abs(R - R_REAL) < 1e-15
    A = np.array(P["per_adapter_A"], float)   # seeds 0-11 in numeric order (ledger note)
    B = np.array(P["per_adapter_B"], float)
    k = len(A)
    assert k == len(B) == P["k_per_arm"] == 12

    add("nLimRreal", R, sig(R, 3), LEDGER, "denominators.R_real", "00",
        "matches RESULTS Entry 00 text 3.56703e-02 and CONVENTIONS 0.0356703",
        "R_real, real-image device contrast of the D200 pair (E2 estimate, 40 held-out photographs per body)")
    add("nLimRrealFull", R, dec(R, 7), LEDGER, "denominators.R_real", "00",
        "matches RESULTS Entry 00 text 3.56703e-02", "R_real printed to 7 decimals (0.0356703)")
    add("nLimRrealLowerNinetyNine", den["R_real_lower99"], sig(den["R_real_lower99"], 3), LEDGER,
        "denominators.R_real_lower99", "00", "ledger only; used by the conservative variant (v4 Table)",
        "lower 99 % confidence limit of R_real")

    add("nLimAdaptersPerArm", k, integer(k), LEDGER, "primary.k_per_arm", "00",
        "len(per_adapter_A) == len(per_adapter_B) == 12", "adapters per arm in the primary design")
    c6s = load("c6_estimator_swap.json")
    nimg = sum(v["v1_n"] for v in c6s["v1_rows_check"]["adapters"].values())
    nper = {v["v1_n"] for v in c6s["v1_rows_check"]["adapters"].values()}
    assert nper == {500} and len(c6s["v1_rows_check"]["adapters"]) == 24
    add("nLimPrimaryImagesPerAdapter", 500, integer(500), OUT + "/c6_estimator_swap.json",
        "v1_rows_check.adapters.*.v1_n", "63", "every ledger adapter used 500 generations (Entry 63)",
        "generations scored per primary adapter")
    add("nLimPrimaryImages", nimg, integer(nimg), OUT + "/c6_estimator_swap.json",
        "sum(v1_rows_check.adapters.*.v1_n)", "50",
        "24 x 500 = 12,000; matches RESULTS Entry 50 '12,000 in the primary study'",
        "generations behind the primary limit (24 adapters x 500)")

    tcrit = stats.t.ppf(0.995, k - 1)
    assert abs(tcrit - P["t_crit_0995"]) < 1e-3
    add("nLimTcrit", tcrit, dec(tcrit, 3), LEDGER, "primary.t_crit_0995 (= t_0.995,11)", "00",
        "scipy t.ppf(0.995, 11) = 3.1058 matches ledger", "per-arm critical value t_0.995 with 11 df")

    def U(x, mult=1.0):
        return x.mean() + mult * stats.t.ppf(0.995, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))

    UA, UB = U(A), U(B)
    assert abs(UA - P["U_A"]) < 1e-8 and abs(UB - P["U_B"]) < 1e-9
    Udev = max(UA, UB)
    add("nLimUdevice", Udev, sci(Udev, 3), LEDGER, "primary.U_device (recomputed from per_adapter_A/B)",
        "00", "recomputed 5.3761e-05 matches ledger and RESULTS Entry 00", "U_device, nominal max-arm limit in NCC units")
    add("nLimUA", UA, sci(UA, 3), LEDGER, "primary.U_A (recomputed)", "00",
        "recomputed matches ledger 3.384e-05", "per-arm 99.5 % limit, arm A, NCC units")
    add("nLimUB", UB, sci(UB, 3), LEDGER, "primary.U_B (recomputed)", "00",
        "recomputed matches ledger 5.3761e-05", "per-arm 99.5 % limit, arm B, NCC units")
    add("nLimUAPct", 100 * UA / R, sig(100 * UA / R, 3), LEDGER, "100*U_A/R_real", "00",
        "% of R_real (E2 0.0356703)", "arm A limit as % of R_real")
    add("nLimUBPct", 100 * UB / R, sig(100 * UB / R, 3), LEDGER, "100*U_B/R_real", "00",
        "% of R_real (E2 0.0356703); equals the nominal limit (arm B is the maximum)", "arm B limit as % of R_real")

    lam_nom = 100 * Udev / R
    assert abs(lam_nom - P["lambda_U_plugin_pct"]) < 1e-4
    add("nLimNom", lam_nom, sig(lam_nom, 3), LEDGER, "primary.lambda_U_plugin_pct", "00",
        "matches RESULTS Entry 00 0.1507 %, verify_v2 check 65 input, FINDINGS 'Primary max-arm limit'; % of R_real (E2)",
        "nominal max-arm one-sided 99 % limit, 12 adapters per arm, E2, 12,000 images")
    add("nLimNomFourDigit", lam_nom, dec(lam_nom, 4), LEDGER, "primary.lambda_U_plugin_pct", "00",
        "matches RESULTS Entry 00 text 0.1507 %; % of R_real (E2)", "nominal max-arm limit to 4 decimals")
    tau = P["tau_U_plugin_pct"]
    assert abs(tau - 100 * Udev / den["R_VAE"]) < 1e-3
    add("nLimTauU", tau, sig(tau, 3), LEDGER, "primary.tau_U_plugin_pct (= U_device/R_VAE)", "00",
        "matches RESULTS Entry 00 0.4117 %; % of R_VAE (post-autoencoder contrast), not of R_real",
        "tau_U, nominal limit as % of the contrast surviving the autoencoder")
    add("nLimTauUFourDigit", tau, dec(tau, 4), LEDGER, "primary.tau_U_plugin_pct", "00",
        "matches RESULTS Entry 00 0.4117 %; % of R_VAE", "tau_U to 4 decimals")
    cons = P["lambda_U_conservative_pct"]
    assert abs(cons - 100 * Udev / den["R_real_lower99"]) < 1e-3
    add("nLimNomConservative", cons, sig(cons, 3), LEDGER, "primary.lambda_U_conservative_pct", "00",
        "U_device / lower-99 % R_real; ledger and v4 table 0.1681 %; not in RESULTS text",
        "conservative variant of the nominal limit (divides by the lower 99 % limit of R_real)")

    th_sym = 0.5 * (A.mean() + B.mean())
    assert abs(th_sym - P["theta_sym"]) < 1e-9
    add("nLimThetaSym", th_sym, sci(th_sym, 3), LEDGER, "primary.theta_sym (recomputed)", "00",
        "matches RESULTS Entry 00 +4.5996e-06, verify_v2 check 1", "theta_sym of the primary design, NCC units (positive)")
    lam_hat = 100 * th_sym / R
    assert abs(lam_hat - P["lambda_hat_pct"]) < 1e-5
    add("nLimPoint", lam_hat, sig(lam_hat, 3), LEDGER, "primary.lambda_hat_pct", "107",
        "matches RESULTS Entry 107 table +0.0129 %; % of R_real (E2)", "point estimate lambda-hat of the primary design")
    add("nLimThetaAMean", A.mean(), sci(A.mean(), 3), LEDGER, "mean(primary.per_adapter_A)", "00",
        "matches ledger theta_A_mean -9.5699e-06", "arm A mean own-body contrast")
    add("nLimThetaBMean", B.mean(), sci(B.mean(), 3), LEDGER, "mean(primary.per_adapter_B)", "00",
        "matches ledger theta_B_mean 1.8769e-05", "arm B mean own-body contrast")
    add("nLimThetaASd", A.std(ddof=1), sci(A.std(ddof=1), 3), LEDGER, "sd(primary.per_adapter_A)", "00",
        "matches ledger theta_A_sd 4.842e-05 and Entry 63 SD_A 4.84e-5", "adapter SD, arm A")
    add("nLimThetaBSd", B.std(ddof=1), sci(B.std(ddof=1), 3), LEDGER, "sd(primary.per_adapter_B)", "00",
        "matches ledger theta_B_sd 3.903e-05 and Entry 63 SD_B 3.90e-5", "adapter SD, arm B")

    signs = np.array(list(itertools.product([1, -1], repeat=k)))

    def signflip(x):
        d = (signs * x).mean(1)
        c = int((d >= x.mean() - 1e-15).sum())
        return c, c / len(signs)

    cA, pA = signflip(A)
    cB, pB = signflip(B)
    assert abs(pA - P["exact_p_A"]) < 1e-4 and abs(pB - P["exact_p_B"]) < 1e-4
    add("nLimSignFlipPA", pA, dec(pA, 4), LEDGER, "primary.exact_p_A (recomputed, 2^12 sign flips)", "00",
        f"exact enumeration {cA}/4096 matches ledger 0.7546; printed exact to 4 decimals", "registered sign-flip p, arm A")
    add("nLimSignFlipPB", pB, dec(pB, 4), LEDGER, "primary.exact_p_B (recomputed, 2^12 sign flips)", "00",
        f"exact enumeration {cB}/4096 matches ledger 0.0625", "registered sign-flip p, arm B")
    add("nLimSignFlipCountA", cA, integer(cA), LEDGER, "count of 4096 sign assignments >= observed, arm A",
        "00", "p_A x 4096", "sign-flip count, arm A (of 4096)")
    add("nLimSignFlipCountB", cB, integer(cB), LEDGER, "count of 4096 sign assignments >= observed, arm B",
        "00", "p_B x 4096", "sign-flip count, arm B (of 4096)")
    floor = P["signflip_floor"]
    assert floor == 1 / 4096
    add("nLimSignFlipFloor", floor, "1/4096", LEDGER, "primary.signflip_floor", "00",
        "= 2^-12, attainable floor of the exact sign-flip test at k = 12", "sign-flip floor as a fraction")
    add("nLimSignFlipFloorDec", floor, sig(floor, 2), LEDGER, "primary.signflip_floor", "00",
        "1/4096 = 0.000244", "sign-flip floor as a decimal")
    add("nLimSignFlipAssignments", 4096, integer(4096), LEDGER, "2**k_per_arm", "00", "2^12",
        "number of sign assignments enumerated")
    spA, spB = stats.shapiro(A).pvalue, stats.shapiro(B).pvalue
    add("nLimShapiroA", spA, dec(spA, 3), LEDGER, "shapiro(primary.per_adapter_A)", "00",
        f"ledger shapiro_p_A {P['shapiro_p_A']}", "Shapiro-Wilk p, arm A")
    add("nLimShapiroB", spB, dec(spB, 3), LEDGER, "shapiro(primary.per_adapter_B)", "00",
        f"ledger shapiro_p_B {P['shapiro_p_B']}", "Shapiro-Wilk p, arm B")

    # leave-one-out (most influential adapter per arm)
    def loo(x):
        u = U(x)
        lo = [U(np.delete(x, i)) for i in range(len(x))]
        return (min(lo) - u) / u

    add("nLimLooUA", -100 * loo(A), sig(-100 * loo(A), 2), LEDGER, "leave-one-out min U_A vs U_A", "00",
        "recomputed 61 % matches v4 s5_results text; % change of U_A, not of R_real",
        "reduction of U_A when its most influential adapter is removed, %")
    add("nLimLooUB", -100 * loo(B), sig(-100 * loo(B), 2), LEDGER, "leave-one-out min U_B vs U_B", "00",
        "recomputed 15 % matches v4 s5_results text; % change of U_B", "reduction of U_B when its most influential adapter is removed, %")

    # tightening history k = 3, 6, 12
    l6 = 100 * max(U(A[:6]), U(B[:6])) / R
    l3 = 100 * max(U(A[:3]), U(B[:3])) / R
    reps = {r["name"]: r for r in L["replications"]}
    assert abs(l6 - reps["Primary at k=6"]["lambda_U_pct"]) < 1e-3 and abs(l3 - reps["Primary at matched n=3"]["lambda_U_pct"]) < 1e-3
    add("nLimNomKsix", l6, sig(l6, 3), LEDGER, "max-arm over seeds 0-5 (= replications 'Primary at k=6')", "00",
        "matches ledger 0.249 and history_k6; FINDINGS 0.25 %; % of R_real (E2)", "max-arm limit at six adapters per arm (pre-specified k)")
    add("nLimNomKthree", l3, sig(l3, 3), LEDGER, "max-arm over seeds 0-2 (= replications 'Primary at matched n=3')", "00",
        "matches ledger 0.6993, FINDINGS 0.70 %; % of R_real (E2)", "max-arm limit at three adapters per arm")
    add("nLimThetaAMeanKsix", A[:6].mean(), sci(A[:6].mean(), 3), LEDGER, "mean(per_adapter_A[0:6])", "00",
        "matches v4 text +1.55e-5", "arm A mean at six adapters per arm")

    # symmetric statistic, primary design
    va, vb = A.var(ddof=1) / k, B.var(ddof=1) / k
    se = 0.5 * math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (k - 1) + vb ** 2 / (k - 1))
    t = th_sym / se
    psym = stats.t.sf(t, df)
    lim99 = th_sym + stats.t.ppf(0.99, df) * se
    lim995 = th_sym + stats.t.ppf(0.995, df) * se
    add("nLimSymSE", se, sci(se, 3), LEDGER, "Welch SE of theta_sym from per_adapter_A/B", "50",
        "matches RESULTS Entry 50 Welch SE 8.98e-06", "Welch SE of theta_sym, primary")
    add("nLimSymDf", df, integer(df), LEDGER, "Welch df", "50", "Entry 50 df 21.0 (v4_offline 21.05); printed as an integer (v4 text: 21 degrees of freedom)",
        "Welch degrees of freedom, primary symmetric statistic")
    add("nLimSymT", t, dec(t, 2), LEDGER, "theta_sym / SE", "50",
        "c6_estimator_swap ledger_values t 0.512", "t of the primary symmetric statistic")
    add("nLimSymP", psym, _p(psym), LEDGER, "one-sided Welch p", "50",
        "c6_estimator_swap ledger_values one_sided_p 0.307", "one-sided p of the primary symmetric statistic")
    add("nLimSymLimRaw", lim99, sci(lim99, 3), LEDGER, "theta_sym + t_0.99,df * SE (NCC units)", "50",
        "matches RESULTS Entry 50 2.72e-05", "symmetric one-sided 99 % limit, NCC units")
    add("nLimSym", 100 * lim99 / R, sig(100 * lim99 / R, 2), LEDGER, "100*(theta_sym + t_0.99 SE)/R_real", "50",
        "matches RESULTS Entry 50 0.076 %, verify_v2 check 2 (0.0762); % of R_real (E2); 500 images per adapter",
        "symmetric one-sided 99 % limit, primary, E2 (secondary construction)")
    add("nLimSymFull", 100 * lim99 / R, sig(100 * lim99 / R, 3), LEDGER, "same", "50",
        "verify_v2 check 2: 0.0762; % of R_real (E2)", "symmetric limit to 3 significant figures")
    add("nLimSymNinetyNineFive", 100 * lim995 / R, sig(100 * lim995 / R, 3), LEDGER,
        "100*(theta_sym + t_0.995 SE)/R_real", "50", "matches RESULTS Entry 50 99.5 %: 0.084 %; % of R_real (E2)",
        "symmetric one-sided 99.5 % limit")
    tau_sym = 100 * lim99 / den["R_VAE"]
    add("nLimSymTau", tau_sym, sig(tau_sym, 3), LEDGER, "100*symmetric limit/R_VAE", "50",
        "v4 table 0.2083 %; % of R_VAE not R_real", "symmetric limit as % of the post-autoencoder contrast")
    v4o = load("v4_offline.json")
    for lp, nm in ((0.06, "Six"), (0.08, "Eight"), (0.1, "Ten"), (0.12, "Twelve")):
        pw = [r["power"] for r in v4o["symmetric"]["power_curve"] if abs(r["lambda_pct"] - lp) < 1e-9][0]
        add(f"nLimSymPower{nm}", pw, dec(pw, 2), OUT + "/v4_offline.json",
            f"symmetric.power_curve[lambda_pct={lp}].power", "50",
            "matches RESULTS Entry 50" + ("; verify_v2 check 3 (0.92)" if nm == "Ten" else ""),
            f"power of the one-sided 0.01 symmetric test at lambda = {lp} % of R_real")

    # instrument calibration (injected fingerprint in generated images)
    cal = L["calibration"]
    lv = [x for x in cal["levels"] if abs(x["alpha"] - 0.02) < 1e-12][0]
    add("nLimInjSlope", cal["slope_quantised"], sci(cal["slope_quantised"], 4), LEDGER, "calibration.slope_quantised",
        "00", "v1 ledger quantity; v4 text 1.414e-2", "calibration slope of contrast on injection coefficient")
    add("nLimInjSlopeLower", cal["slope_lower99"], sci(cal["slope_lower99"], 4), LEDGER, "calibration.slope_lower99",
        "00", "v4 text 1.361e-2", "lower 99 % limit of the calibration slope")
    add("nLimInjImages", cal["n"], integer(cal["n"]), LEDGER, "calibration.n", "00", "v4 text 2500",
        "images in the injection calibration")
    add("nLimInjT", lv["t"], dec(lv["t"], 2), LEDGER, "calibration.levels[alpha=0.02].t", "00", "v4 text t = 6.76",
        "t of detection at injection coefficient 0.02")
    inj_pct = 100 * 0.02 * cal["slope_quantised"] / R
    add("nLimInjPct", inj_pct, sig(inj_pct, 2), LEDGER, "100*0.02*slope_quantised/R_real", "00",
        "v4 text 0.79 %; % of R_real (E2)", "contrast at the smallest detected coefficient, % of R_real")
    add("nLimInjTimesLimit", inj_pct / lam_nom, sig(inj_pct / lam_nom, 2), LEDGER, "inj_pct / nominal limit", "00",
        "v4 text 5.3 times", "detected injection level as a multiple of the nominal limit")

    # ------------------------------------------------------------------ H6 calibrated headline
    h6 = load("h6_calibrated_limit.json")
    pd6 = h6["primary_d200"]
    c = pd6["c"]
    lam_cal = 100 * max(U(A, c), U(B, c)) / R
    assert abs(lam_cal - pd6["lambda_U_calibrated_pct"]) < 1e-6
    add("nLimCal", lam_cal, sig(lam_cal, 3), OUT + "/h6_calibrated_limit.json", "primary_d200.lambda_U_calibrated_pct",
        "107", "matches RESULTS Entry 107 0.1752 % / brief 0.175 %, verify_v2 check 65; % of R_real (E2)",
        "headline: calibrated max-arm one-sided 99 % limit (c = 1.25), E2, 12,000 images")
    add("nLimCalFourDigit", lam_cal, dec(lam_cal, 4), OUT + "/h6_calibrated_limit.json",
        "primary_d200.lambda_U_calibrated_pct", "107", "matches RESULTS Entry 107 0.1752 %", "calibrated limit to 4 decimals")
    add("nLimCalC", c, dec(c, 2), OUT + "/h6_calibrated_limit.json", "primary_d200.c", "107",
        "matches RESULTS Entry 107 c = 1.25, verify_v2 check 64", "calibration multiplier c on the max-arm half-width")
    add("nLimCalUA", pd6["U_A"], sci(pd6["U_A"], 3), OUT + "/h6_calibrated_limit.json", "primary_d200.U_A", "107",
        "recomputed U(A, c) matches", "calibrated per-arm limit, arm A, NCC units")
    add("nLimCalUB", pd6["U_B"], sci(pd6["U_B"], 3), OUT + "/h6_calibrated_limit.json", "primary_d200.U_B", "107",
        "recomputed U(B, c) matches", "calibrated per-arm limit, arm B, NCC units")
    add("nLimCalUAPct", 100 * pd6["U_A"] / R, sig(100 * pd6["U_A"] / R, 3), OUT + "/h6_calibrated_limit.json",
        "100*primary_d200.U_A/R_real", "107", "% of R_real (E2)", "calibrated arm A limit as % of R_real")
    sc = h6["scenarios"]
    w1 = sc["both components (headline)"]["worst_coverage_at_1"]
    add("nLimCalCovAtOne", w1, dec(w1, 4), OUT + "/h6_calibrated_limit.json",
        "scenarios.both components (headline).worst_coverage_at_1", "107",
        "matches RESULTS Entry 107 0.9795 (Entry 97's 0.975 on a different draw)",
        "H6 worst-cell max-arm coverage at c = 1 with both components")
    wc = sc["both components (headline)"]["worst_coverage_at_c"]
    add("nLimCalCovAtC", wc, dec(wc, 3), OUT + "/h6_calibrated_limit.json",
        "scenarios.both components (headline).worst_coverage_at_c", "107", "= 0.990 by construction",
        "worst-cell coverage at c = 1.25")
    we = sc["estimation only"]["worst_coverage_at_1"]
    add("nLimCalCovEstOnly", we, dec(we, 3), OUT + "/h6_calibrated_limit.json",
        "scenarios.estimation only.worst_coverage_at_1", "107", "matches RESULTS Entry 107 0.995 (c = 1.00 suffices)",
        "H6 max-arm coverage at c = 1 with estimation error only")
    add("nLimCalCestOnly", sc["estimation only"]["c"], dec(sc["estimation only"]["c"], 2),
        OUT + "/h6_calibrated_limit.json", "scenarios.estimation only.c", "107", "Entry 107: c = 1.00 already suffices",
        "multiplier needed with estimation error alone")
    add("nLimCalReps", h6["n_rep"], integer(h6["n_rep"]), OUT + "/h6_calibrated_limit.json", "n_rep", "106",
        "Entry 106: 4,000 replications per cell", "Monte Carlo replications per cell (H6/H4)")

    # ------------------------------------------------------------------ H4 coverage (Entries 93, 97)
    h4 = load("h4_coverage2.json")
    wcase = h4["worst_case"]
    rows = h4["rows"]

    def worst(est, tr, key):
        return min(r[key] for r in rows if r["estimation"] == est and r["training"] == tr)

    ncell = wcase["headline"]["n_cells"]
    add("nLimCovCells", ncell, integer(ncell), OUT + "/h4_coverage2.json", "worst_case.headline.n_cells", "97",
        "27 scenario cells (3 spreads x 3 main effects x 3 interactions)", "scenario cells in the coverage simulation")
    hs, hm = wcase["headline"]["min_coverage_sym"], wcase["headline"]["min_coverage_maxarm"]
    assert abs(hs - worst("on", "G5", "coverage_sym")) < 1e-12
    add("nLimCovSym", hs, dec(hs, 3), OUT + "/h4_coverage2.json", "worst_case.headline.min_coverage_sym", "97",
        "matches RESULTS Entry 97 0.8120, verify_v2 check 57", "worst-cell coverage of the nominal 99 % symmetric limit, both components measured")
    add("nLimCovMax", hm, dec(hm, 3), OUT + "/h4_coverage2.json", "worst_case.headline.min_coverage_maxarm", "97",
        "matches RESULTS Entry 97 0.9750, verify_v2 check 58", "worst-cell coverage of the nominal 99 % max-arm limit, both components measured")
    es, em = wcase["estimation_only"]["min_coverage_sym"], wcase["estimation_only"]["min_coverage_maxarm"]
    add("nLimCovEstSym", es, dec(es, 4), OUT + "/h4_coverage2.json", "worst_case.estimation_only.min_coverage_sym", "97",
        "file 0.89225 = Entry 97 0.8922; Entry 93 and brief print 0.8875 (earlier draw) -- file wins",
        "worst-cell symmetric coverage with estimation error only")
    add("nLimCovEstMax", em, dec(em, 4), OUT + "/h4_coverage2.json", "worst_case.estimation_only.min_coverage_maxarm", "97",
        "file 0.9965 = FINDINGS; Entries 93 and 97 and brief print 0.9955 -- file wins",
        "worst-cell max-arm coverage with estimation error only")
    cs, cm_ = wcase["c6_reproduction"]["min_coverage_sym"], wcase["c6_reproduction"]["min_coverage_maxarm"]
    add("nLimCovNeitherSym", cs, dec(cs, 3), OUT + "/h4_coverage2.json", "worst_case.c6_reproduction.min_coverage_sym",
        "97", "matches Entry 97 0.9862", "worst-cell symmetric coverage, neither component (C6 reproduction)")
    add("nLimCovNeitherMax", cm_, dec(cm_, 3), OUT + "/h4_coverage2.json", "worst_case.c6_reproduction.min_coverage_maxarm",
        "97", "matches Entry 97 0.9998", "worst-cell max-arm coverage, neither component")
    for tr, nm in (("0.5xSD", "Half"), ("1xSD", "One"), ("2xSD", "Two")):
        s_, m_ = worst("on", tr, "coverage_sym"), worst("on", tr, "coverage_maxarm")
        add(f"nLimCovSweep{nm}Sym", s_, dec(s_, 3), OUT + "/h4_coverage2.json",
            f"min(rows[estimation=on, training={tr}].coverage_sym)", "93",
            "file (re-run) differs slightly from Entry 93's first draw (0.8155 / 0.7165 / 0.6178) -- file wins",
            f"sensitivity sweep: symmetric coverage, estimation + training-set SD at {tr}")
        add(f"nLimCovSweep{nm}Max", m_, dec(m_, 3), OUT + "/h4_coverage2.json",
            f"min(rows[estimation=on, training={tr}].coverage_maxarm)", "93",
            "file (re-run) differs slightly from Entry 93's first draw (0.9755 / 0.9347 / 0.8752) -- file wins",
            f"sensitivity sweep: max-arm coverage, estimation + training-set SD at {tr}")
    esd, tsd = h4["estimation_sd"], h4["training_sd"]
    add("nLimEstSD", esd, sci(esd, 3), OUT + "/h4_coverage2.json", "estimation_sd", "93",
        "matches RESULTS Entry 93 SD 1.224e-05 (from H2)", "fingerprint-estimation SD (NCC units)")
    add("nLimEstSDRatio", h4["estimation_sd_in_adapter_sd"], dec(h4["estimation_sd_in_adapter_sd"], 2),
        OUT + "/h4_coverage2.json", "estimation_sd_in_adapter_sd", "93", "matches Entry 93 0.28 of the adapter SD",
        "estimation SD as a fraction of the adapter SD")
    add("nLimTrainSD", tsd, sci(tsd, 3), OUT + "/h4_coverage2.json", "training_sd", "97",
        "matches RESULTS Entry 97 2.201e-05", "training-set SD pinned from G5 (NCC units)")
    tr_ratio = tsd / h4["adapter_sd"]["mean"]
    add("nLimTrainSDRatio", tr_ratio, dec(tr_ratio, 2), OUT + "/h4_coverage2.json", "training_sd/adapter_sd.mean", "97",
        "matches Entry 97 0.50 of the adapter SD", "training-set SD as a fraction of the adapter SD")
    td = h4["training_detail"]
    add("nLimTrainShiftA", td["shift_A"], sci(td["shift_A"], 3), OUT + "/h4_coverage2.json", "training_detail.shift_A",
        "97", "matches Entry 97 -4.22e-05", "shift of body A between its two training sets")
    add("nLimTrainShiftB", td["shift_B"], sci(td["shift_B"], 3), OUT + "/h4_coverage2.json", "training_detail.shift_B",
        "97", "matches Entry 97 +2.31e-05", "shift of body B between its two training sets")

    # C6 coverage (Entry 63), neither component
    c6 = load("c6_coverage.json")
    reg = [r for r in c6["rows"] if not r["supplementary"]]
    assert len(reg) == 27
    smin, smax = min(r["sym_coverage"] for r in reg), max(r["sym_coverage"] for r in reg)
    mmin = min(r["maxarm_coverage"] for r in reg)
    nbelow = sum(r["sym_coverage"] < 0.99 for r in reg)
    add("nLimCsixSymMin", smin, dec(smin, 3), OUT + "/c6_coverage.json", "min(rows.sym_coverage), 27 registered cells",
        "63", "matches RESULTS Entry 63 0.987", "C6: lowest symmetric coverage, no estimation or training variance")
    add("nLimCsixSymMax", smax, dec(smax, 3), OUT + "/c6_coverage.json", "max(rows.sym_coverage)", "63",
        "matches RESULTS Entry 63 0.996", "C6: highest symmetric coverage")
    add("nLimCsixMaxMin", mmin, dec(mmin, 3), OUT + "/c6_coverage.json", "min(rows.maxarm_coverage)", "63",
        "matches RESULTS Entry 63 1.000 in all 27 cells", "C6: max-arm coverage (every cell)")
    add("nLimCsixBelow", nbelow, integer(nbelow), OUT + "/c6_coverage.json", "count(rows.sym_coverage < 0.99)", "63",
        "matches RESULTS Entry 63 '12 of 27 cells'", "C6: symmetric cells below 0.99")
    for sp, nm in (("normal", "Normal"), ("t3", "Heavy"), ("resample", "Resampled")):
        v = c6["sym_pooled_by_spread"][sp]["sym_coverage_pooled_9_cells"]
        add(f"nLimCsixPooled{nm}", v, dec(v, 4), OUT + "/c6_coverage.json",
            f"sym_pooled_by_spread.{sp}.sym_coverage_pooled_9_cells", "63",
            "matches RESULTS Entry 63 (0.9899 / 0.9931 / 0.9891)", f"C6: symmetric coverage pooled over nine cells, {sp} spread")
    corr = c6s["corr_E1_E2_per_adapter"]
    add("nLimSwapCorr", corr, dec(corr, 2), OUT + "/c6_estimator_swap.json", "corr_E1_E2_per_adapter", "63",
        "matches RESULTS Entry 63 0.49 (D200; G6 compares 0.35)", "per-adapter E1-E2 correlation, D200 primary adapters")
    madd = c6["inputs"]["m_observed_additive_part"]

    # ------------------------------------------------------------------ G3 estimator sensitivity (Entry 83)
    g3 = load("g3_estimator_scale.json")
    E = g3["estimates"]
    nimg3 = sum(g3["images_per_adapter"].values())
    add("nLimGthreeImages", nimg3, integer(nimg3), OUT + "/g3_estimator_scale.json", "sum(images_per_adapter)", "83",
        "matches RESULTS Entry 83 7,500 images (brief D-list: G3 used 7,500 not 12,000)",
        "generations scored in the estimator-sensitivity analysis")
    h2 = load("h2_estimation_error.json")
    n_e2 = h2["photographs"]["A"]["n_photographs"]
    photos = {"E2": n_e2, "E1": 80, "P": n_e2 + 80, "H1": n_e2 // 2, "H2": n_e2 // 2}
    names = {"E2": "Etwo", "E1": "Eone", "P": "Pooled", "H1": "HalfOne", "H2": "HalfTwo"}
    for key, nm in names.items():
        e = E[key]
        add(f"nLimGthree{nm}Photos", photos[key], integer(photos[key]),
            RES if key != "E2" else OUT + "/h2_estimation_error.json",
            "Entry 83: table column 'photographs'" if key != "E2" else "photographs.A.n_photographs", "83",
            "text-only (E1 = 80, Entry 63/83); pooled and halves derived from E2 = 140" if key != "E2" else "E2 = 140 per body",
            f"photographs per body behind estimate {key}")
        add(f"nLimGthree{nm}Rreal", e["R_real_own_scale"], dec(e["R_real_own_scale"], 4), OUT + "/g3_estimator_scale.json",
            f"estimates.{key}.R_real_own_scale", "83", "matches RESULTS Entry 83 table; the estimate's own held-out contrast",
            f"R_real of estimate {key} on its own scale")
        add(f"nLimGthree{nm}Sym", e["sym_limit99_pct"], sig(e["sym_limit99_pct"], 3), OUT + "/g3_estimator_scale.json",
            f"estimates.{key}.sym_limit99_pct", "83", f"matches RESULTS Entry 83; % of the estimate's own R_real ({e['R_real_own_scale']:.6f})",
            f"G3 symmetric 99 % limit under estimate {key}, 7,500 images")
        add(f"nLimGthree{nm}Max", e["maxarm_U_pct"], sig(e["maxarm_U_pct"], 3), OUT + "/g3_estimator_scale.json",
            f"estimates.{key}.maxarm_U_pct", "83", "matches RESULTS Entry 83; % of the estimate's own R_real",
            f"G3 max-arm limit under estimate {key}")
        add(f"nLimGthree{nm}P", e["one_sided_p"], _p(e["one_sided_p"]), OUT + "/g3_estimator_scale.json",
            f"estimates.{key}.one_sided_p", "83", "matches RESULTS Entry 83", f"G3 one-sided p of theta_sym under estimate {key}")
        add(f"nLimGthree{nm}Theta", e["theta_sym"], sci(e["theta_sym"], 3), OUT + "/g3_estimator_scale.json",
            f"estimates.{key}.theta_sym", "83", "matches RESULTS Entry 83", f"G3 theta_sym under estimate {key} (NCC units)")
    syms = [E[k_]["sym_limit99_pct"] for k_ in names]
    maxs = [E[k_]["maxarm_U_pct"] for k_ in names]
    ps = [E[k_]["one_sided_p"] for k_ in names]
    add("nLimGthreeSymLow", min(syms), sig(min(syms), 2), OUT + "/g3_estimator_scale.json", "min(estimates.*.sym_limit99_pct)",
        "83", "matches RESULTS Entry 83 / brief 0.062 %; own-scale %", "lowest symmetric limit over the five estimates")
    add("nLimGthreeSymHigh", max(syms), sig(max(syms), 3), OUT + "/g3_estimator_scale.json", "max(estimates.*.sym_limit99_pct)",
        "83", "matches RESULTS Entry 83 / brief 0.152 %; own-scale %", "highest symmetric limit over the five estimates")
    add("nLimGthreeMaxLow", min(maxs), sig(min(maxs), 3), OUT + "/g3_estimator_scale.json", "min(estimates.*.maxarm_U_pct)",
        "83", "matches RESULTS Entry 83 / brief 0.221 %", "lowest max-arm limit over the five estimates")
    add("nLimGthreeMaxHigh", max(maxs), sig(max(maxs), 3), OUT + "/g3_estimator_scale.json", "max(estimates.*.maxarm_U_pct)",
        "83", "matches RESULTS Entry 83 / brief 0.646 %", "highest max-arm limit over the five estimates")
    add("nLimGthreePLow", min(ps), _p(min(ps)), OUT + "/g3_estimator_scale.json", "min(estimates.*.one_sided_p)", "83",
        "matches RESULTS Entry 83 p 0.39", "smallest one-sided p over the five estimates")
    add("nLimGthreePHigh", max(ps), _p(max(ps)), OUT + "/g3_estimator_scale.json", "max(estimates.*.one_sided_p)", "83",
        "matches RESULTS Entry 83 p 0.75", "largest one-sided p over the five estimates")
    rc = 100 * g3["pooled_vs_E2_relative_change"]
    add("nLimGthreeChange", rc, sig(rc, 3), OUT + "/g3_estimator_scale.json", "pooled_vs_E2_relative_change", "83",
        "matches RESULTS Entry 83 +31.8 % (registered threshold 25 %); relative change, not % of R_real",
        "change of the pooled-estimate symmetric limit relative to E2's")
    rr = E["E2"]["R_real_own_scale"] / E["E1"]["R_real_own_scale"]
    add("nLimRrealEone", E["E1"]["R_real_own_scale"], sig(E["E1"]["R_real_own_scale"], 3), OUT + "/g3_estimator_scale.json",
        "estimates.E1.R_real_own_scale", "63", "matches RESULTS Entry 63 R_real(E1) = 0.027256", "R_real of the E1 estimate (80 photographs)")
    add("nLimRrealEoneRatio", 1 / rr, dec(1 / rr, 3), OUT + "/g3_estimator_scale.json", "E1/E2 R_real_own_scale", "63",
        "matches RESULTS Entry 63 ratio 0.764", "R_real(E1) / R_real(E2)")
    rspread = max(E[k_]["R_real_own_scale"] for k_ in names) / min(E[k_]["R_real_own_scale"] for k_ in names)
    add("nLimGthreeRrealSpread", rspread, dec(rspread, 1), OUT + "/g3_estimator_scale.json", "max/min R_real_own_scale",
        "83", "matches RESULTS Entry 83 'a factor of 1.5'", "spread of R_real across the five estimates (ratio)")

    # ------------------------------------------------------------------ H2 bootstrap (Entry 90)
    reps_ = h2["replicates"]
    th = np.array([r["theta_sym_pct"] for r in reps_])
    se_ = np.array([r["se_pct"] for r in reps_])
    vA, vE = float(np.mean(se_ ** 2)), float(np.var(th, ddof=1))
    share = vE / (vA + vE)
    infl = float(th.mean() + stats.t.ppf(0.99, np.mean([r["welch_df"] for r in reps_])) * math.sqrt(vA + vE))
    assert abs(infl - h2["inflated_limit_pct"]) < 1e-6
    nrep = h2["n_replicates"]
    add("nLimBootReps", nrep, integer(nrep), OUT + "/h2_estimation_error.json", "n_replicates", "90",
        "matches RESULTS Entry 90 twenty-four", "bootstrap replicates of the E2 photographs")
    nimg_h2 = sum(h2["images_used"].values())
    add("nLimBootImages", nimg_h2, integer(nimg_h2), OUT + "/h2_estimation_error.json", "sum(images_used)", "90",
        "matches RESULTS Entry 90 4,800 (200 per adapter)", "generations re-scored in every replicate")
    add("nLimBootImagesPerAdapter", h2["images_per_adapter"], integer(h2["images_per_adapter"]),
        OUT + "/h2_estimation_error.json", "images_per_adapter", "90", "Entry 90: 200 per adapter", "generations per adapter in H2")
    add("nLimBootPhotos", n_e2, integer(n_e2), OUT + "/h2_estimation_error.json", "photographs.A.n_photographs", "90",
        "E2 = 140 photographs per body", "photographs resampled per body")
    add("nLimBootDistinct", h2["photographs"]["A"]["mean_distinct_per_replicate"],
        dec(h2["photographs"]["A"]["mean_distinct_per_replicate"], 1), OUT + "/h2_estimation_error.json",
        "photographs.A.mean_distinct_per_replicate", "90", "file only", "mean distinct photographs per replicate")
    add("nLimBootShare", 100 * share, sig(100 * share, 3), OUT + "/h2_estimation_error.json",
        "variance_components_pct2.estimation_share (recomputed)", "90",
        "matches RESULTS Entry 90 43.2 %, verify_v2 check 54; share of variance, not of R_real",
        "share of the statistic's variance from fingerprint estimation")
    add("nLimBootAdapterShare", 100 * (1 - share), sig(100 * (1 - share), 3), OUT + "/h2_estimation_error.json",
        "1 - estimation_share", "90", "matches RESULTS Entry 90 56.8 %", "share of variance from adapters")
    add("nLimBootInflated", infl, sig(infl, 3), OUT + "/h2_estimation_error.json", "inflated_limit_pct (recomputed)", "90",
        "matches RESULTS Entry 90 0.1093 %, verify_v2 check 55; % of each replicate's own R_real (mean 0.0287), not of 0.0357",
        "estimation-inflated symmetric one-sided 99 % limit")
    add("nLimBootInflatedFourDigit", infl, dec(infl, 4), OUT + "/h2_estimation_error.json", "inflated_limit_pct", "90",
        "matches RESULTS Entry 90 0.1093 %", "inflated limit to 4 decimals")
    fr = infl / h2["filed_limit_pct"]
    add("nLimBootVsPooled", fr, dec(fr, 2), OUT + "/h2_estimation_error.json", "inflated_limit_pct/filed_limit_pct (0.0815)",
        "90", "matches RESULTS Entry 90 1.34x; baseline = pooled E1+E2 symmetric 0.0815 %", "inflation factor against the pooled-estimate symmetric limit")
    fe2 = infl / (100 * lim99 / R)
    add("nLimBootVsEtwo", fe2, dec(fe2, 2), OUT + "/h2_estimation_error.json", "inflated_limit_pct / primary symmetric 0.0762",
        "90", "brief 1.43x against E2 0.076 % (Entry 90 names only the 1.34x baseline)", "inflation factor against the E2 symmetric limit")
    lp = h2["limit_pct"]
    add("nLimBootLimitMean", lp["mean"], sig(lp["mean"], 3), OUT + "/h2_estimation_error.json", "limit_pct.mean", "90",
        "matches RESULTS Entry 90 0.0675 %; replicate's own R_real", "mean symmetric limit over replicates")
    add("nLimBootLimitSd", lp["sd"], sig(lp["sd"], 3), OUT + "/h2_estimation_error.json", "limit_pct.sd", "90",
        "matches Entry 90 0.0460", "SD of the symmetric limit over replicates")
    add("nLimBootLimitMin", lp["min"], sig(lp["min"], 3), OUT + "/h2_estimation_error.json", "limit_pct.min", "90",
        "matches Entry 90 -0.0522 (one replicate's limit is negative)", "lowest replicate limit")
    add("nLimBootLimitMax", lp["max"], sig(lp["max"], 3), OUT + "/h2_estimation_error.json", "limit_pct.max", "90",
        "matches Entry 90 0.1546", "highest replicate limit")
    nl = h2["null"]
    add("nLimBootPMin", nl["min_p"], dec(nl["min_p"], 3), OUT + "/h2_estimation_error.json", "null.min_p", "90",
        "matches RESULTS Entry 90 0.304, verify_v2 check 56 (3 decimals, as sourced)", "smallest one-sided p over replicates")
    add("nLimBootPMax", nl["max_p"], dec(nl["max_p"], 3), OUT + "/h2_estimation_error.json", "null.max_p", "90",
        "matches Entry 90 0.999", "largest one-sided p over replicates")
    add("nLimBootReject", nl["n_replicates_rejecting_at_01"], integer(nl["n_replicates_rejecting_at_01"]),
        OUT + "/h2_estimation_error.json", "null.n_replicates_rejecting_at_01", "90", "matches Entry 90: 0 of 24",
        "replicates rejecting at one-sided 0.01")
    rrs = [r["R_real"] for r in h2["R_real_per_replicate"]]
    add("nLimBootRrealMean", float(np.mean(rrs)), dec(float(np.mean(rrs)), 4), OUT + "/h2_estimation_error.json",
        "mean(R_real_per_replicate.R_real)", "90", "file only; the H2 percentages use each replicate's own R_real",
        "mean real-image contrast over bootstrap replicates")

    # ------------------------------------------------------------------ M1 pooled (Entry 107)
    m1 = load("m1_pooled.json")
    pdm = m1["per_design"]
    dnames = {"D200 primary": "Primary", "iPhone 5c (E2)": "IphoneNat", "iPhone 5c (FLAT)": "IphoneFlat",
              "P20 (G4b)": "Ptwenty", "D200 second environment (G2)": "SecondEnv",
              "D200 second training set (G5)": "SecondSet", "D200 FLUX.1-dev": "Flux"}
    for dk, nm in dnames.items():
        d = pdm[dk]
        add(f"nLimPool{nm}Lambda", d["lambda_pct"], sig(d["lambda_pct"], 3), OUT + "/m1_pooled.json",
            f"per_design.{dk}.lambda_pct", "107", f"matches RESULTS Entry 107 table; % of this design's own R_real ({d['R']:.6f})",
            f"M1 input: lambda of {dk}")
        add(f"nLimPool{nm}SE", d["se_pct"], sig(d["se_pct"], 3), OUT + "/m1_pooled.json", f"per_design.{dk}.se_pct", "107",
            "matches RESULTS Entry 107 table; own-scale %", f"M1 input: SE of {dk}")
    pr = m1["primary"]
    rnd, fix = pr["random"], pr["fixed"]
    add("nLimPoolLambda", rnd["lambda_pct"], sig(rnd["lambda_pct"], 3), OUT + "/m1_pooled.json", "primary.random.lambda_pct",
        "107", "matches RESULTS Entry 107 +0.0186 %, verify_v2 check 61; pooled over pairs' own R_real", "M1 pooled lambda, random effects, three pairs")
    add("nLimPoolSE", rnd["se_pct"], sig(rnd["se_pct"], 3), OUT + "/m1_pooled.json", "primary.random.se_pct", "107",
        "matches Entry 107 0.0228 %", "M1 pooled SE, random effects")
    add("nLimPoolZ", rnd["z"], dec(rnd["z"], 2), OUT + "/m1_pooled.json", "primary.random.z", "107", "file", "M1 pooled z, random effects")
    add("nLimPoolP", rnd["one_sided_p"], _p(rnd["one_sided_p"]), OUT + "/m1_pooled.json", "primary.random.one_sided_p", "107",
        "matches RESULTS Entry 107 0.207 / brief 0.21, verify_v2 check 63", "M1 pooled one-sided p")
    add("nLimPoolUpper", rnd["upper99_pct"], sig(rnd["upper99_pct"], 2), OUT + "/m1_pooled.json", "primary.random.upper99_pct",
        "107", "matches RESULTS Entry 107 reading 0.072 % (0.0715), verify_v2 check 62", "M1 pooled one-sided 99 % limit, random effects (most general limit)")
    add("nLimPoolUpperFull", rnd["upper99_pct"], sig(rnd["upper99_pct"], 3), OUT + "/m1_pooled.json", "primary.random.upper99_pct",
        "107", "matches Entry 107 table 0.0715 %", "M1 pooled limit to 3 significant figures")
    add("nLimPoolFixedLambda", fix["lambda_pct"], sig(fix["lambda_pct"], 3), OUT + "/m1_pooled.json", "primary.fixed.lambda_pct",
        "107", "file", "M1 pooled lambda, fixed effect")
    add("nLimPoolFixedP", fix["one_sided_p"], dec(fix["one_sided_p"], 2), OUT + "/m1_pooled.json", "primary.fixed.one_sided_p",
        "107", "file", "M1 pooled one-sided p, fixed effect")
    add("nLimPoolFixedUpper", fix["upper99_pct"], sig(fix["upper99_pct"], 2), OUT + "/m1_pooled.json", "primary.fixed.upper99_pct",
        "107", "matches RESULTS Entry 107 0.054 %", "M1 pooled one-sided 99 % limit, fixed effect")
    add("nLimPoolItwo", 100 * pr["I2"], sig(100 * pr["I2"], 2), OUT + "/m1_pooled.json", "primary.I2", "107",
        "matches RESULTS Entry 107 57 %", "M1 heterogeneity I^2, %")
    add("nLimPoolQ", pr["Q"], dec(pr["Q"], 2), OUT + "/m1_pooled.json", "primary.Q", "107", "file", "Cochran's Q (2 df)")
    add("nLimPoolQP", pr["Q_p"], dec(pr["Q_p"], 2), OUT + "/m1_pooled.json", "primary.Q_p", "107", "matches Entry 107 'Q p 0.10'",
        "p of Cochran's Q")
    add("nLimPoolPairs", pr["k"], integer(pr["k"]), OUT + "/m1_pooled.json", "primary.k", "107", "three independent pairs",
        "device pairs pooled")
    sec = m1["secondary"]
    add("nLimPoolSecLambda", sec["random"]["lambda_pct"], sig(sec["random"]["lambda_pct"], 3), OUT + "/m1_pooled.json",
        "secondary.random.lambda_pct", "107", "matches Entry 107 +0.0201 %", "M1 secondary set pooled lambda")
    add("nLimPoolSecP", sec["random"]["one_sided_p"], _p(sec["random"]["one_sided_p"]), OUT + "/m1_pooled.json",
        "secondary.random.one_sided_p", "107", "matches Entry 107 0.163", "M1 secondary set one-sided p")
    add("nLimPoolSecUpper", sec["random"]["upper99_pct"], sig(sec["random"]["upper99_pct"], 2), OUT + "/m1_pooled.json",
        "secondary.random.upper99_pct", "107", "matches Entry 107 0.068 %", "M1 secondary set one-sided 99 % limit")
    dc = sec["d200_combination"]["fixed"]
    add("nLimPoolDtwoHundred", dc["lambda_pct"], sig(dc["lambda_pct"], 3), OUT + "/m1_pooled.json",
        "secondary.d200_combination.fixed.lambda_pct", "107", "matches Entry 107 +0.018 %", "D200's four designs combined (inverse variance)")
    sf = m1["sensitivity_flat_field"]["fixed"]
    add("nLimPoolFlatLambda", sf["lambda_pct"], sig(sf["lambda_pct"], 3), OUT + "/m1_pooled.json",
        "sensitivity_flat_field.fixed.lambda_pct", "107", "matches RESULTS Entry 107 +0.0301 % / brief +0.030 %",
        "M1 sensitivity: iPhone pair with flat-field estimate, pooled lambda")
    add("nLimPoolFlatP", sf["one_sided_p"], _p(sf["one_sided_p"]), OUT + "/m1_pooled.json",
        "sensitivity_flat_field.fixed.one_sided_p", "107", "matches RESULTS Entry 107 0.0109 / brief 0.011",
        "M1 flat-field sensitivity one-sided p")
    add("nLimPoolFlatUpper", sf["upper99_pct"], sig(sf["upper99_pct"], 2), OUT + "/m1_pooled.json",
        "sensitivity_flat_field.fixed.upper99_pct", "107", "matches Entry 107 0.061 %", "M1 flat-field sensitivity limit")
    si = m1["sensitivity_estimation_inflated"]
    sq = si[si["quoted"]]
    add("nLimPoolInflLambda", sq["lambda_pct"], sig(sq["lambda_pct"], 3), OUT + "/m1_pooled.json",
        "sensitivity_estimation_inflated.fixed.lambda_pct", "107", "matches Entry 107 +0.0191 % (fixed, I^2 24 %)",
        "M1 estimation-inflated sensitivity, pooled lambda")
    add("nLimPoolInflP", sq["one_sided_p"], _p(sq["one_sided_p"]), OUT + "/m1_pooled.json",
        "sensitivity_estimation_inflated.fixed.one_sided_p", "107", "matches Entry 107 0.169 / 'p 0.17'",
        "M1 estimation-inflated sensitivity p")
    add("nLimPoolInflUpper", sq["upper99_pct"], sig(sq["upper99_pct"], 2), OUT + "/m1_pooled.json",
        "sensitivity_estimation_inflated.fixed.upper99_pct", "107", "matches RESULTS Entry 107 0.065 %",
        "M1 pooled limit with estimation error carried")
    add("nLimPoolInflFactor", m1["variance_inflation"], dec(m1["variance_inflation"], 2), OUT + "/m1_pooled.json",
        "variance_inflation (= 1/(1-0.432))", "106", "matches Entry 106 1.76", "variance inflation for estimation error")
    allp = [pr["random"]["lambda_pct"], sec["random"]["lambda_pct"], sf["lambda_pct"], sq["lambda_pct"]]
    add("nLimPoolLeanLow", min(allp), sig(min(allp), 2), OUT + "/m1_pooled.json", "min over pooled variants", "107",
        "matches Entry 107 '+0.019 to +0.030 %'", "smallest pooled estimate over variants")
    add("nLimPoolLeanHigh", max(allp), sig(max(allp), 2), OUT + "/m1_pooled.json", "max over pooled variants", "107",
        "matches Entry 107 '+0.019 to +0.030 %'", "largest pooled estimate over variants")

    # ================================================================== Main: shared fingerprint main effect
    t2 = load("t2_summary.json")

    def panel_additive(det):
        arms = t2[det]["arms"]
        tA = np.mean([arms[f"A_raw_s{i}_r16"]["paired_contrast"] for i in range(3)])
        tB = np.mean([arms[f"B_raw_s{i}_r16"]["paired_contrast"] for i in range(3)])
        return 0.5 * (tA - tB), t2[det]["real_paired_contrast"]

    for det, nm, n_, ck in (("ncc", "Ncc", 1, "Entries 10/32: 0.01 %; additive -5.1e-06 over the detector's own real contrast 0.035659"),
                            ("pce", "Pce", 1, "Entries 10/32: 0.08 %; additive -0.303 PCE over 367.8"),
                            ("lowmid", "Lowmid", 2, "Entries 10/32: 2.9 %; additive -6.3e-05 over 0.002154")):
        a, r = panel_additive(det)
        share = 100 * abs(a) / r
        add(f"nMainShare{nm}", share, sig(share, n_), OUT + "/t2_summary.json",
            f"|mean(A arms) - mean(B arms)|/2 / {det}.real_paired_contrast", "32",
            "matches RESULTS " + ck + "; % of the detector's own real contrast",
            f"shared main effect as % of the {det} detector's real contrast (3 adapters per arm, 500 images each)")
    a_ncc, _ = panel_additive("ncc")
    add("nMainNccAdditive", a_ncc, sci(a_ncc, 2), OUT + "/t2_summary.json", "additive part, ncc", "10",
        "matches Entry 10 -5.1e-06", "NCC additive part (b_A - b_B)/2, primary seeds 0-2")
    a_pce, _ = panel_additive("pce")
    add("nMainPceAdditive", a_pce, sig(a_pce, 3), OUT + "/t2_summary.json", "additive part, pce (PCE units)", "10",
        "matches Entry 10 -0.303", "PCE additive part, PCE units")
    b0 = t2["pce"]["arms"]["B_raw_s0_r16"]
    tb0 = b0["paired_contrast"] / b0["paired_contrast_se"]
    add("nMainPceArmT", tb0, dec(tb0, 1), OUT + "/t2_summary.json", "pce.arms.B_raw_s0_r16 contrast/SE", "10",
        "matches Entry 10 t +2.79", "largest unpaired PCE arm t (would be read as a detection)")
    npz = load("t2_noiseprint_summary.json")
    add("nMainShareNoiseprint", npz["additive_over_R_pct"], sig(npz["additive_over_R_pct"], 2), OUT + "/t2_noiseprint_summary.json",
        "additive_over_R_pct", "32", "matches RESULTS Entry 32 87 %; % of Noiseprint's real contrast", "Noiseprint shared main effect, % of its real contrast")
    add("nMainNoiseprintT", npz["max_abs_image_t"], integer(npz["max_abs_image_t"]), OUT + "/t2_noiseprint_summary.json",
        "max_abs_image_t", "32", "Entry 32 / brief 't ~ 100'; file maximum 104", "largest image-level t of the Noiseprint main effect")
    add("nMainNoiseprintThetaSym", npz["theta_sym_over_R_pct"], sig(npz["theta_sym_over_R_pct"], 2), OUT + "/t2_noiseprint_summary.json",
        "theta_sym_over_R_pct", "32", "matches Entry 32 -0.17 %", "Noiseprint symmetric interaction, % of its real contrast")
    lx = load("t2_learned_ext.json")
    add("nMainShareCnn", lx["additive_over_R_pct"], sig(lx["additive_over_R_pct"], 3), OUT + "/t2_learned_ext.json",
        "additive_over_R_pct", "18", "matches RESULTS Entry 18 162 %; % of the CNN's real contrast (saved model)",
        "learned-CNN shared main effect, % of its real contrast")

    # diverse prompts (Entry 22)
    t5 = load(os.path.join(OUT, "t5", "summary_derived.json"))
    bd, bu = t5["diverse_bank"]["base_KB_minus_KA"], t5["uniform_bank_same_adapters_entry10"]["base_KB_minus_KA"]
    add("nMainDiverseBase", bd, sci(bd, 2), OUT + "/t5/summary_derived.json", "diverse_bank.base_KB_minus_KA", "22",
        "matches RESULTS Entry 22 1.4e-04", "base-model K_B - K_A under five diverse captions")
    add("nMainUniformBase", bu, sci(bu, 2), OUT + "/t5/summary_derived.json", "uniform_bank_same_adapters_entry10.base_KB_minus_KA",
        "22", "matches RESULTS Entry 22 0.4e-04", "base-model K_B - K_A under the single caption")
    add("nMainDiverseRatio", bd / bu, dec(bd / bu, 1), OUT + "/t5/summary_derived.json", "ratio of the two", "22",
        "Entry 22 'three times'", "growth of the base-model main effect under diverse prompts")
    add("nMainDiverseAdditive", t5["diverse_bank"]["additive_part"], sci(t5["diverse_bank"]["additive_part"], 3),
        OUT + "/t5/summary_derived.json", "diverse_bank.additive_part", "22", "matches Entry 22 -10.8e-05",
        "additive part under diverse prompts")
    add("nMainDiverseMaxT", t5["diverse_bank"]["max_abs_image_t"], dec(t5["diverse_bank"]["max_abs_image_t"], 1),
        OUT + "/t5/summary_derived.json", "diverse_bank.max_abs_image_t", "22", "matches Entry 22 B_s2 t = 2.6",
        "largest unpaired arm t under diverse prompts")

    # raw argmax (Entry 07)
    t3 = load("t3_attrib.json")
    for grp, path, nm in (("kodak", ("raw_fingerprints", "raw"), "Kodak"), ("p20", ("K_residualised", "raw"), "Ptwenty")):
        g = t3[grp][path[0]]
        sel = g[path[1]]["selection_frequency_G250"]
        b = g["main_effects_b"]
        top = max(b, key=lambda d_: b[d_])
        most = max(sel, key=lambda d_: sel[d_])
        assert top == most
        nad = t3[grp]["n_adapters"]
        add(f"nMainArgmax{nm}", sel[most], integer(sel[most]), OUT + "/t3_attrib.json",
            f"{grp}.{path[0]}.{path[1]}.selection_frequency_G250.{most}", "07",
            f"matches RESULTS Entry 07 ({most} x{sel[most]} of {nad}); {most} has the largest main effect",
            f"adapters (of {nad}) whose raw argmax picks the body with the largest main effect, {grp}")
        if grp == "kodak":
            # t3_attrib.json keys the Kodak bodies by design label (D0-D4); print the Dresden database ID
            # (as Table S2 does), mapped through the group's manifest (D0 = Kodak_M1063_4).
            kman = V2 + "/data/manifests/kodak/manifest.json"
            dbid = load(kman)["picks"][most]["id"]
            assert dbid.startswith("Kodak_M1063_")
            add(f"nMainArgmax{nm}Body", dbid, "\\texttt{" + dbid.replace("_", "\\_") + "}",
                OUT + "/t3_attrib.json; " + kman, f"argmax {grp} selection ({most}); picks.{most}.id", "07",
                f"Entry 07 ({most}); manifest picks.{most}.id = {dbid}, as Table S2",
                f"Dresden body picked by raw argmax, {grp}")
        else:
            add(f"nMainArgmax{nm}Body", most, str(most), OUT + "/t3_attrib.json", f"argmax {grp} selection", "07",
                "Entry 07; Daxing device number, as Table S2", f"body picked by raw argmax, {grp}")
        add(f"nMainArgmax{nm}B", b[top], sci(b[top], 3), OUT + "/t3_attrib.json", f"{grp}.{path[0]}.main_effects_b.{top}", "07",
            "matches Entry 07 (D0 +6.4e-05; 1102 +15.7e-05)", f"fitted main effect of the most-picked body, {grp}")
    add("nMainArgmaxAdapters", t3["kodak"]["n_adapters"], integer(t3["kodak"]["n_adapters"]), OUT + "/t3_attrib.json",
        "kodak.n_adapters", "07", "10 adapters per group", "adapters per five-body group")

    # primary D200 additive part
    add("nMainPrimaryAdditive", madd, sci(madd, 3), OUT + "/c6_coverage.json", "inputs.m_observed_additive_part", "63",
        "matches Entry 63 m = -1.42e-5", "additive part of the D200 primary (NCC units)")
    add("nMainPrimaryAdditivePct", 100 * madd / R, sig(100 * madd / R, 3), OUT + "/c6_coverage.json",
        "100*m_observed_additive_part/R_real", "63", "% of R_real (E2)", "D200 primary additive part, % of R_real")
    kd = load(os.path.join(OUT, "fp", "gates.json"))["kappa_model_E2"]
    add("nMainKappaDtwoHundred", kd, dec(kd, 3), OUT + "/fp/gates.json", "kappa_model_E2", "105",
        "matches RESULTS Entry 105 0.007", "cross-device kappa of the D200 E2 estimates")

    # G6 iPhone 5c (Entry 103)
    g6 = load("g6_p5c.json")
    for est, nm in (("E2", "Nat"), ("FLAT", "Flat")):
        e = g6["estimators"][est]
        a_ = np.array(e["per_adapter_A"])
        b_ = np.array(e["per_adapter_B"])
        add_ = 0.5 * (a_.mean() - b_.mean())
        Rg = e["R_real"]
        add(f"nMainGsix{nm}Additive", add_, sci(add_, 3), OUT + "/g6_p5c.json", f"(mean_A - mean_B)/2, estimators.{est}", "103",
            "matches Entry 103 " + ("-1.198e-04" if est == "E2" else "-1.21e-05"), f"G6 additive part, {est} estimate")
        add(f"nMainGsix{nm}AdditivePct", 100 * abs(add_) / Rg, sig(100 * abs(add_) / Rg, 2), OUT + "/g6_p5c.json",
            f"|additive|/estimators.{est}.R_real", "103",
            ("matches Entry 103 / brief 0.25 %" if est == "E2" else "file") + f"; % of the iPhone pair's own {est} R_real ({Rg:.4f}), not the D200's",
            f"G6 additive part as % of the iPhone {est} real contrast (magnitude)")
        add(f"nMainGsix{nm}PosA", int((a_ > 0).sum()), integer((a_ > 0).sum()), OUT + "/g6_p5c.json",
            f"count(estimators.{est}.per_adapter_A > 0)", "103", "matches Entry 103", f"G6 arm A adapters positive ({est}), of 12")
        add(f"nMainGsix{nm}PosB", int((b_ > 0).sum()), integer((b_ > 0).sum()), OUT + "/g6_p5c.json",
            f"count(estimators.{est}.per_adapter_B > 0)", "103", "matches Entry 103", f"G6 arm B adapters positive ({est}), of 12")
        add(f"nMainGsix{nm}PA", e["iu_signflip"]["p_A"], _p(e["iu_signflip"]["p_A"]), OUT + "/g6_p5c.json",
            f"estimators.{est}.iu_signflip.p_A", "103", "matches Entry 103", f"G6 arm A sign-flip p ({est})")
        add(f"nMainGsix{nm}PB", e["iu_signflip"]["p_B"], _p(e["iu_signflip"]["p_B"]), OUT + "/g6_p5c.json",
            f"estimators.{est}.iu_signflip.p_B", "103",
            "matches Entry 103" + (" 0.0002 (= floor 1/4096)" if est == "E2" else " 0.0547"), f"G6 arm B sign-flip p ({est})")
    en = g6["estimators"]["E2"]
    ef = g6["estimators"]["FLAT"]
    an = 0.5 * (np.mean(en["per_adapter_A"]) - np.mean(en["per_adapter_B"]))
    af = 0.5 * (np.mean(ef["per_adapter_A"]) - np.mean(ef["per_adapter_B"]))
    add("nMainGsixRatio", an / af, dec(an / af, 1), OUT + "/g6_p5c.json", "additive(E2)/additive(FLAT), NCC units", "103",
        "Entry 103 'ten times smaller'", "how many times smaller the main effect is with flat fields (raw NCC units)")
    add("nMainGsixKappaNat", g6["gates"]["kappa_model_E2"], dec(g6["gates"]["kappa_model_E2"], 3), OUT + "/g6_p5c.json",
        "gates.kappa_model_E2", "103", "file", "cross-device kappa, iPhone E2 estimates")

    # G4b Huawei P20 (Entry 105)
    g4 = load("g4b_p20.json")
    a4, b4 = np.array(g4["per_adapter_A"]), np.array(g4["per_adapter_B"])
    ad4 = 0.5 * (a4.mean() - b4.mean())
    assert abs(ad4 - g4["additive_part"]) < 1e-12
    add("nMainPtwentyAdditive", ad4, sci(ad4, 3), OUT + "/g4b_p20.json", "additive_part", "105", "matches Entry 105 +1.158e-04",
        "G4b additive part (NCC units)")
    add("nMainPtwentyAdditivePct", 100 * ad4 / g4["R_real"], sig(100 * ad4 / g4["R_real"], 2), OUT + "/g4b_p20.json",
        "additive_part/R_real", "105", f"matches Entry 105 / brief 0.30 %; % of the P20 pair's own R_real ({g4['R_real']:.4f})",
        "G4b additive part as % of the P20 real contrast")
    ratio4 = ad4 / g4["symmetric"]["theta_sym"]
    add("nMainPtwentyAdditiveOverTheta", ratio4, dec(ratio4, 1), OUT + "/g4b_p20.json", "additive_part/symmetric.theta_sym", "105",
        "Entry 105 'five times theta_sym'", "G4b additive part as a multiple of theta_sym")
    add("nMainPtwentyPosA", int((a4 > 0).sum()), integer((a4 > 0).sum()), OUT + "/g4b_p20.json", "count(per_adapter_A > 0)", "105",
        "matches Entry 105 12 of 12, n_positive.A", "G4b arm A adapters positive, of 12")
    add("nMainPtwentyPosB", int((b4 > 0).sum()), integer((b4 > 0).sum()), OUT + "/g4b_p20.json", "count(per_adapter_B > 0)", "105",
        "matches Entry 105 0 of 12, n_positive.B", "G4b arm B adapters positive, of 12")
    add("nMainPtwentyPA", g4["iu_signflip"]["p_A"], _p(g4["iu_signflip"]["p_A"]), OUT + "/g4b_p20.json", "iu_signflip.p_A", "105",
        "matches Entry 105 0.00024 (floor 1/4096)", "G4b arm A sign-flip p (at the floor)")
    add("nMainPtwentyPB", g4["iu_signflip"]["p_B"], dec(g4["iu_signflip"]["p_B"], 1), OUT + "/g4b_p20.json", "iu_signflip.p_B", "105",
        "matches Entry 105 1.000", "G4b arm B sign-flip p")
    add("nMainKappaPtwenty", g4["gates"]["kappa_model_E2"], dec(g4["gates"]["kappa_model_E2"], 3), OUT + "/g4b_p20.json",
        "gates.kappa_model_E2", "105", "matches Entry 105 0.462", "cross-device kappa of the P20 pair's estimates")
    return M


if __name__ == "__main__":
    ms = macros()
    for m in ms:
        print(f"{m['name']:34s} {m['text']:28s} {m['meaning']}")
    print(len(ms), "macros")
