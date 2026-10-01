"""Number macros, part n9: the closing checks registered in RESULTS.md Entry 116 and logged in Entry 117.

Every value is READ at run time from the result files of the six Entry 116 items:
  item 1  out/fv_seedbank_calib.json        (src/fv/fv_seedbank_calib.py; Entry 116 named fv_seedbank.*)
  item 2  out/fv_examiner_e2e.json          (src/fv/fv_examiner_e2e.py; Entry 116 named fv_examiner.*)
  item 3  out/fv_crossbody_scenes.json      (src/fv/fv_crossbody_scenes.py; Entry 116 named fv_crossbody_scene.*)
  item 4  out/fv_weights_inv.json           (src/fv/fv_weights_inv.py, the file of record; Entry 116 named
          fv_weights_invert.*) and out/fv_weights_inv_recheck.json (independent recheck + post-hoc checks)
  item 5  out/fv_closedset_expect_run2.json (src/fv/fv_closedset_expect_run2.py; identical to the first attempt's
          out/fv_closedset_expect.json on every shared quantity)
  item 6  out/fv_small.json                 (src/fv/fv_small.py, the registered names)
plus FINAL_LEDGER.json for one derived limit (c of the 20,000-replication rerun without the seed-bank term).

Status (Entry 116): item 4 is the only registered test at a level; item 1 is a construction rule, and under it the
seed-bank term CHANGES the calibration: c* = 1.31 and 0.181 % of R_real becomes the bound of record (nLimRec*),
with H6's nLimCal (0.175 %, c = 1.25) and nLimNom beside it. Item 2 is a registered model check (reading: agrees);
item 3 a data-audit rule (reading: the scene overlap does not move the limit; iPhone: no scene shared); items 5 and
6 are descriptive. Macros whose check text says "post hoc" are not pre-specified.

One quantity, one macro: values already carried by another part are not repeated here (H6's nLimCal*, nLimCalReps,
the nominal nLimNom, nLimNomKsix 0.249 %, nGenGfourbMaxArm, nGenGfourbPFine, the Entry 98 nDoseInvThree* values
other than t, nDataTotalAdapters / nDataTotalGens, nDataSceneAuditWorst). The rates at the bound of record carry the
suffix "Rec" and replace the "Cal" rates wherever the text quotes the bound of record.
"""
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_numlib import OUT, LEDGER, R_REAL, sig, dec, sci, integer, macro, load  # noqa: E402

E = "117"
SB = OUT + "/fv_seedbank_calib.json"
EX = OUT + "/fv_examiner_e2e.json"
XB = OUT + "/fv_crossbody_scenes.json"
WT = OUT + "/fv_weights_inv.json"
WR = OUT + "/fv_weights_inv_recheck.json"
CS = OUT + "/fv_closedset_expect_run2.json"
SM = OUT + "/fv_small.json"
POST = "post hoc, not pre-specified"


def g2(x):
    """An image count to two significant figures (3208.2 -> 3200), as num_n8 prints them."""
    return integer(float(f"{x:.2g}"))


def pct1(x, n=2):
    """A relative change (fraction) as a signed percentage, n significant figures."""
    return sig(100.0 * x, n)


def macros():
    L = []

    def M(name, value, text, src, key, check):
        L.append(macro(name, value, text, src, key, E, check))

    # ======================================================================== item 1: seed-bank calibration
    S = load(SB)
    assert S["entry"].startswith("RESULTS.md Entry 116, item 1"), S["entry"]
    rule = S["rule"]
    assert rule["branch"] == "changes" and abs(rule["c_star"] - 1.31) < 1e-12, rule
    assert abs(rule["value"] - 0.98875) < 1e-12
    assert S["zero_term_check"]["equal"] is True and S["h6_reproduction"]["primary_d200"]["equal"] is True
    rec = S["limits"]["bound_of_record"]
    assert rec["binding_arm"] == "B"
    pt = S["crn_run_4000"]["variants"]["point"]
    M("nLimRec", rec["lambda_U_pct_of_R_real"], sig(rec["lambda_U_pct_of_R_real"], 3), SB,
      "limits.bound_of_record.lambda_U_pct_of_R_real",
      "bound of record after Entry 116 item 1's rule: calibrated max-arm one-sided 99 % limit, % of R_real, "
      "c* applied to the ledger values as H6 applied c (U_X = mean_X + c* t_0.995,11 s_X/sqrt 12, U = max); "
      "calibrated for adapter, fingerprint-estimation, training-set and seed-bank variation; arm B binds")
    M("nLimRecFourDigit", rec["lambda_U_pct_of_R_real"], dec(rec["lambda_U_pct_of_R_real"], 4), SB,
      "limits.bound_of_record.lambda_U_pct_of_R_real", "four-digit form, register and S07/S08 only")
    M("nLimRecC", rule["c_star"], dec(rule["c_star"], 2), SB, "rule.c_star",
      "smallest c on H6's grid with worst-cell max-arm coverage >= 0.99 in the 4,000-replication CRN run with the "
      "seed-bank term at the point estimate of Sigma_v")
    M("nLimRecUdevice", rec["U_device"], sci(rec["U_device"], 3), SB, "limits.bound_of_record.U_device",
      "U_device at c* (NCC units)")
    M("nLimRecUAPct", rec["U_A_pct_of_R_real"], sig(rec["U_A_pct_of_R_real"], 3), SB,
      "limits.bound_of_record.U_A_pct_of_R_real", "arm A's limit at c*, % of R_real (arm B binds)")
    M("nLimRecCovAtOne", pt["worst_coverage_at_1"], dec(pt["worst_coverage_at_1"], 3), SB,
      "crn_run_4000.variants.point.worst_coverage_at_1", "worst-cell max-arm coverage at c = 1 with the term")
    M("nLimRecCovAtCalC", pt["worst_coverage_at_1.25"], dec(pt["worst_coverage_at_1.25"], 5), SB,
      "crn_run_4000.variants.point.worst_coverage_at_1.25",
      "the rule statistic: worst-cell coverage at H6's c = 1.25 with the term (3,955 of 4,000; 3,960 needed)")
    M("nLimRecCovAtRecC", pt["worst_coverage_at_c_star"], dec(pt["worst_coverage_at_c_star"], 3), SB,
      "crn_run_4000.variants.point.worst_coverage_at_c_star", "worst-cell coverage at c*")
    sbe = S["seed_bank_estimate"]
    vp = sbe["variants"]["point"]
    M("nLimSeedVarSym", sbe["symmetric"]["sigma_vs2"], sci(sbe["symmetric"]["sigma_vs2"], 3), SB,
      "seed_bank_estimate.symmetric.sigma_vs2",
      "symmetric seed-bank variance sigma_vs^2 (the part that enters theta_sym), 24 primary adapters x 500 seeds; "
      "equals (sigma_vA^2 + sigma_vB^2 + 2 cov)/4")
    bt = sbe["bootstrap"]
    M("nLimSeedVarSymUpper", bt["one_sided_95_upper_registered"], sci(bt["one_sided_95_upper_registered"], 3), SB,
      "seed_bank_estimate.bootstrap.one_sided_95_upper_registered",
      f"one-sided 95 % upper limit, {bt['n']} bootstrap resamples of the 500 seeds (registered item ii)")
    M("nLimSeedVarSymLower", bt["one_sided_95_lower_added"], sci(bt["one_sided_95_lower_added"], 3), SB,
      "seed_bank_estimate.bootstrap.one_sided_95_lower_added", "5th percentile (lower end; requested with the run)")
    M("nLimSeedCorr", sbe["corr_v_A_v_B_implied"], dec(sbe["corr_v_A_v_B_implied"], 2), SB,
      "seed_bank_estimate.corr_v_A_v_B_implied", "implied correlation of the two arms' seed effects")
    prs = vp["per_replication_sd"]
    M("nLimSeedSymSD", prs["symmetric_(b_A+b_B)/2"], sci(prs["symmetric_(b_A+b_B)/2"], 2), SB,
      "seed_bank_estimate.variants.point.per_replication_sd.symmetric_(b_A+b_B)/2",
      "SD per replication of the bank shift's symmetric part (enters theta_sym)")
    M("nLimSeedAddSD", prs["additive_(b_A-b_B)/2"], sci(prs["additive_(b_A-b_B)/2"], 2), SB,
      "seed_bank_estimate.variants.point.per_replication_sd.additive_(b_A-b_B)/2",
      "SD per replication of the bank shift's additive part (moves the arms in opposite directions)")
    ends = S["limits"]["at_interval_ends"]
    lo, hi = ends["lower_one_sided_95 (5th percentile; requested with the run)"], \
        ends["upper_one_sided_95 (registered, descriptive ii)"]
    M("nLimRecCLow", lo["c"], dec(lo["c"], 2), SB, "limits.at_interval_ends.lower_one_sided_95.c",
      "c at the 5th percentile of sigma_vs^2 (2.5th gives the same c)")
    M("nLimRecCHigh", hi["c"], dec(hi["c"], 2), SB, "limits.at_interval_ends.upper_one_sided_95.c",
      "c at the 95th percentile of sigma_vs^2, registered item (ii) (97.5th gives the same c)")
    M("nLimRecLow", lo["lambda_U_pct_of_R_real"], sig(lo["lambda_U_pct_of_R_real"], 3), SB,
      "limits.at_interval_ends.lower_one_sided_95.lambda_U_pct_of_R_real", "limit at nLimRecCLow, % of R_real")
    M("nLimRecHigh", hi["lambda_U_pct_of_R_real"], sig(hi["lambda_U_pct_of_R_real"], 3), SB,
      "limits.at_interval_ends.upper_one_sided_95.lambda_U_pct_of_R_real", "limit at nLimRecCHigh, % of R_real")
    dr = S["descriptive_registered"]
    pr = dr["i_probe_form_symmetric_only"]
    M("nLimSeedProbeC", pr["c_star"], dec(pr["c_star"], 2), SB,
      "descriptive_registered.i_probe_form_symmetric_only.c_star",
      "registered item (i): the review probe's form (symmetric part only, shared by both arms); descriptive; the "
      "independent check gets 1.53 with another drawing of the shared shift")
    M("nLimSeedProbe", pr["limit_at_c_star"]["lambda_U_pct_of_R_real"],
      sig(pr["limit_at_c_star"]["lambda_U_pct_of_R_real"], 3), SB,
      "descriptive_registered.i_probe_form_symmetric_only.limit_at_c_star.lambda_U_pct_of_R_real",
      "limit at the probe form's c; the review's 'about 0.21 %' came from this incomplete form")
    sy = dr["iii_symmetric_construction"]
    M("nLimSeedSymCovWithout", sy["worst_without_term"], dec(sy["worst_without_term"], 3), SB,
      "descriptive_registered.iii_symmetric_construction.worst_without_term",
      "registered item (iii): symmetric construction's worst-cell coverage, H6 model without the term")
    M("nLimSeedSymCovWith", sy["worst_with_term"], dec(sy["worst_with_term"], 3), SB,
      "descriptive_registered.iii_symmetric_construction.worst_with_term", "same, with the seed-bank term")
    rr = dr["iv_rerun_20000"]
    M("nLimSeedRerunCov", rr["worst_coverage_at_c_star"], dec(rr["worst_coverage_at_c_star"], 3), SB,
      "descriptive_registered.iv_rerun_20000.worst_coverage_at_c_star",
      f"registered item (iv): {rr['n_rep']:,}-replication rerun (master seed {rr['master_seed']}) at c*")
    M("nLimSeedRerunSE", rr["mc_se_worst_cell_at_c_star"], dec(rr["mc_se_worst_cell_at_c_star"], 5), SB,
      "descriptive_registered.iv_rerun_20000.mc_se_worst_cell_at_c_star", "its Monte Carlo SE (worst cell)")
    M("nLimSeedRerunCovCal", rr["worst_coverage_at_1.25"], dec(rr["worst_coverage_at_1.25"], 3), SB,
      "descriptive_registered.iv_rerun_20000.worst_coverage_at_1.25", "the rerun at c = 1.25, with the term")
    M("nLimSeedRerunCWith", rr["c_selected_by_rerun_with_term"], dec(rr["c_selected_by_rerun_with_term"], 2), SB,
      "descriptive_registered.iv_rerun_20000.c_selected_by_rerun_with_term", "c the rerun would select, with term")
    cwo = rr["c_selected_by_rerun_without_term"]
    M("nLimSeedRerunCWithout", cwo, dec(cwo, 2), SB, "descriptive_registered.iv_rerun_20000.c_selected_by_rerun_"
      "without_term", "c the rerun would select without the term (H6's model simulated precisely)")
    # the limit at that c, from the ledger exactly as H6 applies c (asserted against the file at c* and c = 1.25)
    led = load(LEDGER)["primary"]
    t = S["inputs"]["t_0995_11"]

    def lim(c):
        u = []
        for k in ("per_adapter_A", "per_adapter_B"):
            x = led[k]
            u.append(statistics.fmean(x) + c * t * statistics.stdev(x) / math.sqrt(len(x)))
        return 100.0 * max(u) / R_REAL
    assert abs(lim(rule["c_star"]) - rec["lambda_U_pct_of_R_real"]) < 1e-9
    assert abs(lim(1.25) - S["limits"]["H6_calibrated_c_1.25"]["lambda_U_pct_of_R_real"]) < 1e-9
    M("nLimSeedRerunWithoutLim", lim(cwo), sig(lim(cwo), 3), SB + " + " + LEDGER,
      "lim(descriptive_registered.iv_rerun_20000.c_selected_by_rerun_without_term) from ledger primary.per_adapter_A/B",
      "the limit H6's model (no seed-bank term) needs when simulated with 20,000 replications; derived, asserted "
      "to reproduce the file at c* and c = 1.25")
    ad = S["added_descriptive"]
    alt = ad["branch_under_alternative_bank_draws"]
    M("nLimSeedRedraws", alt["n"], integer(alt["n"]), SB, "added_descriptive.branch_under_alternative_bank_draws.n",
      f"{POST}: H6's draws held fixed, bank shift redrawn from other generator seeds")
    M("nLimSeedBranchShare", alt["share_worst_at_1.25_below_0.99"], dec(alt["share_worst_at_1.25_below_0.99"], 2), SB,
      "added_descriptive.branch_under_alternative_bank_draws.share_worst_at_1.25_below_0.99",
      f"{POST}: share of redraws in which the rule reads 'changes' (the independent check: 0.495)")
    M("nLimSeedRedrawCMin", alt["c_star"]["min"], dec(alt["c_star"]["min"], 2), SB,
      "added_descriptive.branch_under_alternative_bank_draws.c_star.min", f"{POST}: smallest c* over the redraws")
    M("nLimSeedRedrawCMax", alt["c_star"]["max"], dec(alt["c_star"]["max"], 2), SB,
      "added_descriptive.branch_under_alternative_bank_draws.c_star.max", f"{POST}: largest c* over the redraws")
    M("nLimSeedRedrawShareRec", alt["c_star"]["share_equal_or_above_registered"],
      dec(alt["c_star"]["share_equal_or_above_registered"], 3), SB,
      "added_descriptive.branch_under_alternative_bank_draws.c_star.share_equal_or_above_registered",
      f"{POST}: share of redraws with c* >= 1.31")
    sp = ad["spread_of_c_over_independent_runs"]
    M("nLimSeedSpreadRuns", sp["n_runs"], integer(sp["n_runs"]), SB,
      "added_descriptive.spread_of_c_over_independent_runs.n_runs", f"{POST}: independent 4,000-replication runs")
    M("nLimSeedSpreadCWith", sp["c_with_term"]["mean"], dec(sp["c_with_term"]["mean"], 2), SB,
      "added_descriptive.spread_of_c_over_independent_runs.c_with_term.mean", f"{POST}: mean c with the term")
    M("nLimSeedSpreadCWithout", sp["c_without_term"]["mean"], dec(sp["c_without_term"]["mean"], 2), SB,
      "added_descriptive.spread_of_c_over_independent_runs.c_without_term.mean", f"{POST}: mean c without the term")
    M("nLimSeedSpreadDiff", sp["difference_with_minus_without"]["mean"],
      dec(sp["difference_with_minus_without"]["mean"], 2), SB,
      "added_descriptive.spread_of_c_over_independent_runs.difference_with_minus_without.mean",
      f"{POST}: the term lowers c on average (independent check: -0.056 on its own draws)")
    M("nLimSeedSpreadSD", sp["c_with_term"]["sd"], dec(sp["c_with_term"]["sd"], 2), SB,
      "added_descriptive.spread_of_c_over_independent_runs.c_with_term.sd",
      f"{POST}: Monte Carlo SD of c at 4,000 replications")

    # rates at the bound of record (derived; reproduce the filed calibrated values exactly at 0.1752 %)
    dv = S["derived_at_new_limit"]
    assert dv["reproduction_at_H6_calibrated"]["max_rel_diff"] == 0.0
    an = dv["at_new_limit"]
    na1 = an["N_A1_two_candidates_archive_sigma_mu"]
    for nm, k, d in (("TwoFiveHundred", "TPR_G500", 3), ("TwoFiveThousand", "TPR_G5000", 3), ("TwoInf", "TPR_G_inf", 3)):
        M(f"nAttTpr{nm}Rec", na1[k], dec(na1[k], d), SB, f"derived_at_new_limit.at_new_limit.N_A1_two_candidates_"
          f"archive_sigma_mu.{k}", "two candidates, 1 % FPR, transfer at the bound of record, archive sigma_mu "
          "5.23e-05 (named sensitivity row, as nAttTpr*Cal)")
    z = an["calibrated_rows"]["zero"]["M2"]
    M("nAttTprZeroTwoFiveHundredRec", z["TPR_G500"], dec(z["TPR_G500"], 3), SB,
      "derived_at_new_limit.at_new_limit.calibrated_rows.zero.M2.TPR_G500",
      "two candidates, 1 % FPR, bound of record, sigma = 0 (an upper bound for this examiner)")
    M("nAttTprZeroTwoFiveThousandRec", z["TPR_G5000"], dec(z["TPR_G5000"], 2), SB,
      "derived_at_new_limit.at_new_limit.calibrated_rows.zero.M2.TPR_G5000", "as above, 5000 images")
    M("nAttZeroGFiftyExactRec", z["G_for_TPR50"], g2(z["G_for_TPR50"]), SB,
      "derived_at_new_limit.at_new_limit.calibrated_rows.zero.M2.G_for_TPR50",
      "exact images for TPR 0.5, two candidates, bound of record, sigma = 0; two significant figures")
    M("nAttZeroGNinetyExactRec", z["G_for_TPR90"], g2(z["G_for_TPR90"]), SB,
      "derived_at_new_limit.at_new_limit.calibrated_rows.zero.M2.G_for_TPR90", "exact images for TPR 0.9, as above")
    ph = an["calibrated_rows"]["paired_high_95"]["M2"]
    for nm, k in (("FiveHundred", "TPR_G500"), ("FiveThousand", "TPR_G5000"), ("Inf", "TPR_G_inf")):
        M(f"nAttTprTwo{nm}PairedHighRec", ph[k], dec(ph[k], 3), SB,
          f"derived_at_new_limit.at_new_limit.calibrated_rows.paired_high_95.M2.{k}",
          "two candidates, 1 % FPR, bound of record, persistent term at nAttSigmaPairedHigh; TPR 0.5 unreachable")
    sc = an["sigma_critical"]["M2_TPRinf_50"]
    M("nAttSigmaCritHalfRec", sc, sci(sc, 2), SB, "derived_at_new_limit.at_new_limit.sigma_critical.M2_TPRinf_50",
      "persistent term above which no number of images reaches TPR 0.5 (bound of record, two candidates)")
    smr = an["seed_matched_reference"]
    M("nAttSeedMatchedTprFiveHundredRec", smr["TPR_G500"], dec(smr["TPR_G500"], 3), SB,
      "derived_at_new_limit.at_new_limit.seed_matched_reference.TPR_G500",
      "seed-matched reference, single-SE form (as nAttSeedMatchedTprFiveHundredCal), bound of record")
    M("nAttSeedMatchedGFiftyRec", smr["G_for_TPR50"], g2(smr["G_for_TPR50"]), SB,
      "derived_at_new_limit.at_new_limit.seed_matched_reference.G_for_TPR50", "single-SE form; two significant figures")
    M("nAttSeedMatchedGNinetyRec", smr["G_for_TPR90"], g2(smr["G_for_TPR90"]), SB,
      "derived_at_new_limit.at_new_limit.seed_matched_reference.G_for_TPR90", "single-SE form; two significant figures")
    smn = an["seed_matched_null_specific"]["mean_of_orientations"]
    M("nAttSeedMatchedGFiftyNullRec", smn["G_for_TPR50"], g2(smn["G_for_TPR50"]), SB,
      "derived_at_new_limit.at_new_limit.seed_matched_null_specific.mean_of_orientations.G_for_TPR50",
      "seed-matched, null-specific SE (the corrected form, Entry 114 addendum item 3), bound of record")
    M("nAttSeedMatchedTprFiveHundredNullRec", smn["TPR_G500"], dec(smn["TPR_G500"], 2), SB,
      "derived_at_new_limit.at_new_limit.seed_matched_null_specific.mean_of_orientations.TPR_G500",
      "TPR with 500 images, null-specific SE, bound of record (pairs with nAttSeedMatchedGFiftyNullRec)")
    tr = an["training_set_component_persistent"]["M2"]
    M("nAttTrainTprFiveHundredRec", tr["TPR_G500"], dec(tr["TPR_G500"], 3), SB,
      "derived_at_new_limit.at_new_limit.training_set_component_persistent.M2.TPR_G500",
      "two candidates, 1 % FPR, bound of record, persistent term = H6 training-set sd (nLimTrainSD)")
    M("nAttTrainTprFiveThousandRec", tr["TPR_G5000"], dec(tr["TPR_G5000"], 2), SB,
      "derived_at_new_limit.at_new_limit.training_set_component_persistent.M2.TPR_G5000", "as above, 5000 images")
    M("nAttTrainTprInfRec", tr["TPR_G_inf"], dec(tr["TPR_G_inf"], 2), SB,
      "derived_at_new_limit.at_new_limit.training_set_component_persistent.M2.TPR_G_inf", "as above, unlimited")
    M("nAttTrainGFiftyRec", tr["G_for_TPR50"], g2(tr["G_for_TPR50"]), SB,
      "derived_at_new_limit.at_new_limit.training_set_component_persistent.M2.G_for_TPR50",
      "images for TPR 0.5 with that persistent term; two significant figures")
    x50 = an["transfer_for_TPR50_with_50_images_x"]
    for nm, k in (("FreshRec", "fresh_seeds_sigma_zero"), ("FreshMuRec", "fresh_seeds_archive_sigma_mu"),
                  ("MatchedRec", "seed_matched_sigma_zero")):
        M(f"nAttFiftyImgMult{nm}", x50[k], dec(x50[k], 1), SB,
          f"derived_at_new_limit.at_new_limit.transfer_for_TPR50_with_50_images_x.{k}",
          "multiple of the bound of record at which 50 images give a two-candidate TPR of 0.5 at 1 % FPR")

    # ======================================================================== item 2: end-to-end examiner test
    X = load(EX)
    assert X["entry"].startswith("RESULTS.md Entry 116, item 2"), X["entry"]
    assert X["reading_key"] == "agree" and X["cells_below_band"] == [] and X["cells_above_band"] == []
    gname = {10: "Ten", 20: "Twenty", 50: "Fifty", 100: "Hundred"}
    tname = {"1x": "One", "10x": "Ten"}
    for a in X["agreement"]:
        assert a["relation_to_band"] == "meets"
        b = f"nAttEnd{tname[a['target']]}G{gname[a['G']]}"
        key = f"agreement[target={a['target']}, G={a['G']}]"
        what = f"{a['target']} the nominal limit, G = {a['G']}, two candidates, pooled 99th-percentile null threshold"
        M(b, a["empirical_TPR"], sig(a["empirical_TPR"], 2), EX, key + ".empirical_TPR",
          f"empirical TPR, {what}; 18 held-out adapters x 1,000 subsets; subset-sampling SD about 0.001 (1x) to "
          f"0.004-0.009 (10x), not in the interval (independent check)")
        M(b + "Low", a["simultaneous_interval"][0], sig(a["simultaneous_interval"][0], 2), EX,
          key + ".simultaneous_interval.0", "cluster bootstrap over adapters, 2,000, stratified by body, "
          "simultaneous at 1 - 0.05/8, lower end")
        M(b + "High", a["simultaneous_interval"][1], sig(a["simultaneous_interval"][1], 2), EX,
          key + ".simultaneous_interval.1", "same, upper end")
        M(b + "BandLow", a["band"][0], sig(a["band"][0], 3), EX, key + ".band.0",
          "eq. (power) adjusted to this design at sigma_mu = 4.1e-05 (band lower end)")
        M(b + "BandHigh", a["band"][1], sig(a["band"][1], 3), EX, key + ".band.1",
          "eq. (power) adjusted to this design at sigma_mu = 0 (band upper end)")
        M(b + "Unadj", a["model_unadjusted_sigma_mu_0"], sig(a["model_unadjusted_sigma_mu_0"], 3), EX,
          key + ".model_unadjusted_sigma_mu_0", "eq. (power) as printed (SE_500 7.035e-05), sigma_mu = 0")
    mte = X["descriptive"]["model_threshold_examiner"]["cells"]
    for g in (10, 20, 50, 100):
        v = mte[f"G{g}"]["null"]["pooled"]
        M(f"nAttEndFprG{gname[g]}", v, dec(v, 4), EX, f"descriptive.model_threshold_examiner.cells.G{g}.null.pooled",
          "false-positive rate of an examiner using the model's own threshold z_0.99 sigma_G (adjusted, sigma_mu = 0)")
    st = X["settings"]
    M("nAttEndAdapters", 18, integer(18), EX, "inputs.images.examined", "examined adapters, s3-s11 of each body")
    M("nAttEndImages", 18 * 250, integer(18 * 250), EX, "inputs.images.examined (18 x images 0-249)",
      "examined images per condition")
    M("nAttEndSubsets", st["subsets_per_adapter_per_G"], integer(st["subsets_per_adapter_per_G"]), EX,
      "settings.subsets_per_adapter_per_G", "random subsets per adapter, per G and per condition")
    cal = X["calibration"]
    aA, aB = cal["A"]["1x"]["amplitude"], cal["B"]["1x"]["amplitude"]
    M("nAttEndAmpA", aA, sci(aA, 3), EX, "calibration.A.1x.amplitude", "planting amplitude a_A at 1x (Y(1 + a K_E1))")
    M("nAttEndAmpB", aB, sci(aB, 3), EX, "calibration.B.1x.amplitude", "planting amplitude a_B at 1x")
    M("nAttEndAmpRatio", aB / aA, dec(aB / aA, 2), EX, "calibration.B.1x.amplitude / calibration.A.1x.amplitude",
      "body B needs this multiple of A's amplitude (its two estimates agree less)")
    ps = X["planted_shift_examined"]["1x"]
    for b in ("A", "B"):
        r = ps[b]["mean_shift"] / ps["target"]
        M(f"nAttEndShift{b}", r, dec(r, 3), EX, f"planted_shift_examined.1x.{b}.mean_shift / target",
          "achieved shift on the examined images as a multiple of the target (registered 10 % rule met)")
    sh = X["descriptive"]["decomposition_post_hoc"]["adapter_mean_shift_over_T"]["1x"]
    M("nAttEndShiftAdapterMin", min(sh), dec(min(sh), 2), EX,
      "min(descriptive.decomposition_post_hoc.adapter_mean_shift_over_T.1x)", f"{POST}: per-adapter achieved shift / T")
    M("nAttEndShiftAdapterMax", max(sh), dec(max(sh), 2), EX,
      "max(descriptive.decomposition_post_hoc.adapter_mean_shift_over_T.1x)", f"{POST}: per-adapter achieved shift / T")
    ar = X["checks"]["archive_agreement"]["images_0_249_all_24_adapters"]["pooled"]
    M("nAttEndArchiveR", ar["pearson_r"], dec(ar["pearson_r"], 4), EX,
      "checks.archive_agreement.images_0_249_all_24_adapters.pooled.pearson_r",
      "per-image paired contrast, local scorer with E2 vs the archive rows behind the paper's numbers, 6,000 images")
    pb = X["descriptive"]["per_body"]
    for b in ("A", "B"):
        v = pb[b]["G100"]["10x"]["empirical"]
        M(f"nAttEndBody{b}TenGHundred", v, dec(v, 2), EX, f"descriptive.per_body.{b}.G100.10x.empirical",
          "per-body TPR at 10x, G = 100 (descriptive)")
    dc = X["descriptive"]["decomposition_post_hoc"]["cells"]["10x_G100"]
    M("nAttEndDecConst", dc["empirical_null_plus_constant_T"], dec(dc["empirical_null_plus_constant_T"], 3), EX,
      "descriptive.decomposition_post_hoc.cells.10x_G100.empirical_null_plus_constant_T",
      f"{POST}: measured null plus a constant shift T, 10x, G = 100")
    M("nAttEndDecEach", dc["empirical_null_plus_each_adapters_mean_shift"],
      dec(dc["empirical_null_plus_each_adapters_mean_shift"], 3), EX,
      "descriptive.decomposition_post_hoc.cells.10x_G100.empirical_null_plus_each_adapters_mean_shift",
      f"{POST}: measured null plus each adapter's own achieved shift, 10x, G = 100")
    share = max(a["resolution_descriptive"]["share_of_sigma_G2_from_sigma_mu_4.1e-05"] for a in X["agreement"])
    M("nAttEndSigmaMuShare", 100 * share, sig(100 * share, 2), EX,
      "max(agreement[*].resolution_descriptive.share_of_sigma_G2_from_sigma_mu_4.1e-05) x 100",
      "largest share (%) of sigma_G^2 that sigma_mu = 4.1e-05 contributes at G <= 100: the check does not test it")
    nv = X["descriptive"]["null_vs_model"]
    qs = [nv[f"G{g}"]["empirical_q99_over_sd"] for g in (10, 20, 50, 100)]
    M("nAttEndNullQLow", min(qs), dec(min(qs), 2), EX, "min(descriptive.null_vs_model.G*.empirical_q99_over_sd)",
      "null 99th percentile in SDs (2.33 for a normal)")
    M("nAttEndNullQHigh", max(qs), dec(max(qs), 2), EX, "max(descriptive.null_vs_model.G*.empirical_q99_over_sd)",
      "same, largest")

    # ======================================================================== item 3: cross-body scene audit
    Y = load(XB)
    assert Y["entry"].startswith("RESULTS.md Entry 116, item 3"), Y["entry"]
    rp = Y["reading_per_pair"]
    assert rp["d200"] == rp["p20b"] == "the scene overlap does not move the limit" and rp["p5c"].startswith("no near")
    sp_ = Y["summary"]["pairs"]
    d2, p5, p2 = sp_["d200"], sp_["p5c"], sp_["p20b"]
    M("nDataXbThreshold", Y["settings"]["near_copy_threshold"], dec(Y["settings"]["near_copy_threshold"], 2), XB,
      "settings.near_copy_threshold", "near-copy rule: DINOv2-base cosine >= this in either E2 representation")
    M("nDataXbMaxDtwo", d2["max_cross_body_cosine_either"], dec(d2["max_cross_body_cosine_either"], 3), XB,
      "summary.pairs.d200.max_cross_body_cosine_either", "largest cross-body cosine, D200, T(A) x E2(B), E2 crop")
    M("nDataXbMaxDtwoFull", d2["max_cross_body_cosine_E2_full"], dec(d2["max_cross_body_cosine_E2_full"], 3), XB,
      "summary.pairs.d200.max_cross_body_cosine_E2_full", "same, E2 full photograph (the v1 audit's input)")
    oth = d2["max_by_direction"]["T(B) x E2(A)"]["either"]
    M("nDataXbMaxDtwoOther", oth, dec(oth, 3), XB, "summary.pairs.d200.max_by_direction.T(B) x E2(A).either",
      "largest cosine in the other direction, D200")
    for nm, p in (("Dtwo", d2), ("Ptwenty", p2)):
        M(f"nDataXbNear{nm}", p["n_near_copies"], integer(p["n_near_copies"]), XB,
          f"summary.pairs.{'d200' if nm == 'Dtwo' else 'p20b'}.n_near_copies", "cross-body near-copy pairs")
        M(f"nDataXbAffected{nm}", p["n_affected_E2_photographs"], integer(p["n_affected_E2_photographs"]), XB,
          f"summary.pairs.{'d200' if nm == 'Dtwo' else 'p20b'}.n_affected_E2_photographs",
          "E2 photographs excluded in the re-estimate")
    M("nDataXbNearTotal", Y["n_near_copies_total"], integer(Y["n_near_copies_total"]), XB, "n_near_copies_total",
      "near-copy pairs over the three pairs")
    M("nDataXbMaxPfive", p5["max_cross_body_cosine_either"], dec(p5["max_cross_body_cosine_either"], 3), XB,
      "summary.pairs.p5c.max_cross_body_cosine_either", "iPhone 5c pair: largest cross-body cosine (no near-copy)")
    M("nDataXbMaxPtwenty", p2["max_cross_body_cosine_either"], dec(p2["max_cross_body_cosine_either"], 3), XB,
      "summary.pairs.p20b.max_cross_body_cosine_either", "P20 pair: largest cross-body cosine, T(B) x E2(A), crop")
    nx = p2["max_by_direction"]["T(B) x E2(A)"]["largest_cos_below_threshold"]
    M("nDataXbPtwentyNext", nx, dec(nx, 3), XB,
      "summary.pairs.p20b.max_by_direction.T(B) x E2(A).largest_cos_below_threshold",
      "P20: the next pair below the threshold (the second near-copy clears it by 0.0005; both are textureless walls)")
    tt = Y["descriptive"]["d200"]["T(A) x T(B)"]["counts"]["0.90"]["pairs"]
    M("nDataXbTrainPairsDtwo", tt, integer(tt), XB, "descriptive.d200.T(A) x T(B).counts.0.90.pairs",
      f"{POST}: D200 training-crop pairs across bodies at cosine >= 0.90 (the two training sets share scenes)")
    hmax = max(Y["descriptive"][p][dn]["any_representation"]["max"] for p in ("d200", "p5c", "p20b")
               for dn in ("H(A) x E2(B)", "H(B) x E2(A)"))
    M("nDataXbHeldMax", hmax, dec(hmax, 3), XB, "max(descriptive.*.H(X) x E2(Y).any_representation.max)",
      "largest H x E2 cross-body cosine (R_real's photographs); no near-copy")
    sens = Y["sensitivity"]["pairs"]
    rd = sens["d200"]["rule_exclusion_0.90"]
    for nm, k in (("Nom", "max_arm_nominal"), ("Cal", "max_arm_calibrated_c_H6"), ("Rec", "max_arm_calibrated_c_item1")):
        M(f"nLimXbDtwo{nm}Change", rd[k]["relative_change"], pct1(rd[k]["relative_change"]), XB,
          f"sensitivity.pairs.d200.rule_exclusion_0.90.{k}.relative_change",
          "relative change (%) of the D200 max-arm limit when K_B^E2 is re-estimated without the near-copy "
          "photographs; 24 primary adapters, images 0-249, local instrument (arm A binds there); rule line 10 %")
    M("nLimXbDtwoNomOld", rd["max_arm_nominal"]["old_pct"], sig(rd["max_arm_nominal"]["old_pct"], 3), XB,
      "sensitivity.pairs.d200.rule_exclusion_0.90.max_arm_nominal.old_pct",
      "the 250-image basis's nominal max-arm limit before the exclusion (not the limit of record)")
    M("nLimXbDtwoNomNew", rd["max_arm_nominal"]["new_pct"], sig(rd["max_arm_nominal"]["new_pct"], 3), XB,
      "sensitivity.pairs.d200.rule_exclusion_0.90.max_arm_nominal.new_pct", "same, after the exclusion")
    M("nLimXbDtwoRrealChange", rd["R_real"]["relative_change"], pct1(rd["R_real"]["relative_change"]), XB,
      "sensitivity.pairs.d200.rule_exclusion_0.90.R_real.relative_change", "relative change (%) of R_real, D200")
    rp2 = sens["p20b"]["rule_exclusion_0.90"]
    M("nLimXbPtwentyNomChange", rp2["max_arm_nominal"]["relative_change"],
      pct1(rp2["max_arm_nominal"]["relative_change"]), XB,
      "sensitivity.pairs.p20b.rule_exclusion_0.90.max_arm_nominal.relative_change",
      "relative change (%) of the P20 nominal max-arm limit (all 12 x 250 per arm)")
    M("nLimXbPtwentyNomNew", rp2["max_arm_nominal"]["new_pct"], sig(rp2["max_arm_nominal"]["new_pct"], 3), XB,
      "sensitivity.pairs.p20b.rule_exclusion_0.90.max_arm_nominal.new_pct",
      "P20 nominal max-arm limit after the exclusion (before: nGenGfourbMaxArm)")
    M("nLimXbPtwentyRrealChange", rp2["R_real"]["relative_change"], pct1(rp2["R_real"]["relative_change"]), XB,
      "sensitivity.pairs.p20b.rule_exclusion_0.90.R_real.relative_change", "relative change (%) of R_real, P20")
    M("nLimXbPtwentySymP", rp2["theta_sym"]["new_p"], dec(rp2["theta_sym"]["new_p"], 3), XB,
      "sensitivity.pairs.p20b.rule_exclusion_0.90.theta_sym.new_p",
      "P20 symmetric one-sided p after the exclusion (before: nGenGfourbPFine)")
    bs = sens["d200"]["seed_bootstrap_not_prespecified"]["x90"]["constructions"]["x90:max_arm_nominal"]
    M("nLimXbDtwoBootSD", bs["boot_sd"], pct1(bs["boot_sd"]), XB,
      "sensitivity.pairs.d200.seed_bootstrap_not_prespecified.x90.constructions.x90:max_arm_nominal.boot_sd",
      f"{POST}: SD (percentage points) of the D200 nominal change over 2,000 seed resamples; centred near "
      f"{100 * bs['boot_mean']:.1f} %")
    M("nLimXbDtwoBootBeyond", bs["frac_abs_gt_0.10"], pct1(bs["frac_abs_gt_0.10"]), XB,
      "sensitivity.pairs.d200.seed_bootstrap_not_prespecified.x90.constructions.x90:max_arm_nominal.frac_abs_gt_0.10",
      f"{POST}: % of seed resamples whose D200 nominal change exceeds 10 % in size")
    bp = sens["p20b"]["seed_bootstrap_not_prespecified"]["x90"]["constructions"]["x90:max_arm_nominal"]
    M("nLimXbPtwentyBootBeyond", bp["frac_abs_gt_0.10"], pct1(bp["frac_abs_gt_0.10"]), XB,
      "sensitivity.pairs.p20b.seed_bootstrap_not_prespecified.x90.constructions.x90:max_arm_nominal.frac_abs_gt_0.10",
      f"{POST}: same, P20")
    lt = sens["d200"]["ledger_transport_not_prespecified"]["x90"]
    for nm, k in (("Nom", "nominal"), ("Rec", "calibrated_c_item1")):
        M(f"nLimXbDtwoTransport{nm}", lt[k]["arm_mean_shift_relative_change"],
          pct1(lt[k]["arm_mean_shift_relative_change"]), XB,
          f"sensitivity.pairs.d200.ledger_transport_not_prespecified.x90.{k}.arm_mean_shift_relative_change",
          f"{POST}: the measured arm-mean shifts added to the ledger values (500-image basis, arm B binds): "
          f"relative change (%) of the limit")
        M(f"nLimXbDtwoTransport{nm}PerAdapter", lt[k]["per_adapter_shifts_relative_change"],
          pct1(lt[k]["per_adapter_shifts_relative_change"]), XB,
          f"sensitivity.pairs.d200.ledger_transport_not_prespecified.x90.{k}.per_adapter_shifts_relative_change",
          f"{POST}: same with each adapter's own shift")
    pl = d2["placebo_comparison_not_prespecified"]["max_arm_nominal"]
    M("nLimXbDtwoPlaceboMin", pl["placebo_min"], pct1(pl["placebo_min"]), XB,
      "summary.pairs.d200.placebo_comparison_not_prespecified.max_arm_nominal.placebo_min",
      f"{POST}: six random exclusions of as many kept photographs, smallest change (%)")
    M("nLimXbDtwoPlaceboMax", pl["placebo_max"], pct1(pl["placebo_max"]), XB,
      "summary.pairs.d200.placebo_comparison_not_prespecified.max_arm_nominal.placebo_max",
      f"{POST}: same, largest change (%)")

    # ======================================================================== item 4: normal-versus-inverted weights
    W = load(WT)
    R = load(WR)
    assert W["entry"].startswith("RESULTS.md Entry 116, item 4"), W["entry"]
    assert R["comparison_with_file_of_record"]["all_agree"] is True
    pw = W["primary"]
    assert pw["p_one_sided_exact"] == pw["floor"] == 1 / 1024 and pw["D"] > 0
    M("nWtInvD", pw["D"], sci(pw["D"], 3), WT, "primary.D",
      "registered statistic D = (D_A + D_B)/2: within-condition minus cross-condition mean cosine of the weight "
      "updates, different-seed pairs, seeds 0-5, normal vs inverted 16000-step adapters")
    pbw = W["descriptive"]["per_body"]
    for b in ("A", "B"):
        M(f"nWtInvD{b}", pbw[b]["D_X"], sci(pbw[b]["D_X"], 3), WT, f"descriptive.per_body.{b}.D_X", f"body {b}'s D_X")
    M("nWtInvP", pw["p_one_sided_exact"], sig(pw["p_one_sided_exact"], 2), WT, "primary.p_one_sided_exact",
      "exact one-sided relabelling p (registered level 0.01); equals the floor")
    M("nWtInvFloor", pw["floor"], "1/1024", WT, "primary.floor", "the design's floor (1,024 distinct relabellings)")
    M("nWtInvPerBodyP", pbw["A"]["p_one_sided_exact"], sig(pbw["A"]["p_one_sided_exact"], 2), WT,
      "descriptive.per_body.A.p_one_sided_exact", "each body's own exact p, at its floor 1/32 (both bodies)")
    M("nWtInvZ", pw["null_summary"]["z_of_observed"], dec(pw["null_summary"]["z_of_observed"], 1), WT,
      "primary.null_summary.z_of_observed", "observed D in SDs of the 1,024 relabellings")
    tc = W["descriptive"]["twin_cosines"]
    tmin = min(tc["A"]["min"], tc["B"]["min"])
    tmax = max(tc["A"]["max"], tc["B"]["max"])
    M("nWtInvTwinMin", tmin, dec(tmin, 3), WT, "min(descriptive.twin_cosines.*.min)",
      "same-seed normal-inverted (twin) cosine, smallest over both bodies")
    M("nWtInvTwinMax", tmax, dec(tmax, 3), WT, "max(descriptive.twin_cosines.*.max)", "same, largest")
    mv = W["descriptive"]["matched_variant"]
    M("nWtInvMatched", mv["M"], sig(mv["M"], 3), WT, "descriptive.matched_variant.M",
      "matched variant: mean cosine among twin differences (p at its floor 1/1024); one finding with D")
    s67 = W["descriptive"]["seeds_6_7_kept"]
    M("nWtInvAllSeedsD", s67["D"], sci(s67["D"], 3), WT, "descriptive.seeds_6_7_kept.D",
      "D with inverted seeds 6-7 kept (p at its floor 1/4096)")
    lo_ = R["skeptic_checks_not_prespecified"]["F1_per_pair_and_leave_one_seed_out"]["leave_one_seed_out"]
    loo = [v["D"] for v in lo_.values()]
    M("nWtInvLooMin", min(loo), sci(min(loo), 3), WR,
      "min(skeptic_checks_not_prespecified.F1_per_pair_and_leave_one_seed_out.leave_one_seed_out.*.D)",
      f"{POST}: D leaving one seed pair out, smallest (each at its floor 1/256)")
    M("nWtInvLooMax", max(loo), sci(max(loo), 3), WR,
      "max(skeptic_checks_not_prespecified.F1_per_pair_and_leave_one_seed_out.leave_one_seed_out.*.D)",
      f"{POST}: same, largest")
    nm_ = W["descriptive"]["norms"]
    nd = []
    for b in ("A", "B"):
        nn, ii = nm_[b]["normal"], nm_[b]["inverted"]
        for s, v in zip(nn["seeds"], nn["dW_frobenius"]):
            nd.append(ii["dW_frobenius"][ii["seeds"].index(s)] - v)
    M("nWtInvNormDiff", statistics.fmean(nd), dec(statistics.fmean(nd), 3), WT,
      "mean over bodies and seeds 0-5 of descriptive.norms.*.inverted.dW_frobenius - normal.dW_frobenius",
      "mean twin difference of the weight-update norm (mixed signs)")
    sk = R["skeptic_checks_not_prespecified"]
    f9 = sk["F9_alignment_across_patterns"]
    grp = f9["groups"]
    diag = {g: f9["mean_cosine"][i][i] for i, g in enumerate(grp)}
    fixed = [v for g, v in diag.items() if g not in ("colab", "inverted_16000_A", "inverted_16000_B")]
    M("nWtInvPatternMin", min(fixed), dec(min(fixed), 4), WR,
      "min over fixed-pattern groups of skeptic_checks_not_prespecified.F9_alignment_across_patterns.mean_cosine diag",
      f"{POST}: alignment of twin differences for 2000-step arms that add one fixed pattern to body A's crops "
      f"(random field to tiles and DiffusionShield); 2-3 seeds each")
    M("nWtInvPatternMax", max(fixed), dec(max(fixed), 3), WR,
      "max over fixed-pattern groups of the F9 diagonal", f"{POST}: same, largest")
    M("nWtInvColab", diag["colab"], dec(diag["colab"], 4), WR, "F9 diagonal, group colab",
      f"{POST}: a decoder change with no shared pattern (v1 Colab crops)")
    M("nWtInvAlignA", diag["inverted_16000_A"], dec(diag["inverted_16000_A"], 4), WR,
      "F9 diagonal, group inverted_16000_A", f"{POST}: the inverted condition's alignment, body A")
    M("nWtInvAlignB", diag["inverted_16000_B"], dec(diag["inverted_16000_B"], 4), WR,
      "F9 diagonal, group inverted_16000_B", f"{POST}: same, body B")
    f2 = sk["F2_common_versus_body_specific"]
    M("nWtInvCrossBody", f2["cross_body_delta_alignment_different_seed_mean_30"],
      dec(f2["cross_body_delta_alignment_different_seed_mean_30"], 4), WR,
      "skeptic_checks_not_prespecified.F2_common_versus_body_specific.cross_body_delta_alignment_different_seed_mean_30",
      f"{POST}: alignment of twin differences across bodies (different seeds); a component shared by both bodies")
    M("nWtInvCommonShare", f2["common_share_of_within_alignment"], pct1(f2["common_share_of_within_alignment"]), WR,
      "skeptic_checks_not_prespecified.F2_common_versus_body_specific.common_share_of_within_alignment",
      f"{POST}: that shared part as % of the within-body alignment; the design cannot test a body-specific part")
    f5 = sk["F5_inverted_crops_rebuilt_and_decomposed"]["bodies"]
    for b in ("A", "B"):
        sh_ = f5[b]["share_of_mean_square"]["fingerprint_term"]
        M(f"nWtInvFpShare{b}", sh_, pct1(sh_), WR,
          f"skeptic_checks_not_prespecified.F5_inverted_crops_rebuilt_and_decomposed.bodies.{b}.share_of_mean_square."
          f"fingerprint_term", f"{POST}: % of the stored change's energy that is the reversed-fingerprint term, body {b}")
    dr_ = f5["A"]["rms_gray"]["dither_rounding_clipping"]
    M("nWtInvDitherRms", dr_, dec(dr_, 2), WR,
      "skeptic_checks_not_prespecified.F5_inverted_crops_rebuilt_and_decomposed.bodies.A.rms_gray."
      "dither_rounding_clipping", f"{POST}: RMS (gray) of the dither-plus-rounding part, drawn afresh per crop")

    # ======================================================================== item 5: closed-set expectation
    C = load(CS)
    assert C["entry"].startswith("RESULTS.md Entry 116, item 5"), C["entry"]
    assert C["primary_limits"]["item1"]["included"] is True
    assert abs(C["primary_limits"]["item1"]["pct"] - rec["lambda_U_pct_of_R_real"]) < 1e-12
    rows = {(r["group"], r["level"]): r for r in C["summary_table"]}
    gk = {"Kodak": "kodak", "Ptwenty": "p20"}
    lv = {"Dev": "a_device_limit", "Nom": "b_nominal", "Rec": "b_item1", "Zero": "c_zero"}
    for gn, g in gk.items():
        for ln, l in lv.items():
            r = rows[(g, l)]
            bt_, pa_ = r["bootstrap_two_stage"], r["parametric_sigma_mu_0"]
            src = f"summary_table[group={g}, level={l}]"
            M(f"nAttCs{gn}{ln}E", bt_["E_accuracy"], dec(bt_["E_accuracy"], 3), CS, src + ".bootstrap_two_stage.E_accuracy",
              "expected closed-set accuracy (10 adapters, G = 250, corrected argmax), pre-specified two-stage "
              "bootstrap, 2,000 replicates (MC SE about 0.003); descriptive, no reading")
            if ln != "Zero":
                M(f"nAttCs{gn}{ln}EPar", pa_["E_accuracy"], dec(pa_["E_accuracy"], 3), CS,
                  src + ".parametric_sigma_mu_0.E_accuracy", "same, parametric model at sigma_mu = 0, 100,000 experiments")
            if ln in ("Nom", "Rec"):
                M(f"nAttCs{gn}{ln}LeThree", bt_["P_X_le_observed"], sig(bt_["P_X_le_observed"], 2), CS,
                  src + ".bootstrap_two_stage.P_X_le_observed", "P(3 or fewer correct of 10), bootstrap")
                M(f"nAttCs{gn}{ln}LeThreePar", pa_["P_X_le_observed"], sig(pa_["P_X_le_observed"], 2), CS,
                  src + ".parametric_sigma_mu_0.P_X_le_observed", "same, parametric sigma_mu = 0")
            if ln == "Dev":
                assert bt_["P_X_le_observed"] == 0.0 and pa_["P_X_le_observed"] == 0.0
                assert r["parametric_sigma_mu_4.108e-05"]["P_X_le_observed"] == 0.0
    M("nAttCsBootReps", 2000, integer(2000), CS, "groups.*.bootstrap_two_stage.n_replicates", "bootstrap replicates")
    M("nAttCsParReps", C["settings"]["n_parametric"], integer(C["settings"]["n_parametric"]), CS,
      "settings.n_parametric", "parametric experiments; 3 or fewer correct at the device-level limit in none of them")
    for gn, g in gk.items():
        s4 = C["groups"][g]["S4_lean_removed_first"]["levels"]
        k250 = "first_250_images (the closed-set images)"
        v = s4["b_nominal"][k250]["bootstrap_two_stage"]["E_accuracy"]
        M(f"nAttCs{gn}NomLean", v, dec(v, 3), CS,
          f"groups.{g}.S4_lean_removed_first.levels.b_nominal.{k250}.bootstrap_two_stage.E_accuracy",
          f"{POST}: nominal level with the closed-set images' own lean removed first (bootstrap)")
        v = s4["a_device_limit"][k250]["bootstrap_two_stage"]["E_accuracy"]
        M(f"nAttCs{gn}DevLean", v, dec(v, 3), CS,
          f"groups.{g}.S4_lean_removed_first.levels.a_device_limit.{k250}.bootstrap_two_stage.E_accuracy",
          f"{POST}: device-level limit with the lean removed first (bootstrap)")
        ln_ = C["groups"][g]["group_statistic"]["same_statistic_first_250_images (descriptive)"]["lambda_hat_pct"]
        M(f"nAttCs{gn}LeanPct", ln_, sig(ln_, 2), CS,
          f"groups.{g}.group_statistic.same_statistic_first_250_images (descriptive).lambda_hat_pct",
          "the closed-set images' own lean (group transfer statistic on the first 250 images), % of the group's R_real")
    ch = C["chance_reference"]
    M("nAttCsChanceLeThree", ch["P_X_le_3"], dec(ch["P_X_le_3"], 3), CS, "chance_reference.P_X_le_3",
      "Binomial(10, 0.2): P(X <= 3)")
    M("nAttCsChanceGeFive", ch["P_X_ge_5"], dec(ch["P_X_ge_5"], 3), CS, "chance_reference.P_X_ge_5",
      "Binomial(10, 0.2): P(X >= 5)")
    M("nAttCsChanceGeSix", ch["P_X_ge_6"], dec(ch["P_X_ge_6"], 4), CS, "chance_reference.P_X_ge_6",
      "Binomial(10, 0.2): P(X >= 6)")

    # ======================================================================== item 6: small quantities
    Q = load(SM)
    assert Q["entry"].startswith("RESULTS.md Entry 116, item 6"), Q["entry"]
    qa = Q["a_hostile_r2_04_local_stack_max_arm"]
    M("nLimLocalMaxArm", qa["nominal_six_per_arm"]["lambda_U_pct"], sig(qa["nominal_six_per_arm"]["lambda_U_pct"], 3),
      SM, "a_hostile_r2_04_local_stack_max_arm.nominal_six_per_arm.lambda_U_pct",
      "Table 5 row 3: nominal max-arm limit (c = 1) of the primary design retrained on the local stack, six per "
      "arm, % of R_real; set by arm A")
    M("nLimLocalMaxArmB", qa["nominal_six_per_arm"]["lambda_U_B_pct"],
      sig(qa["nominal_six_per_arm"]["lambda_U_B_pct"], 3), SM,
      "a_hostile_r2_04_local_stack_max_arm.nominal_six_per_arm.lambda_U_B_pct", "arm B's limit alone")
    M("nLimLocalMaxArmNoFive", qa["post_hoc_without_nomark_s5"]["lambda_U_pct"],
      sig(qa["post_hoc_without_nomark_s5"]["lambda_U_pct"], 3), SM,
      "a_hostile_r2_04_local_stack_max_arm.post_hoc_without_nomark_s5.lambda_U_pct",
      f"{POST} (labelled so in Entry 116): without nomark_s5, arm A at five adapters; still set by arm A")
    qb = Q["b_hostile_r2_14_inflated_p"]["results"]
    for nm, k in (("Rep", "entry68_replication_new_three_per_body"), ("Sixteen", "entry68_pooled_six_per_body"),
                  ("Inv", "g1_inversion_eight_per_arm")):
        M(f"nDose{nm}PInfl", qb[k]["inflated"]["one_sided_p_infl"], sig(qb[k]["inflated"]["one_sided_p_infl"], 2), SM,
          f"b_hostile_r2_14_inflated_p.results.{k}.inflated.one_sided_p_infl",
          "one-sided p with the E2 estimation shift added (SE_infl = sqrt(SE_Welch^2 + 1.224e-05^2), Welch df); "
          "estimation SD transported from 2000 steps / 200 images to 16000 steps / 250 images")
        M(f"nDose{nm}PInflHigh", qb[k]["inflated_at_sd_est_upper95"]["one_sided_p_infl"],
          sig(qb[k]["inflated_at_sd_est_upper95"]["one_sided_p_infl"], 2), SM,
          f"b_hostile_r2_14_inflated_p.results.{k}.inflated_at_sd_est_upper95.one_sided_p_infl",
          "same with the estimation SD at its one-sided 95 % upper limit (1.62e-05; not registered)")
    qd = Q["d_figures_r2_11_nDoseInvThreeT"]
    M("nDoseInvThreeT", qd["t"], dec(qd["t"], 2), SM, "d_figures_r2_11_nDoseInvThreeT.t",
      "Welch t of the first inversion reading, seeds 0-2 per arm (reproduces Entry 98)")
    qe = Q["e_trace_r2_09_counts"]
    # 1 Oct 2026: the verifier was extended with the Entry 114-119 checks; the count is read from the re-run
    # (src/fv/fv_verifier_count.py -> out/fv_verifier_count.json); fv_small.json keeps the earlier count (76)
    VC = OUT + "/fv_verifier_count.json"
    qv = load(VC)["verifier"]
    assert qv["checks_agreeing"] == qv["checks_executed"] and qv["exit_code"] == 0, qv
    M("nDataVerifierChecks", qv["checks_agreeing"], integer(qv["checks_agreeing"]), VC,
      "verifier.checks_agreeing",
      f"verify_v2.py checks agreeing of {qv['checks_executed']} executed (working tree; not yet pushed; was "
      f"{qe['verifier']['checks_agreeing']} in fv_small.json before the Entry 114-119 checks were added)")
    M("nDataVerifierChecksPublic", qe["verifier"]["committed_HEAD_verifier"]["checks_agreeing"],
      integer(qe["verifier"]["committed_HEAD_verifier"]["checks_agreeing"]), SM,
      "e_trace_r2_09_counts.verifier.committed_HEAD_verifier.checks_agreeing",
      "the committed (public) verifier's checks, all agreeing; becomes nDataVerifierChecks once pushed")
    bsk = qe["table2b_by_stack"]["by_stack"]
    assert qe["table2b_by_stack"]["total_adapters"] == 211 and qe["table2b_by_stack"]["total_generations"] == 95500
    for nm, k in (("Archive", "archive"), ("Local", "local")):
        M(f"nData{nm}Adapters", bsk[k]["adapters"], integer(bsk[k]["adapters"]), SM,
          f"e_trace_r2_09_counts.table2b_by_stack.by_stack.{k}.adapters", f"Table 2(b) adapters, stack '{k}'")
        M(f"nData{nm}Gens", bsk[k]["generations"], integer(bsk[k]["generations"]), SM,
          f"e_trace_r2_09_counts.table2b_by_stack.by_stack.{k}.generations", f"Table 2(b) generations, stack '{k}'")
    M("nDataArchiveLocalGens", bsk["archive/local"]["generations"], integer(bsk["archive/local"]["generations"]), SM,
      "e_trace_r2_09_counts.table2b_by_stack.by_stack.archive/local.generations",
      "Table 2(b) generations, stack 'archive/local' (its adapters are counted under archive)")
    qc = Q["c_trace_r2_12_gpu_hours"]
    na5 = qc["registered_estimate"]["N-A5 (211 adapters, 95,500 generations)"]
    col = na5["A100 archive and Colab runs, including FLUX.1-dev"]
    loc = na5["L40S local"]
    allg = na5["all GPUs"]
    M("nDataGpuColabRule", col["total_low_h"], sig(col["total_low_h"], 2), SM,
      "c_trace_r2_12_gpu_hours.registered_estimate.N-A5.A100 archive and Colab runs, including FLUX.1-dev.total_low_h",
      "Colab GPU-hours under the registered rule (recorded training + S3 per-unit generation); TOO LOW: S3's A100 "
      "generation time (20 min per 500) is contradicted by the write times (about 55 min); FLUX and full fine-tuning "
      "generation not estimated")
    M("nDataGpuLocalRuleLow", loc["total_low_h"], integer(loc["total_low_h"]), SM,
      "c_trace_r2_12_gpu_hours.registered_estimate.N-A5.L40S local.total_low_h",
      "L40S GPU-hours under the registered rule, low end (generation part a floor)")
    M("nDataGpuLocalRuleHigh", loc["total_high_h"], integer(loc["total_high_h"]), SM,
      "c_trace_r2_12_gpu_hours.registered_estimate.N-A5.L40S local.total_high_h", "same, high end")
    M("nDataGpuTrainRecorded", allg["training_recorded_h"], integer(allg["training_recorded_h"]), SM,
      "c_trace_r2_12_gpu_hours.registered_estimate.N-A5.all GPUs.training_recorded_h",
      "recorded training hours, all cards (210 of 211 adapters have a record)")
    bes = qc["beside_not_registered"]["N-A5"]
    M("nDataGpuColabMeasured", bes["A100 archive and Colab runs, including FLUX.1-dev"]["total_h"],
      integer(bes["A100 archive and Colab runs, including FLUX.1-dev"]["total_h"]), SM,
      "c_trace_r2_12_gpu_hours.beside_not_registered.N-A5.A100 archive and Colab runs, including FLUX.1-dev.total_h",
      f"{POST}: Colab GPU-hours with generation measured from the images' write times (includes FLUX and full "
      f"fine-tuning generation)")
    M("nDataGpuLocalMeasured", bes["L40S local"]["total_h"], integer(bes["L40S local"]["total_h"]), SM,
      "c_trace_r2_12_gpu_hours.beside_not_registered.N-A5.L40S local.total_h",
      f"{POST}: L40S job-hours with generation measured from write times")
    M("nDataGpuLocalOccupied", bes["L40S_occupancy"]["union_hours_training_and_generation"],
      integer(bes["L40S_occupancy"]["union_hours_training_and_generation"]), SM,
      "c_trace_r2_12_gpu_hours.beside_not_registered.N-A5.L40S_occupancy.union_hours_training_and_generation",
      f"{POST}: hours the L40S was occupied (jobs overlapped)")
    a5 = qc["beside_not_registered"]["per_job_generation_minutes_vs_S3"]["A100_40GB_500"]
    M("nDataGpuAHundredGenMin", a5["median"], integer(a5["median"]), SM,
      "c_trace_r2_12_gpu_hours.beside_not_registered.per_job_generation_minutes_vs_S3.A100_40GB_500.median",
      f"{POST}: median minutes per 500 generations on the A100-40GB ({a5['jobs']} jobs; S3 says about 20)")
    return L


if __name__ == "__main__":
    ms = macros()
    names = [m["name"] for m in ms]
    assert len(names) == len(set(names)), "duplicate within n9"
    for m in ms:
        print(f"{m['name']:40s} {m['text']}")
    print(len(ms), "macros")
