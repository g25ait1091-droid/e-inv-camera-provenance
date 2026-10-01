"""Number macros, part n8: post-hoc checks from the manuscript review (RESULTS.md Entry 114).

Every value is READ at run time from out/fv_sigma.json (src/fv/fv_sigma.py: the persistent term of the attribution
power model, estimated from the paired contrast of the 24 primary adapters) and out/fv_weights_posthoc.json
(src/fv/fv_weights_posthoc.py: the H5 weight comparison with same-seed pairs set aside). None of these quantities
was pre-specified; they are descriptive or sensitivity analyses, and no registered reading changes.

Quantities that already have a macro are not duplicated (OUTLINE 8.1, one quantity one macro):
  * sigma = 0 at the nominal limit: nAttTprZeroTwoFiveHundred, nAttTprZeroTwoFiveThousand, nAttZeroGFiftyExact,
    nAttZeroGNinetyExact (num_n4). fv_sigma.json reproduces them (reproduction.numbers_json_macros). The
    paired-contrast point estimate of the persistent term is 0, so its rates ARE these macros; no separate
    "Paired" rate macro exists.
  * same-body mean cosines (nWtTwoSameBody, nWtSixteenSameBody) and different-batch means (nWtTwoCrossBatch,
    nWtSixteenCrossBatch): no same-body or different-batch pair shares a seed, so setting same-seed pairs aside
    leaves them unchanged (asserted in fv_weights_posthoc.py).
  * the training-set component of H6: nLimTrainSD.
Addendum to Entry 114 (the independent check, 30 Sep 2026): nAttGtwoOutlierZ now uses the outlier's own per-seed
deviations (3.0, not 4.0); coverage macros print two decimals; nWtSameBatchNoSeed* print three significant
figures; new macros carry the heavy-tail p of the archive value, the null-specific seed-matched figures, the H6
training-set component as a persistent term, and the Table S16 columns at the paired upper limit.
Image counts are given to two significant figures, as nAttZeroGFiftyExact (4600) and nAttZeroGNinetyExact (11,000)
are; counts at ten times the limit are rounded up, as nAttTenxG*Exact are.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_numlib import OUT, sig, dec, sci, integer, macro  # noqa: E402

SIG = OUT + "/fv_sigma.json"
WTS = OUT + "/fv_weights_posthoc.json"
E = "114"


def _load(p):
    import json
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def g2(x):
    """An image count to two significant figures (4633.5 -> 4600, 11144.8 -> 11000)."""
    return integer(float(f"{x:.2g}"))


def tprtxt(x):
    return dec(x, 3) if x < 0.5 else dec(x, 2)


def macros():
    S = _load(SIG)
    W = _load(WTS)
    assert S.get("entry") == "RESULTS.md Entry 114", S.get("entry")
    assert W.get("entry") == "RESULTS.md Entry 114", W.get("entry")
    L = []

    def M(name, value, text, src, key, check):
        L.append(macro(name, value, text, src, key, E, check))

    pc = S["primary_crossed_model"]
    pool = pc["pooled"]
    ci = pc["sigma_u_interval_exact"]
    sn = S["seed_effect_and_image_noise"]
    pw = S["power"]
    post = "post hoc, not pre-specified"

    # ---------------------------------------------------------------- the persistent term, primary design
    M("nAttSigmaPaired", pc["sigma_u_estimate"], sci(pc["sigma_u_estimate"], 2), SIG,
      "primary_crossed_model.sigma_u_estimate",
      f"pooled REML sigma_u of the paired contrast, 24 primary adapters x 500 images, crossed adapter x seed model "
      f"(MoM variance {pool['sigma_u2_mom']:.2e}, on the boundary); statsmodels MixedLM agrees; {post}")
    M("nAttSigmaPairedLow", ci["two_sided_95"][0], sci(ci["two_sided_95"][0], 2), SIG,
      "primary_crossed_model.sigma_u_interval_exact.two_sided_95.0", "exact normal-theory 95 % interval, lower end")
    M("nAttSigmaPairedHigh", ci["two_sided_95"][1], sci(ci["two_sided_95"][1], 2), SIG,
      "primary_crossed_model.sigma_u_interval_exact.two_sided_95.1",
      "exact normal-theory 95 % interval, upper end (F on 22 and 10,978 df); calibrated in the coverage simulation")
    M("nAttSigmaPairedUpperNinetyNine", ci["one_sided_99_upper"], sci(ci["one_sided_99_upper"], 2), SIG,
      "primary_crossed_model.sigma_u_interval_exact.one_sided_99_upper", "exact one-sided 99 % upper limit")
    ups95 = [ci["one_sided_95_upper"], pc["sigma_u_bootstrap_adapters"]["percentile_95_upper_one_sided"],
             pc["sigma_u_bootstrap_adapters_and_seeds"]["percentile_95_upper_one_sided"]]
    M("nAttSigmaUpperNinetyFiveMax", max(ups95), sci(max(ups95), 2), SIG,
      "max(exact one_sided_95_upper, bootstrap_adapters and bootstrap_adapters_and_seeds percentile_95_upper_one_sided)",
      f"largest one-sided 95 % upper limit over the three methods ({', '.join(f'{u:.2e}' for u in ups95)}); "
      f"the archive sigma_mu lies above it")
    M("nAttSigmaPairedF", pool["F_adapter"], dec(pool["F_adapter"], 2), SIG, "primary_crossed_model.pooled.F_adapter",
      f"MS_adapter / MS_resid on {pool['df_adapter']} and {pool['df_resid']} df")
    M("nAttSigmaPairedP", pool["F_p_upper"], dec(pool["F_p_upper"], 2), SIG, "primary_crossed_model.pooled.F_p_upper",
      "upper-tail F p for a persistent component > 0")
    p_mu = pc["archive_sigma_mu_one_sided_p"]
    M("nAttSigmaMuP", p_mu, dec(p_mu, 3), SIG, "primary_crossed_model.archive_sigma_mu_one_sided_p",
      "one-sided p of sigma_u >= 5.23e-05 (the archive value) given the 24 adapters, normal theory")
    sd = pc["sd_adapter_means"]
    for nm, k in (("Obs", "observed_pooled_within_arm"), ("Noise", "predicted_image_noise_only"),
                  ("Archive", "predicted_with_archive_sigma_mu")):
        M(f"nAttAdapterSd{nm}", sd[k], sci(sd[k], 2), SIG, f"primary_crossed_model.sd_adapter_means.{k}",
          "spread of the 500-image adapter means within arm (observed; predicted by image noise alone; predicted "
          "with the archive sigma_mu)")
    M("nAttSplitHalfR", pc["split_half_r"], dec(pc["split_half_r"], 2), SIG, "primary_crossed_model.split_half_r",
      "even-seed vs odd-seed adapter means, 24 adapters, centred within arm")
    ba, bs = pc["sigma_u_bootstrap_adapters"], pc["sigma_u_bootstrap_adapters_and_seeds"]
    M("nAttBootAdaptersHigh", ba["percentile_95"][1], sci(ba["percentile_95"][1], 2), SIG,
      "primary_crossed_model.sigma_u_bootstrap_adapters.percentile_95.1",
      f"bootstrap over adapters, {ba['n_boot']} replicates, 97.5th percentile (lower end 0)")
    M("nAttBootTwoWayHigh", bs["percentile_95"][1], sci(bs["percentile_95"][1], 2), SIG,
      "primary_crossed_model.sigma_u_bootstrap_adapters_and_seeds.percentile_95.1",
      f"bootstrap over adapters and seeds, {bs['n_boot']} replicates, 97.5th percentile (lower end 0)")
    cells = pc["interval_calibration_simulation"]["cells"]
    pos = {k: v["coverage_one_sided_95"] for k, v in cells.items() if not k.endswith("0e+00")}
    c_ex_n = min(v["exact"] for k, v in pos.items() if k.startswith("normal"))
    c_ex_t = min(v["exact"] for k, v in pos.items() if k.startswith("t3"))
    c_ba = max(v["bootstrap_adapters"] for v in pos.values())
    c_bs = min(v["bootstrap_adapters_and_seeds"] for v in pos.values())
    src_c = "primary_crossed_model.interval_calibration_simulation.cells"
    # two decimals: 200 data sets per cell, Monte Carlo SE 0.015-0.03 (Entry 114 addendum)
    M("nAttCovExactNormal", c_ex_n, dec(c_ex_n, 2), SIG, src_c + " (min exact, normal u, sigma_u > 0)",
      "coverage of the exact one-sided 95 % upper limit, 200 simulated data sets per cell (MC SE about 0.015)")
    M("nAttCovExactHeavy", c_ex_t, dec(c_ex_t, 2), SIG, src_c + " (min exact, t3 u, sigma_u > 0)",
      "same, heavy-tailed persistent component")
    M("nAttCovBootAdapters", c_ba, dec(c_ba, 2), SIG, src_c + " (max bootstrap_adapters, sigma_u > 0)",
      "the adapter bootstrap under-covers in every cell")
    M("nAttCovBootTwoWay", c_bs, dec(c_bs, 2), SIG, src_c + " (min bootstrap_adapters_and_seeds, sigma_u > 0)",
      "the two-way bootstrap over-covers in every cell")
    # heavy-tailed persistent term (Entry 114 addendum): p of the archive value and coverage of the exact limit
    ht = pc["heavy_tail_checks"]["cells"]
    kmu = f"{S['archive_sigma_mu']['value']:.4g}"
    khi = f"{ci['two_sided_95'][1]:.4g}"
    src_h = "primary_crossed_model.heavy_tail_checks.cells"
    for dist, nm in (("t5", "Five"), ("t3", "Three")):
        v = ht[dist][kmu]["p_F_le_Fobs"]
        M(f"nAttSigmaMuPt{nm}", v, dec(v, 3), SIG, f"{src_h}.{dist}.{kmu}.p_F_le_Fobs",
          f"one-sided p of a persistent component of 5.23e-05 when u is Student t with {dist[1:]} df "
          f"(200,000 simulated data sets); nAttSigmaMuP is the normal-theory value")
    for k, nm in ((khi, "High"), (kmu, "Mu")):
        v = ht["t3"][k]["coverage_exact_one_sided_95"]
        M(f"nAttCovExactTthree{nm}", v, dec(v, 2), SIG, f"{src_h}.t3.{k}.coverage_exact_one_sided_95",
          "coverage of the exact one-sided 95 % upper limit when u is Student t with 3 df and sigma_u equals "
          + ("the interval's upper end" if nm == "High" else "the archive value"))

    # ---------------------------------------------------------------- seed effect and image noise
    M("nAttSigmaSeed", sn["sigma_v_per_image"], sci(sn["sigma_v_per_image"], 2), SIG,
      "seed_effect_and_image_noise.sigma_v_per_image",
      "per-image SD of the seed (content) effect shared by an arm's adapters, pooled over arms")
    M("nAttSigmaResid", sn["sigma_e_per_image"], sci(sn["sigma_e_per_image"], 2), SIG,
      "seed_effect_and_image_noise.sigma_e_per_image", "per-image SD of the adapter-by-seed residual")
    share = 100 * sn["seed_share_of_per_image_variance"]
    M("nAttSeedSharePct", share, sig(share, 2), SIG, "100 x seed_effect_and_image_noise.seed_share_of_per_image_variance",
      "share of the per-image variance of one adapter's paired contrast that the seed carries")
    r = sn["corr_seed_means_A_vs_B"]
    M("nAttSeedArmCorr", r, dec(r, 2), SIG, "seed_effect_and_image_noise.corr_seed_means_A_vs_B",
      "correlation over the 500 seeds of the two arms' seed means, own-minus-other terms")
    se24 = sn["SE_500_single_adapter"]["rms_over_24"]
    M("nAttSEImgAll", se24, sci(se24, 3), SIG, "seed_effect_and_image_noise.SE_500_single_adapter.rms_over_24",
      f"500-image SE of one adapter's paired contrast, RMS over the 24 archive adapters "
      f"({sn['SE_500_single_adapter']['min']:.2e}-{sn['SE_500_single_adapter']['max']:.2e}); the model uses "
      f"nAttSEImg")
    g = pw["sensitivity_SE_500_all_24"]["nominal_sigma_zero_M2"]["G_for_TPR50"]
    M("nAttZeroGFiftyExactAll", g, g2(g), SIG, "power.sensitivity_SE_500_all_24.nominal_sigma_zero_M2.G_for_TPR50",
      "as nAttZeroGFiftyExact with nAttSEImgAll as the image term")

    # ---------------------------------------------------------------- power: sigma = 0 at the calibrated limit
    z = pw["calibrated"]["zero"]["M2"]
    M("nAttTprZeroTwoFiveHundredCal", z["TPR_G500"], tprtxt(z["TPR_G500"]), SIG,
      "power.calibrated.zero.M2.TPR_G500", "two candidates, 1 % FPR, calibrated limit, sigma = 0; verify_v2.tpr()")
    M("nAttTprZeroTwoFiveThousandCal", z["TPR_G5000"], tprtxt(z["TPR_G5000"]), SIG,
      "power.calibrated.zero.M2.TPR_G5000", "as above, 5000 images")
    M("nAttZeroGFiftyExactCal", z["G_for_TPR50"], g2(z["G_for_TPR50"]), SIG, "power.calibrated.zero.M2.G_for_TPR50",
      "exact images for TPR 0.5, two candidates, calibrated limit, sigma = 0; two significant figures")
    M("nAttZeroGNinetyExactCal", z["G_for_TPR90"], g2(z["G_for_TPR90"]), SIG, "power.calibrated.zero.M2.G_for_TPR90",
      "exact images for TPR 0.9, as above")

    # ---------------------------------------------------------------- power at the upper end of the interval
    for lim, suf in (("nominal", ""), ("calibrated", "Cal")):
        h = pw[lim]["paired_high_95"]["M2"]
        assert h["G_for_TPR50"] is None and h["G_for_TPR90"] is None
        for k, nm in (("TPR_G500", "FiveHundred"), ("TPR_G5000", "FiveThousand"), ("TPR_G_inf", "Inf")):
            M(f"nAttTprTwo{nm}PairedHigh{suf}", h[k], tprtxt(h[k]), SIG, f"power.{lim}.paired_high_95.M2.{k}",
              f"two candidates, 1 % FPR, {lim} limit, persistent term at nAttSigmaPairedHigh; TPR 0.5 unattainable")
        sc = pw["sigma_critical"][f"{lim}_M2_TPRinf_50"]
        M(f"nAttSigmaCritHalf{suf}", sc, sci(sc, 2), SIG, f"power.sigma_critical.{lim}_M2_TPRinf_50",
          f"persistent term above which no number of images reaches TPR 0.5 ({lim} limit, two candidates): "
          f"U / z_0.99")

    # ---------------------------------------------------------------- ten times the limit, sigma = 0
    tz = pw["ten_times_nominal_G_for_TPR50"]["zero"]
    for M_, nm in (("M2", "Two"), ("M5", "Five"), ("M50", "Fifty")):
        M(f"nAttTenxG{nm}ExactZero", tz[M_], integer(math.ceil(tz[M_])), SIG,
          f"power.ten_times_nominal_G_for_TPR50.zero.{M_}",
          "exact images for TPR 0.5 at ten times the nominal limit, sigma = 0, rounded up (nAttTenxG*Exact: archive "
          "sigma_mu)")

    # ---------------------------------------------------------------- seed-matched reference
    sm = pw["seed_matched_reference"]
    M("nAttSESeedMatched", sm["SE_img_500"], sci(sm["SE_img_500"], 2), SIG, "power.seed_matched_reference.SE_img_500",
      "500-image SE of (d_A + d_B)/2 for one adapter per body on the same seeds, RMS over 144 pairs")
    for lim, suf in (("nominal", ""), ("calibrated", "Cal")):
        b = sm[lim]
        M(f"nAttSeedMatchedTprFiveHundred{suf}", b["TPR_G500"], tprtxt(b["TPR_G500"]), SIG,
          f"power.seed_matched_reference.{lim}.TPR_G500", f"two candidates, 1 % FPR, {lim} limit, persistent 0")
        M(f"nAttSeedMatchedGFifty{suf}", b["G_for_TPR50"], g2(b["G_for_TPR50"]), SIG,
          f"power.seed_matched_reference.{lim}.G_for_TPR50", "exact images for TPR 0.5; two significant figures")
        M(f"nAttSeedMatchedGNinety{suf}", b["G_for_TPR90"], g2(b["G_for_TPR90"]), SIG,
          f"power.seed_matched_reference.{lim}.G_for_TPR90", "exact images for TPR 0.9; two significant figures")
    # null-specific SE (Entry 114 addendum): the corrected seed-matched figures the manuscript quotes
    ns = sm["null_specific"]["mean_of_orientations"]
    src_n = "power.seed_matched_reference.null_specific.mean_of_orientations"
    M("nAttSESeedMatchedNull", ns["SE_null_500"], sci(ns["SE_null_500"], 2), SIG, src_n + ".SE_null_500",
      "500-image SE of the seed-matched contrast under the null (suspect and reference from one camera: the seed "
      "term cancels exactly), RMS of the two reference cameras")
    M("nAttSeedMatchedGFiftyNull", ns["G_for_TPR50"], g2(ns["G_for_TPR50"]), SIG, src_n + ".G_for_TPR50",
      "images for a two-candidate TPR of 0.5 at 1 % FPR, nominal limit, seed-matched, null-specific SE, "
      "persistent term 0; mean of the two reference cameras; two significant figures")
    M("nAttSeedMatchedGFiftyNullCal", ns["G_for_TPR50_calibrated"], g2(ns["G_for_TPR50_calibrated"]), SIG,
      src_n + ".G_for_TPR50_calibrated", "as nAttSeedMatchedGFiftyNull at the calibrated limit")
    M("nAttSeedMatchedTprFiveHundredNull", ns["TPR_G500"], dec(ns["TPR_G500"], 2), SIG, src_n + ".TPR_G500",
      "two-candidate TPR with 500 images, nominal limit, seed-matched, null-specific SE")
    M("nAttFiftyImgMultMatchedNull", ns["x_nominal_50_images"], dec(ns["x_nominal_50_images"], 1), SIG,
      src_n + ".x_nominal_50_images",
      "multiple of the nominal limit at which 50 images give a two-candidate TPR of 0.5, seed-matched, null SE")
    M("nAttSeedMatchedGFiftyMany", ns["G_for_TPR50_many_references"], g2(ns["G_for_TPR50_many_references"]), SIG,
      src_n + ".G_for_TPR50_many_references", "as nAttSeedMatchedGFiftyNull with many reference adapters")
    M("nAttFiftyImgMultMatchedMany", ns["x_nominal_50_images_many_references"],
      dec(ns["x_nominal_50_images_many_references"], 1), SIG, src_n + ".x_nominal_50_images_many_references",
      "as nAttFiftyImgMultMatchedNull with many reference adapters")
    M("nAttSeedMatchedGFiftyTrain", ns["G_for_TPR50_with_training_sets"], g2(ns["G_for_TPR50_with_training_sets"]),
      SIG, src_n + ".G_for_TPR50_with_training_sets",
      "as nAttSeedMatchedGFiftyNull with the suspect's and reference's training sets differing (H6 training sd)")

    # the H6 training-set component as a persistent term, fresh-seed examiner (Entry 114 addendum)
    tr = pw["training_set_component_persistent"]
    for lim, suf in (("nominal", ""), ("calibrated", "Cal")):
        b = tr[lim]["M2"]
        for k, nm in (("TPR_G500", "FiveHundred"), ("TPR_G5000", "FiveThousand"), ("TPR_G_inf", "Inf")):
            M(f"nAttTrainTpr{nm}{suf}", b[k], tprtxt(b[k]) if k == "TPR_G500" else dec(b[k], 2), SIG,
              f"power.training_set_component_persistent.{lim}.M2.{k}",
              f"two candidates, 1 % FPR, {lim} limit, persistent term = H6 training-set sd (nLimTrainSD)")
        M(f"nAttTrainGFifty{suf}", b["G_for_TPR50"], g2(b["G_for_TPR50"]), SIG,
          f"power.training_set_component_persistent.{lim}.M2.G_for_TPR50",
          "images for TPR 0.5 with that persistent term; two significant figures")

    # Table S16 columns at the paired upper limit and sigma = 0 for five and fifty candidates
    for M_, nm in (("M5", "Five"), ("M50", "Fifty")):
        h = pw["nominal"]["paired_high_95"][M_]
        assert h["G_for_TPR50"] is None
        for k, kn in (("TPR_G500", "FiveHundred"), ("TPR_G5000", "FiveThousand"), ("TPR_G_inf", "Inf")):
            M(f"nAttTpr{nm}{kn}PairedHigh", h[k], tprtxt(h[k]), SIG, f"power.nominal.paired_high_95.{M_}.{k}",
              f"{M_[1:]} candidates, 1 % FPR, nominal limit, persistent term at nAttSigmaPairedHigh")
        z0 = pw["nominal"]["zero"][M_]["G_for_TPR50"]
        M(f"nAttZeroGFiftyExact{nm}", z0, g2(z0), SIG, f"power.nominal.zero.{M_}.G_for_TPR50",
          f"exact images for TPR 0.5, {M_[1:]} candidates, nominal limit, sigma = 0; two significant figures")

    t50 = pw["transfer_for_TPR50_with_50_images"]
    for nm, k, x in (("FiftyImgMultFresh", "fresh_seeds_sigma_zero", "x_nominal"),
                     ("FiftyImgMultFreshMu", "fresh_seeds_archive_sigma_mu", "x_nominal"),
                     ("FiftyImgMultMatched", "seed_matched_sigma_zero", "x_nominal"),
                     ("FiftyImgMultFreshCal", "fresh_seeds_sigma_zero", "x_calibrated"),
                     ("FiftyImgMultMatchedCal", "seed_matched_sigma_zero", "x_calibrated")):
        M(f"nAtt{nm}", t50[k][x], dec(t50[k][x], 1), SIG, f"power.transfer_for_TPR50_with_50_images.{k}.{x}",
          "multiple of the limit at which 50 images give a two-candidate TPR of 0.5 at 1 % FPR")

    # ---------------------------------------------------------------- secondary designs (local stack)
    g2d = S["secondary_designs_descriptive"]["G2_local_2000_steps"]
    M("nAttSigmaGtwo", g2d["pooled"]["sigma_u_reml"], sci(g2d["pooled"]["sigma_u_reml"], 2), SIG,
      "secondary_designs_descriptive.G2_local_2000_steps.pooled.sigma_u_reml",
      "G2 second environment, 6 + 6 adapters x 500 images; descriptive")
    M("nAttSigmaGtwoP", g2d["pooled"]["F_p_upper"], dec(g2d["pooled"]["F_p_upper"], 3), SIG,
      "secondary_designs_descriptive.G2_local_2000_steps.pooled.F_p_upper", "upper-tail F p")
    out_tag = g2d["largest_deviation_adapter"]
    M("nAttGtwoOutlierSeed", int(out_tag.split("_s")[-1]), str(int(out_tag.split("_s")[-1])), SIG,
      "secondary_designs_descriptive.G2_local_2000_steps.largest_deviation_adapter", f"{out_tag} (body A)")
    M("nAttGtwoOutlierPct", g2d["largest_deviation_pct_of_R_real"], sig(g2d["largest_deviation_pct_of_R_real"], 3),
      SIG, "secondary_designs_descriptive.G2_local_2000_steps.largest_deviation_pct_of_R_real",
      f"{out_tag} own-minus-other contrast {g2d['largest_deviation_value']:.3e} as % of R_real")
    od = g2d["largest_deviation_detail"]
    src_o = "secondary_designs_descriptive.G2_local_2000_steps.largest_deviation_detail"
    M("nAttGtwoOutlierZ", od["z_own"], dec(od["z_own"], 1), SIG, src_o + ".z_own",
      "its distance from the mean of the other five arm-A adapters, in SEs of its own per-seed deviations "
      "(corrected in the Entry 114 addendum; the single-mean image-noise SE gave "
      f"{g2d['largest_deviation_z_vs_image_noise']:.1f})")
    M("nAttGtwoOutlierZDiff", od["z_pooled_difference_se"], dec(od["z_pooled_difference_se"], 1), SIG,
      src_o + ".z_pooled_difference_se", "the same distance with the pooled model's SE of a difference")
    M("nAttGtwoOutlierFwP", od["familywise_p_most_extreme_of_48"], dec(od["familywise_p_most_extreme_of_48"], 2), SIG,
      src_o + ".familywise_p_most_extreme_of_48",
      "family-wise p for the most extreme of the 48 adapters examined: 1 - (1 - p_two-sided)^48")
    dp = g2d["leave_one_out_arm_A"][out_tag]["F_p"]
    M("nAttSigmaGtwoDropP", dp, dec(dp, 2), SIG,
      f"secondary_designs_descriptive.G2_local_2000_steps.leave_one_out_arm_A.{out_tag}.F_p",
      "F p for a persistent component with that adapter left out (sigma_u 0)")
    hp = S["secondary_designs_descriptive"]["dose_16000_steps"]["pooled"]["F_p_upper"]
    M("nAttSigmaSixteenP", hp, dec(hp, 2), SIG, "secondary_designs_descriptive.dose_16000_steps.pooled.F_p_upper",
      "16000 steps, 6 + 6 adapters x 250 images; sigma_u 0")

    # ================================================================ weights (H5, same-seed pairs set aside)
    d2, d16 = W["doses"]["2000"], W["doses"]["16000"]
    for d, nm in ((d2, "Two"), (d16, "Sixteen")):
        dk = "2000" if nm == "Two" else "16000"
        perm = d["D_strat_permutation"]
        assert perm["n_relabellings"] == 64 and perm["n_distinct_partitions"] == 32
        assert d["every_same_body_above_every_cross_different_seed"]
        M(f"nWtSameSeed{nm}", d["same_seed"]["mean"], dec(d["same_seed"]["mean"], 3), WTS, f"doses.{dk}.same_seed.mean",
          "mean cosine of the six same-seed (cross-body) pairs")
        M(f"nWtSameSeed{nm}Min", d["same_seed"]["min"], dec(d["same_seed"]["min"], 3), WTS,
          f"doses.{dk}.same_seed.min", "smallest same-seed cosine")
        M(f"nWtOtherMax{nm}", d["largest_cosine_other_than_same_seed"], dec(d["largest_cosine_other_than_same_seed"], 3),
          WTS, f"doses.{dk}.largest_cosine_other_than_same_seed", "largest cosine of any pair that does not share a seed")
        M(f"nWtSameSeedRatio{nm}", d["same_seed_over_same_body_mean"], sig(d["same_seed_over_same_body_mean"], 2), WTS,
          f"doses.{dk}.same_seed_over_same_body_mean", "same-seed mean / same-body mean")
        M(f"nWtCrossNoSeed{nm}", d["cross_body_different_seed"]["mean"], dec(d["cross_body_different_seed"]["mean"], 3),
          WTS, f"doses.{dk}.cross_body_different_seed.mean", "mean cosine of the 30 cross-body pairs of different seeds")
        M(f"nWtCrossNoSeedMax{nm}", d["cross_body_different_seed"]["max"], dec(d["cross_body_different_seed"]["max"], 3),
          WTS, f"doses.{dk}.cross_body_different_seed.max", "largest of them")
        M(f"nWtSameBodyMin{nm}", d["same_body"]["min"], dec(d["same_body"]["min"], 3), WTS, f"doses.{dk}.same_body.min",
          "smallest same-body cosine (the mean is nWt*SameBody, unchanged)")
        M(f"nWtStratD{nm}", d["D_strat"], sig(d["D_strat"], 2), WTS, f"doses.{dk}.D_strat",
          "same-body mean minus cross-body different-seed mean; post hoc, descriptive")
        M(f"nWtStratP{nm}", perm["p_one_sided"], dec(perm["p_one_sided"], 3), WTS,
          f"doses.{dk}.D_strat_permutation.p_one_sided",
          f"exact one-sided p over the within-seed-pair relabellings; the observed value is the largest of the 32 "
          f"(next {perm['second_largest_value']:.4f}); equals the floor")
        bb = d["batch_without_same_seed_pairs"]
        assert bb["different_batch_mean_equals_entry95"]
        # three significant figures (Entry 114 addendum): at three decimals it printed 0.013 beside nWtTwoCrossBatch
        # 0.014, a gap of 0.001 where the true difference is -0.00015
        M(f"nWtSameBatchNoSeed{nm}", bb["same_batch_different_seed_mean"], sig(bb["same_batch_different_seed_mean"], 3),
          WTS, f"doses.{dk}.batch_without_same_seed_pairs.same_batch_different_seed_mean",
          "mean cosine of same-batch pairs of different seeds (the different-batch mean is nWt*CrossBatch)")
        M(f"nWtDBatchNoSeed{nm}", bb["difference"], sig(bb["difference"], 1), WTS,
          f"doses.{dk}.batch_without_same_seed_pairs.difference",
          "same-batch different-seed mean minus different-batch mean")
    M("nWtStratFloor", W["floor"], dec(W["floor"], 3), WTS, "floor",
      "smallest attainable one-sided p of the within-seed-pair relabelling: 2/64 = 1/32")
    M("nWtStratRelabelings", d2["D_strat_permutation"]["n_relabellings"],
      integer(d2["D_strat_permutation"]["n_relabellings"]), WTS, "doses.2000.D_strat_permutation.n_relabellings",
      "2^6 swaps of body labels within the six seed pairs")
    M("nWtStratDistinct", d2["D_strat_permutation"]["n_distinct_partitions"],
      integer(d2["D_strat_permutation"]["n_distinct_partitions"]), WTS,
      "doses.2000.D_strat_permutation.n_distinct_partitions", "swapping every pair returns the same partition")
    return L
