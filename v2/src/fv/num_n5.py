"""Number macros, part n5: groups Dose (adaptation dose, H1, G1 inversion), Gen (generalisation rows),
Pone (P1, five everyday captions) and Est (fingerprint estimability gates).

Every value is read from a result file at run time (out/, out/t1, out/t5, the primary ledger, or a local
snapshot of three Drive-only result files) or computed here from those files. Only two gate thresholds, which
are design constants of the registrations, come from RESULTS.md text.

Units: "Lam" macros are percentages of a real-image device contrast. For the D200 pair (dose, G1, G5, G2,
FLUX, full FT, captions, random crop) that is R_real = 0.0356703 (E2, 40 held-out photographs per body); for
G6/P1 (iPhone 5c) and G4b (P20) it is that pair's own R_real, stated in each macro's check field.
Adapter tags are always ordered by integer seed.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_numlib import OUT, LEDGER, R_REAL, load, get, sig, dec, sci, integer, pct_of_rreal, macro  # noqa: E402

T1 = OUT + "/t1"
T5 = OUT + "/t5"
SNAP = (EINV.V2 + "/paper/fv/work/drive_snapshot")
DRIVE = (EINV.MYDRIVE + "/inv_channel")


def _drive(sub, name):
    """Drive-only result file: the live Drive copy if mounted, else the local snapshot taken 29 Sep 2026."""
    live = f"{DRIVE}/{sub}/{name}"
    return live if os.path.exists(live) else f"{SNAP}/{name}"


# ----------------------------------------------------------------------------- formatting helpers
def pval(p):
    """p-value: 2 significant figures, at most 3 decimals; below 0.001 keep 2 significant figures."""
    if p < 0.001:
        return sig(p, 2)
    s = sig(p, 2)
    return dec(p, 3) if len(s.split(".")[-1]) > 3 else s


def pfine(p, n=2):
    """p-value at n significant figures without the 3-decimal cap (for values near a threshold)."""
    return sig(p, n)


def tval(t):
    return dec(t, 2)


def dfv(d):
    return dec(d, 1)


def welch_sym(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    th = 0.5 * (a.mean() + b.mean())
    se = 0.5 * math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return th, se, df


def seed_of(tag):
    """Integer seed from a tag such as 'inv16kext_A_s7', 's7' or 'A_raw_s10' (never lexicographic)."""
    return int(str(tag).rsplit("s", 1)[-1])


# ----------------------------------------------------------------------------- the part
def macros():
    L = []

    def M(name, value, text, source, key, entry, check=""):
        L.append(macro(name, value, text, source, key, entry, check))

    # ============================================================== Dose: 2000 / 8000 / 16000 steps
    f_dose = T1 + "/dose_stats.json"
    D = load(f_dose)["doses"]
    D8m2 = load(f_dose)["dose_8000_minus_2000"]
    for tag, key, entry in (("TwoThousand", "2000", "47"), ("EightThousand", "8000", "47"),
                            ("Sixteen", "16000", "68"), ("Rep", "16000_new", "68")):
        d = D[key]
        n = len(d["A_own"])
        pv = float(stats.t.sf(d["t"], d["welch_df"]))
        lab = {"TwoThousand": "2000 steps (v2 environment, 3 per body)", "EightThousand": "8000 steps (2 per body)",
               "Sixteen": "16000 steps, six per body pooled", "Rep": "16000 steps, registered replication (seeds 3-5)"}[tag]
        M(f"nDose{tag}Lam", d["lambda_sym_pct"], sig(d["lambda_sym_pct"], 3), f_dose, f"doses.{key}.lambda_sym_pct", entry,
          f"{lab}: theta_sym as % of R_real (E2, 0.0356703); RESULTS Entry {entry} and FINDINGS agree")
        M(f"nDose{tag}Theta", d["theta_sym"], sci(d["theta_sym"], 3), f_dose, f"doses.{key}.theta_sym", entry,
          f"{lab}: theta_sym in NCC units")
        M(f"nDose{tag}SE", d["adapter_SE"], sci(d["adapter_SE"], 3), f_dose, f"doses.{key}.adapter_SE", entry,
          f"{lab}: Welch adapter-level SE")
        M(f"nDose{tag}Df", d["welch_df"], dfv(d["welch_df"]), f_dose, f"doses.{key}.welch_df", entry,
          f"{lab}: Welch df")
        M(f"nDose{tag}T", d["t"], tval(d["t"]), f_dose, f"doses.{key}.t", entry, f"{lab}: t = theta_sym / SE")
        M(f"nDose{tag}P", pv, pval(pv), f_dose, f"stats.t.sf(doses.{key}.t, welch_df)", entry,
          f"{lab}: one-sided p (theta_sym > 0), computed here")
        M(f"nDose{tag}Limit", d["lambda_U_plugin_pct"], sig(d["lambda_U_plugin_pct"], 3), f_dose,
          f"doses.{key}.lambda_U_plugin_pct", entry,
          f"{lab}: SYMMETRIC one-sided 99 % plug-in limit (theta + t_0.99,df SE)/R_real, % of R_real (E2)")
        M(f"nDose{tag}ThetaA", d["theta_A"], sci(d["theta_A"], 3), f_dose, f"doses.{key}.theta_A", entry,
          f"{lab}: arm A mean own-minus-other")
        M(f"nDose{tag}ThetaB", d["theta_B"], sci(d["theta_B"], 2), f_dose, f"doses.{key}.theta_B", entry,
          f"{lab}: arm B mean own-minus-other")
        M(f"nDose{tag}Additive", d["additive_part"], sci(d["additive_part"], 3), f_dose, f"doses.{key}.additive_part",
          entry, f"{lab}: additive part (theta_A - theta_B)/2, shared lean toward K_A, NCC units")
        M(f"nDose{tag}Adapters", n, integer(n), f_dose, f"len(doses.{key}.A_own)", entry, f"{lab}: adapters per body")
        M(f"nDose{tag}APos", int((np.array(d["A_own"]) > 0).sum()), integer((np.array(d["A_own"]) > 0).sum()), f_dose,
          f"count(doses.{key}.A_own > 0)", entry, f"{lab}: arm-A adapters with positive own-contrast")
        M(f"nDose{tag}BPos", int((np.array(d["B_own"]) > 0).sum()), integer((np.array(d["B_own"]) > 0).sum()), f_dose,
          f"count(doses.{key}.B_own > 0)", entry, f"{lab}: arm-B adapters with positive own-contrast")
    # fine p-values near thresholds (the 3-decimal cap would print 0.010 / 0.002)
    for tag, key in (("Sixteen", "16000"), ("Rep", "16000_new")):
        d = D[key]
        pv = float(stats.t.sf(d["t"], d["welch_df"]))
        M(f"nDose{tag}PFine", pv, pfine(pv, 3), f_dose, f"stats.t.sf(doses.{key}.t, welch_df)", "68",
          "three significant figures, no decimal cap; matches RESULTS Entry 68 text 0.00198 / 0.00955 "
          "(the brief's 0.0096 is a double rounding of 0.009549)")
    a16 = np.array(D["16000"]["A_own"])
    M("nDoseSixteenAMin", a16.min(), sci(a16.min(), 2), f_dose, "min(doses.16000.A_own)", "68",
      "matches Entry 68 '4.9e-5 to 1.9e-4'")
    M("nDoseSixteenAMax", a16.max(), sci(a16.max(), 2), f_dose, "max(doses.16000.A_own)", "68",
      "matches Entry 68 '4.9e-5 to 1.9e-4'")
    M("nDoseEightMinusTwoDiff", D8m2["difference"], sci(D8m2["difference"], 2), f_dose,
      "dose_8000_minus_2000.difference", "47", "matches Entry 47 -4.6e-5")
    M("nDoseEightMinusTwoSE", D8m2["se"], sci(D8m2["se"], 2), f_dose, "dose_8000_minus_2000.se", "47",
      "matches Entry 47 2.9e-5")
    M("nDoseEightMinusTwoZ", D8m2["z"], dec(D8m2["z"], 2), f_dose, "dose_8000_minus_2000.z", "47",
      "matches Entry 47 z -1.58")

    # first three 16000-step adapters per body (F9, Entry 54), seeds 0-2
    a3, b3 = D["16000"]["A_own"][:3], D["16000"]["B_own"][:3]
    th, se, df = welch_sym(a3, b3)
    p3 = float(stats.t.sf(th / se, df))
    lim3 = 100 * (th + stats.t.ppf(0.99, df) * se) / R_REAL
    k3 = "welch_sym(doses.16000.A_own[0:3], B_own[0:3]) (seeds 0-2)"
    M("nDoseFirstThreeTheta", th, sci(th, 3), f_dose, k3, "54", "matches Entry 54 5.55e-05 and verify_v2 check 15")
    M("nDoseFirstThreeLam", 100 * th / R_REAL, sig(100 * th / R_REAL, 3), f_dose, k3, "54",
      "% of R_real (E2); matches Entry 54 0.156 %")
    M("nDoseFirstThreeSE", se, sci(se, 3), f_dose, k3, "54", "matches Entry 54 2.80e-05")
    M("nDoseFirstThreeDf", df, dfv(df), f_dose, k3, "54", "matches Entry 54 df 3.30")
    M("nDoseFirstThreeT", th / se, tval(th / se), f_dose, k3, "54", "matches Entry 54 t 1.98 and verify_v2 check 16")
    M("nDoseFirstThreeP", p3, pval(p3), f_dose, k3, "54", "one-sided; matches Entry 54 p 0.067")
    M("nDoseFirstThreeLimit", lim3, sig(lim3, 3), f_dose, k3, "54",
      "symmetric one-sided 99 % plug-in, % of R_real (E2); matches Entry 54 0.487 %")

    # leave-one-adapter-out over the twelve 16000-step adapters (Entry 81)
    A6, B6 = np.array(D["16000"]["A_own"]), np.array(D["16000"]["B_own"])
    loo = []
    for arm in "AB":
        for i in range(6):
            a = np.delete(A6, i) if arm == "A" else A6
            b = np.delete(B6, i) if arm == "B" else B6
            t_, s_, d_ = welch_sym(a, b)
            loo.append((t_, float(stats.t.sf(t_ / s_, d_))))
    th_l = [x[0] for x in loo]
    p_l = [x[1] for x in loo]
    kl = "leave-one-adapter-out over doses.16000.{A,B}_own, Welch one-sided"
    M("nDoseLooThetaMin", min(th_l), sci(min(th_l), 3), f_dose, kl, "81",
      "file gives 4.925e-5 -> 4.92; Entry 81 prints +4.93e-5 (rounding of 4.925)")
    M("nDoseLooThetaMax", max(th_l), sci(max(th_l), 3), f_dose, kl, "81", "matches Entry 81 +6.36e-5")
    M("nDoseLooPMin", min(p_l), pval(min(p_l)), f_dose, kl, "81", "matches Entry 81 0.0018 (3-decimal cap)")
    M("nDoseLooPMax", max(p_l), pval(max(p_l)), f_dose, kl, "81", "matches Entry 81 0.0081 (3-decimal cap)")
    M("nDoseLooPMinFine", min(p_l), pfine(min(p_l)), f_dose, kl, "81", "matches Entry 81 0.0018")
    M("nDoseLooPMaxFine", max(p_l), pfine(max(p_l)), f_dose, kl, "81", "matches Entry 81 0.0081")

    # ============================================================== Dose: H1, strength vs body at 16000 steps
    f_h1 = OUT + "/h1_strength.json"
    H1 = load(f_h1)
    sd = H1["arm_strength_difference"]["lora_B_norm"]
    fits = H1["models"]["lora_B_norm"]["fits"]
    lfits = H1["models"]["loss_tail"]["fits"]
    M("nDoseHoneNormA", sd["mean_A"], dec(sd["mean_A"], 2), f_h1, "arm_strength_difference.lora_B_norm.mean_A",
      "89", "matches Entry 81/89 17.42")
    M("nDoseHoneNormB", sd["mean_B"], dec(sd["mean_B"], 2), f_h1, "arm_strength_difference.lora_B_norm.mean_B",
      "89", "matches Entry 81/89 17.11")
    M("nDoseHoneNormT", sd["welch_t"], tval(sd["welch_t"]), f_h1, "arm_strength_difference.lora_B_norm.welch_t",
      "81", "matches Entry 81 t 3.85")
    M("nDoseHoneNormP", sd["p_two_sided"], pval(sd["p_two_sided"]), f_h1,
      "arm_strength_difference.lora_B_norm.p_two_sided", "89", "two-sided; Entry 81 0.005, Entry 89 0.0055")
    M("nDoseHoneBodyP", fits["body_only"]["body_A"]["p"], pval(fits["body_only"]["body_A"]["p"]), f_h1,
      "models.lora_B_norm.fits.body_only.body_A.p", "89", "matches Entry 89 p 0.003 and verify_v2 check 51")
    M("nDoseHoneStrengthP", fits["strength_only"]["strength"]["p"], pval(fits["strength_only"]["strength"]["p"]),
      f_h1, "models.lora_B_norm.fits.strength_only.strength.p", "89", "matches Entry 89 p 0.012, verify_v2 check 52")
    M("nDoseHoneBothBodyP", fits["body_plus_strength"]["body_A"]["p"],
      pval(fits["body_plus_strength"]["body_A"]["p"]), f_h1, "models.lora_B_norm.fits.body_plus_strength.body_A.p",
      "89", "matches Entry 89 0.087")
    M("nDoseHoneBothStrengthP", fits["body_plus_strength"]["strength"]["p"],
      pval(fits["body_plus_strength"]["strength"]["p"]), f_h1,
      "models.lora_B_norm.fits.body_plus_strength.strength.p", "89", "matches Entry 89 0.500")
    for nm, k in (("Body", "body_only"), ("Strength", "strength_only"), ("Both", "body_plus_strength")):
        v = fits[k]["r2"]
        M(f"nDoseHoneRsq{nm}", v, dec(v, 3), f_h1, f"models.lora_B_norm.fits.{k}.r2", "89",
          "matches Entry 89 R^2 0.614 / 0.483 / 0.634")
    v = H1["models"]["lora_B_norm"]["corr_strength_body"]
    M("nDoseHoneCorrBody", v, dec(v, 2), f_h1, "models.lora_B_norm.corr_strength_body", "89",
      "collinearity of strength with body; Entry 89 0.773, verify_v2 check 53 (printed 2 decimals: 0.77)")
    v = H1["models"]["lora_B_norm"]["corr_strength_theta"]
    M("nDoseHoneCorrTheta", v, dec(v, 2), f_h1, "models.lora_B_norm.corr_strength_theta", "89",
      "matches Entry 81/89 r +0.695")
    v = lfits["body_plus_strength"]["body_A"]["p"]
    M("nDoseHoneLossBodyP", v, pval(v), f_h1, "models.loss_tail.fits.body_plus_strength.body_A.p", "89",
      "loss_tail as the strength measure; matches Entry 89 0.012")
    v = lfits["body_plus_strength"]["strength"]["p"]
    M("nDoseHoneLossStrengthP", v, pval(v), f_h1, "models.loss_tail.fits.body_plus_strength.strength.p", "89",
      "matches Entry 89 0.556")
    M("nDoseHoneAdapters", H1["n_adapters"], integer(H1["n_adapters"]), f_h1, "n_adapters", "89", "design count")

    # ============================================================== Dose: G1, three inverted per arm (Entry 98)
    f_g1 = OUT + "/g1_covariate.json"
    G1 = load(f_g1)
    M("nDoseInvThreeTheta", G1["theta_sym"], sci(G1["theta_sym"], 3), f_g1, "theta_sym", "98",
      "matches Entry 98 -5.089e-05")
    M("nDoseInvThreeLam", G1["theta_sym_pct"], sig(G1["theta_sym_pct"], 3), f_g1, "theta_sym_pct", "98",
      "% of R_real (E2); matches Entry 98 -0.1427 % and verify_v2 check 47")
    M("nDoseInvThreeSE", G1["welch_se"], sci(G1["welch_se"], 3), f_g1, "welch_se", "98", "matches Entry 98 4.009e-05")
    M("nDoseInvThreeDf", G1["welch_df"], dfv(G1["welch_df"]), f_g1, "welch_df", "98", "matches Entry 98 df 3.15")
    M("nDoseInvThreeP", G1["one_sided_p_lt0"], pval(G1["one_sided_p_lt0"]), f_g1, "one_sided_p_lt0", "98",
      "one-sided (theta<0); Entry 98 0.145, verify_v2 check 48; printed at 2 s.f.")
    M("nDoseInvThreePFine", G1["one_sided_p_lt0"], pfine(G1["one_sided_p_lt0"], 3), f_g1, "one_sided_p_lt0", "98",
      "3 s.f., as Entry 98 prints it")
    pw = G1["power"]
    M("nDoseInvThreePower", pw["achieved_power_at_target"], dec(pw["achieved_power_at_target"], 2), f_g1,
      "power.achieved_power_at_target", "98", "matches Entry 98 power 0.22")
    M("nDoseInvThreeSERequired", pw["se_required"], sci(pw["se_required"], 3), f_g1, "power.se_required", "98",
      "matches Entry 98 2.47e-05")
    M("nDoseInvThreeAdjP", G1["strength_adjusted"]["one_sided_p_lt0"],
      pval(G1["strength_adjusted"]["one_sided_p_lt0"]), f_g1, "strength_adjusted.one_sided_p_lt0", "98",
      "matches Entry 98 0.106")
    M("nDoseInvThreeCorrTheta", G1["corr_strength_theta"], dec(G1["corr_strength_theta"], 2), f_g1,
      "corr_strength_theta", "98", "matches Entry 98 +0.810")
    M("nDoseInvThreeCorrBody", G1["corr_strength_body"], dec(G1["corr_strength_body"], 2), f_g1,
      "corr_strength_body", "98", "matches Entry 98 +0.711")
    asd = G1["arm_strength_difference"]["lora_B_norm"]
    M("nDoseInvThreeNormP", asd["p_two_sided"], pval(asd["p_two_sided"]), f_g1,
      "arm_strength_difference.lora_B_norm.p_two_sided", "98", "matches Entry 98 0.126")
    sic = G1["stored_inverted_contrast"]
    M("nDoseInvStoredA", abs(sic["A"]), dec(abs(sic["A"]), 2), f_g1, "abs(stored_inverted_contrast.A)", "98",
      "stored inverted contrast magnitude, multiple of natural (sign negative); Entry 98/110 -2.85x")
    M("nDoseInvStoredB", abs(sic["B"]), dec(abs(sic["B"]), 2), f_g1, "abs(stored_inverted_contrast.B)", "98",
      "stored inverted contrast magnitude, multiple of natural (sign negative); Entry 98/110 -1.83x")

    # ============================================================== Dose: G1 extension, eight per arm (Entry 110)
    f_ge = OUT + "/g1_ext.json"
    GE = load(f_ge)
    a8 = [GE["per_adapter_A"][k] for k in sorted(GE["per_adapter_A"], key=seed_of)]
    b8 = [GE["per_adapter_B"][k] for k in sorted(GE["per_adapter_B"], key=seed_of)]
    th8, se8, df8 = welch_sym(a8, b8)
    ae = GE["all_eight"]
    assert abs(th8 - ae["theta_sym"]) < 1e-12 and abs(se8 - ae["welch_se"]) < 1e-12, "g1_ext all_eight mismatch"
    M("nDoseInvLam", ae["theta_sym_pct"], sig(ae["theta_sym_pct"], 3), f_ge, "all_eight.theta_sym_pct", "110",
      "% of R_real (E2); matches Entry 110 -0.163 % and verify_v2 check 66 (-0.1629)")
    M("nDoseInvTheta", ae["theta_sym"], sci(ae["theta_sym"], 3), f_ge, "all_eight.theta_sym", "110",
      "matches Entry 110 -5.810e-05; recomputed from per-adapter values (seed order)")
    M("nDoseInvSE", ae["welch_se"], sci(ae["welch_se"], 3), f_ge, "all_eight.welch_se", "110",
      "matches Entry 110 2.173e-05")
    M("nDoseInvDf", ae["welch_df"], dfv(ae["welch_df"]), f_ge, "all_eight.welch_df", "110", "Welch df")
    M("nDoseInvT", ae["t"], tval(ae["t"]), f_ge, "all_eight.t", "110", "matches Entry 110 t -2.67")
    M("nDoseInvP", ae["one_sided_p_lt0"], pval(ae["one_sided_p_lt0"]), f_ge, "all_eight.one_sided_p_lt0", "110",
      "one-sided (theta<0); 2 s.f. = 0.011; Entry 110 0.0106, verify_v2 check 67")
    M("nDoseInvPFine", ae["one_sided_p_lt0"], pfine(ae["one_sided_p_lt0"], 3), f_ge, "all_eight.one_sided_p_lt0",
      "110", "3 s.f. (0.0106), needed beside the 0.01 threshold; matches Entry 110 and verify_v2 check 67")
    M("nDoseInvThetaA", ae["theta_A"], sci(ae["theta_A"], 3), f_ge, "all_eight.theta_A", "110",
      "arm A mean inverted own-contrast; matches Entry 110 +1.05e-05")
    M("nDoseInvThetaB", ae["theta_B"], sci(ae["theta_B"], 3), f_ge, "all_eight.theta_B", "110",
      "arm B mean inverted own-contrast; matches Entry 110 -1.267e-04")
    M("nDoseInvAdapters", ae["n_per_arm"], integer(ae["n_per_arm"]), f_ge, "all_eight.n_per_arm", "110",
      "adapters per arm (design)")
    fn = GE["five_new_alone"]
    M("nDoseInvNewAdapters", fn["n_per_arm"], integer(fn["n_per_arm"]), f_ge, "five_new_alone.n_per_arm", "110",
      "new adapters per arm (seeds 3-7)")
    M("nDoseInvFiveLam", fn["theta_sym_pct"], sig(fn["theta_sym_pct"], 3), f_ge, "five_new_alone.theta_sym_pct",
      "110", "% of R_real (E2); matches Entry 110 -0.175 %")
    M("nDoseInvFiveTheta", fn["theta_sym"], sci(fn["theta_sym"], 3), f_ge, "five_new_alone.theta_sym", "110",
      "matches Entry 110 -6.243e-05")
    M("nDoseInvFiveSE", fn["welch_se"], sci(fn["welch_se"], 3), f_ge, "five_new_alone.welch_se", "110",
      "matches Entry 110 2.848e-05")
    M("nDoseInvFiveDf", fn["welch_df"], dfv(fn["welch_df"]), f_ge, "five_new_alone.welch_df", "110", "Welch df")
    M("nDoseInvFiveT", fn["t"], tval(fn["t"]), f_ge, "five_new_alone.t", "110", "matches Entry 110 t -2.19")
    M("nDoseInvFiveP", fn["one_sided_p_lt0"], pval(fn["one_sided_p_lt0"]), f_ge, "five_new_alone.one_sided_p_lt0",
      "110", "matches Entry 110 0.034 and verify_v2 check 68")
    sc = GE["strength_covariate"]
    adj = sc["strength_adjusted"]
    M("nDoseInvAdjT", adj["t"], tval(adj["t"]), f_ge, "strength_covariate.strength_adjusted.t", "110",
      "matches Entry 110 t -3.03")
    M("nDoseInvAdjSE", adj["welch_se"], sci(adj["welch_se"], 3), f_ge,
      "strength_covariate.strength_adjusted.welch_se", "110", "strength-adjusted Welch SE")
    M("nDoseInvAdjP", adj["one_sided_p_lt0"], pval(adj["one_sided_p_lt0"]), f_ge,
      "strength_covariate.strength_adjusted.one_sided_p_lt0", "110", "3-decimal cap gives 0.005; Entry 110 0.0047")
    M("nDoseInvAdjPFine", adj["one_sided_p_lt0"], pfine(adj["one_sided_p_lt0"]), f_ge,
      "strength_covariate.strength_adjusted.one_sided_p_lt0", "110", "matches Entry 110 0.0047")
    M("nDoseInvNormA", sc["lora_B_norm_A"], dec(sc["lora_B_norm_A"], 2), f_ge, "strength_covariate.lora_B_norm_A",
      "110", "mean adapted-weight norm, inverted arm A")
    M("nDoseInvNormB", sc["lora_B_norm_B"], dec(sc["lora_B_norm_B"], 2), f_ge, "strength_covariate.lora_B_norm_B",
      "110", "mean adapted-weight norm, inverted arm B")
    M("nDoseInvNormP", sc["p_two_sided"], pval(sc["p_two_sided"]), f_ge, "strength_covariate.p_two_sided", "110",
      "two-sided Welch, arms differ in strength (text-only in RESULTS; computed by g1_ext_score.py)")
    M("nDoseInvCorrTheta", sc["corr_strength_theta"], dec(sc["corr_strength_theta"], 2), f_ge,
      "strength_covariate.corr_strength_theta", "110", "r(weight norm, per-adapter contrast), 16 adapters")
    M("nDoseInvCorrBody", sc["corr_strength_body"], dec(sc["corr_strength_body"], 2), f_ge,
      "strength_covariate.corr_strength_body", "110", "r(weight norm, body indicator), 16 adapters")
    pw = GE["power_at_mirror_of_unsuppressed"]
    M("nDoseInvPower", pw, dec(pw, 2), f_ge, "power_at_mirror_of_unsuppressed", "110", "matches Entry 110 power 0.79")
    nimg = sum(v["n_images"] for v in GE["adapters"].values())
    M("nDoseInvImages", nimg, integer(nimg), f_ge, "sum(adapters.*.n_images)", "110",
      "all 16 inverted adapters x 250; Entry 110 '4,000 images'")
    nper = sorted({v["n_images"] for v in GE["adapters"].values()})
    M("nDoseInvImagesPerAdapter", nper[0], integer(nper[0]), f_ge, "adapters.*.n_images (all equal)", "110",
      "generations per inverted adapter")
    # additive parts (shared lean toward K_A), normal 16000 vs inverted
    add_inv = 0.5 * (ae["theta_A"] - ae["theta_B"])
    M("nDoseAdditiveInverted", add_inv, sci(add_inv, 2), f_ge, "0.5*(all_eight.theta_A - all_eight.theta_B)", "110",
      "matches Entry 110 +6.9e-05")
    M("nDoseAdditiveNormal", D["16000"]["additive_part"], sci(D["16000"]["additive_part"], 2), f_dose,
      "doses.16000.additive_part", "113",
      "dose_stats.json is Entry 68's output (pooled six per body, as nDoseSixteen*); Entry 68 does not print the "
      "additive part. Entry 110 item 2 prints '+5.0e-05 normal', a register misprint of 5.756e-05 (file wins), "
      "corrected by Entry 113")

    # within-body inverted minus normal (descriptive, Entry 110)
    f_wb = OUT + "/g1_within_body.json"
    WB = load(f_wb)
    for b, pre in (("A", "A"), ("B", "B")):
        pb = WB["per_body"][b]
        inv, nor = np.array(WB["inverted_own"][b]), np.array(WB["normal_16k_own"][b])
        assert abs((inv.mean() - nor.mean()) - pb["difference"]) < 1e-12
        M(f"nDoseWithin{pre}Lam", pb["difference_pct"], sig(pb["difference_pct"], 3), f_wb,
          f"per_body.{b}.difference_pct", "110",
          f"% of R_real (E2); Entry 110 {'-0.29' if b == 'A' else '-0.35'} %; verify_v2 check {69 if b == 'A' else 70}")
        M(f"nDoseWithin{pre}Diff", pb["difference"], sci(pb["difference"], 3), f_wb, f"per_body.{b}.difference",
          "110", f"inverted minus normal own-contrast, NCC units; verify_v2 check {69 if b == 'A' else 70}")
        M(f"nDoseWithin{pre}SE", pb["se"], sci(pb["se"], 3), f_wb, f"per_body.{b}.se", "110", "Welch SE")
        M(f"nDoseWithin{pre}T", pb["t"], tval(pb["t"]), f_wb, f"per_body.{b}.t", "110",
          f"matches Entry 110 t {'-3.15' if b == 'A' else '-3.13'}")
        M(f"nDoseWithin{pre}Df", pb["df"], dfv(pb["df"]), f_wb, f"per_body.{b}.df", "110", "Welch df")
        M(f"nDoseWithin{pre}P", pb["one_sided_p_drop"], pval(pb["one_sided_p_drop"]), f_wb,
          f"per_body.{b}.one_sided_p_drop", "110",
          f"one-sided; Entry 110 {'0.0046' if b == 'A' else '0.0060'} (brief 0.005/0.006)")
        M(f"nDoseWithin{pre}PFine", pb["one_sided_p_drop"], pfine(pb["one_sided_p_drop"]), f_wb,
          f"per_body.{b}.one_sided_p_drop", "110", "two significant figures, as Entry 110 table")
        M(f"nDoseWithin{pre}NormalMean", nor.mean(), sci(nor.mean(), 3 if abs(nor.mean()) > 1e-6 else 2), f_wb,
          f"mean(normal_16k_own.{b})", "110", "normal 16000-step own-contrast mean (6 adapters)")
        M(f"nDoseWithin{pre}InvertedMean", inv.mean(), sci(inv.mean(), 3), f_wb, f"mean(inverted_own.{b})", "110",
          "inverted own-contrast mean (8 adapters)")
    bb = WB["both_bodies"]
    M("nDoseWithinBothLam", bb["difference_pct"], sig(bb["difference_pct"], 3), f_wb, "both_bodies.difference_pct",
      "110", "% of R_real (E2); matches Entry 110 -0.32 %")
    M("nDoseWithinBothDiff", bb["difference"], sci(bb["difference"], 3), f_wb, "both_bodies.difference", "110",
      "matches Entry 110 -1.15e-04; equals the change in theta_sym between conditions")
    M("nDoseWithinBothZ", bb["z"], dec(bb["z"], 2), f_wb, "both_bodies.z", "110", "matches Entry 110 z -4.42")
    M("nDoseWithinBothP", bb["one_sided_p"], sci(bb["one_sided_p"], 2), f_wb, "both_bodies.one_sided_p", "110",
      "DISAGREES with Entry 110 table '1e-05': file gives 5.0e-06; file wins")
    # adapted-weight norm within body, inverted vs normal (Entry 110: A +0.056 p 0.42; B +0.033 p 0.71)
    h1ad = load(f_h1)["adapters"]
    for b in "AB":
        nor = np.array([r["lora_B_norm"] for r in sorted(h1ad, key=lambda r: r["seed"]) if r["body"] == b])
        tags = sorted([k for k in GE["adapters"] if f"_{b}_s" in k], key=seed_of)
        inv = np.array([GE["adapters"][k]["lora_B_norm"] for k in tags])
        tt, pp = stats.ttest_ind(inv, nor, equal_var=False)
        k = f"inverted (g1_ext adapters.*_{b}_s*.lora_B_norm) minus normal (h1_strength adapters body {b}), Welch"
        M(f"nDoseWithinNormDiff{b}", inv.mean() - nor.mean(), dec(inv.mean() - nor.mean(), 3), f_ge + " ; " + f_h1, k,
          "110", f"matches Entry 110 {'+0.056' if b == 'A' else '+0.033'}")
        M(f"nDoseWithinNormP{b}", float(pp), pval(float(pp)), f_ge + " ; " + f_h1, k + ", two-sided p", "110",
          f"matches Entry 110 {'0.42' if b == 'A' else '0.71'}")

    # ============================================================== Gen: G5, second training set
    f_g5 = OUT + "/g5_alt_training.json"
    G5 = load(f_g5)
    g = G5["alt"]
    M("nGenGfiveLam", g["theta_sym_pct"], sig(g["theta_sym_pct"], 3), f_g5, "alt.theta_sym_pct", "96",
      "% of R_real (E2); matches Entry 96 -0.0139 % and verify_v2 check 36")
    M("nGenGfiveTheta", g["theta_sym"], sci(g["theta_sym"], 3), f_g5, "alt.theta_sym", "96", "matches -4.943e-06")
    M("nGenGfiveSE", g["welch_se"], sci(g["welch_se"], 3), f_g5, "alt.welch_se", "96", "Welch SE")
    M("nGenGfiveDf", g["welch_df"], dfv(g["welch_df"]), f_g5, "alt.welch_df", "96", "Welch df")
    M("nGenGfiveT", g["t"], tval(g["t"]), f_g5, "alt.t", "96", "matches Entry 96 t -0.309")
    M("nGenGfiveP", g["one_sided_p"], pval(g["one_sided_p"]), f_g5, "alt.one_sided_p", "96",
      "one-sided (theta>0); Entry 96 0.608 (2 s.f. 0.61)")
    M("nGenGfiveLimit", g["limit99_pct"], sig(g["limit99_pct"], 3), f_g5, "alt.limit99_pct", "96",
      "symmetric one-sided 99 %, % of R_real (E2); Entry 96 0.2629, verify_v2 check 37")
    M("nGenGfiveMaxArm", g["maxarm_U_pct"], sig(g["maxarm_U_pct"], 3), f_g5, "alt.maxarm_U_pct", "96",
      "max-arm, % of R_real (E2); matches Entry 96 0.9872 %")
    M("nGenGfiveThetaA", g["theta_A"], sci(g["theta_A"], 3), f_g5, "alt.theta_A", "96", "matches -5.178e-05")
    M("nGenGfiveThetaB", g["theta_B"], sci(g["theta_B"], 3), f_g5, "alt.theta_B", "96", "matches +4.190e-05")
    c = G5["comparison"]
    M("nGenGfiveDiff", c["difference"], sci(c["difference"], 3), f_g5, "comparison.difference", "96",
      "G5 minus primary theta_sym; matches -9.54e-06")
    M("nGenGfiveDiffSE", c["se_difference"], sci(c["se_difference"], 3), f_g5, "comparison.se_difference", "96",
      "matches 1.84e-05")
    M("nGenGfiveDiffZ", c["z"], dec(c["z"], 2), f_g5, "comparison.z", "96", "matches z -0.52")
    M("nGenGfiveDiffP", c["two_sided_p_normal"], pval(c["two_sided_p_normal"]), f_g5, "comparison.two_sided_p_normal",
      "96", "two-sided; matches Entry 96 0.603 / brief 0.60")
    M("nGenGfiveAdapters", G5["n_per_arm"], integer(G5["n_per_arm"]), f_g5, "n_per_arm", "96", "adapters per arm")
    ni = sum(g["images_per_adapter"].values())
    M("nGenGfiveImages", ni, integer(ni), f_g5, "sum(alt.images_per_adapter)", "96", "6 x 500")

    # ============================================================== Gen: FLUX.1-dev, six per arm
    f_fx = OUT + "/flux_seed_ext_summary.json"
    FX = load(f_fx)
    s = FX["symmetric"]
    M("nGenFluxMaxArm", FX["max_arm"]["lambda_U_pct"], sig(FX["max_arm"]["lambda_U_pct"], 3), f_fx,
      "max_arm.lambda_U_pct", "70", "max-arm, % of R_real (E2); Entry 70 0.3353 %, verify_v2 check 34")
    M("nGenFluxSym", s["lambda_sym_U_pct"], sig(s["lambda_sym_U_pct"], 3), f_fx, "symmetric.lambda_sym_U_pct", "70",
      "symmetric one-sided 99 %, % of R_real (E2); Entry 70 0.1456 %, verify_v2 check 35")
    M("nGenFluxTheta", s["theta_sym"], sci(s["theta_sym"], 3), f_fx, "symmetric.theta_sym", "70",
      "matches Entry 70 +5.5611e-6")
    M("nGenFluxLam", s["lambda_hat_pct"], sig(s["lambda_hat_pct"], 3), f_fx, "symmetric.lambda_hat_pct", "70",
      "% of R_real (E2); matches Entry 70 0.0156 %")
    M("nGenFluxSE", s["welch_se"], sci(s["welch_se"], 3), f_fx, "symmetric.welch_se", "70", "matches 1.615e-5")
    M("nGenFluxDf", s["welch_df"], dfv(s["welch_df"]), f_fx, "symmetric.welch_df", "70", "matches df 8.30")
    M("nGenFluxT", s["t"], tval(s["t"]), f_fx, "symmetric.t", "70", "matches t 0.34")
    M("nGenFluxP", s["p_one_sided"], pval(s["p_one_sided"]), f_fx, "symmetric.p_one_sided", "70",
      "one-sided; matches Entry 70 0.3696")
    M("nGenFluxArmAMean", FX["arms"]["A"]["mean"], sci(FX["arms"]["A"]["mean"], 3), f_fx, "arms.A.mean", "70",
      "matches +8.57e-6")
    M("nGenFluxArmBMean", FX["arms"]["B"]["mean"], sci(FX["arms"]["B"]["mean"], 3), f_fx, "arms.B.mean", "70",
      "matches +2.55e-6")
    M("nGenFluxSignflipPA", FX["iut"]["p_A"], pval(FX["iut"]["p_A"]), f_fx, "iut.p_A", "70", "exact, 0.4062")
    M("nGenFluxSignflipPB", FX["iut"]["p_B"], pval(FX["iut"]["p_B"]), f_fx, "iut.p_B", "70", "exact, 0.4219")
    M("nGenFluxSignflipFloor", FX["iut"]["floor"], dec(FX["iut"]["floor"], 4), f_fx, "iut.floor", "70",
      "1/64 = 0.0156, the attainable floor at k = 6")
    M("nGenFluxAdapters", FX["arms"]["A"]["k"], integer(FX["arms"]["A"]["k"]), f_fx, "arms.A.k", "70",
      "adapters per arm")
    nimg = sum(a["n_gen"] for arm in ("A", "B") for a in FX["arms"][arm]["adapters"])
    M("nGenFluxImages", nimg, integer(nimg), f_fx, "sum(arms.*.adapters.*.n_gen)", "70", "12 x 500")
    sbd = FX["e0_record"]["seed_bank"]["images"]
    sb = [sbd[k] for k in sorted(sbd, key=int)] if isinstance(sbd, dict) else sbd
    for i, w in ((0, "One"), (1, "Two")):
        M(f"nGenFluxSeedR{w}", sb[i]["r_same"], dec(sb[i]["r_same"], 4), f_fx,
          f"e0_record.seed_bank.images.{i}.r_same", "70", f"matches Entry 70 {'0.9991' if i == 0 else '0.9830'}")
        M(f"nGenFluxSeedROther{w}", sb[i]["r_other_seed"], dec(sb[i]["r_other_seed"], 3), f_fx,
          f"e0_record.seed_bank.images.{i}.r_other_seed", "70", f"matches Entry 70 {'0.063' if i == 0 else '0.060'}")
        M(f"nGenFluxSeedDiff{w}", sb[i]["mean_abs_diff_255"], dec(sb[i]["mean_abs_diff_255"], 2), f_fx,
          f"e0_record.seed_bank.images.{i}.mean_abs_diff_255", "70",
          f"mean |diff| of 255; matches Entry 70 {'0.93' if i == 0 else '4.51'}")
    v = FX["v1_reproduced"]["lambda_U_n3_pct"]
    M("nGenFluxThreeMaxArm", v, sig(v, 3), f_fx, "v1_reproduced.lambda_U_n3_pct", "70",
      "max-arm at n = 3 (notebook 07), % of R_real (E2); matches 0.777 %")
    v = FX["context"]["sd35_maxarm_k6_lambda_pct"]
    M("nGenSdSixMaxArm", v, sig(v, 3), f_fx, "context.sd35_maxarm_k6_lambda_pct", "70",
      "SD-3.5 primary at six adapters per arm, max-arm, % of R_real (E2); = ledger history_k6 0.249")

    # ============================================================== Gen: ledger rows (full FT, five-body, low/mid)
    LED = load(LEDGER)
    reps = {r["name"]: r for r in LED["replications"]}
    ff = reps["Full fine-tuning"]
    M("nGenFullSym", ff["lambda_U_pct"], sig(ff["lambda_U_pct"], 3), LEDGER, "replications[Full fine-tuning].lambda_U_pct",
      "(v1)", "symmetric per-seed paired t, one-sided 99.5 % (S99.5), % of R_real (E2); brief prints 0.75")
    f_ff = _drive("E_FULLFT", "E_FULLFT_results.json")
    FF = load(f_ff)
    ffmax = 100 * max(FF["arms"]["A"]["U"], FF["arms"]["B"]["U"]) / R_REAL
    M("nGenFullMaxArm", ffmax, sig(ffmax, 3), f_ff, "max(arms.A.U, arms.B.U) / R_real", "(v1)",
      "per-arm (max-arm) limit, % of R_real (E2); matches v4 text 2.72 %")
    M("nGenFullTheta", FF["symmetric"]["theta_sym"], sci(FF["symmetric"]["theta_sym"], 2), f_ff, "symmetric.theta_sym",
      "(v1)", "matches v4 text -2.3e-5")
    M("nGenFullT", FF["symmetric"]["t"], tval(FF["symmetric"]["t"]), f_ff, "symmetric.t", "(v1)",
      "matches v4 text t -0.77")
    ffs = 100 * FF["symmetric"]["lambda_U"]
    assert abs(ffs - ff["lambda_U_pct"]) < 0.001, "full-FT ledger and Drive file disagree"
    M("nGenFullAdapters", FF["arms"]["A"]["n"], integer(FF["arms"]["A"]["n"]), f_ff, "arms.A.n", "(v1)",
      "adapters per arm")
    kd = reps["Kodak M1063, CCD"]
    M("nGenKodakLimit", kd["lambda_U_pct"], sig(kd["lambda_U_pct"], 3), LEDGER,
      "replications[Kodak M1063, CCD].lambda_U_pct", "50",
      "limit over five device means (D), R_real at its lower 99 % limit; Kodak's own R_real, not E2")
    f_v4o = OUT + "/v4_offline.json"
    ks = load(f_v4o)["kodak_exact_signflip"]
    M("nGenKodakSignflipCount", ks["count"], integer(ks["count"]), f_v4o, "kodak_exact_signflip.count", "50",
      "matches Entry 50 14/32")
    M("nGenKodakSignflipOf", ks["of"], integer(ks["of"]), f_v4o, "kodak_exact_signflip.of", "50", "2^5 sign patterns")
    M("nGenKodakP", ks["p"], pval(ks["p"]), f_v4o, "kodak_exact_signflip.p", "50", "exact 0.4375")
    f_md = _drive("E_MULTIDEV", "C4_multidev.json")
    raw = load(f_md)["variants"]["raw"]
    M("nGenKodakIcc", raw["icc"], dec(raw["icc"], 3), f_md, "variants.raw.icc", "(v1)", "matches v4 text ICC 0.943")
    M("nGenKodakGrandMean", raw["grand_mean"], sci(raw["grand_mean"], 2), f_md, "variants.raw.grand_mean", "(v1)",
      "matches v4 text +1.3e-5")
    M("nGenKodakBodies", raw["n_devices"], integer(raw["n_devices"]), f_md, "variants.raw.n_devices", "(v1)", "5")
    M("nGenKodakSeeds", raw["seeds_per_device"], integer(raw["seeds_per_device"]), f_md,
      "variants.raw.seeds_per_device", "(v1)", "2")
    M("nGenKodakSignAgree", raw["n_sign_agree"], integer(raw["n_sign_agree"]), f_md, "variants.raw.n_sign_agree",
      "(v1)", "bodies whose two seeds agree in sign")
    p20 = reps["Huawei P20, smartphone"]
    sm = LED["smartphone"]
    M("nGenPtwentyLimit", p20["lambda_U_pct"], sig(p20["lambda_U_pct"], 3), LEDGER,
      "replications[Huawei P20, smartphone].lambda_U_pct", "(v1)", "limit over five device means (D); own R_real")
    M("nGenPtwentyP", sm["signflip_p"], pval(sm["signflip_p"]), LEDGER, "smartphone.signflip_p", "(v1)",
      "exact sign-flip over five bodies, 9/32 = 0.28")
    M("nGenPtwentyIcc", sm["icc"], dec(sm["icc"], 3), LEDGER, "smartphone.icc", "(v1)", "ICC")
    M("nGenPtwentyEta", sm["eta"], dec(sm["eta"], 3), LEDGER, "smartphone.eta", "(v1)", "autoencoder retention")
    M("nGenPtwentyEtaLo", sm["eta_ci"][0], dec(sm["eta_ci"][0], 3), LEDGER, "smartphone.eta_ci.0", "(v1)", "95 % CI")
    M("nGenPtwentyEtaHi", sm["eta_ci"][1], dec(sm["eta_ci"][1], 3), LEDGER, "smartphone.eta_ci.1", "(v1)", "95 % CI")
    M("nGenPtwentyLowmidP", sm["lowmid_signflip_p"], pval(sm["lowmid_signflip_p"]), LEDGER,
      "smartphone.lowmid_signflip_p", "(v1)", "low/mid representation on P20, exact sign-flip 0.094")
    lm = reps["Low/mid band"]
    M("nGenLowmidLimit", lm["lambda_U_pct"], sig(lm["lambda_U_pct"], 3), LEDGER,
      "replications[Low/mid band].lambda_U_pct", "(v1)",
      "S99.5 limit as % of the low/mid representation's own real contrast (not E2 NCC)")
    M("nGenLowmidAdapters", lm["n"], integer(lm["n"]), LEDGER, "replications[Low/mid band].n", "(v1)", "6 per arm")
    M("nGenPrimaryMatchedThree", reps["Primary at matched n=3"]["lambda_U_pct"],
      sig(reps["Primary at matched n=3"]["lambda_U_pct"], 3), LEDGER,
      "replications[Primary at matched n=3].lambda_U_pct", "(v1)", "max-arm at n = 3, % of R_real (E2); 0.70 %")

    # ============================================================== Gen: five captions (E-PROMPT, Entry 22)
    f_t5 = T5 + "/summary_derived.json"
    T5d = load(f_t5)
    dv = T5d["diverse_bank"]
    M("nGenCaptionsTheta", dv["theta_sym"], sci(dv["theta_sym"], 2), f_t5, "diverse_bank.theta_sym", "22",
      "matches Entry 22 +2.1e-05")
    M("nGenCaptionsLam", dv["lambda_sym_pct"], sig(dv["lambda_sym_pct"], 3), f_t5, "diverse_bank.lambda_sym_pct",
      "22", "% of R_real (E2); Entry 22 0.06 %")
    M("nGenCaptionsSE", dv["adapter_level_SE"], sci(dv["adapter_level_SE"], 2), f_t5, "diverse_bank.adapter_level_SE",
      "22", "matches 2.1e-05")
    M("nGenCaptionsDf", dv["welch_df"], dfv(dv["welch_df"]), f_t5, "diverse_bank.welch_df", "22", "matches df 2.2")
    M("nGenCaptionsT", dv["t"], tval(dv["t"]), f_t5, "diverse_bank.t", "22", "Entry 22 t = 1.0")
    M("nGenCaptionsLimit", dv["lambda_U_plugin_n3_pct"], sig(dv["lambda_U_plugin_n3_pct"], 3), f_t5,
      "diverse_bank.lambda_U_plugin_n3_pct", "22",
      "symmetric Welch one-sided 99.5 % (S99.5), % of R_real (E2); Entry 22 / v4 0.56 %")
    M("nGenCaptionsAdditive", dv["additive_part"], sci(dv["additive_part"], 3), f_t5, "diverse_bank.additive_part",
      "22", "matches Entry 22 -10.8e-05")
    M("nGenCaptionsBaseLean", dv["base_KB_minus_KA"], sci(dv["base_KB_minus_KA"], 2), f_t5,
      "diverse_bank.base_KB_minus_KA", "22", "base model K_B - K_A, diverse bank; Entry 22 1.4e-04")
    ub = T5d["uniform_bank_same_adapters_entry10"]["base_KB_minus_KA"]
    M("nGenCaptionsBaseLeanUniform", ub, sci(ub, 2), f_t5, "uniform_bank_same_adapters_entry10.base_KB_minus_KA",
      "22", "base model K_B - K_A, single caption; Entry 22 0.4e-04")
    M("nGenCaptionsBaseLeanRatio", dv["base_KB_minus_KA"] / ub, dec(dv["base_KB_minus_KA"] / ub, 1), f_t5,
      "diverse / uniform base_KB_minus_KA", "22", "Entry 22 'three times'")
    M("nGenCaptionsMaxImageT", dv["max_abs_image_t"], dec(dv["max_abs_image_t"], 1), f_t5,
      "diverse_bank.max_abs_image_t", "22", "largest unpaired arm image-level t (B_s2); Entry 22 2.6")
    M("nGenCaptionsAdapters", len(dv["per_arm_own_contrast"]) // 2, integer(len(dv["per_arm_own_contrast"]) // 2),
      f_t5, "len(diverse_bank.per_arm_own_contrast)/2", "22", "adapters per arm")

    # ============================================================== Gen: G2 second environment (six per arm)
    f_g2 = OUT + "/g2_pooled_six.json"
    G2 = load(f_g2)
    lam2 = 100 * G2["theta_sym"] / R_REAL
    M("nGenGtwoLam", lam2, sig(lam2, 3), f_g2, "100*theta_sym/R_real", "79",
      "% of R_real (E2); matches Entry 79 +0.0894 % and verify_v2 check 49")
    M("nGenGtwoTheta", G2["theta_sym"], sci(G2["theta_sym"], 3), f_g2, "theta_sym", "79", "matches +3.188e-5")
    M("nGenGtwoSE", G2["welch_se"], sci(G2["welch_se"], 3), f_g2, "welch_se", "79", "matches 1.929e-5")
    M("nGenGtwoDf", G2["welch_df"], dfv(G2["welch_df"]), f_g2, "welch_df", "79", "matches df 5.67")
    M("nGenGtwoT", G2["t"], tval(G2["t"]), f_g2, "t", "79", "matches t 1.65")
    M("nGenGtwoP", G2["one_sided_p"], pval(G2["one_sided_p"]), f_g2, "one_sided_p", "79",
      "one-sided; matches 0.076 and verify_v2 check 50")
    M("nGenGtwoLimit", G2["limit99_pct"], sig(G2["limit99_pct"], 3), f_g2, "limit99_pct", "79",
      "symmetric one-sided 99 %, % of R_real (E2); Entry 79 0.2627 %")
    M("nGenGtwoAdapters", len(G2["per_adapter_A"]), integer(len(G2["per_adapter_A"])), f_g2, "len(per_adapter_A)",
      "79", "six per arm (brief warns the old table said 3)")
    f_c12 = OUT + "/g_chain12.json"
    g2n = load(f_c12)["G2"]
    M("nGenGtwoNewLam", g2n["lambda_pct"], sig(g2n["lambda_pct"], 3), f_c12, "G2.lambda_pct", "79",
      "new three alone, % of R_real (E2); matches +0.0994 %")
    M("nGenGtwoNewT", g2n["t"], tval(g2n["t"]), f_c12, "G2.t", "79", "matches t 0.89")
    M("nGenGtwoNewDf", g2n["welch_df"], dfv(g2n["welch_df"]), f_c12, "G2.welch_df", "79",
      "Welch df of the three new adapters per arm (Table S15 row 'its seeds 3-5'); added 29 Sep 2026")
    M("nGenGtwoNewP", g2n["one_sided_p_gt0"], pval(g2n["one_sided_p_gt0"]), f_c12, "G2.one_sided_p_gt0", "79",
      "matches 0.233")

    # ============================================================== Gen: random-crop training (Entry 34)
    f_e3 = T1 + "/e3_stats.json"
    E3 = load(f_e3)
    M("nGenRcropLam", E3["rcrop_lambda_pct"], sig(E3["rcrop_lambda_pct"], 3), f_e3, "rcrop_lambda_pct", "34",
      "% of R_real (E2), unpaired A-body mean; Entry 34 0.05 %")
    M("nGenRcropMean", E3["rcrop_mean"], sci(E3["rcrop_mean"], 2), f_e3, "rcrop_mean", "34", "matches +1.6e-5")
    M("nGenRcropSE", E3["rcrop_se"], sci(E3["rcrop_se"], 2), f_e3, "rcrop_se", "34", "matches 5.0e-5")
    M("nGenRcropFixedMean", E3["nomark_mean"], sci(E3["nomark_mean"], 2), f_e3, "nomark_mean", "34", "matches +6.2e-5")
    M("nGenRcropFixedSE", E3["nomark_se"], sci(E3["nomark_se"], 2), f_e3, "nomark_se", "34", "matches 2.6e-5")
    M("nGenRcropDiff", E3["difference_rcrop_minus_nomark"], sci(E3["difference_rcrop_minus_nomark"], 2), f_e3,
      "difference_rcrop_minus_nomark", "34", "matches -4.6e-5")
    M("nGenRcropT", E3["welch_t"], tval(E3["welch_t"]), f_e3, "welch_t", "34", "matches t -0.82 (file -0.807)")
    M("nGenRcropP", E3["p_two_sided"], pval(E3["p_two_sided"]), f_e3, "p_two_sided", "34",
      "two-sided; file 0.478 -> 0.48, Entry 34 prints 0.47")
    M("nGenRcropAdapters", len(E3["rcrop_values"]), integer(len(E3["rcrop_values"])), f_e3, "len(rcrop_values)",
      "34", "3")

    # ============================================================== Gen: G6 iPhone 5c (uniform caption)
    f_g6 = OUT + "/g6_p5c.json"
    G6 = load(f_g6)
    for tag, est in (("Nat", "E2"), ("Flat", "FLAT")):
        e = G6["estimators"][est]
        R = e["R_real"]
        th, se, df = welch_sym(e["per_adapter_A"], e["per_adapter_B"])
        lam = 100 * th / R
        rr = f"of this pair's own {est} R_real ({R:.6f}), not E2"
        chk = {"E2": 38, "FLAT": 41}[est]
        M(f"nGenGsix{tag}Lam", lam, sig(lam, 3), f_g6, f"welch_sym(estimators.{est}.per_adapter_*)/R_real", "103",
          f"% {rr}; verify_v2 check {chk}; Entry 103 {'-0.019' if est == 'E2' else '+0.023'} %")
        M(f"nGenGsix{tag}Theta", th, sci(th, 3), f_g6, f"estimators.{est}.symmetric.theta_sym", "103", "NCC units")
        M(f"nGenGsix{tag}SE", se, sci(se, 3), f_g6, f"estimators.{est}.symmetric.welch_se", "103", "Welch SE")
        M(f"nGenGsix{tag}Df", df, dfv(df), f_g6, f"estimators.{est}.symmetric.welch_df", "103", "Welch df")
        M(f"nGenGsix{tag}T", th / se, tval(th / se), f_g6, f"estimators.{est}.symmetric.t", "103",
          f"Entry 103 t {'-0.70' if est == 'E2' else '+1.22'}")
        pp = e["symmetric"]["one_sided_p"]
        M(f"nGenGsix{tag}P", pp, pval(pp), f_g6, f"estimators.{est}.symmetric.one_sided_p", "103",
          f"one-sided; Entry 103 {'0.753' if est == 'E2' else '0.118'}")
        M(f"nGenGsix{tag}PFine", pp, pfine(pp, 3), f_g6, f"estimators.{est}.symmetric.one_sided_p", "103",
          "3 s.f., as Entry 103 prints it")
        M(f"nGenGsix{tag}Sym", e["symmetric"]["lambda_sym_pct"], sig(e["symmetric"]["lambda_sym_pct"], 3), f_g6,
          f"estimators.{est}.symmetric.lambda_sym_pct", "103",
          f"symmetric one-sided 99 %, % {rr}; verify_v2 check {chk + 1}")
        M(f"nGenGsix{tag}MaxArm", e["max_arm"]["lambda_U_pct"], sig(e["max_arm"]["lambda_U_pct"], 3), f_g6,
          f"estimators.{est}.max_arm.lambda_U_pct", "103", f"max-arm, % {rr}; verify_v2 check {chk + 2}")
        M(f"nGenGsix{tag}ArmA", e["mean_A"], sci(e["mean_A"], 3), f_g6, f"estimators.{est}.mean_A", "103",
          "arm A mean own-minus-other")
        M(f"nGenGsix{tag}ArmB", e["mean_B"], sci(e["mean_B"], 3), f_g6, f"estimators.{est}.mean_B", "103",
          "arm B mean own-minus-other")
        na = int((np.array(e["per_adapter_A"]) > 0).sum())
        nb = int((np.array(e["per_adapter_B"]) > 0).sum())
        M(f"nGenGsix{tag}APos", na, integer(na), f_g6, f"count(estimators.{est}.per_adapter_A > 0)", "103",
          f"Entry 103 {'0' if est == 'E2' else '8'} of 12")
        M(f"nGenGsix{tag}BPos", nb, integer(nb), f_g6, f"count(estimators.{est}.per_adapter_B > 0)", "103",
          f"Entry 103 {'12' if est == 'E2' else '9'} of 12")
        add = 0.5 * (e["mean_A"] - e["mean_B"])
        M(f"nGenGsix{tag}Additive", add, sci(add, 3), f_g6, f"0.5*(estimators.{est}.mean_A - mean_B)", "103",
          f"Entry 103 {'-1.198e-04' if est == 'E2' else '-1.21e-05'}")
        M(f"nGenGsix{tag}AdditivePct", 100 * abs(add) / R, sig(100 * abs(add) / R, 2), f_g6,
          f"100*|additive|/estimators.{est}.R_real", "103",
          f"magnitude as % {rr}; Entry 103 prints 0.25 % for E2")
        iu = e["iu_signflip"]
        M(f"nGenGsix{tag}SignflipPA", iu["p_A"], pval(iu["p_A"]), f_g6, f"estimators.{est}.iu_signflip.p_A", "103",
          "exact sign-flip, arm A")
        M(f"nGenGsix{tag}SignflipPB", iu["p_B"], pval(iu["p_B"]), f_g6, f"estimators.{est}.iu_signflip.p_B", "103",
          "exact sign-flip, arm B (E2: at the 1/4096 floor, Entry 103 0.0002)")
    ed = G6["estimator_dependence"]
    M("nGenGsixSymRatio", ed["ratio_FLAT_over_E2"], dec(ed["ratio_FLAT_over_E2"], 2), f_g6,
      "estimator_dependence.ratio_FLAT_over_E2", "103", "symmetric limits FLAT/E2; matches Entry 103 x1.42")
    mr = G6["estimators"]["FLAT"]["max_arm"]["lambda_U_pct"] / G6["estimators"]["E2"]["max_arm"]["lambda_U_pct"]
    M("nGenGsixMaxArmRatio", mr, dec(mr, 2), f_g6, "FLAT.max_arm / E2.max_arm", "103", "matches Entry 103 x0.32")
    M("nGenGsixCorr", ed["per_adapter_corr"], dec(ed["per_adapter_corr"], 2), f_g6, "estimator_dependence.per_adapter_corr",
      "103", "per-adapter r between estimators; matches Entry 103 r = 0.35")
    ar = (0.5 * abs(G6["estimators"]["E2"]["mean_A"] - G6["estimators"]["E2"]["mean_B"])
          / (0.5 * abs(G6["estimators"]["FLAT"]["mean_A"] - G6["estimators"]["FLAT"]["mean_B"])))
    M("nGenGsixAdditiveRatio", ar, dec(ar, 1), f_g6, "|additive E2| / |additive FLAT|", "103",
      "Entry 103 'ten times smaller'")
    M("nGenGsixAdapters", G6["estimators"]["E2"]["n_adapters_per_arm"], integer(G6["estimators"]["E2"]["n_adapters_per_arm"]),
      f_g6, "estimators.E2.n_adapters_per_arm", "103", "12 per arm")

    # ============================================================== Gen: G4b Huawei P20 pair
    f_4b = OUT + "/g4b_p20.json"
    G4 = load(f_4b)
    R4 = G4["R_real"]
    th, se, df = welch_sym(G4["per_adapter_A"], G4["per_adapter_B"])
    rr4 = f"of this pair's own E2 R_real ({R4:.6f}), not the D200 E2"
    M("nGenGfourbLam", 100 * th / R4, sig(100 * th / R4, 3), f_4b, "welch_sym(per_adapter_*)/R_real", "105",
      f"% {rr4}; verify_v2 check 44 (0.0605); Entry 105 +0.060 %")
    M("nGenGfourbTheta", th, sci(th, 3), f_4b, "symmetric.theta_sym", "105", "matches +2.373e-05")
    M("nGenGfourbSE", se, sci(se, 3), f_4b, "symmetric.welch_se", "105", "matches 1.013e-05")
    M("nGenGfourbDf", df, dfv(df), f_4b, "symmetric.welch_df", "105", "matches df 22.0")
    M("nGenGfourbT", th / se, tval(th / se), f_4b, "symmetric.t", "105", "matches t 2.34")
    pp = G4["symmetric"]["one_sided_p"]
    M("nGenGfourbP", pp, pval(pp), f_4b, "symmetric.one_sided_p", "105", "one-sided; verify_v2 check 45 (0.0143)")
    M("nGenGfourbPFine", pp, pfine(pp, 3), f_4b, "symmetric.one_sided_p", "105", "3 s.f., as Entry 105")
    M("nGenGfourbSym", G4["symmetric"]["lambda_sym_pct"], sig(G4["symmetric"]["lambda_sym_pct"], 3), f_4b,
      "symmetric.lambda_sym_pct", "105", f"symmetric one-sided 99 %, % {rr4}; Entry 105 0.1252 %")
    M("nGenGfourbMaxArm", G4["max_arm"]["lambda_U_pct"], sig(G4["max_arm"]["lambda_U_pct"], 3), f_4b,
      "max_arm.lambda_U_pct", "105", f"max-arm, % {rr4}; verify_v2 check 46 (0.468)")
    M("nGenGfourbArmA", G4["mean_A"], sci(G4["mean_A"], 3), f_4b, "mean_A", "105", "matches +1.395e-04")
    M("nGenGfourbArmB", G4["mean_B"], sci(G4["mean_B"], 3), f_4b, "mean_B", "105", "matches -9.206e-05")
    M("nGenGfourbAPos", G4["n_positive"]["A"], integer(G4["n_positive"]["A"]), f_4b, "n_positive.A", "105", "12 of 12")
    M("nGenGfourbBPos", G4["n_positive"]["B"], integer(G4["n_positive"]["B"]), f_4b, "n_positive.B", "105", "0 of 12")
    M("nGenGfourbAdditive", G4["additive_part"], sci(G4["additive_part"], 3), f_4b, "additive_part", "105",
      "matches +1.158e-04")
    ap = 100 * G4["additive_part"] / R4
    M("nGenGfourbAdditivePct", ap, sig(ap, 2), f_4b, "100*additive_part/R_real", "105",
      f"% {rr4}; matches Entry 105 0.30 % (file 0.295)")
    ratio = G4["additive_part"] / G4["symmetric"]["theta_sym"]
    M("nGenGfourbAdditiveOverTheta", ratio, dec(ratio, 1), f_4b, "additive_part / symmetric.theta_sym", "105",
      "Entry 105 'five times theta_sym'")
    M("nGenGfourbSignflipPA", G4["iu_signflip"]["p_A"], pval(G4["iu_signflip"]["p_A"]), f_4b, "iu_signflip.p_A",
      "105", "exact sign-flip at the 1/4096 floor; Entry 105 0.00024")
    M("nGenGfourbSignflipPB", G4["iu_signflip"]["p_B"], pval(G4["iu_signflip"]["p_B"]), f_4b, "iu_signflip.p_B",
      "105", "1.000")
    M("nGenGfourbAdapters", len(G4["per_adapter_A"]), integer(len(G4["per_adapter_A"])), f_4b, "len(per_adapter_A)",
      "105", "12 per arm")
    M("nGenSignflipFloor", G4["iu_signflip"]["floor"], sig(G4["iu_signflip"]["floor"], 2), f_4b, "iu_signflip.floor",
      "105", "1/4096 at twelve adapters per arm")

    # ============================================================== Pone: five everyday captions on the G6 adapters
    f_p1 = OUT + "/p1_prompts.json"
    f_dv = OUT + "/g6_p5c_div.json"
    P1 = load(f_p1)["estimators"]
    DVv = load(f_dv)["estimators"]
    for tag, est in (("Nat", "E2"), ("Flat", "FLAT")):
        e = DVv[est]
        R = e["R_real"]
        th, se, df = welch_sym(e["per_adapter_A"], e["per_adapter_B"])
        pd_ = P1[est]["diverse"]
        assert abs(th - pd_["theta_sym"]) < 1e-12
        rr = f"of the iPhone 5c pair's own {est} R_real ({R:.6f}), not E2"
        chk = {"E2": 71, "FLAT": 73}[est]
        lam = 100 * th / R
        M(f"nPone{tag}Lam", lam, dec(lam, 3), f_p1, f"estimators.{est}.diverse.theta_sym_pct", "111",
          f"% {rr}; printed to 3 decimals as Entry 111 ({'-0.004' if est == 'E2' else '+0.067'}); verify_v2 check {chk}")
        M(f"nPone{tag}Theta", th, sci(th, 3), f_dv, f"estimators.{est}.symmetric.theta_sym", "111",
          f"matches Entry 111 {'-1.72e-06' if est == 'E2' else '+3.91e-05'}")
        M(f"nPone{tag}SE", se, sci(se, 3), f_dv, f"estimators.{est}.symmetric.welch_se", "111", "Welch SE")
        M(f"nPone{tag}Df", df, dfv(df), f_dv, f"estimators.{est}.symmetric.welch_df", "111", "Welch df")
        M(f"nPone{tag}T", th / se, tval(th / se), f_dv, f"estimators.{est}.symmetric.t", "111",
          f"matches Entry 111 t {'-0.10' if est == 'E2' else '+3.20'}")
        M(f"nPone{tag}P", pd_["one_sided_p"], pval(pd_["one_sided_p"]), f_p1, f"estimators.{est}.diverse.one_sided_p",
          "111", f"one-sided; verify_v2 check {chk + 1}; Entry 111 {'0.539' if est == 'E2' else '0.0020'}")
        M(f"nPone{tag}PFine", pd_["one_sided_p"], pfine(pd_["one_sided_p"], 3 if est == "E2" else 2), f_p1,
          f"estimators.{est}.diverse.one_sided_p", "111", "as Entry 111 prints it")
        M(f"nPone{tag}Sym", pd_["symmetric_limit_pct"], sig(pd_["symmetric_limit_pct"], 3), f_p1,
          f"estimators.{est}.diverse.symmetric_limit_pct", "111", f"symmetric one-sided 99 %, % {rr}")
        M(f"nPone{tag}MaxArm", pd_["max_arm_pct"], sig(pd_["max_arm_pct"], 3), f_p1,
          f"estimators.{est}.diverse.max_arm_pct", "111", f"max-arm, % {rr}" + (
              "; Entry 111 table prints 0.198 %, a register rounding misprint of 0.19746 (file wins); "
              "corrected by Entry 113" if est == "FLAT" else "; matches Entry 111 0.311 %"))
        M(f"nPone{tag}ArmA", e["mean_A"], sci(e["mean_A"], 3), f_dv, f"estimators.{est}.mean_A", "111",
          "arm A mean own-minus-other")
        M(f"nPone{tag}ArmB", e["mean_B"], sci(e["mean_B"], 3), f_dv, f"estimators.{est}.mean_B", "111",
          "arm B mean own-minus-other")
        M(f"nPone{tag}SignflipPA", pd_["iu_signflip"]["p_A"], pval(pd_["iu_signflip"]["p_A"]), f_p1,
          f"estimators.{est}.diverse.iu_signflip.p_A", "111",
          f"Entry 111 arm A {'0.99' if est == 'E2' else '0.16'}")
        M(f"nPone{tag}SignflipPB", pd_["iu_signflip"]["p_B"], pval(pd_["iu_signflip"]["p_B"]), f_p1,
          f"estimators.{est}.diverse.iu_signflip.p_B", "111",
          f"Entry 111 arm B {'0.0032 (not in text)' if est == 'E2' else '0.004'}")
        M(f"nPone{tag}Additive", pd_["additive_part"], sci(pd_["additive_part"], 2), f_p1,
          f"estimators.{est}.diverse.additive_part", "111",
          f"Entry 111 {'-8.2e-05' if est == 'E2' else '-2.2e-05'} (diverse bank)")
        M(f"nPone{tag}AdditiveUniform", P1[est]["uniform"]["additive_part"], sci(P1[est]["uniform"]["additive_part"], 2),
          f_p1, f"estimators.{est}.uniform.additive_part", "111",
          f"Entry 111 {'-1.2e-04' if est == 'E2' else '-1.2e-05'} (uniform bank)")
        pdd = P1[est]["paired_difference_diverse_minus_uniform"]
        M(f"nPone{tag}DiffLam", pdd["theta_sym_pct"], dec(pdd["theta_sym_pct"], 3), f_p1,
          f"estimators.{est}.paired_difference_diverse_minus_uniform.theta_sym_pct", "111",
          f"diverse minus uniform, same adapters, % {rr}; Entry 111 {'+0.015' if est == 'E2' else '+0.043'} %")
        M(f"nPone{tag}DiffT", pdd["t"], tval(pdd["t"]), f_p1,
          f"estimators.{est}.paired_difference_diverse_minus_uniform.t", "111", "Welch t")
        M(f"nPone{tag}DiffP", pdd["two_sided_p"], pval(pdd["two_sided_p"]), f_p1,
          f"estimators.{est}.paired_difference_diverse_minus_uniform.two_sided_p", "111",
          f"two-sided; Entry 111 {'0.75' if est == 'E2' else '0.20'}")
        M(f"nPone{tag}RealContrast", R, dec(R, 4), f_dv, f"estimators.{est}.R_real", "111",
          f"iPhone 5c {est} R_real; Entry 111 {'0.0474' if est == 'E2' else '0.0586'}")
    pflat = P1["FLAT"]["diverse"]["one_sided_p"]
    ncomb = 4  # estimator x bank combinations, as Entry 111 counts them: {E2, FLAT} x {uniform, diverse}
    padj = min(1.0, ncomb * pflat)
    M("nPoneFlatPadj", padj, pval(padj), f_p1, "4 * estimators.FLAT.diverse.one_sided_p (Bonferroni)", "111",
      "matches Entry 111 'Bonferroni over four gives p 0.008'")
    M("nPoneCombos", ncomb, integer(ncomb), f_p1, "len({E2, FLAT} x {uniform, diverse})", "111",
      "estimator x bank combinations on the same adapters")
    M("nPoneCorr", load(f_dv)["estimator_dependence"]["per_adapter_corr"],
      dec(load(f_dv)["estimator_dependence"]["per_adapter_corr"], 2), f_dv, "estimator_dependence.per_adapter_corr",
      "111", "per-adapter r between estimators, diverse bank; matches Entry 111 0.42")
    fw = load(f_p1)["firearm"]
    for tag, k in (("Uniform", "uniform_bank"), ("Diverse", "diverse_bank")):
        b = fw[k]
        M(f"nPoneRifle{tag}Share", 100 * b["share"], dec(100 * b["share"], 1), f_p1, f"firearm.{k}.share", "111",
          f"percent of 250 sampled generations; verify_v2 check {75 if tag == 'Uniform' else 76}")
        M(f"nPoneRifle{tag}Count", b["firearm"], integer(b["firearm"]), f_p1, f"firearm.{k}.firearm", "111",
          f"Entry 111 {'243' if tag == 'Uniform' else '0'}/250")
        M(f"nPoneRifle{tag}N", b["n"], integer(b["n"]), f_p1, f"firearm.{k}.n", "111", "sample size")
        M(f"nPoneRifle{tag}Lo", 100 * b["wilson95"][0], dec(100 * b["wilson95"][0], 1), f_p1,
          f"firearm.{k}.wilson95.0", "111", "Wilson 95 % lower, percent")
        M(f"nPoneRifle{tag}Hi", 100 * b["wilson95"][1], dec(100 * b["wilson95"][1], 1), f_p1,
          f"firearm.{k}.wilson95.1", "111", "Wilson 95 % upper, percent; diverse matches Entry 111 1.5")
    na = DVv["E2"]["n_adapters_per_arm"]
    M("nPoneAdapters", na, integer(na), f_dv, "estimators.E2.n_adapters_per_arm", "111", "12 per arm")
    M("nPoneImages", 2 * na * 250, integer(2 * na * 250), f_dv, "2 * n_adapters_per_arm * 250 (Entry 86 budget)", "111",
      "Entry 111 6,000 images")

    # ============================================================== Est: fingerprint estimability gates
    M("nEstGateAuc", 0.90, "0.90", "RESULTS.md", "Entry 78: 'If the held-out AUC is below 0.90'", "78",
      "text-only (registered design constant)")
    M("nEstGateSplitHalf", 0.15, "0.15", "RESULTS.md", "Entry 78: 'or split-half reliability below 0.15'", "78",
      "text-only (registered design constant)")
    f_p10 = OUT + "/g4_p10.json"
    P10 = load(f_p10)["gates"]
    M("nEstPtenAuc", P10["AUC_held_out"], dec(P10["AUC_held_out"], 3), f_p10, "gates.AUC_held_out", "85",
      "corrected (D9) held-out AUC; matches Entry 85 0.809 (table 0.8094)")
    M("nEstPtenAucFine", P10["AUC_held_out"], dec(P10["AUC_held_out"], 4), f_p10, "gates.AUC_held_out", "85",
      "4 decimals, Entry 85 table")
    M("nEstPtenRreal", P10["R_real"], dec(P10["R_real"], 4), f_p10, "gates.R_real", "85", "matches +0.0179")
    M("nEstPtenContrastA", P10["per_body_contrast"]["A"], dec(P10["per_body_contrast"]["A"], 4), f_p10,
      "gates.per_body_contrast.A", "85", "body 1604 own-minus-other; matches +0.0328 (brief +0.033)")
    M("nEstPtenContrastB", P10["per_body_contrast"]["B"], dec(P10["per_body_contrast"]["B"], 4), f_p10,
      "gates.per_body_contrast.B", "85", "body 1601 own-minus-other; matches +0.0030 (brief +0.003)")
    M("nEstPtenSplitHalfA", P10["splithalf_A"], dec(P10["splithalf_A"], 2), f_p10, "gates.splithalf_A", "85",
      "body 1604; matches 0.510")
    M("nEstPtenSplitHalfB", P10["splithalf_B"], dec(P10["splithalf_B"], 2), f_p10, "gates.splithalf_B", "85",
      "body 1601; matches 0.568")
    M("nEstPtenKappa", P10["kappa_model_E2"], dec(P10["kappa_model_E2"], 3), f_p10, "gates.kappa_model_E2", "85",
      "cross-device correlation; matches -0.002")
    M("nEstPtenPairedOwn", 100 * P10["paired_own_gt_other"], dec(100 * P10["paired_own_gt_other"], 1), f_p10,
      "gates.paired_own_gt_other", "85", "percent of held-out photographs scoring own above other")
    f_m10 = OUT + "/fp_p10/manifest.json"
    MP = load(f_m10)["roles"]
    for r, dev in (("A", "A"), ("B", "B")):
        u, raw_ = MP[r]["unique_images"], MP[r]["raw_files"]
        M(f"nEstPtenUnique{dev}", u, integer(u), f_m10, f"roles.{r}.unique_images", "78",
          f"body {MP[r]['device']}; Entry 78 {'257' if r == 'A' else '252'}")
        M(f"nEstPtenRaw{dev}", raw_, integer(raw_), f_m10, f"roles.{r}.raw_files", "78",
          f"body {MP[r]['device']}; Entry 78 {'514' if r == 'A' else '504'}")
        M(f"nEstPtenDup{dev}", 100 * (1 - u / raw_), dec(100 * (1 - u / raw_), 1), f_m10,
          f"1 - roles.{r}.unique_images/raw_files", "78", "percent byte-identical duplicates; Entry 78 'about 50 %'")
    g6g = G6["gates"]
    M("nEstGsixAucNat", g6g["AUC_held_out"], dec(g6g["AUC_held_out"], 4), f_g6, "gates.AUC_held_out", "103",
      "matches Entry 103 0.9988")
    M("nEstGsixAucFlat", g6g["AUC_held_out_FLAT"], dec(g6g["AUC_held_out_FLAT"], 4), f_g6, "gates.AUC_held_out_FLAT",
      "103", "matches Entry 103 0.9940")
    M("nEstGsixRrealNat", g6g["R_real"], dec(g6g["R_real"], 4), f_g6, "gates.R_real", "103", "matches 0.047409")
    M("nEstGsixRrealFlat", g6g["R_real_FLAT"], dec(g6g["R_real_FLAT"], 4), f_g6, "gates.R_real_FLAT", "103",
      "matches 0.058578")
    M("nEstGsixSplitHalfA", g6g["splithalf_A"], dec(g6g["splithalf_A"], 2), f_g6, "gates.splithalf_A", "87",
      "rebuilt splits (Entry 87); supersedes Entry 85's 0.365")
    M("nEstGsixSplitHalfB", g6g["splithalf_B"], dec(g6g["splithalf_B"], 2), f_g6, "gates.splithalf_B", "87",
      "rebuilt splits (Entry 87); supersedes Entry 85's 0.323")
    M("nEstGsixKappaNat", g6g["kappa_model_E2"], dec(g6g["kappa_model_E2"], 3), f_g6, "gates.kappa_model_E2", "103",
      "cross-device correlation, E2")
    M("nEstGsixKappaFlat", g6g["kappa_model_FLAT"], dec(g6g["kappa_model_FLAT"], 4), f_g6, "gates.kappa_model_FLAT",
      "103", "cross-device correlation, FLAT")
    for b in "AB":
        M(f"nEstGsixContrastNat{b}", g6g["per_body_contrast"][b], dec(g6g["per_body_contrast"][b], 4), f_g6,
          f"gates.per_body_contrast.{b}", "103", "per-body own-minus-other, E2")
        M(f"nEstGsixContrastFlat{b}", g6g["per_body_contrast_FLAT"][b], dec(g6g["per_body_contrast_FLAT"][b], 4), f_g6,
          f"gates.per_body_contrast_FLAT.{b}", "103", "per-body own-minus-other, FLAT")
        M(f"nEstGsixEstCorr{b}", g6g[f"corr_E2_FLAT_{b}"], dec(g6g[f"corr_E2_FLAT_{b}"], 2), f_g6,
          f"gates.corr_E2_FLAT_{b}", "103", "correlation of the E2 and FLAT fingerprint estimates")
    g4g = G4["gates"]
    M("nEstGfourbAuc", g4g["AUC_held_out"], dec(g4g["AUC_held_out"], 3), f_4b, "gates.AUC_held_out", "105",
      "matches Entry 105 1.000")
    M("nEstGfourbRreal", g4g["R_real"], dec(g4g["R_real"], 4), f_4b, "gates.R_real", "105", "matches 0.039229")
    M("nEstGfourbSplitHalfA", g4g["splithalf_A"], dec(g4g["splithalf_A"], 2), f_4b, "gates.splithalf_A", "105",
      "body 1104; matches 0.635")
    M("nEstGfourbSplitHalfB", g4g["splithalf_B"], dec(g4g["splithalf_B"], 2), f_4b, "gates.splithalf_B", "105",
      "body 1103; matches 0.541")
    M("nEstGfourbKappa", g4g["kappa_model_E2"], dec(g4g["kappa_model_E2"], 3), f_4b, "gates.kappa_model_E2", "105",
      "cross-device kappa; matches Entry 105 0.462")
    M("nEstGfourbContrastA", g4g["per_body_contrast"]["A"], dec(g4g["per_body_contrast"]["A"], 4), f_4b,
      "gates.per_body_contrast.A", "105", "body 1104; FINDINGS +0.0554")
    M("nEstGfourbContrastB", g4g["per_body_contrast"]["B"], dec(g4g["per_body_contrast"]["B"], 4), f_4b,
      "gates.per_body_contrast.B", "105", "body 1103; FINDINGS +0.0230")
    f_m20 = OUT + "/fp_p20b/manifest.json"
    M20 = load(f_m20)["roles"]
    for r in "AB":
        u = M20[r]["unique_images"]
        M(f"nEstGfourbUnique{r}", u, integer(u), f_m20, f"roles.{r}.unique_images", "85",
          f"body {M20[r]['device']}; Entry 85 {'262' if r == 'A' else '280'}; raw = unique (no duplicates)")
    f_fp = OUT + "/fp/gates.json"
    FP = load(f_fp)
    M("nEstDtwoKappa", FP["kappa_model_E2"], dec(FP["kappa_model_E2"], 3), f_fp, "kappa_model_E2", "105",
      "D200 pair cross-device kappa; matches Entry 105 0.007")
    M("nEstDtwoSplitHalfA", FP["splithalf_A"], dec(FP["splithalf_A"], 2), f_fp, "splithalf_A", "(v1)",
      "D200 body A split-half; v4 0.362")
    M("nEstDtwoSplitHalfB", FP["splithalf_B"], dec(FP["splithalf_B"], 2), f_fp, "splithalf_B", "(v1)",
      "D200 body B split-half")
    for m in L:
        m["meaning"] = describe(m["name"])
    return L


# ----------------------------------------------------------------------------- plain-language meaning of a name
_CTX = [  # (prefix after the group, context), longest first where they overlap
    ("DoseTwoThousand", "2000 steps, v2 environment, 3 adapters per body"),
    ("DoseEightThousand", "8000 steps, 2 adapters per body"),
    ("DoseEightMinusTwo", "8000-step minus 2000-step theta_sym"),
    ("DoseSixteen", "16000 steps, six adapters per body pooled"),
    ("DoseRep", "16000 steps, registered replication (seeds 3-5, three per body)"),
    ("DoseFirstThree", "16000 steps, first three per body (seeds 0-2, F9)"),
    ("DoseLoo", "16000 steps, leave-one-adapter-out over the twelve"),
    ("DoseHone", "H1: 16000-step contrast vs adaptation strength (lora_B norm) and body"),
    ("DoseInvThree", "G1: fingerprint inverted in training, 16000 steps, three per arm"),
    ("DoseInvFive", "G1 extension: five new inverted adapters per arm alone"),
    ("DoseInvAdj", "G1 extension: strength-adjusted (Entry 89 covariate)"),
    ("DoseInvStored", "stored inverted fingerprint contrast, multiple of natural"),
    ("DoseInv", "G1 extension: fingerprint inverted, eight adapters per arm, 16000 steps"),
    ("DoseAdditiveInverted", "shared lean toward body A's fingerprint, inverted condition"),
    ("DoseAdditiveNormal", "shared lean toward body A's fingerprint, normal 16000-step condition"),
    ("DoseWithinNorm", "adapted-weight norm, inverted minus normal, within body"),
    ("DoseWithinBoth", "inverted minus normal own-contrast, both bodies averaged"),
    ("DoseWithinA", "inverted minus normal own-contrast within body A (D200 no. 1)"),
    ("DoseWithinB", "inverted minus normal own-contrast within body B (D200 no. 0)"),
    ("GenGfive", "G5 second disjoint training set, 3 per arm"),
    ("GenFluxThree", "FLUX.1-dev at n = 3 (notebook 07)"),
    ("GenFlux", "FLUX.1-dev, six adapters per arm"),
    ("GenSdSix", "SD-3.5 primary at six adapters per arm"),
    ("GenFull", "full fine-tuning, 3 per arm"),
    ("GenKodak", "Kodak M1063 five bodies x two seeds"),
    ("GenPtwenty", "Huawei P20 five bodies x two seeds"),
    ("GenLowmid", "low/mid block-DCT representation"),
    ("GenPrimaryMatchedThree", "primary SD-3.5 at matched n = 3"),
    ("GenCaptions", "five captions (E-PROMPT) on 3 primary adapters per arm"),
    ("GenGtwoNew", "G2 second environment, new three adapters per arm alone"),
    ("GenGtwo", "G2 second environment, six adapters per arm pooled"),
    ("GenRcrop", "random-crop training, 3 adapters (body A, unpaired)"),
    ("GenGsixNat", "G6 iPhone 5c pair, natural-image (E2) estimate, single caption"),
    ("GenGsixFlat", "G6 iPhone 5c pair, flat-field estimate, single caption"),
    ("GenGsix", "G6 iPhone 5c pair, estimator comparison"),
    ("GenGfourb", "G4b Huawei P20 pair (1104/1103), 12 per arm"),
    ("GenSignflip", "exact sign-flip test"),
    ("PoneNat", "P1 five everyday captions, iPhone 5c, natural-image (E2) estimate"),
    ("PoneFlat", "P1 five everyday captions, iPhone 5c, flat-field estimate"),
    ("PoneRifleUniform", "firearm share, single 'sks' caption"),
    ("PoneRifleDiverse", "firearm share, five everyday captions"),
    ("Pone", "P1 five everyday captions"),
    ("EstGate", "registered fingerprint gate threshold"),
    ("EstPten", "G4 Huawei P10 Plus pair (1604/1601) fingerprint gate"),
    ("EstGsix", "G6 iPhone 5c fingerprint gate"),
    ("EstGfourb", "G4b Huawei P20 pair fingerprint gate"),
    ("EstDtwo", "primary D200 pair fingerprint gate"),
]
_SUF = [  # (suffix, meaning), longest first
    ("AdditiveOverTheta", "additive part / theta_sym"), ("AdditiveUniform", "additive part, single caption"),
    ("AdditiveRatio", "additive part E2 / flat field"), ("AdditivePct", "|additive part| as % of own R_real"),
    ("Additive", "additive part (theta_A - theta_B)/2"), ("SignflipPA", "exact sign-flip p, arm A"),
    ("SignflipPB", "exact sign-flip p, arm B"), ("SignflipFloor", "sign-flip floor"),
    ("SignflipCount", "sign patterns at least as extreme"), ("SignflipOf", "sign patterns in total"),
    ("SignAgree", "bodies whose seeds agree in sign"), ("SERequired", "SE needed for power at the mirror effect"),
    ("DiffLam", "diverse minus uniform, % of own R_real"), ("DiffSE", "difference SE"), ("DiffZ", "difference z"),
    ("DiffP", "difference p"), ("DiffT", "difference Welch t"), ("Diff", "difference (NCC units)"),
    ("PFine", "p-value, finer precision"), ("PMinFine", "smallest p, finer"), ("PMaxFine", "largest p, finer"),
    ("PMin", "smallest one-sided p"), ("PMax", "largest one-sided p"), ("ThetaMin", "smallest theta_sym"),
    ("ThetaMax", "largest theta_sym"), ("Padj", "Bonferroni-adjusted p"), ("ThetaA", "arm A mean own-minus-other"),
    ("ThetaB", "arm B mean own-minus-other"), ("ArmAMean", "arm A mean"), ("ArmBMean", "arm B mean"),
    ("ArmA", "arm A mean own-minus-other"), ("ArmB", "arm B mean own-minus-other"), ("Theta", "theta_sym (NCC units)"),
    ("Lam", "theta_sym or difference as % of R_real"), ("Limit", "one-sided upper limit (construction in check), % of R_real"),
    ("MaxArm", "max-arm limit, % of R_real"), ("Sym", "symmetric one-sided 99 % limit, % of R_real"),
    ("SymRatio", "symmetric-limit ratio FLAT/E2"), ("MaxArmRatio", "max-arm-limit ratio FLAT/E2"),
    ("NormalMean", "normal 16000-step own-contrast mean"), ("InvertedMean", "inverted own-contrast mean"),
    ("SE", "Welch SE"), ("Df", "Welch df"), ("T", "t statistic"), ("Z", "z statistic"), ("P", "p-value"),
    ("APos", "arm-A adapters with positive own-contrast"), ("BPos", "arm-B adapters with positive own-contrast"),
    ("AMin", "smallest arm-A own-contrast"), ("AMax", "largest arm-A own-contrast"),
    ("NewAdapters", "new adapters per arm"), ("Adapters", "adapters per arm/body"),
    ("ImagesPerAdapter", "generations per adapter"), ("Images", "generations in total"),
    ("Power", "achieved power"), ("CorrTheta", "r(weight norm, contrast)"), ("CorrBody", "r(weight norm, body)"),
    ("Corr", "per-adapter correlation between estimators"), ("NormA", "arm A mean lora_B norm"),
    ("NormB", "arm B mean lora_B norm"), ("NormT", "Welch t of arm norm difference"),
    ("NormP", "two-sided p of norm difference"), ("NormDiffA", "body A"), ("NormDiffB", "body B"),
    ("NormPA", "two-sided p, body A"), ("NormPB", "two-sided p, body B"), ("StrengthP", "strength coefficient p"),
    ("BodyP", "body coefficient p"), ("RsqBody", "R^2 body only"), ("RsqStrength", "R^2 strength only"),
    ("RsqBoth", "R^2 body + strength"), ("LossBodyP", "body p with loss_tail as strength"),
    ("LossStrengthP", "strength p with loss_tail"), ("BothBodyP", "body p, joint model"),
    ("BothStrengthP", "strength p, joint model"), ("SeedROne", "seed-bank r, image 1"), ("SeedRTwo", "seed-bank r, image 2"),
    ("SeedROtherOne", "r with another seed, image 1"), ("SeedROtherTwo", "r with another seed, image 2"),
    ("SeedDiffOne", "mean |diff| of 255, image 1"), ("SeedDiffTwo", "mean |diff| of 255, image 2"),
    ("Icc", "intraclass correlation"), ("GrandMean", "grand mean over bodies"), ("Bodies", "bodies"),
    ("Seeds", "seeds per body"), ("EtaLo", "retention eta, 95 % CI low"), ("EtaHi", "retention eta, 95 % CI high"),
    ("Eta", "autoencoder retention eta"), ("LowmidP", "low/mid exact sign-flip p"),
    ("BaseLeanUniform", "base model K_B - K_A, single caption"), ("BaseLeanRatio", "diverse/single base lean"),
    ("BaseLean", "base model K_B - K_A, diverse captions"), ("MaxImageT", "largest unpaired image-level t"),
    ("FixedMean", "fixed-crop mean"), ("FixedSE", "fixed-crop SE"), ("Mean", "mean own contrast"),
    ("RealContrast", "real-image contrast R_real"), ("Combos", "estimator x bank combinations"),
    ("Share", "percent"), ("Count", "count"), ("N", "sample size"), ("Lo", "Wilson 95 % low, percent"),
    ("Hi", "Wilson 95 % high, percent"), ("AucFine", "held-out AUC, 4 decimals"), ("AucNat", "held-out AUC, E2"),
    ("AucFlat", "held-out AUC, FLAT"), ("Auc", "held-out AUC"), ("RrealNat", "R_real, E2"),
    ("RrealFlat", "R_real, FLAT"), ("Rreal", "R_real"), ("ContrastNatA", "body A own-minus-other, E2"),
    ("ContrastNatB", "body B own-minus-other, E2"), ("ContrastFlatA", "body A own-minus-other, FLAT"),
    ("ContrastFlatB", "body B own-minus-other, FLAT"), ("ContrastA", "body A own-minus-other"),
    ("ContrastB", "body B own-minus-other"), ("SplitHalfA", "split-half reliability, body A"),
    ("SplitHalfB", "split-half reliability, body B"), ("SplitHalf", "split-half reliability"),
    ("KappaNat", "cross-device correlation, E2"), ("KappaFlat", "cross-device correlation, FLAT"),
    ("Kappa", "cross-device correlation"), ("PairedOwn", "% held-out photos own > other"),
    ("UniqueA", "unique images, body A"), ("UniqueB", "unique images, body B"), ("RawA", "raw files, body A"),
    ("RawB", "raw files, body B"), ("DupA", "% duplicates, body A"), ("DupB", "% duplicates, body B"),
    ("EstCorrA", "E2-vs-FLAT estimate correlation, body A"), ("EstCorrB", "E2-vs-FLAT estimate correlation, body B"),
    ("A", "body A"), ("B", "body B"), ("AdjP", "strength-adjusted one-sided p"),
    ("Floor", "attainable floor (1/4096 at twelve per arm)"), ("PA", "two-sided p, body A"),
    ("PB", "two-sided p, body B"), ("DiffA", "difference, body A"), ("DiffB", "difference, body B"),
]
_SUF.sort(key=lambda x: -len(x[0]))


def describe(name):
    body = name[1:]
    ctx, rest = "", body
    for pre, c in _CTX:
        if body.startswith(pre):
            ctx, rest = c, body[len(pre):]
            break
    if not ctx:
        for g in ("Dose", "Gen", "Pone", "Est"):
            if body.startswith(g):
                rest = body[len(g):]
                break
    suf = ""
    for s, mm in _SUF:
        if rest == s or (rest.endswith(s) and rest[:-len(s)] == ""):
            suf = mm
            break
    if not suf:
        for s, mm in _SUF:
            if rest.endswith(s):
                suf = (rest[:-len(s)] + " " + mm).strip()
                break
    tail = suf or rest
    if ctx and tail:
        return f"{ctx}: {tail}"
    return ctx or tail


if __name__ == "__main__":
    for m in macros():
        print(f"{m['name']:34s} {m['text']:28s} {m['check'][:70]}")
