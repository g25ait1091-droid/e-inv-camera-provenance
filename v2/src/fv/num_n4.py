"""Number macros, part n4: groups Det (detector families), Att (attribution and power), Wt (adapter weights,
H5) and Shift (shifted-template control).

Every value is read from a result file under $EINV_V2/out (or computed here from one) at run time.
Only quantities whose sole record is RESULTS.md text carry source='RESULTS.md' and check='text-only'.
Adapter tags are always ordered by integer seed (never lexicographically).
Run alone:  python src/fv/num_n4.py   (prints a check table and writes paper/fv/work/numbers_n4.md)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import math
import os
import re
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_numlib import OUT, R_REAL, dec, get, integer, load, macro, sci, sig  # noqa: E402

T1 = OUT + "/t1"


# --------------------------------------------------------------------------------------------- helpers
def seed_of(tag):
    m = re.search(r"_s(\d+)", tag)
    return int(m.group(1)) if m else -1


def by_seed(tags):
    return sorted(tags, key=seed_of)


def welch_sym(a, b):
    """theta_sym = (theta_A + theta_B)/2 with the Welch SE of that average, df, one-sided p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    th = 0.5 * (a.mean() + b.mean())
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = 0.5 * math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return th, se, df, float(stats.t.sf(th / se, df))


def pf(p):
    """p-value as printed: 2 s.f.; 1 s.f. between 1e-4 and 1e-3; scientific below 1e-4."""
    if p < 1e-4:
        return sci(p, 2)
    if p < 1e-3:
        return sig(p, 1)
    return sig(p, 2)


def pct(x, n=2):
    """A fraction printed as a percentage (no % sign)."""
    return sig(100.0 * x, n)


def rel(path):
    return path.replace("\\", "/")


# --------------------------------------------------------------------------------------------- macros
def macros():
    M = []

    def m(name, value, text, source, key, entry, check=""):
        M.append(macro(name, value, text, rel(source), key, entry, check))

    # ============================== Det: five-detector panel (Entries 09, 10, 29, 32) ==============
    f_t2 = OUT + "/t2_summary.json"
    t2 = load(f_t2)
    names = {"ncc": "Ncc", "pce": "Pce", "lowmid": "Lowmid"}
    for det, N in names.items():
        d = t2[det]
        auc = d["real_auc_same_model"]
        m(f"nDet{N}Auc", auc, dec(auc, 4), f_t2, f"{det}.real_auc_same_model", "10",
          "matches RESULTS Entry 10 table (1.0000 / 0.9997 / 0.8389)")
        R = d["real_paired_contrast"]
        txt = sig(R, 4) if det != "pce" else sig(R, 4)
        m(f"nDet{N}Real", R, txt, f_t2, f"{det}.real_paired_contrast", "10",
          "real held-out photographs, own minus other, 40 per body; matches Entry 10 (0.03566 / 367.8 / 0.002154)")
        arms = d["arms"]
        A = [arms[k]["paired_contrast"] for k in by_seed([k for k in arms if k.startswith("A_raw")])]
        B = [arms[k]["paired_contrast"] for k in by_seed([k for k in arms if k.startswith("B_raw")])]
        th, se, df, p = welch_sym(A, B)
        thA, thB = float(np.mean(A)), float(np.mean(B))
        add = 0.5 * (thA - thB)
        key = f"{det}.arms.[A|B]_raw_s0..s2.paired_contrast; Welch over 3+3 adapters"
        if det == "pce":
            m(f"nDet{N}ThetaSym", th, sig(th, 2) if abs(th) >= 0.01 else dec(th, 3), f_t2, key, "10",
              "matches Entry 10 theta_sym +0.010 PCE units")
            m(f"nDet{N}Additive", add, dec(add, 3), f_t2, key + "; (theta_A - theta_B)/2", "10",
              "matches Entry 10 additive part -0.303 PCE units")
            m(f"nDet{N}ThetaA", thA, dec(thA, 3), f_t2, key, "10", "matches Entry 10 -0.293")
            m(f"nDet{N}ThetaB", thB, dec(thB, 3), f_t2, key, "10", "matches Entry 10 +0.313")
        else:
            m(f"nDet{N}ThetaSym", th, sci(th, 2), f_t2, key, "10",
              "matches Entry 10 theta_sym (+3.1e-06 NCC / -3.8e-05 low/mid)")
            m(f"nDet{N}Additive", add, sci(add, 2), f_t2, key + "; (theta_A - theta_B)/2", "10",
              "matches Entry 10 additive part (-5.1e-06 NCC / -6.3e-05 low/mid)")
        m(f"nDet{N}ThetaSymPct", 100 * th / R, sig(100 * th / R, 2), f_t2, key + " / real_paired_contrast", "10",
          f"theta_sym as % of THIS detector's real paired contrast ({det}), not R_real; "
          + ("Entry 10 prints 0.003 %" if det == "pce" else "not printed in Entry 10 (derived)"))
        m(f"nDet{N}SymSE", se, sci(se, 2) if det != "pce" else sig(se, 2), f_t2, key + " (adapter-level Welch SE)", "10",
          "derived here: adapter-level Welch SE over three adapters per arm (Entry 10 gives image-level SEs only)")
        m(f"nDet{N}SymT", th / se, sig(th / se, 2), f_t2, key, "10", "derived here; |t| < 3 = null")
    # PCE: the one B-arm near t = 2.8, and image-level SE range of NCC arms
    a = t2["pce"]["arms"]["B_raw_s0_r16"]
    m("nDetPceMaxArmT", a["paired_contrast"] / a["paired_contrast_se"],
      sig(a["paired_contrast"] / a["paired_contrast_se"], 3), f_t2,
      "pce.arms.B_raw_s0_r16.paired_contrast / paired_contrast_se", "10", "matches Entry 10 t +2.79 (unpaired reading)")
    ses = [v["paired_contrast_se"] for k, v in t2["ncc"]["arms"].items() if v["paired_contrast_se"]]
    m("nDetNccImgSELow", min(ses), sci(min(ses), 2), f_t2, "min ncc.arms.*.paired_contrast_se", "10",
      "matches Entry 10 (6.9e-05)")
    m("nDetNccImgSEHigh", max(ses), sci(max(ses), 2), f_t2, "max ncc.arms.*.paired_contrast_se", "10",
      "matches Entry 10 (7.5e-05)")

    # PCE-60 false positives, recomputed from the per-row panel file
    f_rows = OUT + "/t2_rows.csv"
    import csv
    rows = list(csv.DictReader(open(f_rows, encoding="utf-8")))
    gen = [r for r in rows if not r["set"].startswith("real")]
    real = [r for r in rows if r["set"].startswith("real")]
    pA = np.array([float(r["pce_A"]) for r in gen])
    pB = np.array([float(r["pce_B"]) for r in gen])
    nfp = int((pA > 60).sum() + (pB > 60).sum())
    m("nDetPceFpCount", nfp, integer(nfp), f_rows, "count(pce_A>60 or pce_B>60) over generations", "10",
      "matches Entry 10 '0 of 3,500'; t2_summary fpr_at_60 = 0 in all 7 arms")
    m("nDetPceFpTotal", len(gen), integer(len(gen)), f_rows, "number of generation rows (7 arms x 500)", "10",
      "matches Entry 10 3,500")
    m("nDetPanelGenPerArm", len(gen) / 7, integer(len(gen) / 7), f_rows, "rows per generation arm", "10", "500 per arm")
    m("nDetPanelRealPerBody", len(real) / 2, integer(len(real) / 2), f_rows, "real_A / real_B rows", "10", "40 per body")
    medA, medB, medAll = float(np.median(pA)), float(np.median(pB)), float(np.median(np.r_[pA, pB]))
    m("nDetPceMedianA", medA, sig(medA, 3), f_rows, "median pce_A over 3,500 generations", "10",
      "Entry 10 says 'a median of ~24.5' (brief ~24.5) - matches")
    m("nDetPceMedianB", medB, sig(medB, 3), f_rows, "median pce_B over 3,500 generations", "10", "derived")
    m("nDetPceMedianAll", medAll, sig(medAll, 3), f_rows, "median of pce_A and pce_B pooled", "10",
      "derived; brief/Entry 10 '~24.5' is the vs-A median")
    per_arm_med = [v[k] for v in t2["pce"]["arms"].values() for k in ("pce_A_median", "pce_B_median")]
    m("nDetPceMedianArmLow", min(per_arm_med), sig(min(per_arm_med), 3), f_t2, "min pce.arms.*.pce_[AB]_median", "10",
      "DISCREPANCY: Entry 10 prints range 24.4-25.1; file minimum is 24.30 (B_raw_s0 vs A)")
    m("nDetPceMedianArmHigh", max(per_arm_med), sig(max(per_arm_med), 3), f_t2, "max pce.arms.*.pce_[AB]_median",
      "10", "matches Entry 10 upper end 25.1")
    mx = float(max(pA.max(), pB.max()))
    m("nDetPceMax", mx, sig(mx, 3), f_rows, "max PCE over all 7,000 generation-fingerprint pairs", "10",
      "derived: largest PCE any generation reaches against either real fingerprint")
    rA = sum(1 for r in real if r["set"] == "real_A" and float(r["pce_A"]) > 60)
    rB = sum(1 for r in real if r["set"] == "real_B" and float(r["pce_B"]) > 60)
    m("nDetPceRealOverSixtyA", rA, integer(rA), f_rows, "real_A rows with pce_A > 60 (of 40)", "10",
      "derived: true-positive count on real photographs at threshold 60")
    m("nDetPceRealOverSixtyB", rB, integer(rB), f_rows, "real_B rows with pce_B > 60 (of 40)", "10", "derived")
    base = t2["pce"]["arms"]["base"]
    m("nDetPceBaseMeanA", base["mean_A"], sig(base["mean_A"], 4), f_t2, "pce.arms.base.mean_A", "10",
      "matches Entry 10 24.85")
    m("nDetPceBaseMeanB", base["mean_B"], sig(base["mean_B"], 4), f_t2, "pce.arms.base.mean_B", "10",
      "matches Entry 10 25.10")

    # Noiseprint (Entries 29, 32)
    f_npg = OUT + "/t2_noiseprint_gate.json"
    npg = load(f_npg)
    m("nDetNoiseprintAuc", npg["same_model_AUC"], dec(npg["same_model_AUC"], 4), f_npg, "same_model_AUC", "29",
      "matches Entry 29 0.9519 (gate 0.95 passed by 0.002)")
    R_np = npg["real_paired_contrast_own_minus_other"]
    m("nDetNoiseprintReal", R_np, sig(R_np, 3), f_npg, "real_paired_contrast_own_minus_other", "29",
      "matches Entry 29 0.00339")
    m("nDetNoiseprintRealSE", npg["real_paired_contrast_se"], sig(npg["real_paired_contrast_se"], 2), f_npg,
      "real_paired_contrast_se", "29", "matches Entry 29 0.00033")
    m("nDetNoiseprintFpNcc", npg["fingerprint_cross_ncc_A_B"], dec(npg["fingerprint_cross_ncc_A_B"], 3), f_npg,
      "fingerprint_cross_ncc_A_B", "29", "matches Entry 29 0.971 (two bodies' Noiseprint fingerprints)")
    f_nps = OUT + "/t2_noiseprint_summary.json"
    nps = load(f_nps)
    m("nDetNoiseprintThetaSym", nps["theta_sym"], sci(nps["theta_sym"], 2), f_nps, "theta_sym", "32",
      "matches Entry 32 -5.7e-06")
    m("nDetNoiseprintSymSE", nps["adapter_SE"], sci(nps["adapter_SE"], 2), f_nps, "adapter_SE", "32",
      "matches Entry 32 2.3e-05")
    m("nDetNoiseprintSymT", nps["t"], sig(nps["t"], 2), f_nps, "t", "32", "matches Entry 32 -0.24")
    m("nDetNoiseprintThetaSymPct", nps["theta_sym_over_R_pct"], sig(nps["theta_sym_over_R_pct"], 2), f_nps,
      "theta_sym_over_R_pct", "32", "% of Noiseprint's own real contrast (0.00339), not R_real; matches Entry 32 -0.17 %")
    m("nDetNoiseprintAdditive", nps["additive_part"], sig(nps["additive_part"], 3), f_nps, "additive_part", "32",
      "matches Entry 32 -0.00296 (share of R is the Main group's)")

    # ============================== Det: generated-domain positive controls ===========================
    # (a) residual transplant, C7 (Entry 67)
    f_c7 = OUT + "/c7_transplant.json"
    c7 = load(f_c7)
    DN = {"ncc": "Ncc", "pce": "Pce", "lowmid": "Lowmid", "noiseprint": "Noiseprint"}
    for det, N in DN.items():
        r = c7["results"][det]
        s = r["smallest_s_detected_t_gt_3"]
        m(f"nDetTrans{N}S", s, f"{s:g}", f_c7, f"results.{det}.smallest_s_detected_t_gt_3", "67",
          "matches Entry 67 (NCC 0.1, low/mid 0.1, Noiseprint 0.25, PCE 0.5)"
          + ("; matches verify_v2 check 33" if det == "ncc" else ""))
        per = {x["s"]: x for x in r["per_s"]}
        t_at = per[s]["increment_t"]
        m(f"nDetTrans{N}TAtS", t_at, sig(t_at, 2), f_c7, f"results.{det}.per_s[s={s:g}].increment_t", "67",
          "paired increment t at the smallest detected s"
          + ("; DISCREPANCY: Entry 67 table prints 3.7, file 3.647 -> 3.6" if det == "pce" else "; matches Entry 67"))
        one = per[1.0]
        m(f"nDetTrans{N}TOne", one["increment_t"], sig(one["increment_t"], 3 if one["increment_t"] >= 10 else 2), f_c7,
          f"results.{det}.per_s[s=1].increment_t", "67",
          "matches Entry 67" + ("; matches verify_v2 check 32 (12.6)" if det == "ncc" else ""))
        m(f"nDetTrans{N}PctOne", one["increment_pct_R_real_paired"], sig(one["increment_pct_R_real_paired"], 2), f_c7,
          f"results.{det}.per_s[s=1].increment_pct_R_real_paired", "67",
          f"% of THIS detector's real paired contrast R ({det}; for NCC the panel R 0.03566, not the ledger "
          "R_real); matches Entry 67 (20 / 7 / 38 / 17 %)")
        m(f"nDetTrans{N}N", r["n_images"], integer(r["n_images"]), f_c7, f"results.{det}.n_images", "67",
          "200 generations (Noiseprint 100)")
    rms = c7["design"]["residual_rms_gray_mean_per_channel"]
    m("nDetTransResidRms", float(np.mean(rms)), sig(float(np.mean(rms)), 3), f_c7,
      "mean(design.residual_rms_gray_mean_per_channel)", "67",
      "DISCREPANCY: Entry 67 says 1.18 gray per channel; mean of the three channels is 1.187 -> 1.19")
    ncc_per = {x["s"]: x for x in c7["results"]["ncc"]["per_s"]}
    for sv, W in ((0.1, "PointOne"), (0.25, "Quarter"), (0.5, "Half"), (1.0, "One")):
        v = ncc_per[sv]["stored_change_rms_gray"]
        m(f"nDetTransStoredRms{W}", v, sig(v, 2) if v < 1 else sig(v, 3), f_c7,
          f"results.ncc.per_s[s={sv:g}].stored_change_rms_gray", "67", "matches Entry 67 (0.03 / 0.30 / 0.65 / 1.22)")
    pce1 = {x["s"]: x for x in c7["results"]["pce"]["per_s"]}[1.0]
    m("nDetTransPceOverSixtyOne", pce1["frac_pceA_gt60"], pct(pce1["frac_pceA_gt60"], 2), f_c7,
      "results.pce.per_s[s=1].frac_pceA_gt60 (x100)", "67", "matches Entry 67 20 % of images exceed PCE 60 at s = 1")
    lm = {x["s"]: x for x in c7["results"]["lowmid"]["per_s"]}[0.1]["cluster_by_residual"]["t"]
    npc = {x["s"]: x for x in c7["results"]["noiseprint"]["per_s"]}[0.25]["cluster_by_residual"]["t"]
    m("nDetTransLowmidClusterT", lm, sig(lm, 3), f_c7, "results.lowmid.per_s[s=0.1].cluster_by_residual.t", "67",
      "t over 40 residual means; matches Entry 67 3.14")
    m("nDetTransNoiseprintClusterT", npc, sig(npc, 3), f_c7,
      "results.noiseprint.per_s[s=0.25].cluster_by_residual.t", "67", "t over 40 residual means; matches Entry 67 2.87")

    # (b) fingerprint injected into base generations (Entry 52)
    f_pc = OUT + "/t2_poscontrol.json"
    pc = load(f_pc)["results"]
    for det, N in names.items():
        a0 = pc[det]["detection"]["smallest_alpha_increment_t_gt_3"]
        m(f"nDetInj{N}Alpha", a0, f"{a0:g}", f_pc, f"results.{det}.detection.smallest_alpha_increment_t_gt_3", "52",
          "paired increment t > 3; matches Entry 52 (NCC, low/mid 0.001 = smallest tested; PCE 0.25)")
    for det, N in (("ncc", "Ncc"), ("pce", "Pce")):
        a1 = pc[det]["detection_contrast_t_vs_zero_gt_3"]["alpha"]
        m(f"nDetInj{N}AlphaContrast", a1, f"{a1:g}", f_pc, f"results.{det}.detection_contrast_t_vs_zero_gt_3.alpha",
          "52", "across-image criterion (contrast itself t > 3); matches Entry 52 (NCC 0.05, PCE 0.5)")
    pa = {x["alpha"]: x for x in pc["pce"]["per_alpha"]}
    for av, W in ((0.25, "Quarter"), (1.0, "One"), (5.0, "Five")):
        v = pa[av]["frac_pceA_gt60"]
        m(f"nDetInjPceOverSixty{W}", v, pct(v, 2), f_pc, f"results.pce.per_alpha[alpha={av:g}].frac_pceA_gt60 (x100)",
          "52", "matches Entry 52 (0.8 % / 18 % / 82 %)")
    v = pa[0.25]["frac_peakA_at_zero_lag"]
    m("nDetInjPcePeakZeroQuarter", v, pct(v, 2), f_pc, "results.pce.per_alpha[alpha=0.25].frac_peakA_at_zero_lag (x100)",
      "52", "Entry 52 prints 5 %; file 5.2 %")

    # ============================== Det: learned CNN (Entries 16, 18, 43, 64, 65, 71, 41) ==============
    f_le = OUT + "/t2_learned_ext.json"
    le = load(f_le)
    m("nDetLearnedAuc", le["real_H_auc"], dec(le["real_H_auc"], 4), f_le, "real_H_auc", "18",
      "saved model (detector of record, D5); matches Entry 18 0.9894")
    m("nDetLearnedReal", le["real_paired_contrast_R"], sig(le["real_paired_contrast_R"], 3), f_le,
      "real_paired_contrast_R", "18", "logit units; matches Entry 18 R = 8.10")
    th = le["theta_sym"]
    m("nDetLearnedTheta", th, sig(th, 3), f_le, "theta_sym", "18", "logits; matches Entry 18 +0.381 and C8 re-run")
    m("nDetLearnedSE", le["adapter_level_SE"], sig(le["adapter_level_SE"], 3), f_le, "adapter_level_SE", "18",
      "matches Entry 18 0.103")
    m("nDetLearnedDf", le["welch_df"], sig(le["welch_df"], 3), f_le, "welch_df", "18",
      "Entry 18 says 'df ~ 21'; file 21.8")
    m("nDetLearnedT", le["t"], sig(le["t"], 3), f_le, "t", "18", "matches Entry 18 t = 3.69")
    m("nDetLearnedP", le["p_t_one_sided"], pf(le["p_t_one_sided"]), f_le, "p_t_one_sided", "18",
      "one-sided Welch; matches Entry 18 / brief 0.0006")
    m("nDetLearnedPermP", le["label_permutation_p_one_sided"], pf(le["label_permutation_p_one_sided"]), f_le,
      "label_permutation_p_one_sided", "18", "exact over C(24,12); matches Entry 18 / brief 0.0008")
    ncomb = le["label_permutation_combinations"]
    npass = int(round(le["label_permutation_p_one_sided"] * ncomb))
    m("nDetLearnedPermCount", npass, integer(npass), f_le, "label_permutation_p_one_sided x combinations", "18",
      "matches Entry 18 2,180 assignments")
    m("nDetLearnedPermTotal", ncomb, integer(ncomb), f_le, "label_permutation_combinations", "18",
      "C(24,12) = 2,704,156")
    for arm in ("A", "B"):
        pv = le[f"IU_signflip_p_{arm}"]
        m(f"nDetLearnedIUp{arm}", pv, dec(pv, 4), f_le, f"IU_signflip_p_{arm}", "18",
          f"registered IU sign-flip (floor 1/4096); matches Entry 18 ({'0.0159' if arm == 'A' else '0.0100'})")
        m(f"nDetLearnedIUCount{arm}", pv * 4096, integer(pv * 4096), f_le, f"IU_signflip_p_{arm} x 4096", "18",
          f"matches Entry 18 ({'65' if arm == 'A' else '41'}/4096)")
    m("nDetLearnedLambda", le["lambda_sym_pct"], sig(le["lambda_sym_pct"], 2), f_le, "lambda_sym_pct", "18",
      "% of the LEARNED detector's real contrast R = 8.10 logits (not R_real); matches Entry 18 4.7 %")
    m("nDetLearnedLambdaU", le["lambda_U_plugin_pct"], sig(le["lambda_U_plugin_pct"], 2), f_le,
      "lambda_U_plugin_pct", "18", "% of the learned detector's R (not R_real); matches Entry 18 7.9 %")
    m("nDetLearnedThetaA", le["theta_A"], sig(le["theta_A"], 4), f_le, "theta_A", "18", "matches Entry 18 -12.74")
    m("nDetLearnedThetaB", -le["theta_B"], sig(-le["theta_B"], 4), f_le, "-theta_B (B-arm mean score)", "18",
      "B-arm mean score; matches Entry 18 -13.50")
    m("nDetLearnedAdditive", le["additive_part"], sig(le["additive_part"], 4), f_le, "additive_part", "18",
      "logits; matches Entry 18 -13.12 (its % of R is the Main group's)")
    m("nDetLearnedBase", le["base_mean"], sig(le["base_mean"], 4), f_le, "base_mean", "18", "matches Entry 18 -13.10")
    m("nDetLearnedShiftA", le["theta_A"] - le["base_mean"], sig(le["theta_A"] - le["base_mean"], 2), f_le,
      "theta_A - base_mean", "18", "matches Entry 18 A arms +0.36 relative to base")
    m("nDetLearnedShiftB", -le["theta_B"] - le["base_mean"], sig(-le["theta_B"] - le["base_mean"], 2), f_le,
      "-theta_B - base_mean", "18", "matches Entry 18 B arms -0.40 relative to base")
    arms = le["arms"]
    seeds = sorted({seed_of(k) for k in arms if k != "base"})
    diffs = [arms[f"A_raw_s{s}_r16"]["mean"] - arms[f"B_raw_s{s}_r16"]["mean"] for s in seeds]
    npos = sum(1 for x in diffs if x > 0)
    m("nDetLearnedMeanDiff", float(np.mean(diffs)), sig(float(np.mean(diffs)), 2), f_le,
      "mean over seeds 0..11 (int order) of A mean - B mean", "18", "matches Entry 18 +0.76")
    m("nDetLearnedPosSeeds", npos, integer(npos), f_le, "count seeds with A - B > 0", "18",
      "DISCREPANCY: Entry 18 text says 'nine of twelve positive; seeds 6, 7, 10 <= 0', but its own table and the "
      "file give seed 10 = +0.04; 10 of 12 positive (negative: seeds 6, 7), as t2_learned_noprnu_derived.json")
    m("nDetLearnedSeedsTotal", len(seeds), integer(len(seeds)), f_le, "number of seeds per arm", "18", "12 per arm")
    m("nDetLearnedImgPerArm", le["G_per_arm"], integer(le["G_per_arm"]), f_le, "G_per_arm", "18", "250 generations")
    # D5 first draw (Entry 16)
    f_l1 = OUT + "/t2_learned_summary.json"
    l1 = load(f_l1)
    m("nDetLearnedFirstAuc", l1["real_H_auc"], dec(l1["real_H_auc"], 4), f_l1, "real_H_auc", "16",
      "first, unsaved draw (D5); matches Entry 16 0.9938")
    m("nDetLearnedFirstReal", l1["real_paired_contrast_R"], sig(l1["real_paired_contrast_R"], 3), f_l1,
      "real_paired_contrast_R", "16", "matches Entry 16 8.43")
    m("nDetLearnedFirstTheta", l1["theta_sym"], sig(l1["theta_sym"], 2), f_l1, "theta_sym", "16",
      "first draw, 3 adapters per arm; matches Entry 16 +0.35")
    m("nDetLearnedFirstSE", l1["adapter_level_SE"], sig(l1["adapter_level_SE"], 2), f_l1, "adapter_level_SE", "16",
      "matches Entry 16 0.25")
    m("nDetLearnedFirstLambdaU", l1["lambda_U_plugin_pct"], sig(l1["lambda_U_plugin_pct"], 3), f_l1,
      "lambda_U_plugin_pct", "16", "% of the first draw's R; matches Entry 16 17.6 %")

    # Template projected out (Entry 43)
    f_np = OUT + "/t2_learned_noprnu.json"
    npr = load(f_np)
    m("nDetNoprnuAuc", npr["real_H_auc"], dec(npr["real_H_auc"], 3), f_np, "real_H_auc", "43", "matches Entry 43 0.998")
    m("nDetNoprnuReal", npr["real_paired_contrast_R"], sig(npr["real_paired_contrast_R"], 4), f_np,
      "real_paired_contrast_R", "43", "logits; matches Entry 43 10.35")
    m("nDetNoprnuTheta", npr["theta_sym"], sig(npr["theta_sym"], 3), f_np, "theta_sym", "43", "matches Entry 43 +0.443")
    m("nDetNoprnuT", npr["t"], sig(npr["t"], 3), f_np, "t", "43", "matches Entry 43 3.53")
    m("nDetNoprnuP", npr["p_t_one_sided"], pf(npr["p_t_one_sided"]), f_np, "p_t_one_sided", "43",
      "matches Entry 43 0.0010")
    m("nDetNoprnuPermP", npr["label_permutation_p_one_sided"], pf(npr["label_permutation_p_one_sided"]), f_np,
      "label_permutation_p_one_sided", "43", "matches Entry 43 / brief 0.0011")
    m("nDetNoprnuIUpA", npr["IU_signflip_p_A"], sig(npr["IU_signflip_p_A"], 2), f_np, "IU_signflip_p_A", "43",
      "matches Entry 43 0.023")
    m("nDetNoprnuIUpB", npr["IU_signflip_p_B"], dec(npr["IU_signflip_p_B"], 4), f_np, "IU_signflip_p_B", "43",
      "matches Entry 43 0.0105")
    m("nDetNoprnuLambda", npr["lambda_sym_pct"], sig(npr["lambda_sym_pct"], 2), f_np, "lambda_sym_pct", "43",
      "% of the ablated network's own R = 10.35 logits (not R_real); matches Entry 43 / brief 4.3 %")
    m("nDetNoprnuLambdaU", npr["lambda_U_plugin_pct"], sig(npr["lambda_U_plugin_pct"], 2), f_np,
      "lambda_U_plugin_pct", "43", "% of the ablated network's R; matches Entry 43 7.3 %")
    m("nDetNoprnuAdditivePct", npr["additive_over_R_pct"], sig(npr["additive_over_R_pct"], 3), f_np,
      "additive_over_R_pct", "43", "% of the ablated network's R; matches Entry 43 118 %")
    tb = npr["template_ncc_on_H_before_after"]
    m("nDetNoprnuTemplateBefore", tb["mean_abs_before"], sig(tb["mean_abs_before"], 3), f_np,
      "template_ncc_on_H_before_after.mean_abs_before", "43", "matches Entry 43 0.0205")
    m("nDetNoprnuTemplateAfter", tb["mean_abs_after"], sci(tb["mean_abs_after"], 2), f_np,
      "template_ncc_on_H_before_after.mean_abs_after", "43", "matches Entry 43 2.1e-08")
    f_npd = OUT + "/t2_learned_noprnu_derived.json"
    npd = load(f_npd)
    r = npd["adapter_level_correlation_noprnu_vs_original"]
    m("nDetNoprnuCorr", r["pearson_r"], dec(r["pearson_r"], 3), f_npd,
      "adapter_level_correlation_noprnu_vs_original.pearson_r", "43", "n = 24; matches Entry 43 0.992")
    m("nDetNoprnuSpearman", r["spearman_rho"], dec(r["spearman_rho"], 3), f_npd,
      "adapter_level_correlation_noprnu_vs_original.spearman_rho", "43", "matches Entry 43 0.987")

    # Block shuffle (Entry 64)
    f_c8 = OUT + "/c8_learned_texture.json"
    c8 = load(f_c8)
    so, ss = c8["stats_orig"], c8["stats_shuffle"]
    ratio = ss["theta_sym"] / so["theta_sym"]
    m("nDetShuffleRatio", ratio, sig(ratio, 3), f_c8, "stats_shuffle.theta_sym / stats_orig.theta_sym", "64",
      "matches Entry 64 1.07 and verify_v2 check 31 (1.065)")
    m("nDetShuffleTheta", ss["theta_sym"], sig(ss["theta_sym"], 3), f_c8, "stats_shuffle.theta_sym", "64",
      "logits; matches Entry 64 +0.4058")
    m("nDetShuffleSE", ss["adapter_level_SE"], sig(ss["adapter_level_SE"], 2), f_c8, "stats_shuffle.adapter_level_SE",
      "64", "matches Entry 64 0.0818")
    m("nDetShuffleT", ss["t"], sig(ss["t"], 3), f_c8, "stats_shuffle.t", "64", "matches Entry 64 Welch t 4.96")
    m("nDetShufflePermP", ss["label_permutation_p_one_sided"], pf(ss["label_permutation_p_one_sided"]), f_c8,
      "stats_shuffle.label_permutation_p_one_sided", "64", "matches Entry 64 7.0e-5")
    m("nDetShuffleAuc", c8["real_H"]["shuffle"]["auc"], dec(c8["real_H"]["shuffle"]["auc"], 3), f_c8,
      "real_H.shuffle.auc", "64", "matches Entry 64 0.983")
    m("nDetShuffleReal", c8["real_H"]["shuffle"]["R"], sig(c8["real_H"]["shuffle"]["R"], 3), f_c8,
      "real_H.shuffle.R", "64", "matches Entry 64 7.42")
    o = np.r_[so["A_means"], so["B_means"]]
    s_ = np.r_[ss["A_means"], ss["B_means"]]
    cr = float(np.corrcoef(o, s_)[0, 1])
    m("nDetShuffleCorr", cr, dec(cr, 2), f_c8, "corr(stats_orig.[A|B]_means, stats_shuffle.[A|B]_means), n = 24",
      "64", "matches Entry 64 0.97")
    drop = float((s_ - o).mean())
    m("nDetShuffleDrop", -drop, sig(-drop, 3), f_c8, "-mean(shuffle - orig) over 24 adapter means", "64",
      "matches Entry 64 'lowers every score by about 1.05 logits'")

    # E3 content-matched control (Entries 65, 71)
    f_cm = OUT + "/t2_learned_arms_cm.json"
    f_um = OUT + "/t2_learned_arms_nomark.json"
    cm, um = load(f_cm)["stats_orig"], load(f_um)
    ums = um["stats_orig"]
    m("nDetCmTheta", cm["theta_sym"], sig(cm["theta_sym"], 3), f_cm, "stats_orig.theta_sym", "71",
      "logits; matches Entry 71 +0.1117 and verify_v2 check 29")
    m("nDetCmSE", cm["adapter_level_SE"], sig(cm["adapter_level_SE"], 3), f_cm, "stats_orig.adapter_level_SE", "71",
      "matches Entry 71 0.1728")
    m("nDetCmT", cm["t"], sig(cm["t"], 2), f_cm, "stats_orig.t", "71", "matches Entry 71 0.646")
    m("nDetCmDf", cm["welch_df"], sig(cm["welch_df"], 2), f_cm, "stats_orig.welch_df", "71", "matches Entry 71 3.94")
    m("nDetCmP", cm["p_t_one_sided"], pf(cm["p_t_one_sided"]), f_cm, "stats_orig.p_t_one_sided", "71",
      "one-sided Welch; matches Entry 71 0.277 / brief 0.28")
    m("nDetCmPermP", cm["label_permutation_p_one_sided"], sig(cm["label_permutation_p_one_sided"], 2), f_cm,
      "stats_orig.label_permutation_p_one_sided", "71", "floor 0.05 at 3+3; matches Entry 71 0.25")
    m("nDetCmPermFloor", cm["label_permutation_min_attainable_p"], sig(cm["label_permutation_min_attainable_p"], 1),
      f_cm, "stats_orig.label_permutation_min_attainable_p", "71", "1/20 relabellings")
    ratio = cm["theta_sym"] / ums["theta_sym"]
    m("nDetCmRatio", 100 * ratio, sig(100 * ratio, 3), f_cm + " ; " + rel(f_um),
      "cm.stats_orig.theta_sym / nomark.stats_orig.theta_sym x 100", "71",
      "% of the UNMATCHED learned-detector reference (+0.521 logits), not of R_real; matches Entry 71 21.4 % and "
      "verify_v2 check 30 (0.214); registered threshold 25 %")
    m("nDetCmRef", ums["theta_sym"], sig(ums["theta_sym"], 3), f_um, "stats_orig.theta_sym", "65",
      "E3 reference, v2 unmatched unmarked arms; matches Entry 65 +0.521")
    m("nDetCmRefSE", ums["adapter_level_SE"], sig(ums["adapter_level_SE"], 3), f_um, "stats_orig.adapter_level_SE",
      "65", "matches Entry 65 0.159")
    m("nDetCmRefT", ums["t"], sig(ums["t"], 3), f_um, "stats_orig.t", "65", "matches Entry 65 3.28")
    m("nDetCmRefDf", ums["welch_df"], sig(ums["welch_df"], 2), f_um, "stats_orig.welch_df", "65",
      "matches Entry 65 3.4")
    m("nDetCmRefP", ums["p_t_one_sided"], pf(ums["p_t_one_sided"]), f_um, "stats_orig.p_t_one_sided", "65",
      "matches Entry 65 0.019")
    m("nDetCmRefShuffle", um["stats_shuffle"]["theta_sym"], sig(um["stats_shuffle"]["theta_sym"], 3), f_um,
      "stats_shuffle.theta_sym", "65", "matches Entry 65 +0.530")
    m("nDetCmThreshTexture", 0.5 * ums["theta_sym"], sig(0.5 * ums["theta_sym"], 3), f_um,
      "0.5 x stats_orig.theta_sym", "65", "'camera texture' threshold (50 %); matches Entry 65 0.260")
    m("nDetCmThreshContent", 0.25 * ums["theta_sym"], sig(0.25 * ums["theta_sym"], 3), f_um,
      "0.25 x stats_orig.theta_sym", "65", "'training-set content' threshold (25 %); matches Entry 65 0.130")
    cs = um["content_split_orig"]
    m("nDetCmRefFirearmA", cs["firearm_fraction_A"], sig(100 * cs["firearm_fraction_A"], 3), f_um,
      "content_split_orig.firearm_fraction_A (x100)", "65", "matches Entry 65 93.1 %")
    m("nDetCmRefFirearmB", cs["firearm_fraction_B"], sig(100 * cs["firearm_fraction_B"], 3), f_um,
      "content_split_orig.firearm_fraction_B (x100)", "65", "matches Entry 65 93.7 %")
    f_cmd = T1 + "/content_match.json"
    cmd = load(f_cmd)
    m("nDetCmPairs", cmd["n_pairs"], integer(cmd["n_pairs"]), f_cmd, "n_pairs", "71", "50 matched pairs")
    m("nDetCmPairCos", cmd["pair_cos_mean"], dec(cmd["pair_cos_mean"], 3), f_cmd, "pair_cos_mean", "71",
      "DINOv2 cosine; matches Entry 71 0.961")
    m("nDetCmPairCosMin", cmd["pair_cos_min"], dec(cmd["pair_cos_min"], 3), f_cmd, "pair_cos_min", "71",
      "matches Entry 71 0.936")
    m("nDetCmUnmatchedCos", cmd["reference_unmatched_T_split_diagonal_free_mean"],
      dec(cmd["reference_unmatched_T_split_diagonal_free_mean"], 3), f_cmd,
      "reference_unmatched_T_split_diagonal_free_mean", "71", "matches Entry 71 0.177 (unmatched primary splits)")
    # NCC in the content-matched arms (descriptive only; Entry 71 says do not promote to a finding)
    f_scm = T1 + "/summary_cm.json"
    scm = load(f_scm)["arms"]
    Acm = [scm[k]["natural_paired_KA_minus_KB"] for k in by_seed([k for k in scm if k.startswith("cm_A")])]
    Bcm = [-scm[k]["natural_paired_KA_minus_KB"] for k in by_seed([k for k in scm if k.startswith("cm_B")])]
    th, se, df, p = welch_sym(Acm, Bcm)
    m("nDetCmNccLambda", 100 * th / R_REAL, sig(100 * th / R_REAL, 3), f_scm,
      "Welch theta_sym over arms.cm_[A|B]_s*.natural_paired_KA_minus_KB (B sign-flipped) / R_real", "71",
      "% of R_real (E2, 0.0356703); DESCRIPTIVE ONLY (Entry 71: do not promote to a finding); matches Entry 71 0.1542 %")
    m("nDetCmNccP", p, pf(p), f_scm, "same, one-sided Welch p", "71", "descriptive; matches Entry 71 0.0152")
    m("nDetCmNccT", th / se, sig(th / se, 3), f_scm, "same, t", "71", "descriptive; matches Entry 71 3.56")

    # Five-way learned attribution (Entry 41)
    for grp, G in (("kodak", "Kodak"), ("p20", "Ptwenty")):
        f5 = OUT + f"/t2_learned_5body_{grp}.json"
        d5 = load(f5)
        m(f"nDetFiveway{G}Top", d5["H_top1"], dec(d5["H_top1"], 3), f5, "H_top1", "41",
          "gate 0.90 failed; matches Entry 41 / brief (0.205 / 0.207)")
        cc = d5["attribution"]["corrected"]["250"]["adapters_majority_correct"]
        rc = d5["attribution"]["raw"]["250"]["adapters_majority_correct"]
        m(f"nDetFiveway{G}Corrected", cc, integer(cc), f5, "attribution.corrected.250.adapters_majority_correct", "41",
          "descriptive (gate failed); of 10; matches Entry 41 (Kodak 5, P20 3)")
        m(f"nDetFiveway{G}Raw", rc, integer(rc), f5, "attribution.raw.250.adapters_majority_correct", "41",
          "descriptive; of 10; matches Entry 41 (Kodak 4, P20 1)")
    f5p = OUT + "/t2_learned_5body_pooled.json"
    d5p = load(f5p)
    m("nDetFivewayPooled", d5p["corrected"]["adapters_correct_of_20"], integer(d5p["corrected"]["adapters_correct_of_20"]),
      f5p, "corrected.adapters_correct_of_20", "41", "descriptive; matches Entry 41 8/20")
    m("nDetFivewayPooledP", d5p["corrected"]["p_one_sided"], pf(d5p["corrected"]["p_one_sided"]), f5p,
      "corrected.p_one_sided", "41", "matches Entry 41 0.032")
    m("nDetFivewayPooledNeeded", d5p["corrected"]["k_needed_for_p_below_0.01"],
      integer(d5p["corrected"]["k_needed_for_p_below_0.01"]), f5p, "corrected.k_needed_for_p_below_0.01", "41",
      "matches Entry 41 9 needed")

    # ============================== Att: closed-set attribution (Entries 06, 07) =======================
    f_at = OUT + "/t3_attrib.json"
    at = load(f_at)
    cases = (("Kodak", "kodak.raw_fingerprints.corrected", "kodak.raw_fingerprints.raw"),
             ("PtwentyK", "p20.K_residualised.corrected", "p20.K_residualised.raw"),
             ("PtwentyL", "p20.L_lowmid.corrected", "p20.L_lowmid.raw"))
    g1 = []
    for N, kc, kr in cases:
        c = get(at, kc)["G"]["250"]
        k = c["adapters_majority_correct"]
        m(f"nAtt{N}Correct", k, integer(k), f_at, kc + ".G.250.adapters_majority_correct", "07",
          "of 10 adapters, main-effect-corrected, G = 250; matches Entry 07 / brief (3 / 3 / 4)")
        pb = float(stats.binom.sf(k - 1, 10, 0.2))
        m(f"nAtt{N}BinomP", pb, pf(pb), f_at, f"binom.sf({k}-1, 10, 0.2) (computed)", "07",
          "exact one-sided binomial against chance 0.2; Entry 07 prints P(X>=4) = 0.12")
        lo, hi = c["adapter_level_CI95"]
        m(f"nAtt{N}CILow", lo, dec(lo, 2), f_at, kc + ".G.250.adapter_level_CI95[0]", "07",
          "exact (Clopper-Pearson) 95 % interval; Entry 07 prints 0.07-0.65 (3/10), 0.12-0.74 (4/10)")
        m(f"nAtt{N}CIHigh", hi, dec(hi, 2), f_at, kc + ".G.250.adapter_level_CI95[1]", "07", "as above")
        r = get(at, kr)["G"]["250"]["adapters_majority_correct"]
        m(f"nAtt{N}RawCorrect", r, integer(r), f_at, kr + ".G.250.adapters_majority_correct", "07",
          "raw argmax, of 10; matches Entry 07 (3 / 2 / 4)")
        g1 += [get(at, kc)["G"]["1"]["block_accuracy"], get(at, kr)["G"]["1"]["block_accuracy"]]
    m("nAttGOneLow", min(g1), dec(min(g1), 3), f_at, "min G.1.block_accuracy over the six scores", "07",
      "matches Entry 07 0.206")
    m("nAttGOneHigh", max(g1), dec(max(g1), 3), f_at, "max G.1.block_accuracy over the six scores", "07",
      "matches Entry 07 0.216")
    m("nAttAdapters", at["kodak"]["n_adapters"], integer(at["kodak"]["n_adapters"]), f_at, "kodak.n_adapters", "07",
      "10 adapters per group (5 bodies x 2 seeds)")
    m("nAttChanceCount", 0.2 * at["kodak"]["n_adapters"], integer(0.2 * at["kodak"]["n_adapters"]), f_at,
      "0.2 x n_adapters", "06", "expected correct under chance (2/10)")
    selK = at["kodak"]["raw_fingerprints"]["raw"]["selection_frequency_G250"]
    selP = at["p20"]["K_residualised"]["raw"]["selection_frequency_G250"]
    selL = at["p20"]["L_lowmid"]["raw"]["selection_frequency_G250"]
    m("nAttKodakRawPicksTop", max(selK.values()), integer(max(selK.values())), f_at,
      "kodak.raw_fingerprints.raw.selection_frequency_G250 (max = D0)", "07", "raw argmax picks D0 for 7 of 10; Entry 07")
    m("nAttPtwentyRawPicksTop", max(selP.values()), integer(max(selP.values())), f_at,
      "p20.K_residualised.raw.selection_frequency_G250 (max = 1102)", "07", "raw argmax picks 1102 for 8 of 10; Entry 07")
    m("nAttPtwentyLRawPicksTop", max(selL.values()), integer(max(selL.values())), f_at,
      "p20.L_lowmid.raw.selection_frequency_G250 (max = 1103)", "07", "1103 x 6; Entry 07")
    bK = at["kodak"]["raw_fingerprints"]["main_effects_b"]["D0"]
    bP = at["p20"]["K_residualised"]["main_effects_b"]["1102"]
    m("nAttKodakMainDZero", bK, sci(bK, 2), f_at, "kodak.raw_fingerprints.main_effects_b.D0", "07",
      "matches Entry 07 +6.4e-05")
    m("nAttPtwentyMainOneOneZeroTwo", bP, sci(bP, 3), f_at, "p20.K_residualised.main_effects_b.1102", "07",
      "matches Entry 07 +15.7e-05")
    rc = at["kodak"]["real_photo_ceiling"]
    m("nAttKodakCeilingOne", rc["1"]["accuracy"], dec(rc["1"]["accuracy"], 2), f_at,
      "kodak.real_photo_ceiling.1.accuracy", "07", "real-photo ceiling at G = 1 (200 images); matches Entry 07 0.99")
    m("nAttKodakCeilingTen", rc["10"]["accuracy"], dec(rc["10"]["accuracy"], 2), f_at,
      "kodak.real_photo_ceiling.10.accuracy", "07", "matches Entry 07 1.00")
    rp = at["p20"]["real_photo_ceiling_K"]
    m("nAttPtwentyCeilingOne", rp["1"]["accuracy"], dec(rp["1"]["accuracy"], 2), f_at,
      "p20.real_photo_ceiling_K.1.accuracy", "07", "matches Entry 07 1.00 (150 images)")

    # Reproduction (Entry 13)
    f_rp = OUT + "/t3_repro.json"
    rpd = load(f_rp)
    mxd = max(rpd["kodak"]["max_abs_delta"], rpd["p20"]["max_abs_delta"])
    m("nAttReproMaxDelta", mxd, sci(mxd, 2), f_rp, "max(kodak.max_abs_delta, p20.max_abs_delta)", "13",
      "matches Entry 13 / brief 4.6e-08")
    m("nAttReproMaxDeltaPtwenty", rpd["p20"]["max_abs_delta"], sci(rpd["p20"]["max_abs_delta"], 2), f_rp,
      "p20.max_abs_delta", "13", "matches Entry 13 2.7e-08")
    nrows = rpd["kodak"]["n_rows"] + rpd["p20"]["n_rows"]
    m("nAttReproRows", nrows, integer(nrows), f_rp, "kodak.n_rows + p20.n_rows", "13", "matches Entry 13 25,000")
    nimg = rpd["kodak"]["n_images"] + rpd["p20"]["n_images"]
    m("nAttReproImages", nimg, integer(nimg), f_rp, "kodak.n_images + p20.n_images", "13", "matches Entry 13 5,000")
    m("nAttReproSd", rpd["kodak"]["archive_rho_sd"], sci(rpd["kodak"]["archive_rho_sd"], 3), f_rp,
      "kodak.archive_rho_sd", "13", "sd of archived rho (Kodak); matches Entry 13 1.33e-03")
    md = max(rpd["kodak"]["median_abs_delta"], rpd["p20"]["median_abs_delta"])
    m("nAttReproMedianDelta", md, sci(md, 2), f_rp, "max of the two median_abs_delta", "13",
      "matches Entry 13 3.2e-09 (Kodak)")

    # ============================== Att: power at the limit, D7-corrected (Entry 50) ====================
    f_pw = OUT + "/t3_power_v4.json"
    pw = load(f_pw)
    inp = pw["inputs"]
    m("nAttSEImg", inp["SE_img_500"], sci(inp["SE_img_500"], 4), f_pw, "inputs.SE_img_500", "50",
      "D7-corrected (R7 supersedes 4.67e-05); matches Entry 50 / brief 7.035e-05")
    m("nAttSEImgA", t2["ncc"]["arms"]["A_raw_s0_r16"]["paired_contrast_se"],
      sci(t2["ncc"]["arms"]["A_raw_s0_r16"]["paired_contrast_se"], 4), f_t2,
      "ncc.arms.A_raw_s0_r16.paired_contrast_se", "50", "matches Entry 50 7.165e-05")
    m("nAttSEImgB", t2["ncc"]["arms"]["B_raw_s0_r16"]["paired_contrast_se"],
      sci(t2["ncc"]["arms"]["B_raw_s0_r16"]["paired_contrast_se"], 4), f_t2,
      "ncc.arms.B_raw_s0_r16.paired_contrast_se", "50", "matches Entry 50 6.905e-05")
    bse = [float(r["ncc_A"]) - float(r["ncc_B"]) for r in rows if r["set"] == "base"]
    bse = float(np.std(bse, ddof=1) / math.sqrt(len(bse)))
    m("nAttSEImgBase", bse, sci(bse, 3), f_rows, "sd(ncc_A - ncc_B)/sqrt(500) over base rows", "50",
      "matches Entry 50 base 5.71e-05")
    m("nAttSigmaMu", inp["sigma_mu"], sci(inp["sigma_mu"], 3), f_pw, "inputs.sigma_mu", "50",
      "persistent per-adapter component (v1 sanity block); Entry 03 / 50")
    snr = pw["asymptotics"]["signal_to_structured_noise_at_U"]
    m("nAttSignalSigmaMu", snr, sig(snr, 3), f_pw, "asymptotics.signal_to_structured_noise_at_U", "50",
      "U_device / sigma_mu; Entry 03 prints 1.03")
    rowsP = {(r["transfer"], r["M_candidates"]): r for r in pw["rows"]}
    U = "at the upper limit U_device"
    Mw = {2: "Two", 5: "Five", 50: "Fifty"}
    for M_, W in Mw.items():
        r = rowsP[(U, M_)]
        for kk, G in (("TPR_G500", "FiveHundred"), ("TPR_G5000", "FiveThousand"), ("TPR_G_inf", "Inf")):
            v = r[kk]
            txt = dec(v, 3) if v >= 0.0095 or kk == "TPR_G_inf" else sig(v, 2)
            if M_ == 50 and kk != "TPR_G_inf":
                txt = sig(v, 2)
            chk = {("TPR_G500", 2): "matches Entry 50 / brief 0.043 and verify_v2 check 20",
                   ("TPR_G5000", 2): "matches Entry 50 / brief 0.084",
                   ("TPR_G_inf", 2): "matches Entry 50 / brief 0.097",
                   ("TPR_G_inf", 5): "matches Entry 50 / brief 0.038",
                   ("TPR_G_inf", 50): "matches Entry 50 / brief 0.006"}.get((kk, M_), "derived (file row)")
            m(f"nAttTpr{W}{G}", v, txt, f_pw, f"rows[{U}, M={M_}].{kk}", "50",
              "TPR at 1 % FPR if transfer sat at the nominal limit; " + chk)
    rz = {r["M_candidates"]: r for r in pw["rows_sigma_mu_zero"]}
    for M_, W in Mw.items():
        r = rz[M_]
        m(f"nAttTprZero{W}FiveHundred", r["TPR_G500"], dec(r["TPR_G500"], 3) if r["TPR_G500"] >= 0.0095 else sig(r["TPR_G500"], 2),
          f_pw, f"rows_sigma_mu_zero[M={M_}].TPR_G500", "50",
          "sigma_mu = 0; " + ("matches Entry 50 / brief 0.059 and verify_v2 check 21" if M_ == 2 else "derived"))
        m(f"nAttTprZero{W}FiveThousand", r["TPR_G5000"], sig(r["TPR_G5000"], 2), f_pw,
          f"rows_sigma_mu_zero[M={M_}].TPR_G5000", "50",
          "sigma_mu = 0; " + ("matches Entry 50 / brief 0.54" if M_ == 2 else "derived"))
    g90 = rz[2]["G_for_TPR90"]
    m("nAttZeroGNinety", g90, integer(g90), f_pw, "rows_sigma_mu_zero[M=2].G_for_TPR90", "50",
      "DISCREPANCY: grid value (grid ...5000, 10000, 20000); the exact requirement is nAttZeroGNinetyExact (~11,000); "
      "Entry 50 / brief 'about 20 000' inherit the grid")
    # exact continuous solutions of the same model
    z = lambda M_: stats.norm.ppf(1 - 0.01 / (M_ - 1))
    Ud, se5 = inp["U_device"], inp["SE_img_500"]

    def g_exact(theta, M_, target, sm):
        s_need = theta / (z(M_) + stats.norm.ppf(target))
        v = s_need ** 2 - sm ** 2
        return 500 * se5 ** 2 / v if v > 0 else float("inf")

    ge90 = g_exact(Ud, 2, 0.9, 0.0)
    m("nAttZeroGNinetyExact", ge90, integer(round(ge90, -3)), f_pw,
      "computed: G = 500 SE_img^2 / (U/(z_0.99 + z_0.9))^2, sigma_mu = 0, M = 2", "50",
      "exact solution of t3_power_v4.py's model (%.0f images), rounded to the nearest thousand" % ge90)
    ge50 = g_exact(Ud, 2, 0.5, 0.0)
    m("nAttZeroGFifty", rz[2]["G_for_TPR50"], integer(rz[2]["G_for_TPR50"]), f_pw,
      "rows_sigma_mu_zero[M=2].G_for_TPR50", "50", "grid value; exact %.0f" % ge50)
    m("nAttZeroGFiftyExact", ge50, integer(round(ge50, -2)), f_pw, "computed as above, target 0.5", "50",
      "exact solution, rounded to the nearest hundred")
    T10 = "10x the upper limit"
    for M_, W in Mw.items():
        g = rowsP[(T10, M_)]["G_for_TPR50"]
        m(f"nAttTenxG{W}", g, integer(g), f_pw, f"rows[{T10}, M={M_}].G_for_TPR50", "50",
          "images for TPR 0.5 at 10x the limit (grid 10, 50, 100, 250, 500 ...); brief 50 / 100 / 250; exact %.0f"
          % g_exact(10 * Ud, M_, 0.5, inp["sigma_mu"]))
        ge = g_exact(10 * Ud, M_, 0.5, inp["sigma_mu"])
        m(f"nAttTenxG{W}Exact", ge, integer(math.ceil(ge)), f_pw, "computed exact G for TPR 0.5 at 10 U_device", "50",
          "exact solution of the same model (rounded up)")
    m("nAttHundredxG", rowsP[("100x the upper limit", 2)]["G_for_TPR50"],
      integer(rowsP[("100x the upper limit", 2)]["G_for_TPR50"]), f_pw, "rows[100x, M=2].G_for_TPR50", "50",
      "10 images (smallest grid value) at 100x the limit, any M")
    m("nAttTenxLambda", rowsP[(T10, 2)]["lambda_pct"], sig(rowsP[(T10, 2)]["lambda_pct"], 3), f_pw,
      "rows[10x, M=2].lambda_pct", "50", "10 x nominal limit, % of R_real (E2, 0.0356703)")

    # ============================== Wt: adapter weights, H5 (Entries 94, 95) =============================
    f_h5 = OUT + "/h5_weight_signature.json"
    h5 = load(f_h5)
    for dose, W in (("2000", "Two"), ("16000", "Sixteen")):
        d = h5["doses"][dose]
        tags = d["adapters"]
        assert tags == sorted(tags, key=lambda t: (t.split("_s")[0], seed_of(t))), "adapter order"
        m(f"nWt{W}SameBody", d["mean_cos_same_body"], dec(d["mean_cos_same_body"], 3), f_h5,
          f"doses.{dose}.mean_cos_same_body", "95", "matches Entry 95 (+0.0163 / +0.0251)")
        m(f"nWt{W}CrossBody", d["mean_cos_diff_body"], dec(d["mean_cos_diff_body"], 3), f_h5,
          f"doses.{dose}.mean_cos_diff_body", "95", "matches Entry 95 (+0.0967 / +0.0927)")
        m(f"nWt{W}D", d["D"], dec(d["D"], 3), f_h5, f"doses.{dose}.D", "95",
          "matches Entry 95 (-0.0804 / -0.0676)" + ("; verify_v2 check 59" if dose == "2000" else ""))
        m(f"nWt{W}P", d["perm_p_one_sided"], pf(d["perm_p_one_sided"]), f_h5, f"doses.{dose}.perm_p_one_sided", "95",
          "exact over 462 relabellings; matches Entry 95 / brief 0.93" + ("; verify_v2 check 60" if dose == "2000" else ""))
        m(f"nWt{W}Z", d["z_vs_perm"], sig(d["z_vs_perm"], 2), f_h5, f"doses.{dose}.z_vs_perm", "95",
          "D in permutation SDs; Entry 95 'about two permutation SDs'")
        b = d["batch_diagnostic_descriptive"]
        m(f"nWt{W}SameBatch", b["mean_cos_same_batch"], dec(b["mean_cos_same_batch"], 3), f_h5,
          f"doses.{dose}.batch_diagnostic_descriptive.mean_cos_same_batch", "95",
          "DESCRIPTIVE (grouping chosen after the result); matches Entry 95 / brief (0.116 / 0.112)")
        m(f"nWt{W}CrossBatch", b["mean_cos_diff_batch"], dec(b["mean_cos_diff_batch"], 3), f_h5,
          f"doses.{dose}.batch_diagnostic_descriptive.mean_cos_diff_batch", "95",
          "descriptive; matches Entry 95 / brief (0.014 / 0.020)")
        m(f"nWt{W}DBatch", b["D_batch"], dec(b["D_batch"], 3), f_h5,
          f"doses.{dose}.batch_diagnostic_descriptive.D_batch", "95", "descriptive; matches Entry 95 (+0.1024 / +0.0920)")
    d2 = h5["doses"]["2000"]
    m("nWtPerms", d2["n_permutations"], integer(d2["n_permutations"]), f_h5, "doses.2000.n_permutations", "94",
      "C(12,6)/2 = 462")
    m("nWtPermFloor", d2["perm_floor"], sig(d2["perm_floor"], 2), f_h5, "doses.2000.perm_floor", "94",
      "1/462; Entry 94 0.0022")
    m("nWtPairs", d2["n_pairs"], integer(d2["n_pairs"]), f_h5, "doses.2000.n_pairs", "94", "66 pairs of 12 adapters")
    m("nWtAdapters", len(d2["adapters"]), integer(len(d2["adapters"])), f_h5, "len(doses.2000.adapters)", "94",
      "12 per dose, six per body")
    m("nWtBatchGapDays", 10, "ten", "RESULTS.md",
      "Entry 95: 'seeds 0-2 of both arms were trained 9-12 September and seeds 3-5 on 15-19 September, ten days apart'",
      "95", "text-only")

    # ============================== Shift: shifted-template control (Entries 62, 91, 92) =================
    f_c5 = OUT + "/c5_shift_grid.json"
    c5 = load(f_c5)
    ar = c5["arm_results"]
    bse_ = ar["local_base"]["excess_on8_vs_off"]
    m("nShiftBaseZ", bse_["z"], sig(bse_["z"], 3), f_c5, "arm_results.local_base.excess_on8_vs_off.z", "62",
      "matches Entry 62 / brief z 3.56")
    m("nShiftBaseExcess", bse_["excess"], sci(bse_["excess"], 3), f_c5, "arm_results.local_base.excess_on8_vs_off.excess",
      "62", "matches Entry 62 +8.43e-05")
    m("nShiftBaseSE", bse_["se"], sci(bse_["se"], 3), f_c5, "arm_results.local_base.excess_on8_vs_off.se", "62",
      "matches Entry 62 2.37e-05")
    adapted = [k for k in c5["arms"] if k != "local_base"]
    zs = [ar[k]["excess_on8_vs_off"]["z"] for k in adapted]
    ex = [ar[k]["excess_on8_vs_off"]["excess"] for k in adapted]
    m("nShiftAdaptedZLow", min(zs), sig(min(zs), 3), f_c5, "min over 8 adapted arms of excess_on8_vs_off.z", "62",
      "matches Entry 62 4.25")
    m("nShiftAdaptedZHigh", max(zs), sig(max(zs), 3), f_c5, "max over 8 adapted arms", "62", "matches Entry 62 6.29")
    m("nShiftAdaptedExcessLow", min(ex), sci(min(ex), 2), f_c5, "min adapted excess", "62", "matches Entry 62 +9.5e-05")
    m("nShiftAdaptedExcessHigh", max(ex), sci(max(ex), 3), f_c5, "max adapted excess", "62", "matches Entry 62 +1.46e-04")
    pad = c5["pooled_adapted"]
    m("nShiftPooledExcess", pad["excess"], sci(pad["excess"], 3), f_c5, "pooled_adapted.excess", "62",
      "matches Entry 62 +1.19e-04")
    m("nShiftPooledZ", pad["z"], sig(pad["z"], 3), f_c5, "pooled_adapted.z", "62", "matches Entry 62 6.45")
    pmb = c5["pooled_adapted_minus_base"]
    m("nShiftAdaptedMinusBase", pmb["excess"], sci(pmb["excess"], 2), f_c5, "pooled_adapted_minus_base.excess", "62",
      "matches Entry 62 +3.4e-05")
    m("nShiftAdaptedMinusBaseZ", pmb["z"], sig(pmb["z"], 3), f_c5, "pooled_adapted_minus_base.z", "62",
      "matches Entry 62 1.37")
    ranks = [ar[k]["v1_shift"]["rank_among_sampled"] for k in c5["arms"]]
    m("nShiftVoneRankLow", min(ranks), integer(min(ranks)), f_c5, "min arm_results.*.v1_shift.rank_among_sampled", "62",
      "of 769; matches Entry 62 41")
    m("nShiftVoneRankHigh", max(ranks), integer(max(ranks)), f_c5, "max arm_results.*.v1_shift.rank_among_sampled", "62",
      "matches Entry 62 198")
    m("nShiftDisplacements", c5["design"]["n_displacements"], integer(c5["design"]["n_displacements"]), f_c5,
      "design.n_displacements", "62", "769")
    m("nShiftOnGrid", c5["design"]["n_on8"], integer(c5["design"]["n_on8"]), f_c5, "design.n_on8", "62", "12 on the 8-px grid")
    m("nShiftImgPerArm", c5["n_images_per_arm"], integer(c5["n_images_per_arm"]), f_c5, "n_images_per_arm", "62",
      "150 per arm (C5)")
    m("nShiftArms", len(c5["arms"]), integer(len(c5["arms"])), f_c5, "len(arms)", "62", "nine arms (base + 8 adapted)")
    ka = c5["K_autocorrelation_diagnostic"]["sampled"]
    m("nShiftKAutoExcess", ka["excess"], sci(ka["excess"], 2), f_c5, "K_autocorrelation_diagnostic.sampled.excess", "62",
      "descriptive; matches Entry 62 +2.1e-4")
    m("nShiftKAutoZ", ka["z"], sig(ka["z"], 2), f_c5, "K_autocorrelation_diagnostic.sampled.z", "62",
      "matches Entry 62 0.98")

    f_h3 = OUT + "/h3_shift_model.json"
    h3 = load(f_h3)
    arms3 = h3["arms"]
    ob3 = float(np.mean([a["observed_excess"] for a in arms3.values()]))
    pr3 = float(np.mean([a["predicted_excess"] for a in arms3.values()]))
    se3 = float(math.sqrt(sum(a["observed_excess_se"] ** 2 for a in arms3.values())) / len(arms3))
    m("nShiftHthreeObs", ob3, sci(ob3, 3), f_h3, "mean over 9 arms of arms.*.observed_excess", "91",
      "DISCREPANCY: Entry 92 / brief quote 6.18e-05 (~6.2e-5) 'pooled over H3's arms', which is local_base alone; "
      "the nine-arm pooled mean is 6.15e-05 (FINDINGS.md agrees)")
    m("nShiftHthreePred", pr3, sci(pr3, 3), f_h3, "mean over 9 arms of arms.*.predicted_excess", "91",
      "Entry 92 prints '+7.0e-05 in H3'; file pooled mean 6.81e-05")
    m("nShiftHthreeSE", se3, sci(se3, 2), f_h3, "sqrt(sum observed_excess_se^2)/9 (same pooling as h3b pooled_on_grid.se)",
      "91", "derived; reproduces h3b's pooled SE construction exactly")
    m("nShiftHthreeDiffSE", (pr3 - ob3) / se3, sig((pr3 - ob3) / se3, 2), f_h3, "(pred - obs)/SE, pooled", "91",
      "derived; within 2 SE")
    m("nShiftHthreeMedianRatio", float(np.median([a["predicted_over_observed"] for a in arms3.values()])),
      sig(float(np.median([a["predicted_over_observed"] for a in arms3.values()])), 3), f_h3,
      "median arms.*.predicted_over_observed", "91", "matches Entry 91 1.10")
    m("nShiftHthreeSpearman", h3["median_spearman"], dec(h3["median_spearman"], 2), f_h3, "median_spearman", "91",
      "registered criterion >= 0.8 not met; matches Entry 91 0.36")
    m("nShiftHthreeWithin", h3["arms_within_2se"], integer(h3["arms_within_2se"]), f_h3, "arms_within_2se", "91",
      "9 of 9 arms within 2 SE")
    m("nShiftHthreeImages", h3["images_per_arm"], integer(h3["images_per_arm"]), f_h3, "images_per_arm", "91",
      "60 per arm per sample (H3: 0-59, H3b: 60-119)")
    f_h3b = OUT + "/h3b_shift_model.json"
    h3b = load(f_h3b)["pooled_on_grid"]
    m("nShiftHthreebObs", h3b["observed_excess"], sci(h3b["observed_excess"], 3), f_h3b, "pooled_on_grid.observed_excess",
      "92", "matches Entry 92 +1.017e-05 (brief 1.0e-5)")
    m("nShiftHthreebPred", h3b["predicted_excess"], sci(h3b["predicted_excess"], 3), f_h3b,
      "pooled_on_grid.predicted_excess", "92", "matches Entry 92 +6.897e-05")
    m("nShiftHthreebSE", h3b["se"], sci(h3b["se"], 2), f_h3b, "pooled_on_grid.se", "92", "derived from file")
    m("nShiftHthreebDiffSE", h3b["difference_in_se"], sig(h3b["difference_in_se"], 3), f_h3b,
      "pooled_on_grid.difference_in_se", "92", "matches Entry 92 1.28 SE")
    m("nShiftHthreebMedianRatio", h3b["median_ratio"], sig(h3b["median_ratio"], 3), f_h3b, "pooled_on_grid.median_ratio",
      "92", "registered window [0.67, 1.5] not met; matches Entry 92 1.88")
    predr = 0.5 * (pr3 + h3b["predicted_excess"])
    m("nShiftPredRound", predr, sci(predr, 1), f_h3 + " ; " + rel(f_h3b), "mean of the two pooled predictions, 1 s.f.",
      "92", "brief '~7e-5'")
    m("nShiftImageFactor", 100, "100", "RESULTS.md",
      "Entry 92: 'needs of order a hundred times the sample, which is thousands of generations per arm'", "92",
      "text-only")
    for x in M:
        x["meaning"] = meaning_of(x["name"])
    return M


# --------------------------------------------------------------------------------------------- meanings
DETN = {"Ncc": "zero-lag NCC (the study's statistic)", "Pce": "PCE (Goljan, prnu-python)",
        "Lowmid": "low/mid 8x8-DCT signature", "Noiseprint": "Noiseprint"}
PATTERNS = [
    (r"^nDet(Ncc|Pce|Lowmid)Auc$", "same-model real-photo AUC (held-out, 40 per body) of the {d}"),
    (r"^nDet(Ncc|Pce|Lowmid)Real$", "real paired contrast R (own minus other) of the {d}, its own units"),
    (r"^nDet(Ncc|Pce|Lowmid)ThetaSym$", "paired (symmetric) contrast toward the training body, {d}, 3+3 primary adapters, 500 gens each"),
    (r"^nDet(Ncc|Pce|Lowmid)Additive$", "additive (shared main-effect) part (theta_A - theta_B)/2, {d}"),
    (r"^nDet(Ncc|Pce|Lowmid)ThetaSymPct$", "theta_sym as % of the {d}'s own real contrast"),
    (r"^nDet(Ncc|Pce|Lowmid)SymSE$", "adapter-level Welch SE of theta_sym, {d}"),
    (r"^nDet(Ncc|Pce|Lowmid)SymT$", "t of theta_sym, {d} (null)"),
    (r"^nDetPceTheta(A|B)$", "PCE arm-{d} mean paired contrast toward own body (3 adapters)"),
    (r"^nDetTrans(Ncc|Pce|Lowmid|Noiseprint)S$", "C7 transplant: smallest residual scale s detected at t > 3, {d}"),
    (r"^nDetTrans(Ncc|Pce|Lowmid|Noiseprint)TAtS$", "C7: paired increment t at that smallest s, {d}"),
    (r"^nDetTrans(Ncc|Pce|Lowmid|Noiseprint)TOne$", "C7: paired increment t at s = 1, {d}"),
    (r"^nDetTrans(Ncc|Pce|Lowmid|Noiseprint)PctOne$", "C7: increment at s = 1 as % of the {d}'s real paired contrast"),
    (r"^nDetTrans(Ncc|Pce|Lowmid|Noiseprint)N$", "C7: number of base-model generations scored, {d}"),
    (r"^nDetTransStoredRms(\w+)$", "C7: RMS change of the stored generation (gray levels) at s = {d}"),
    (r"^nDetInj(Ncc|Pce|Lowmid)Alpha$", "fingerprint injected into base generations: smallest alpha with paired increment t > 3, {d}"),
    (r"^nDetInj(Ncc|Pce)AlphaContrast$", "injection control: smallest alpha at which the contrast itself has t > 3, {d}"),
    (r"^nDetInjPceOverSixty(\w+)$", "injection control: % of generations with PCE > 60 against A at alpha = {d}"),
    (r"^nDetLearnedIUp(A|B)$", "learned CNN: registered IU sign-flip p, arm {d} (floor 1/4096)"),
    (r"^nDetLearnedIUCount(A|B)$", "learned CNN: sign-flip assignments reaching the observed value, arm {d} (of 4096)"),
    (r"^nDetFiveway(Kodak|Ptwenty)Top$", "five-way learned detector: held-out real top-1 accuracy, {d} group (gate 0.90, failed)"),
    (r"^nDetFiveway(Kodak|Ptwenty)Corrected$", "five-way learned detector, descriptive: adapters correct of 10, main-effect-corrected, {d}"),
    (r"^nDetFiveway(Kodak|Ptwenty)Raw$", "five-way learned detector, descriptive: adapters correct of 10, raw argmax, {d}"),
    (r"^nAtt(Kodak|PtwentyK|PtwentyL)Correct$", "closed-set attribution, main-effect-corrected, G = 250: adapters correct of 10, {d}"),
    (r"^nAtt(Kodak|PtwentyK|PtwentyL)BinomP$", "exact one-sided binomial p of that count against chance 0.2, {d}"),
    (r"^nAtt(Kodak|PtwentyK|PtwentyL)CILow$", "exact 95 % interval, lower end, of that accuracy, {d}"),
    (r"^nAtt(Kodak|PtwentyK|PtwentyL)CIHigh$", "exact 95 % interval, upper end, of that accuracy, {d}"),
    (r"^nAtt(Kodak|PtwentyK|PtwentyL)RawCorrect$", "closed-set attribution, raw argmax, G = 250: adapters correct of 10, {d}"),
    (r"^nAttTpr(Two|Five|Fifty)(FiveHundred|FiveThousand|Inf)$",
     "TPR at 1 % FPR if transfer sat at the nominal limit, {d} candidates, G = {e} images (sigma_mu = 5.23e-5)"),
    (r"^nAttTprZero(Two|Five|Fifty)(FiveHundred|FiveThousand)$",
     "TPR at 1 % FPR at the nominal limit with sigma_mu = 0, {d} candidates, G = {e} images"),
    (r"^nAttTenxG(Two|Five|Fifty)$", "images for TPR 0.5 at 10x the limit, {d} candidates (grid value)"),
    (r"^nAttTenxG(Two|Five|Fifty)Exact$", "images for TPR 0.5 at 10x the limit, {d} candidates (exact solution)"),
    (r"^nWt(Two|Sixteen)SameBody$", "H5: mean cosine of adapter updates dW = BA, same-body pairs, {d} steps"),
    (r"^nWt(Two|Sixteen)CrossBody$", "H5: mean cosine, different-body pairs, {d} steps"),
    (r"^nWt(Two|Sixteen)D$", "H5: D = same-body minus cross-body cosine, {d} steps"),
    (r"^nWt(Two|Sixteen)P$", "H5: exact permutation p (462 relabellings), {d} steps"),
    (r"^nWt(Two|Sixteen)Z$", "H5: D in permutation SDs, {d} steps"),
    (r"^nWt(Two|Sixteen)SameBatch$", "H5 descriptive: mean cosine, same training batch, {d} steps"),
    (r"^nWt(Two|Sixteen)CrossBatch$", "H5 descriptive: mean cosine, different training batch, {d} steps"),
    (r"^nWt(Two|Sixteen)DBatch$", "H5 descriptive: same-batch minus cross-batch cosine, {d} steps"),
]
WORD = {"Ncc": DETN["Ncc"], "Pce": DETN["Pce"], "Lowmid": DETN["Lowmid"], "Noiseprint": DETN["Noiseprint"],
        "A": "A", "B": "B", "Kodak": "Kodak M1063", "Ptwenty": "Huawei P20", "PtwentyK": "Huawei P20, fingerprint K",
        "PtwentyL": "Huawei P20, low/mid L", "Two": "2", "Five": "5", "Fifty": "50", "FiveHundred": "500",
        "FiveThousand": "5,000", "Inf": "unlimited", "Sixteen": "16000", "PointOne": "0.1", "Quarter": "0.25",
        "Half": "0.5", "One": "1"}
WTDOSE = {"Two": "2000", "Sixteen": "16000"}
EXPLICIT = {
    "nDetPceMaxArmT": "PCE, B_raw_s0 arm alone: unpaired t toward its body (main-effect artefact)",
    "nDetNccImgSELow": "NCC image-level SE of a 500-image arm mean, smallest arm",
    "nDetNccImgSEHigh": "NCC image-level SE of a 500-image arm mean, largest arm",
    "nDetPceFpCount": "generations exceeding PCE 60 against either real fingerprint",
    "nDetPceFpTotal": "generations scored by the panel (base + 6 primary adapters x 500)",
    "nDetPanelGenPerArm": "generations per arm in the detector panel",
    "nDetPanelRealPerBody": "held-out real photographs per body in the panel",
    "nDetPceMedianA": "median PCE of the 3,500 generations against body A's fingerprint",
    "nDetPceMedianB": "median PCE of the 3,500 generations against body B's fingerprint",
    "nDetPceMedianAll": "median PCE, both fingerprints pooled",
    "nDetPceMedianArmLow": "smallest per-arm median PCE",
    "nDetPceMedianArmHigh": "largest per-arm median PCE",
    "nDetPceMax": "largest PCE any generation reaches against either real fingerprint",
    "nDetPceRealOverSixtyA": "real body-A photographs with PCE > 60 against own fingerprint (of 40)",
    "nDetPceRealOverSixtyB": "real body-B photographs with PCE > 60 against own fingerprint (of 40)",
    "nDetPceBaseMeanA": "base-model mean PCE against fingerprint A",
    "nDetPceBaseMeanB": "base-model mean PCE against fingerprint B",
    "nDetNoiseprintAuc": "Noiseprint same-model real-photo AUC (gate 0.95)",
    "nDetNoiseprintReal": "Noiseprint real paired contrast R",
    "nDetNoiseprintRealSE": "SE of Noiseprint's real paired contrast (80 images)",
    "nDetNoiseprintFpNcc": "NCC between the two bodies' Noiseprint fingerprints",
    "nDetNoiseprintThetaSym": "Noiseprint paired contrast toward training body (3+3 adapters)",
    "nDetNoiseprintSymSE": "adapter-level SE of Noiseprint theta_sym",
    "nDetNoiseprintSymT": "t of Noiseprint theta_sym (null)",
    "nDetNoiseprintThetaSymPct": "Noiseprint theta_sym as % of its real contrast",
    "nDetNoiseprintAdditive": "Noiseprint additive (main-effect) part",
    "nDetTransResidRms": "C7: mean RMS of the transplanted real residual (gray levels per channel)",
    "nDetTransPceOverSixtyOne": "C7: % of generations with PCE > 60 against A at s = 1",
    "nDetTransLowmidClusterT": "C7: low/mid t at s = 0.1 over the 40 residual means",
    "nDetTransNoiseprintClusterT": "C7: Noiseprint t at s = 0.25 over the 40 residual means",
    "nDetInjPcePeakZeroQuarter": "injection control: % of generations with the PCE peak at zero shift, alpha 0.25",
    "nDetLearnedAuc": "learned CNN (saved model): held-out real AUC",
    "nDetLearnedReal": "learned CNN real paired contrast R (logits)",
    "nDetLearnedTheta": "learned CNN symmetric interaction, 12 adapters per arm (logits)",
    "nDetLearnedSE": "adapter-level SE of the learned interaction",
    "nDetLearnedDf": "Welch df of the learned interaction",
    "nDetLearnedT": "Welch t of the learned interaction",
    "nDetLearnedP": "one-sided Welch p of the learned interaction",
    "nDetLearnedPermP": "exact label-permutation p of the learned interaction",
    "nDetLearnedPermCount": "permutation assignments reaching the observed difference",
    "nDetLearnedPermTotal": "total label assignments C(24,12)",
    "nDetLearnedLambda": "learned interaction as % of the learned detector's real contrast",
    "nDetLearnedLambdaU": "plug-in one-sided 99 % upper limit, % of the learned detector's real contrast",
    "nDetLearnedThetaA": "learned CNN mean score of A arms (logits)",
    "nDetLearnedThetaB": "learned CNN mean score of B arms (logits)",
    "nDetLearnedAdditive": "learned CNN additive part (logits)",
    "nDetLearnedBase": "learned CNN base-model mean score (logits)",
    "nDetLearnedShiftA": "A arms minus base model (logits)",
    "nDetLearnedShiftB": "B arms minus base model (logits)",
    "nDetLearnedMeanDiff": "mean per-seed A minus B score (logits)",
    "nDetLearnedPosSeeds": "seeds (of 12) with A minus B > 0",
    "nDetLearnedSeedsTotal": "adapters per arm scored by the learned CNN",
    "nDetLearnedImgPerArm": "generations per adapter scored by the learned CNN",
    "nDetLearnedFirstAuc": "first (unsaved, D5) draw of the learned CNN: real AUC",
    "nDetLearnedFirstReal": "first draw: real contrast R (logits)",
    "nDetLearnedFirstTheta": "first draw: interaction at 3 adapters per arm",
    "nDetLearnedFirstSE": "first draw: adapter-level SE",
    "nDetLearnedFirstLambdaU": "first draw: plug-in 99 % upper limit, % of its R",
    "nDetNoprnuAuc": "learned CNN with PRNU template projected out: real AUC",
    "nDetNoprnuReal": "template-projected network: real contrast R (logits)",
    "nDetNoprnuTheta": "template-projected network: interaction (logits)",
    "nDetNoprnuT": "template-projected network: Welch t",
    "nDetNoprnuP": "template-projected network: one-sided Welch p",
    "nDetNoprnuPermP": "template-projected network: exact permutation p",
    "nDetNoprnuIUpA": "template-projected network: IU sign-flip p, arm A",
    "nDetNoprnuIUpB": "template-projected network: IU sign-flip p, arm B",
    "nDetNoprnuLambda": "template-projected interaction as % of its own real contrast",
    "nDetNoprnuLambdaU": "template-projected plug-in 99 % limit, % of its own real contrast",
    "nDetNoprnuAdditivePct": "template-projected additive part, % of its own real contrast",
    "nDetNoprnuTemplateBefore": "mean |NCC(residual, Y*K_A)| on real H images before projection",
    "nDetNoprnuTemplateAfter": "same after projection",
    "nDetNoprnuCorr": "Pearson r of per-adapter own-body contrasts, projected vs original network (n = 24)",
    "nDetNoprnuSpearman": "Spearman rho of the same",
    "nDetShuffleRatio": "block-shuffled / original learned interaction",
    "nDetShuffleTheta": "block-shuffled learned interaction (logits)",
    "nDetShuffleSE": "adapter-level SE, block-shuffled",
    "nDetShuffleT": "Welch t, block-shuffled",
    "nDetShufflePermP": "exact permutation p, block-shuffled",
    "nDetShuffleAuc": "real AUC with block-shuffled residuals",
    "nDetShuffleReal": "real contrast R with block-shuffled residuals (logits)",
    "nDetShuffleCorr": "correlation of the 24 adapter means with and without shuffling",
    "nDetShuffleDrop": "mean drop of every score under shuffling (logits)",
    "nDetCmTheta": "E3 content-matched learned interaction (logits, 3+3 adapters)",
    "nDetCmSE": "E3 adapter-level SE",
    "nDetCmT": "E3 Welch t",
    "nDetCmDf": "E3 Welch df",
    "nDetCmP": "E3 one-sided Welch p (registered test)",
    "nDetCmPermP": "E3 exact permutation p",
    "nDetCmPermFloor": "smallest attainable permutation p at 3+3",
    "nDetCmRatio": "E3 matched interaction as % of the unmatched reference",
    "nDetCmRef": "E3 reference: unmatched v2 unmarked arms, learned interaction (logits)",
    "nDetCmRefSE": "E3 reference SE",
    "nDetCmRefT": "E3 reference Welch t",
    "nDetCmRefDf": "E3 reference Welch df",
    "nDetCmRefP": "E3 reference one-sided p",
    "nDetCmRefShuffle": "E3 reference, block-shuffled (logits)",
    "nDetCmThreshTexture": "E3 'camera texture' threshold (50 % of reference)",
    "nDetCmThreshContent": "E3 'training-set content' threshold (25 % of reference)",
    "nDetCmRefFirearmA": "firearm share of reference A-arm generations (%)",
    "nDetCmRefFirearmB": "firearm share of reference B-arm generations (%)",
    "nDetCmPairs": "matched training pairs (DINOv2 Hungarian)",
    "nDetCmPairCos": "mean DINOv2 cosine of matched pairs",
    "nDetCmPairCosMin": "minimum DINOv2 cosine of matched pairs",
    "nDetCmUnmatchedCos": "mean DINOv2 cosine of the unmatched primary splits",
    "nDetCmNccLambda": "NCC theta_sym in the content-matched arms, % of R_real (descriptive only)",
    "nDetCmNccP": "its one-sided Welch p (descriptive only)",
    "nDetCmNccT": "its t (descriptive only)",
    "nDetFivewayPooled": "five-way learned, both groups: adapters correct of 20 (corrected, descriptive)",
    "nDetFivewayPooledP": "its one-sided binomial p",
    "nDetFivewayPooledNeeded": "correct adapters needed for p < 0.01 of 20",
    "nAttGOneLow": "block accuracy at G = 1, lowest of the six scores (chance 0.2)",
    "nAttGOneHigh": "block accuracy at G = 1, highest of the six scores",
    "nAttAdapters": "adapters per five-body group",
    "nAttChanceCount": "expected correct adapters under chance (of 10)",
    "nAttKodakRawPicksTop": "Kodak raw argmax: adapters (of 10) assigned to D0, the largest main effect",
    "nAttPtwentyRawPicksTop": "P20 raw argmax (K): adapters assigned to 1102, the largest main effect",
    "nAttPtwentyLRawPicksTop": "P20 raw argmax (low/mid): adapters assigned to 1103",
    "nAttKodakMainDZero": "Kodak fitted main effect of D0",
    "nAttPtwentyMainOneOneZeroTwo": "P20 fitted main effect of 1102",
    "nAttKodakCeilingOne": "Kodak real-photo attribution accuracy at G = 1",
    "nAttKodakCeilingTen": "Kodak real-photo attribution accuracy at G = 10",
    "nAttPtwentyCeilingOne": "P20 real-photo attribution accuracy at G = 1",
    "nAttReproMaxDelta": "largest |delta rho| re-measuring archived rows from images",
    "nAttReproMaxDeltaPtwenty": "same, P20 group",
    "nAttReproRows": "rows re-measured",
    "nAttReproImages": "images re-measured",
    "nAttReproSd": "sd of the archived rho (Kodak) for scale",
    "nAttReproMedianDelta": "median |delta rho|",
    "nAttSEImg": "image-level SE of a 500-image paired-contrast mean (D7-corrected)",
    "nAttSEImgA": "same, A_raw_s0",
    "nAttSEImgB": "same, B_raw_s0",
    "nAttSEImgBase": "same, base model",
    "nAttSigmaMu": "persistent per-adapter structured component sigma_mu",
    "nAttSignalSigmaMu": "signal at the nominal limit in units of sigma_mu",
    "nAttZeroGNinety": "images for TPR 0.9, two candidates, sigma_mu = 0 (grid value, as in the file)",
    "nAttZeroGNinetyExact": "images for TPR 0.9, two candidates, sigma_mu = 0 (exact, nearest thousand)",
    "nAttZeroGFifty": "images for TPR 0.5, two candidates, sigma_mu = 0 (grid)",
    "nAttZeroGFiftyExact": "images for TPR 0.5, two candidates, sigma_mu = 0 (exact)",
    "nAttHundredxG": "images for TPR 0.5 at 100x the limit, any M",
    "nAttTenxLambda": "10x the nominal limit, % of R_real",
    "nWtPerms": "H5 distinct balanced relabellings",
    "nWtPermFloor": "H5 smallest attainable p",
    "nWtPairs": "H5 adapter pairs per dose",
    "nWtAdapters": "H5 adapters per dose (six per body)",
    "nWtBatchGapDays": "days between the two training batches",
    "nShiftBaseZ": "C5: on-grid excess of the shifted-template statistic in base-model generations, z",
    "nShiftBaseExcess": "C5: base-model on-grid excess",
    "nShiftBaseSE": "C5: its SE across displacements",
    "nShiftAdaptedZLow": "C5: smallest z over the 8 adapted arms",
    "nShiftAdaptedZHigh": "C5: largest z over the 8 adapted arms",
    "nShiftAdaptedExcessLow": "C5: smallest adapted-arm excess",
    "nShiftAdaptedExcessHigh": "C5: largest adapted-arm excess",
    "nShiftPooledExcess": "C5: pooled adapted excess",
    "nShiftPooledZ": "C5: pooled adapted z",
    "nShiftAdaptedMinusBase": "C5: pooled adapted minus base excess",
    "nShiftAdaptedMinusBaseZ": "C5: its z",
    "nShiftVoneRankLow": "C5: best rank of the v1 shift among 769 displacements",
    "nShiftVoneRankHigh": "C5: worst rank of the v1 shift",
    "nShiftDisplacements": "C5/H3: displacements tested",
    "nShiftOnGrid": "displacements on the 8-px grid on both axes",
    "nShiftImgPerArm": "C5 generations per arm",
    "nShiftArms": "arms in C5/H3 (base + 8 adapted)",
    "nShiftKAutoExcess": "fingerprint's own on-grid autocorrelation excess (descriptive)",
    "nShiftKAutoZ": "its z",
    "nShiftHthreeObs": "H3: pooled observed on-grid excess (first 60 images per arm)",
    "nShiftHthreePred": "H3: pooled grid-periodic prediction (nothing fitted)",
    "nShiftHthreeSE": "H3: pooled SE of the observation",
    "nShiftHthreeDiffSE": "H3: prediction minus observation in SE",
    "nShiftHthreeMedianRatio": "H3: median per-arm predicted/observed",
    "nShiftHthreeSpearman": "H3: median rank correlation over 769 displacements (criterion 0.8)",
    "nShiftHthreeWithin": "H3: arms (of 9) with prediction within 2 SE",
    "nShiftHthreeImages": "images per arm per sample (H3, H3b)",
    "nShiftHthreebObs": "H3b: pooled observed on-grid excess (disjoint images 60-119)",
    "nShiftHthreebPred": "H3b: pooled prediction",
    "nShiftHthreebSE": "H3b: pooled SE",
    "nShiftHthreebDiffSE": "H3b: prediction minus observation in SE",
    "nShiftHthreebMedianRatio": "H3b: median per-arm predicted/observed (window 0.67-1.5)",
    "nShiftPredRound": "grid-periodic prediction, both samples, one significant figure",
    "nShiftImageFactor": "factor more images needed to pin the observed excess to a tenth of its size",
}


def meaning_of(name):
    if name in EXPLICIT:
        return EXPLICIT[name]
    for pat, tmpl in PATTERNS:
        mm = re.match(pat, name)
        if mm:
            g = mm.groups()
            d = WTDOSE.get(g[0], WORD.get(g[0], g[0])) if name.startswith("nWt") else WORD.get(g[0], g[0])
            e = WORD.get(g[1], g[1]) if len(g) > 1 else ""
            return tmpl.format(d=d, e=e)
    raise KeyError(f"no meaning for {name}")


# --------------------------------------------------------------------------------------------- self-test


def write_md(ms, path=(EINV.V2 + "/paper/fv/work/numbers_n4.md")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["# Number macros, part n4 (Det, Att, Wt, Shift)", "",
             "Generated by `src/fv/num_n4.py`. `printed` is the LaTeX text of the macro; the check column says how "
             "it was verified and names every discrepancy (search DISCREPANCY).", "",
             "| name | printed | meaning | source :: key | entry | check |", "|---|---|---|---|---|---|"]
    for x in ms:
        src = os.path.basename(str(x["source"]).split(" ; ")[0]) + (" +" if " ; " in str(x["source"]) else "")
        key = str(x["key"]).replace("|", "/")
        chk = str(x["check"]).replace("|", "/")
        lines.append(f"| `\\{x['name']}` | `{x['text']}` | {x['meaning']} | `{src}` :: {key} | {x['entry']} | {chk} |")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    ms = macros()
    seen = set()
    for x in ms:
        assert x["name"] not in seen, x["name"]
        seen.add(x["name"])
        print(f"{x['name']:34s} {x['text']:28s} [E{x['entry']}] {x['check'][:70]}")
    print(len(ms), "macros;", write_md(ms))
