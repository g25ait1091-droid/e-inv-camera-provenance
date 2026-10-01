"""Number macros, part n6: post-hoc quantities computed for the manuscript (RESULTS.md Entry 112).

Every value is READ at run time from out/fv_derived.json, which src/fv/fv_derived.py computes from the result
files (N-A1 ... N-A6 of OUTLINE.md section 9.1). None of these quantities was pre-specified: the TPR at the
calibrated limit and the G2 additive part are descriptive, the Hartung-Knapp interval is a small-sample
sensitivity beside the pre-specified DerSimonian-Laird limit, and the rest are counts.

The Q1 schedule macros (nLimSched*) are deliberately absent: the training records do not confirm the seeds 0-2
versus 3-11 learning-rate schedule split (out/fv_derived.json Q1_schedule), so no comparison by schedule exists.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_numlib import OUT, load, sig, dec, integer, macro  # noqa: E402

SRC = OUT + "/fv_derived.json"
E = "112"


def macros():
    D = load(SRC)
    assert D.get("entry") == "RESULTS.md Entry 112", D.get("entry")
    L = []

    def M(name, value, text, key, check):
        L.append(macro(name, value, text, SRC, key, E, check))

    # ---------------------------------------------------------------- N-A1: TPR at the calibrated limit
    a1 = D["N_A1_tpr_calibrated"]
    tc = a1["tpr_calibrated"]
    U = a1["inputs"]["lambda_U_calibrated_pct"]
    for k, nm in (("G_inf", "Inf"), ("G500", "FiveHundred"), ("G5000", "FiveThousand")):
        M(f"nAttTprTwo{nm}Cal", tc[k], dec(tc[k], 3), f"N_A1_tpr_calibrated.tpr_calibrated.{k}",
          f"two candidates, 1 % FPR, transfer at the calibrated limit ({U:.4f} % of R_real, H6 U_device); "
          f"verify_v2.tpr() imported; post hoc, descriptive; nominal-limit values reproduced "
          f"(nAttTprTwo{nm})")

    # ---------------------------------------------------------------- N-A2: Hartung-Knapp upper limit
    h = D["N_A2_hksj"]
    M("nLimPoolHksjUpper", h["upper99_pct"], sig(h["upper99_pct"], 2), "N_A2_hksj.upper99_pct",
      f"Hartung-Knapp-Sidik-Jonkman one-sided 99 % upper limit, % of each pair's R_real, over the three pairs "
      f"(t_0.99,{h['df']} = {h['t_0.99']:.2f}, q = {h['q_scale']:.3f}); DL {h['dl_upper99_pct_reproduced']:.4f} "
      f"reproduces nLimPoolUpper; post hoc sensitivity; CLAIMS T8 audit value about 0.18 %")

    # ---------------------------------------------------------------- N-A3: D200 four-design homogeneity
    d = D["N_A3_d200_combination"]
    M("nLimPoolDtwoHundredItwo", d["I2_pct"], sig(d["I2_pct"], 2), "N_A3_d200_combination.I2_pct",
      "I^2 (per cent) of the four D200 designs combined (m1_pooled.json secondary.d200_combination.I2); "
      "CLAIMS T8 'I2 0'")
    M("nLimPoolDtwoHundredQP", d["Q_p"], dec(d["Q_p"], 2), "N_A3_d200_combination.Q_p",
      f"Cochran Q p of the four D200 designs (Q = {d['Q']:.2f}, {d['Q_df']} df); CLAIMS T8 'Q p 0.51'")

    # ---------------------------------------------------------------- N-A4: 2000-step local arms
    n4 = D["N_A4_dose2000"]
    M("nDoseTwoThousandImages", n4["total_gens"], integer(n4["total_gens"]), "N_A4_dose2000.total_gens",
      "generations scored in the 2000-step local arms (nomark s0-2, nomarkB s0-2); a subset of the G2 design")
    M("nDoseTwoThousandGensPerAdapter", n4["gens_per_adapter"], integer(n4["gens_per_adapter"]),
      "N_A4_dose2000.gens_per_adapter", "scored rows per adapter (summary_nomark*.json arms.*.n) = folder counts")
    M("nDoseTwoThousandAdaptersPerBody", n4["adapters_per_body"], integer(n4["adapters_per_body"]),
      "N_A4_dose2000.adapters_per_body", "len(dose_stats.json doses.2000.A_arms) = len(B_arms)")

    # ---------------------------------------------------------------- N-A5: study totals
    n5 = D["N_A5_totals"]
    M("nDataTotalAdapters", n5["total_adapters"], integer(n5["total_adapters"]), "N_A5_totals.total_adapters",
      "adapters trained over the designs of Table 2 block B (re-used adapters counted once); excludes the "
      "objective arms, the designed-mark ladder and the environment chain (listed in not_counted)")
    M("nDataTotalGens", n5["total_gens"], integer(n5["total_gens"]), "N_A5_totals.total_gens",
      "generations scored over the designs of Table 2 block B, each image counted once (rescored images and "
      "subsets not added)")

    # ---------------------------------------------------------------- N-A6: G2 additive part
    n6 = D["N_A6_g2_additive"]
    M("nGenGtwoAdditivePct", n6["additive_pct_of_R_real"], sig(n6["additive_pct_of_R_real"], 2),
      "N_A6_g2_additive.additive_pct_of_R_real",
      f"G2 additive part (mean_A - mean_B)/2 = {n6['additive_part']:.3e} as % of R_real (estimates b_A - b_B); "
      f"CLAIMS N5 '0.077 %'; descriptive")
    return L
