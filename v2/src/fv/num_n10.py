"""Number macros, part n10: the post-hoc sensitivities of RESULTS.md Entries 118-119 (fourth review round).

Every value is READ at run time from out/fv_r4_sens.json (src/fv/fv_r4_sens.py). Both items are post hoc and
descriptive; neither changes the bound of record (nLimRec) or any registered reading.
  A  (hostile-r4-02) the calibration multiplier and the limit if the training-set component were twice its estimate or
     at its chi-square(2) one-sided 95 % upper confidence limit, with and without the seed-bank term;
  B  (hostile-r4-01) one body's transfer under the max-arm limit: eq. (power) at twice the nominal limit and twice the
     bound of record (the limit bounds the two bodies' average; one body is bounded only by 2U).
Also (numbers-r4-04) values already in out/fv_closedset_expect_run2.json (Entry 117 item 5) that the review asked to be
printed: the device-level floor 'at least 0.95', the sigma_mu = 4.1e-05 column and the P20 low/mid device-level row.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_numlib import OUT, sig, dec, sci, integer, macro, load  # noqa: E402

E = "119"
F = OUT + "/fv_r4_sens.json"
POST = "post hoc, not pre-specified (Entry 118)"


def g2(x):
    return integer(float(f"{x:.2g}"))


def macros():
    L = []

    def M(name, value, text, key, check):
        L.append(macro(name, value, text, F, key, E, check))

    R = load(F)
    assert R["entry"].startswith("RESULTS.md Entry 118")
    A = R["A_training_sd_sensitivity"]
    rows = A["rows"]
    one, two = rows["x1 (measured)"], rows["x2"]
    up = rows["x f95 (chi2(2) one-sided 95 % upper limit)"]
    assert one["with_seed_bank_term"]["c_star"] == 1.31 and one["without_seed_bank_term"]["c_star"] == 1.25
    M("nLimTrainSensFactor", A["f95"], sig(A["f95"], 2), "A_training_sd_sensitivity.f95",
      f"{POST}: sqrt(2 / chi2.ppf(0.05, 2)), the one-sided 95 % upper confidence factor of an SD on two df")
    for nm, r in (("Two", two), ("Upper", up)):
        w, wo = r["with_seed_bank_term"], r["without_seed_bank_term"]
        assert w["c_star"] is not None and wo["c_star"] is not None
        base = f"A_training_sd_sensitivity.rows[{'x2' if nm == 'Two' else 'x f95'}]"
        what = "twice its estimate" if nm == "Two" else "its chi-square(2) 95 % upper confidence limit"
        M(f"nLimTrainSens{nm}C", w["c_star"], dec(w["c_star"], 2), base + ".with_seed_bank_term.c_star",
          f"{POST}: smallest c with worst-cell coverage >= 0.99, training-set SD at {what}, seed-bank term included "
          f"(as the bound of record)")
        M(f"nLimTrainSens{nm}", w["limit_at_c_star"]["lambda_U_pct_of_R_real"],
          sig(w["limit_at_c_star"]["lambda_U_pct_of_R_real"], 3),
          base + ".with_seed_bank_term.limit_at_c_star.lambda_U_pct_of_R_real",
          f"{POST}: max-arm limit at that c, % of R_real (arm {w['limit_at_c_star']['binding_arm']} binds)")
        M(f"nLimTrainSens{nm}CovAtOne", w["worst_coverage_at_1"], dec(w["worst_coverage_at_1"], 3),
          base + ".with_seed_bank_term.worst_coverage_at_1", f"{POST}: worst-cell coverage at c = 1, with the term")
        M(f"nLimTrainSens{nm}CovAtC", w["worst_coverage_at_c_star"], dec(w["worst_coverage_at_c_star"], 3),
          base + ".with_seed_bank_term.worst_coverage_at_c_star", f"{POST}: worst-cell coverage at that c")
        M(f"nLimTrainSens{nm}CNoBank", wo["c_star"], dec(wo["c_star"], 2), base + ".without_seed_bank_term.c_star",
          f"{POST}: as nLimTrainSens{nm}C without the seed-bank term (H6's model)")
        M(f"nLimTrainSens{nm}NoBank", wo["limit_at_c_star"]["lambda_U_pct_of_R_real"],
          sig(wo["limit_at_c_star"]["lambda_U_pct_of_R_real"], 3),
          base + ".without_seed_bank_term.limit_at_c_star.lambda_U_pct_of_R_real",
          f"{POST}: limit at that c without the term, % of R_real")

    B = R["B_one_body_worst_case"]["limits"]
    for nm, k in (("Nom", "nominal"), ("Rec", "bound_of_record")):
        d = B[k]["at_2U"]
        M(f"nAttOneBodyPct{nm}", d["theta_pct_of_R_real"], sig(d["theta_pct_of_R_real"], 3),
          f"B_one_body_worst_case.limits.{k}.at_2U.theta_pct_of_R_real",
          f"{POST}: twice the {k.replace('_', ' ')} limit, % of R_real: the most one body's transfer can be when the "
          f"other body's is non-negative (the max-arm limit bounds the two bodies' average)")
        M(f"nAttOneBodyTpr{nm}", d["TPR_G500"], dec(d["TPR_G500"], 2),
          f"B_one_body_worst_case.limits.{k}.at_2U.TPR_G500",
          f"{POST}: eq. (power) TPR, 500 images, two candidates, FPR 0.01, sigma_mu = 0, transfer at that 2U")
        M(f"nAttOneBodyG{nm}", d["G_for_TPR50"], g2(d["G_for_TPR50"]),
          f"B_one_body_worst_case.limits.{k}.at_2U.G_for_TPR50",
          f"{POST}: exact images for TPR 0.5 at 2U (sigma_mu = 0, two candidates); two significant figures")
        av = B[k]["two_body_average_TPR_G500_one_at_2U_other_0"]
        M(f"nAttOneBodyAvgTpr{nm}", av, dec(av, 2),
          f"B_one_body_worst_case.limits.{k}.two_body_average_TPR_G500_one_at_2U_other_0",
          f"{POST}: TPR at 500 images averaged over the two bodies when one carries 2U and the other none")

    # ---- closed-set yardstick (Entry 117 item 5, out/fv_closedset_expect_run2.json): values already in the file that
    # the fourth review round asked to be printed (numbers-r4-04). Entry "117".
    CS = OUT + "/fv_closedset_expect_run2.json"
    C = load(CS)
    st = {(r["group"], r["level"]): r for r in C["summary_table"]}
    sm = C["settings"]["sigma_mu_values"][1]

    def MC(name, value, text, key, check):
        L.append(macro(name, value, text, CS, key, "117", check))

    dev = [st[(g, "a_device_limit")] for g in ("kodak", "p20")]
    meth = ("bootstrap_two_stage", "parametric_sigma_mu_0", f"parametric_sigma_mu_{sm:.3e}")
    for r in dev:
        assert all(m in r for m in meth), list(r)
    floor = min(r[m]["E_accuracy"] for r in dev for m in meth)
    MC("nAttCsDevEFloor", floor, dec(floor, 2),
       "min over summary_table[group in (kodak, p20), level=a_device_limit] of E_accuracy (bootstrap, parametric at "
       "sigma_mu 0 and 4.108e-05)",
       "the register's floor 'at least 0.95' (Entry 117 item 5 correction 1): lowest expected accuracy at the device-"
       "level limits over both fingerprint scorings, both methods and both persistent terms (Kodak, parametric, "
       "sigma_mu 4.1e-05: 0.954)")
    for g, nm in (("kodak", "Kodak"), ("p20", "Ptwenty")):
        r = st[(g, "a_device_limit")][meth[2]]
        MC(f"nAttCsDevEParMu{nm}", r["E_accuracy"], dec(r["E_accuracy"], 3),
           f"summary_table[group={g}, level=a_device_limit].{meth[2]}.E_accuracy",
           "parametric model at sigma_mu = 4.1e-05 on each of the five scores (a paired sigma_mu of about 5.8e-05, "
           "conservative; Entry 117 item 5 correction 3)")
    MC("nAttCsSigmaPairedEquiv", math.sqrt(2) * sm, sci(math.sqrt(2) * sm, 2), "sqrt(2) * settings.sigma_mu_values[1]",
       "sigma_mu^2 I on each of five scores gives each two-candidate margin 2 sigma_mu^2 (Entry 117 item 5 "
       "correction 3)")
    lm = st[("p20_lowmid", "a_device_limit")]
    assert lm["observed_correct"] == 4 and "not pre-specified" in lm["status"]
    for nm, m in (("", "bootstrap_two_stage"), ("Par", "parametric_sigma_mu_0")):
        MC(f"nAttCsLowmidDevE{nm}", lm[m]["E_accuracy"], dec(lm[m]["E_accuracy"], 2),
           f"summary_table[group=p20_lowmid, level=a_device_limit].{m}.E_accuracy",
           "P20 low/mid representation (S6 supplement, not pre-specified), device-level limit, expected accuracy")
        MC(f"nAttCsLowmidDevLeFour{nm}", lm[m]["P_X_le_observed"], sig(lm[m]["P_X_le_observed"], 2),
           f"summary_table[group=p20_lowmid, level=a_device_limit].{m}.P_X_le_observed",
           "P20 low/mid, device-level limit: probability of the observed four or fewer correct")
    return L


if __name__ == "__main__":
    for m in macros():
        print(f"{m['name']:34s} {m['text']}")
