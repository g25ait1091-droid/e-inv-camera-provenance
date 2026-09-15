"""Tier 3 (v4) — what the bound means operationally, with the measured image-level SE.

Change from t3_power.py (review follow-up, Entry 50): SE_img(G=500) is the MEASURED standard error of
the 500-image paired-contrast mean on the re-measured primary adapters (out/t1/measure_rows.csv:
A_raw_s0 7.165e-05, B_raw_s0 6.905e-05, mean 7.035e-05), replacing 4.67e-05, which did not match the
per-image rows. Output: out/t3_power_v4.json.


Question: if device transfer sat exactly at the v1 upper limit, how many generated images from
one adapter would an examiner need to attribute that adapter to its training camera, and what
true-positive rate is attainable at a fixed false-positive rate?

Inputs (all frozen v1 quantities; none re-estimated here):
  U_device      5.3761e-05   FINAL_LEDGER primary.U_device  — the largest transfer consistent with the data
  R_real        3.56703e-02  FINAL_LEDGER denominators.R_real
  SE_img(G=500) 4.67e-05     E_INV_RESULTS_v2 §8.3 (sanity block): image-noise SE of a 500-image arm mean
  sigma_mu      5.23e-05     E_INV_RESULTS_v2 §8.3: persistent per-adapter structured component (does NOT shrink with G)
  inflation     1.50         E_INV_RESULTS_v2 §8.3: measured dependence inflation over 1/sqrt(mG)

Model. For one adapter, the calibrated paired contrast over G images is
   theta_hat ~ N(theta_true, sigma_mu^2 + (SE_img(500) * sqrt(500/G))^2)
with theta_true = lambda * R_real. A two-candidate attribution (own body vs the other body of
the same model) declares "own" when theta_hat exceeds a threshold set for FPR = 1 % under
theta_true = 0. For M candidates, the FPR is per wrong candidate and the threshold is
Bonferroni-adjusted to 1 %/(M-1). This is the most favourable case for the examiner: fingerprints
of every candidate known exactly, main effects perfectly calibrated out, no scene confound.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import json, math, os
import numpy as np
from scipy.stats import norm

OUT = os.path.join(EINV.V2, 'out')
U_DEVICE, R_REAL = 5.3761e-05, 3.56703e-02
SE_IMG_500, SIGMA_MU, INFL = 7.035e-05, 5.23e-05, 1.50
SE_IMG_500 *= 1.0   # inflation is already in the measured SE (it is an empirical SE); INFL kept for the analytic path

def tpr(theta_true, G, M, fpr=0.01, sigma_mu=None):
    sm = SIGMA_MU if sigma_mu is None else sigma_mu
    se = math.sqrt(sm**2 + (SE_IMG_500 * math.sqrt(500.0 / G))**2)
    z = norm.ppf(1 - fpr / max(M - 1, 1))
    return float(1 - norm.cdf(z - theta_true / se))

def images_needed(theta_true, M, target_tpr=0.5, fpr=0.01, sigma_mu=None):
    for G in [10, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 100000, 1000000]:
        if tpr(theta_true, G, M, fpr, sigma_mu) >= target_tpr: return G
    return None

rows = []
for mult, label in [(1, "at the upper limit U_device"), (10, "10x the upper limit"), (100, "100x the upper limit")]:
    th = mult * U_DEVICE
    for M in (2, 5, 50):
        rows.append({"transfer": label, "lambda_pct": 100 * th / R_REAL, "theta_true": th, "M_candidates": M,
                     "TPR_G500": tpr(th, 500, M), "TPR_G5000": tpr(th, 5000, M), "TPR_G_inf": tpr(th, 10**9, M),
                     "G_for_TPR50": images_needed(th, M)})
asym = {"asymptotic_TPR_M2_at_U": tpr(U_DEVICE, 10**9, 2),
        "signal_to_structured_noise_at_U": U_DEVICE / SIGMA_MU,
        "note": "sigma_mu is the per-adapter structured component that does not average out with more images; at the upper limit the signal is about one sigma_mu."}
# Bracketing case: the persistent component measured on the pooled correlation surface may
# partly cancel in the paired contrast. sigma_mu = 0 is the most favourable limit for an
# examiner: only image noise remains and it averages out with G.
rows0 = []
for M in (2, 5, 50):
    rows0.append({"transfer": "at the upper limit U_device, sigma_mu = 0", "M_candidates": M,
                  "TPR_G500": tpr(U_DEVICE, 500, M, sigma_mu=0.0), "TPR_G5000": tpr(U_DEVICE, 5000, M, sigma_mu=0.0),
                  "G_for_TPR50": images_needed(U_DEVICE, M, sigma_mu=0.0),
                  "G_for_TPR90": images_needed(U_DEVICE, M, target_tpr=0.9, sigma_mu=0.0)})
res = {"inputs": {"U_device": U_DEVICE, "R_real": R_REAL, "SE_img_500": SE_IMG_500, "sigma_mu": SIGMA_MU, "inflation_measured": INFL},
       "rows": rows, "asymptotics": asym, "rows_sigma_mu_zero": rows0}
json.dump(res, open(os.path.join(OUT, "t3_power_v4.json"), "w"), indent=1)

print(f"{'transfer':28s} {'lambda%':>8s} {'M':>3s} {'TPR@G=500':>10s} {'TPR@G=5000':>11s} {'TPR@G=inf':>10s} {'G for TPR 0.5':>14s}")
for r in rows:
    print(f"{r['transfer']:28s} {r['lambda_pct']:8.3f} {r['M_candidates']:3d} {r['TPR_G500']:10.3f} {r['TPR_G5000']:11.3f} {r['TPR_G_inf']:10.3f} {str(r['G_for_TPR50']):>14s}")
print("\nasymptotics:", json.dumps(asym, indent=1))
