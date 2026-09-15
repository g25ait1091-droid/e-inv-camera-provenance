"""Entry 46 (secondary, derived): combine the full-pipeline band transmission (Entry 37 A2) with each
pattern's measured spectrum to predict what a NON-repeating pattern with that spectrum would transmit.
prediction = sum_b e_b * T_full(b); SE propagated as sqrt(sum_b e_b^2 SE_b^2); energy beyond radial
0.5 cycles/px (corners) is given T_full(b0). Compared with the measured transmission of the repeating
tiles (Entries 44-45) and with the natural fingerprint's point estimate and upper limit.
Reads band_summary.json, band_fields.json, periodic_fields.json, periodic_summary_final.json.
Writes out/t1/band_derived.json."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import json, math
T1 = os.path.join(EINV.V2, 'out', 't1')
B = json.load(open(f"{T1}/band_summary.json")); F = json.load(open(f"{T1}/band_fields.json"))
PF = json.load(open(f"{T1}/periodic_fields.json")); PS = json.load(open(f"{T1}/periodic_summary_final.json"))["fields"]
T = [c["T_full"] for c in B["curve"]]; SE = [c["T_full_se"] for c in B["curve"]]
def predict(e):
    e = list(e); corner = max(0.0, 1.0 - sum(e)); e0 = e[:]; e0[0] = e0[0] + corner
    p = sum(ei * ti for ei, ti in zip(e0, T)); s = math.sqrt(sum((ei * si) ** 2 for ei, si in zip(e0, SE)))
    return {"energy_by_band": e, "corner_energy": corner, "predicted_T_nonrepeating": p, "predicted_se": s,
            "predicted_pct": 100 * p, "predicted_hi2se_pct": 100 * (p + 2 * s)}
out = {"T_full_by_band": T, "T_full_se_by_band": SE,
       "K_A_fingerprint": predict(F["energy_fraction_K_A_E2"]),
       "DiffusionShield": predict(F["energy_fraction_DiffusionShield_lum"]),
       "per32": predict(PF["per32"]["energy_fraction_by_band_b0_b5"]),
       "per36": predict(PF["per36"]["energy_fraction_by_band_b0_b5"])}
for k in ("per32", "per36"):
    m = PS[k]["lambda_pct"] / 100.0; out[k]["measured_T"] = m
    out[k]["measured_over_predicted"] = m / out[k]["predicted_T_nonrepeating"] if out[k]["predicted_T_nonrepeating"] > 0 else None
    lo = out[k]["predicted_T_nonrepeating"] + 2 * out[k]["predicted_se"]
    out[k]["measured_over_predicted_hi2se"] = m / lo
out["DiffusionShield"]["measured_T"] = 0.032705; out["DiffusionShield"]["measured_over_predicted"] = 0.032705 / out["DiffusionShield"]["predicted_T_nonrepeating"]
out["natural_fingerprint_reference"] = {"lambda_hat_pct": 0.012894, "lambda_U_pct": 0.15072}
json.dump(out, open(f"{T1}/band_derived.json", "w"), indent=1)
for k in ("K_A_fingerprint", "DiffusionShield", "per32", "per36"):
    v = out[k]; print(k, {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "energy_by_band"})
