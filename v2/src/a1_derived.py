"""Entry 42 (secondary, derived) — does the autoencoder's band transmission predict the measured
Stage-1 retention of the natural fingerprint (eta = 0.3661)?
Prediction = sum over bands of (energy fraction of K_A in band b) x T_vae(b); the energy beyond radial
0.5 cycles/px (diagonal corners, not covered by any band) is bracketed between 0 and T_vae(b0).
Reads out/t1/band_fields.json, band_vae.json, periodic_vae.json; writes out/t1/a1_derived.json."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import json, os
T1 = os.path.join(EINV.V2, 'out', 't1')
f = json.load(open(os.path.join(T1, "band_fields.json"))); v = json.load(open(os.path.join(T1, "band_vae.json")))
p = json.load(open(os.path.join(T1, "periodic_vae.json")))
T = [v["T_vae"][f"band{b}"] for b in range(6)]
out = {"T_vae_by_band": T, "T_vae_periodic": p["T_vae"], "eta_measured": 0.36611}
for name, key in (("K_A_E2", "energy_fraction_K_A_E2"), ("DiffusionShield", "energy_fraction_DiffusionShield_lum")):
    e = f[key]; inband = sum(e); corner = 1.0 - inband
    base = sum(ei * ti for ei, ti in zip(e, T))
    out[name] = {"energy_in_bands": inband, "energy_corners": corner, "predicted_retention_low": base,
                 "predicted_retention_high": base + corner * T[0]}
json.dump(out, open(os.path.join(T1, "a1_derived.json"), "w"), indent=1); print(json.dumps(out, indent=1))
