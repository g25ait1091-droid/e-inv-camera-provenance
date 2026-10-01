"""M1 (RESULTS.md Entry 106) - is there a small transfer common to the device pairs that no single design resolves?

Every paired design at the primary dose (2000 steps), standard training, natural-image fingerprint estimates,
each on its own scale: lambda = theta_sym / R_real and SE = Welch SE / R_real, in per cent.

  primary set    one design per independent device pair: Nikon D200 primary (12/arm, ledger), iPhone 5c
                 (G6, 12/arm, E2), Huawei P20 (G4b, 12/arm)
  secondary set  the D200 pair's four designs (primary, second environment G2, second training set G5,
                 FLUX.1-dev) combined by inverse variance first - an optimistic SE, since they share the
                 fingerprint estimates - then pooled with the iPhone and P20 pairs
  sensitivities  (i) the iPhone pair with its flat-field estimate; (ii) every variance x 1/(1 - 0.432), the
                 share of variance fingerprint estimation contributed in H2 (Entry 90)

Fixed-effect inverse-variance pooling with a normal reference (the per-design variances are treated as
known), DerSimonian-Laird random effects, Cochran's Q and I^2; the random-effects estimate is quoted when
I^2 > 50 %. Readings exactly as Entry 106. Writes out/m1_pooled.json.

Registered after the individual estimates were seen (stated in Entry 106); the design set is fixed by rule.
"""
import os, sys, json
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2; OUT_DIR = os.path.join(V2, "out")
OUT = os.path.join(OUT_DIR, "m1_pooled.json")
LEDGER = os.path.join(EINV.REPO, 'analysis', 'FINAL_LEDGER.json')
EST_SHARE = 0.432          # H2, Entry 90
Z99 = stats.norm.ppf(0.99)


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    vA, vB = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return 0.5 * (a.mean() + b.mean()), 0.5 * np.sqrt(vA + vB)


def load(name):
    return json.load(open(os.path.join(OUT_DIR, name)))


def designs():
    led = json.load(open(LEDGER))
    R_d200 = led["denominators"]["R_real"]
    th, se = welch(led["primary"]["per_adapter_A"], led["primary"]["per_adapter_B"])
    assert abs(th - led["primary"]["theta_sym"]) < 1e-9, "ledger theta_sym not reproduced"
    d = {"D200 primary": {"pair": "Nikon D200", "theta": th, "se": se, "R": R_d200, "n": 12,
                          "source": "FINAL_LEDGER.json primary"}}
    g6 = load("g6_p5c.json")["estimators"]
    for tag, key in (("E2", "iPhone 5c (E2)"), ("FLAT", "iPhone 5c (FLAT)")):
        s = g6[tag]["symmetric"]
        d[key] = {"pair": "Apple iPhone 5c", "theta": s["theta_sym"], "se": s["welch_se"],
                  "R": g6[tag]["R_real"], "n": 12, "source": f"g6_p5c.json {tag}"}
    g4b = load("g4b_p20.json")
    d["P20 (G4b)"] = {"pair": "Huawei P20", "theta": g4b["symmetric"]["theta_sym"],
                      "se": g4b["symmetric"]["welch_se"], "R": g4b["R_real"], "n": 12, "source": "g4b_p20.json"}
    g2 = load("g2_pooled_six.json")
    d["D200 second environment (G2)"] = {"pair": "Nikon D200", "theta": g2["theta_sym"], "se": g2["welch_se"],
                                         "R": R_d200, "n": 6, "source": "g2_pooled_six.json"}
    g5 = load("g5_alt_training.json")
    d["D200 second training set (G5)"] = {"pair": "Nikon D200", "theta": g5["alt"]["theta_sym"],
                                          "se": g5["alt"]["welch_se"], "R": g5["R_real"], "n": 3,
                                          "source": "g5_alt_training.json"}
    fx = load("flux_seed_ext_summary.json")
    d["D200 FLUX.1-dev"] = {"pair": "Nikon D200", "theta": fx["symmetric"]["theta_sym"],
                            "se": fx["symmetric"]["welch_se"], "R": fx["R_real"], "n": 6,
                            "source": "flux_seed_ext_summary.json"}
    for v in d.values():
        v["lambda_pct"] = 100 * v["theta"] / v["R"]
        v["se_pct"] = 100 * v["se"] / v["R"]
    return d


def pool(lam, se, labels):
    lam, se = np.asarray(lam, float), np.asarray(se, float)
    w = 1 / se ** 2
    fe = float((w * lam).sum() / w.sum()); fe_se = float(1 / np.sqrt(w.sum()))
    k = len(lam)
    Q = float((w * (lam - fe) ** 2).sum()); dfq = k - 1
    C = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (Q - dfq) / C) if C > 0 else 0.0
    I2 = max(0.0, (Q - dfq) / Q) if Q > 0 else 0.0
    ws = 1 / (se ** 2 + tau2)
    re = float((ws * lam).sum() / ws.sum()); re_se = float(1 / np.sqrt(ws.sum()))
    out = {"designs": labels, "k": k,
           "fixed": {"lambda_pct": fe, "se_pct": fe_se, "z": fe / fe_se,
                     "one_sided_p": float(stats.norm.sf(fe / fe_se)), "upper99_pct": fe + Z99 * fe_se},
           "random": {"lambda_pct": re, "se_pct": re_se, "z": re / re_se, "tau2": tau2,
                      "one_sided_p": float(stats.norm.sf(re / re_se)), "upper99_pct": re + Z99 * re_se},
           "Q": Q, "Q_df": dfq, "Q_p": float(stats.chi2.sf(Q, dfq)) if dfq > 0 else None, "I2": I2}
    out["quoted"] = "random" if I2 > 0.5 else "fixed"
    return out


def main():
    d = designs()
    for k, v in d.items():
        print(f"[m1] {k:32s} n {v['n']:2d}  lambda {v['lambda_pct']:+.4f} %  SE {v['se_pct']:.4f} %", flush=True)

    prim = ["D200 primary", "iPhone 5c (E2)", "P20 (G4b)"]
    primary = pool([d[x]["lambda_pct"] for x in prim], [d[x]["se_pct"] for x in prim], prim)

    # secondary: combine the four D200 designs first (optimistic SE: they share K)
    d200 = ["D200 primary", "D200 second environment (G2)", "D200 second training set (G5)", "D200 FLUX.1-dev"]
    c = pool([d[x]["lambda_pct"] for x in d200], [d[x]["se_pct"] for x in d200], d200)
    comb = {"lambda_pct": c["fixed"]["lambda_pct"], "se_pct": c["fixed"]["se_pct"]}
    secondary = pool([comb["lambda_pct"], d["iPhone 5c (E2)"]["lambda_pct"], d["P20 (G4b)"]["lambda_pct"]],
                     [comb["se_pct"], d["iPhone 5c (E2)"]["se_pct"], d["P20 (G4b)"]["se_pct"]],
                     ["D200 (four designs combined)", "iPhone 5c (E2)", "P20 (G4b)"])
    secondary["d200_combination"] = c

    s1 = ["D200 primary", "iPhone 5c (FLAT)", "P20 (G4b)"]
    sens_flat = pool([d[x]["lambda_pct"] for x in s1], [d[x]["se_pct"] for x in s1], s1)
    infl = 1 / np.sqrt(1 - EST_SHARE)
    sens_infl = pool([d[x]["lambda_pct"] for x in prim], [d[x]["se_pct"] * infl for x in prim], prim)

    q = primary["quoted"]
    pp = primary[q]["one_sided_p"]; lam = primary[q]["lambda_pct"]
    pinf = sens_infl[sens_infl["quoted"]]["one_sided_p"]
    if lam > 0 and pp < 0.01 and pinf < 0.05:
        reading = (f"a small transfer common to the device pairs, lambda of about {lam:.3f} %; "
                   f"at most {primary[q]['upper99_pct']:.3f} % (one-sided 99 %)")
    elif pp >= 0.01:
        reading = (f"no transfer common to the device pairs; pooled one-sided 99 % limit "
                   f"{primary[q]['upper99_pct']:.3f} %")
    else:
        reading = "a pooled lean that does not survive fingerprint-estimation error"

    res = {"entry": "RESULTS.md Entry 106 (M1)", "per_design": d, "primary": primary, "secondary": secondary,
           "sensitivity_flat_field": sens_flat, "sensitivity_estimation_inflated": sens_infl,
           "variance_inflation": float(infl ** 2), "reading": reading,
           "note": ("fixed-effect pooling treats each design's Welch variance as known and uses a normal "
                    "reference; designs with few adapters (G5, n = 3) have poorly estimated variances")}
    json.dump(res, open(OUT, "w"), indent=2)
    for lab, r in (("primary (3 pairs)", primary), ("secondary (D200 combined)", secondary),
                   ("sens: iPhone flat-field", sens_flat), ("sens: estimation-inflated", sens_infl)):
        f, rr = r["fixed"], r["random"]
        print(f"[m1] {lab:28s} fixed {f['lambda_pct']:+.4f} % (SE {f['se_pct']:.4f}, p {f['one_sided_p']:.4f}, "
              f"U99 {f['upper99_pct']:.4f} %) | random {rr['lambda_pct']:+.4f} % (p {rr['one_sided_p']:.4f}) | "
              f"I2 {100 * r['I2']:.0f} % Q p {r['Q_p']:.3f} -> quoted {r['quoted']}", flush=True)
    print(f"[m1] READING: {reading}", flush=True)


if __name__ == "__main__":
    main()
