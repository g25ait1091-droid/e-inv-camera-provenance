"""H1 (RESULTS.md Entry 88) - does the 16000-step interaction track how hard each adapter trained?

The one resolved positive in this study is carried by one of the two bodies, and arm A's adapters have
larger adapted weights than arm B's. That is a candidate explanation for the whole effect that has nothing
to do with the camera: if adapters that train harder show a larger interaction, then a body whose adapters
happened to train harder will look like it transfers.

Twelve adapters, each with a per-adapter contrast already measured and its own training record. The
contrast is regressed on adaptation strength and a body indicator by ordinary least squares, with the
pre-specified readings of Entry 88. Nothing here is re-measured from images: this is the archived
per-adapter output read against the archived training metadata. Writes out/h1_strength.json.
"""
import os, sys, json, itertools
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2; T1 = os.path.join(V2, "out", "t1")
OUT = os.path.join(V2, "out", "h1_strength.json")
SUMS = ("summary_dose16k", "summary_dose16krep", "summary_dose16krep2")
R_REAL = 0.0356703


def ols(X, y):
    """Least squares with t statistics; X includes its own intercept column."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = n - k
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    return beta, se, t, p, dof


def main():
    # per-adapter contrasts, and the training record of the same adapter
    arms = {}
    for f in SUMS:
        arms.update(json.load(open(os.path.join(T1, f + ".json")))["arms"])
    rows = []
    for tag, v in sorted(arms.items()):
        meta_p = os.path.join(T1, "adapters", tag, "train_meta.json")
        if not os.path.exists(meta_p):
            print(f"[h1] no train_meta for {tag}, skipped", flush=True); continue
        m = json.load(open(meta_p))
        # the statistic that produces the 16000-step result, oriented own-minus-other per body exactly as
        # dose_stats.py does: nat(x) for arm A, -nat(x) for arm B. contrast_M is a different quantity.
        body = "A" if "_A_" in tag else "B"
        nat = float(v["natural_paired_KA_minus_KB"])
        rows.append({"arm": tag, "body": body,
                     "theta": nat if body == "A" else -nat,
                     "n_images": int(v["n"]), "lora_B_norm": float(m["lora_B_norm"]),
                     "loss_tail": float(m["loss_tail"]), "minutes": float(m["minutes"]),
                     "seed": int(m["seed"])})
    assert rows, "no 16000-step adapters with training metadata"
    y = np.array([r["theta"] for r in rows])
    body = np.array([1.0 if r["body"] == "A" else 0.0 for r in rows])  # arm indicator
    n = len(rows)
    print(f"[h1] {n} adapters: {sum(body):.0f} in arm A, {n - sum(body):.0f} in arm B", flush=True)

    # how different are the two arms' adaptation strengths in the first place
    strength_gap = {}
    for k in ("lora_B_norm", "loss_tail", "minutes"):
        a = np.array([r[k] for r in rows if r["body"] == "A"])
        b = np.array([r[k] for r in rows if r["body"] == "B"])
        t, p = stats.ttest_ind(a, b, equal_var=False)
        strength_gap[k] = {"mean_A": float(a.mean()), "mean_B": float(b.mean()),
                           "welch_t": float(t), "p_two_sided": float(p)}
        print(f"[h1] {k}: A {a.mean():.4f} vs B {b.mean():.4f}  (p {p:.4f})", flush=True)

    models = {}
    for k in ("lora_B_norm", "loss_tail"):
        x = np.array([r[k] for r in rows])
        xc = (x - x.mean()) / x.std(ddof=1)          # standardised, so the slope is per SD of strength
        # body alone, strength alone, and both together
        fits = {}
        for name, X in (("body_only", np.column_stack([np.ones(n), body])),
                        ("strength_only", np.column_stack([np.ones(n), xc])),
                        ("body_plus_strength", np.column_stack([np.ones(n), body, xc]))):
            beta, se, t, p, dof = ols(X, y)
            names = ["intercept"] + (["body_A"] if "body" in name else []) + (["strength"] if "strength" in name else [])
            fits[name] = {nm: {"coef": float(b), "se": float(s), "t": float(tt), "p": float(pp)}
                          for nm, b, s, tt, pp in zip(names, beta, se, t, p)}
            fits[name]["dof"] = dof
            resid = y - X @ beta
            fits[name]["r2"] = float(1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean())))
        models[k] = {"fits": fits,
                     "corr_strength_theta": float(np.corrcoef(x, y)[0, 1]),
                     "corr_strength_body": float(np.corrcoef(x, body)[0, 1])}
        joint = fits["body_plus_strength"]
        print(f"[h1] {k}: corr(strength, theta) {models[k]['corr_strength_theta']:+.3f}; "
              f"adjusted body {joint['body_A']['coef']:+.3e} (p {joint['body_A']['p']:.3f}), "
              f"strength {joint['strength']['coef']:+.3e} (p {joint['strength']['p']:.3f})", flush=True)

    # pre-specified reading, on the primary strength measure
    j = models["lora_B_norm"]["fits"]["body_plus_strength"]
    b_only = models["lora_B_norm"]["fits"]["body_only"]["body_A"]
    strength_resolved = j["strength"]["p"] < 0.05
    body_resolved_adj = j["body_A"]["p"] < 0.05
    if strength_resolved and not body_resolved_adj:
        reading = ("the 16000-step interaction tracks adaptation strength, not body identity; "
                   "G1's inversion must be read alongside this")
    elif body_resolved_adj and not strength_resolved:
        reading = ("adaptation strength does not explain it; the fingerprint, training-set content and "
                   "training-environment explanations stand and G1 discriminates")
    else:
        reading = "inconclusive: both coefficients reported, neither explanation promoted"
    print(f"[h1] body term unadjusted p {b_only['p']:.3f} -> adjusted p {j['body_A']['p']:.3f}", flush=True)
    print(f"[h1] READING: {reading}", flush=True)

    res = {"entry": "RESULTS.md Entry 88 (H1)", "n_adapters": n, "R_real": R_REAL,
           "adapters": rows, "arm_strength_difference": strength_gap, "models": models,
           "reading": reading,
           "caveat": ("twelve adapters give this little power; it is a covariate check, not a test that "
                      "clears or convicts on its own")}
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"[h1] written {OUT}", flush=True)


if __name__ == "__main__":
    main()
