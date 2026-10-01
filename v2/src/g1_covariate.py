"""G1 read-out (RESULTS.md Entries 77, 89) - the inversion arms with the adaptation-strength covariate,
and what power the test actually had.

Entry 89 registered, before G1 landed, that its sign reading is reported beside a strength-adjusted
estimate: H1 showed that at 16000 steps the per-adapter interaction tracks how far an adapter trained as
well as it tracks body identity, and the two are collinear. If the inverted arms also differ in
adapted-weight norm, the same confound recurs and must be visible.

Also computes what the test could have resolved. The registered reading needed one-sided p < 0.05 on three
adapters per arm; this reports the standard error it achieved, the effect it was trying to detect (the
mirror of the unsuppressed +5.710e-5), and the adapters per arm that would have been needed.

CPU only, from the archived per-adapter output and training metadata. Writes out/g1_covariate.json.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2; T1 = os.path.join(V2, "out", "t1")
OUT = os.path.join(V2, "out", "g1_covariate.json")
CHAIN12 = os.path.join(V2, "out", "g_chain12.json")
R_REAL = 0.0356703416571125
UNSUPPRESSED = 5.710e-5          # the 16000-step theta_sym this test was built to mirror


def meta(tag):
    return json.load(open(os.path.join(T1, "adapters", tag, "train_meta.json")))


def main():
    g1 = json.load(open(CHAIN12))["G1"]
    A = np.array(g1["per_adapter_A"]); B = np.array(g1["per_adapter_B"])
    tags_A = [f"inv16k_A_s{s}" for s in range(len(A))]
    tags_B = [f"inv16k_B_s{s}" for s in range(len(B))]
    rows = []
    for tag, th in list(zip(tags_A, A)) + list(zip(tags_B, B)):
        m = meta(tag)
        rows.append({"arm": tag, "body": "A" if "_A_" in tag else "B", "theta": float(th),
                     "lora_B_norm": float(m["lora_B_norm"]), "loss_tail": float(m["loss_tail"]),
                     "minutes": float(m["minutes"]), "seed": int(m["seed"])})

    # does the confound recur? the arms' adaptation strengths, compared as in H1
    strength = {}
    for k in ("lora_B_norm", "loss_tail", "minutes"):
        a = np.array([r[k] for r in rows if r["body"] == "A"])
        b = np.array([r[k] for r in rows if r["body"] == "B"])
        t, p = stats.ttest_ind(a, b, equal_var=False)
        strength[k] = {"mean_A": float(a.mean()), "mean_B": float(b.mean()),
                       "welch_t": float(t), "p_two_sided": float(p)}
        print(f"[g1] {k}: A {a.mean():.4f} vs B {b.mean():.4f}  (p {p:.4f})", flush=True)

    y = np.array([r["theta"] for r in rows])
    x = np.array([r["lora_B_norm"] for r in rows])
    body = np.array([1.0 if r["body"] == "A" else 0.0 for r in rows])
    corr_sy = float(np.corrcoef(x, y)[0, 1]); corr_sb = float(np.corrcoef(x, body)[0, 1])
    print(f"[g1] corr(strength, theta) {corr_sy:+.3f}; corr(strength, body) {corr_sb:+.3f}", flush=True)

    # strength-adjusted symmetric estimate: remove the strength trend, recompute theta_sym on residuals
    X = np.column_stack([np.ones(len(y)), (x - x.mean()) / x.std(ddof=1)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta + beta[0]                    # keep the intercept, drop only the strength trend
    rA = resid[body == 1]; rB = resid[body == 0]
    vA, vB = rA.var(ddof=1) / len(rA), rB.var(ddof=1) / len(rB)
    th_adj = 0.5 * (rA.mean() + rB.mean()); se_adj = 0.5 * np.sqrt(vA + vB)
    df_adj = (vA + vB) ** 2 / (vA ** 2 / (len(rA) - 1) + vB ** 2 / (len(rB) - 1))
    p_adj_lt0 = float(stats.t.cdf(th_adj / se_adj, df_adj))

    # what the test could have resolved
    se = g1["welch_se"]; df = g1["welch_df"]
    t_need = stats.t.ppf(0.95, df)                    # one-sided 0.05
    target = -UNSUPPRESSED                            # the mirror of the unsuppressed effect
    se_need = abs(target) / t_need
    n_need = int(np.ceil(len(A) * (se / se_need) ** 2))
    achieved_power = float(stats.t.sf(t_need - abs(target) / se, df))
    print(f"[g1] achieved SE {se:.3e}; to resolve {target:+.3e} at one-sided 0.05 needs SE <= {se_need:.3e}",
          flush=True)
    print(f"[g1] that is about {n_need} adapters per arm against the {len(A)} run; "
          f"power as run was about {achieved_power:.2f}", flush=True)

    res = {"entry": "RESULTS.md Entries 77, 89 (G1)", "adapters": rows,
           "registered_reading": g1["reading"],
           "theta_sym": g1["theta_sym"], "theta_sym_pct": 100 * g1["theta_sym"] / R_REAL,
           "welch_se": se, "welch_df": df, "one_sided_p_lt0": g1["one_sided_p_lt0"],
           "stored_inverted_contrast": g1["stored_inverted_contrast"],
           "arm_strength_difference": strength,
           "corr_strength_theta": corr_sy, "corr_strength_body": corr_sb,
           "strength_adjusted": {"theta_sym": float(th_adj), "se": float(se_adj), "df": float(df_adj),
                                 "one_sided_p_lt0": p_adj_lt0,
                                 "theta_sym_pct": float(100 * th_adj / R_REAL)},
           "power": {"target_effect": target, "se_achieved": se, "se_required": float(se_need),
                     "n_per_arm_run": len(A), "n_per_arm_required": n_need,
                     "achieved_power_at_target": achieved_power},
           "unsuppressed_theta_sym": UNSUPPRESSED}
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"[g1] strength-adjusted theta_sym {th_adj:+.3e} (p<0 {p_adj_lt0:.3f}) "
          f"against unadjusted {g1['theta_sym']:+.3e} (p<0 {g1['one_sided_p_lt0']:.3f})", flush=True)


if __name__ == "__main__":
    main()
