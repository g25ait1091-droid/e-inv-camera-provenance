"""G1 extension readings (RESULTS.md Entries 77, 89, 99) - eight inverted adapters per arm at 16000 steps.

The inverted arms (training crops carrying each body's own fingerprint with its sign reversed and its
amplitude multiplied, Entry 77) were run at three adapters per arm and returned "inconclusive" at a power of
0.22 (Entry 98). Entry 99 extended them to eight per arm on byte-identical crops, with the reading unchanged:

  theta_sym < 0 with one-sided Welch p < 0.05                -> the 16000-step signal follows the fingerprint's sign
  theta_sym > 0 and within 2 adapter-level SE of +5.710e-5   -> the 16000-step signal does not follow the fingerprint
  otherwise                                                  -> inconclusive

applied to all eight per arm, with the p from the five new adapters alone reported beside it (the extension was
decided after an inconclusive result). If the two disagree in reading, both are reported and the arm stays
"inconclusive". The adaptation-strength covariate of Entry 89 is reported with it, and the power achieved.

Per-adapter statistic exactly as g_analyze.py: arm A own-minus-other = natural_paired_KA_minus_KB, arm B = its
negative. The first three per arm must reproduce Entry 98's values (checked). Writes out/g1_ext.json.
"""
import os, sys, json
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2; T1 = os.path.join(V2, "out", "t1")
OUT = os.path.join(V2, "out", "g1_ext.json")
R_REAL = 0.0356703416571125
UNSUPPRESSED = 5.710e-5
OLD, NEW = (0, 1, 2), (3, 4, 5, 6, 7)


def per_adapter():
    s0 = json.load(open(os.path.join(T1, "summary_inv16k.json")))["arms"]
    s1 = json.load(open(os.path.join(T1, "summary_inv16kext.json")))["arms"]
    A, B, meta = {}, {}, {}
    for s in OLD + NEW:
        src, pre = (s0, "inv16k") if s in OLD else (s1, "inv16kext")
        for body, store in (("A", A), ("B", B)):
            tag = f"{pre}_{body}_s{s}"
            nat = float(src[tag]["natural_paired_KA_minus_KB"])
            store[s] = nat if body == "A" else -nat
            m = json.load(open(os.path.join(T1, "adapters", tag, "train_meta.json")))
            meta[tag] = {"lora_B_norm": float(m["lora_B_norm"]), "loss_tail": float(m["loss_tail"]),
                         "n_images": int(src[tag]["n"])}
    return A, B, meta


def sym(a, b):
    a, b = np.asarray(a), np.asarray(b)
    vA, vB = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    th = 0.5 * (a.mean() + b.mean()); se = 0.5 * np.sqrt(vA + vB)
    df = (vA + vB) ** 2 / (vA ** 2 / (len(a) - 1) + vB ** 2 / (len(b) - 1))
    t = th / se
    return {"n_per_arm": len(a), "theta_A": float(a.mean()), "theta_B": float(b.mean()),
            "theta_sym": float(th), "theta_sym_pct": float(100 * th / R_REAL), "welch_se": float(se),
            "welch_df": float(df), "t": float(t),
            "one_sided_p_lt0": float(stats.t.cdf(t, df)), "one_sided_p_gt0": float(stats.t.sf(t, df))}


def reading(st):
    if st["theta_sym"] < 0 and st["one_sided_p_lt0"] < 0.05:
        return "the 16000-step signal follows the fingerprint's sign"
    if st["theta_sym"] > 0 and abs(st["theta_sym"] - UNSUPPRESSED) <= 2 * st["welch_se"]:
        return "the 16000-step signal does not follow the fingerprint"
    return "inconclusive"


def main():
    A, B, meta = per_adapter()
    # reproduction check against Entry 98 (g_chain12.json), the first three adapters per arm
    g1 = json.load(open(os.path.join(V2, "out", "g_chain12.json")))["G1"]
    old_A, old_B = [A[s] for s in OLD], [B[s] for s in OLD]
    assert np.allclose(old_A, g1["per_adapter_A"], rtol=1e-9, atol=1e-12), "arm A does not reproduce Entry 98"
    assert np.allclose(old_B, g1["per_adapter_B"], rtol=1e-9, atol=1e-12), "arm B does not reproduce Entry 98"

    all8 = sym([A[s] for s in OLD + NEW], [B[s] for s in OLD + NEW])
    new5 = sym([A[s] for s in NEW], [B[s] for s in NEW])
    r_all, r_new = reading(all8), reading(new5)
    final = r_all if r_all == r_new else "inconclusive"

    # Entry 89 covariate: does adapted-weight norm still track the per-adapter value?
    tags = [(f"{'inv16k' if s in OLD else 'inv16kext'}_{b}_s{s}", b, (A if b == "A" else B)[s])
            for s in OLD + NEW for b in ("A", "B")]
    y = np.array([v for _, _, v in tags]); x = np.array([meta[t]["lora_B_norm"] for t, _, _ in tags])
    body = np.array([1.0 if b == "A" else 0.0 for _, b, _ in tags])
    xa, xb = x[body == 1], x[body == 0]
    tn, pn = stats.ttest_ind(xa, xb, equal_var=False)
    X = np.column_stack([np.ones(len(y)), (x - x.mean()) / x.std(ddof=1)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta + beta[0]
    adj = sym(resid[body == 1], resid[body == 0])

    # power achieved against the effect it mirrors
    tcrit = stats.t.ppf(0.95, all8["welch_df"])
    power = float(stats.t.sf(tcrit - UNSUPPRESSED / all8["welch_se"], all8["welch_df"]))

    res = {"entry": "RESULTS.md Entries 77, 89, 99 (G1 extension)",
           "per_adapter_A": {f"s{s}": A[s] for s in OLD + NEW}, "per_adapter_B": {f"s{s}": B[s] for s in OLD + NEW},
           "adapters": meta, "all_eight": all8, "five_new_alone": new5,
           "reading_all_eight": r_all, "reading_five_new": r_new, "reading": final,
           "sequential_caveat": ("the extension was decided after an inconclusive result at n = 3; the p from the "
                                 "five new adapters alone is reported beside the pooled one, and the arm is "
                                 "'inconclusive' unless both readings agree"),
           "strength_covariate": {"lora_B_norm_A": float(xa.mean()), "lora_B_norm_B": float(xb.mean()),
                                  "welch_t": float(tn), "p_two_sided": float(pn),
                                  "corr_strength_theta": float(np.corrcoef(x, y)[0, 1]),
                                  "corr_strength_body": float(np.corrcoef(x, body)[0, 1]),
                                  "strength_adjusted": adj},
           "power_at_mirror_of_unsuppressed": power, "unsuppressed_theta_sym": UNSUPPRESSED,
           "reproduces_entry_98": True}
    json.dump(res, open(OUT, "w"), indent=2)
    for lab, st, r in (("all eight", all8, r_all), ("five new ", new5, r_new)):
        print(f"[g1ext] {lab}: theta_sym {st['theta_sym']:+.3e} ({st['theta_sym_pct']:+.4f} %), "
              f"SE {st['welch_se']:.3e}, t {st['t']:+.2f}, p(<0) {st['one_sided_p_lt0']:.4f}  -> {r}")
    print(f"[g1ext] strength: A {xa.mean():.3f} vs B {xb.mean():.3f} (p {pn:.3f}); r(strength, theta) "
          f"{res['strength_covariate']['corr_strength_theta']:+.2f}; adjusted p(<0) {adj['one_sided_p_lt0']:.4f}")
    print(f"[g1ext] power at the mirror of +5.71e-5: {power:.2f}")
    print(f"[g1ext] READING: {final}")


if __name__ == "__main__":
    main()
