"""Entry 39 — the adaptation-dose axis, paired by body, plus descriptive memorization metrics.

theta_A, theta_B, theta_sym, additive part at 2000 (nomark / nomarkB), 8000 (dose8k) and 16000
(dose16k) steps. Memorization: per generation, max NCC of its 256^2 grayscale thumbnail against the
50 training crops of its body (mean, 95th percentile, fraction > 0.5); adaptation strength = mean
|generation - local_base| at matched seeds. First 250 generations of every arm. Writes
out/t1/dose_stats.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, json, glob, numpy as np
from PIL import Image
from scipy import stats
T1 = os.path.join(EINV.V2, 'out', 't1'); R_REAL = 0.0356703416571125; U_DEVICE = 5.3761e-05
S = {}
for k in ("nomark", "nomarkB", "dose8k", "dose16k", "dose16krep", "dose16krep2"):
    p = os.path.join(T1, f"summary_{k}.json")
    if os.path.exists(p): S.update(json.load(open(p))["arms"])
nat = lambda a: S[a]["natural_paired_KA_minus_KB"]
DOSES = {2000: ([f"nomark_s{i}" for i in range(3)], [f"nomarkB_s{i}" for i in range(3)]),
         8000: (["dose8k_A_s0", "dose8k_A_s1"], ["dose8k_B_s0", "dose8k_B_s1"]),
         16000: ([f"dose16k_A_s{i}" for i in range(6)], [f"dose16k_B_s{i}" for i in range(6)]),
         # Entry 55: the pre-specified primary reading is on the new adapters alone (seeds 3-5)
         "16000_new": ([f"dose16k_A_s{i}" for i in range(3, 6)], [f"dose16k_B_s{i}" for i in range(3, 6)])}
out = {"doses": {}}
for steps, (As, Bs) in DOSES.items():
    As = [a for a in As if a in S]; Bs = [b for b in Bs if b in S]
    if not As or not Bs: continue
    a = np.array([nat(x) for x in As]); b = np.array([-nat(x) for x in Bs])
    th = 0.5 * (a.mean() + b.mean()); rec = {"A_arms": As, "B_arms": Bs, "A_own": a.tolist(), "B_own": b.tolist(),
            "theta_A": float(a.mean()), "theta_B": float(b.mean()), "theta_sym": float(th), "additive_part": float(0.5 * (a.mean() - b.mean())),
            "lambda_sym_pct": float(100 * th / R_REAL), "below_U_device": bool(th < U_DEVICE)}
    if len(a) > 1 and len(b) > 1:
        vA, vB = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b); se = 0.5 * np.sqrt(vA + vB)
        df = (vA + vB) ** 2 / (vA ** 2 / (len(a) - 1) + vB ** 2 / (len(b) - 1))
        rec.update(adapter_SE=float(se), welch_df=float(df), t=float(th / se), lambda_U_plugin_pct=float(100 * (th + stats.t.ppf(0.99, df) * se) / R_REAL))
    out["doses"][str(steps)] = rec
r2, r8 = out["doses"].get("2000"), out["doses"].get("8000")
if r2 and r8 and "adapter_SE" in r2 and "adapter_SE" in r8:
    diff = r8["theta_sym"] - r2["theta_sym"]; se = float(np.sqrt(r2["adapter_SE"] ** 2 + r8["adapter_SE"] ** 2))
    out["dose_8000_minus_2000"] = {"difference": diff, "se": se, "z": diff / se, "within_2se": bool(abs(diff) <= 2 * r8["adapter_SE"])}

def thumb(path):
    a = np.asarray(Image.open(path).convert("L").resize((256, 256), Image.BOX), np.float32).ravel(); a -= a.mean(); return a / (np.linalg.norm(a) + 1e-12)
train = {body: np.stack([thumb(f) for f in sorted(glob.glob(os.path.join(T1, "train_png", d, "*.png")))]) for body, d in (("A", "none_a0"), ("B", "noneB_a0"))}
base_files = sorted(glob.glob(os.path.join(T1, "gens", "local_base", "*.png")))[:250]
base_gray = None
mem = {}
for arm in [x for steps in DOSES.values() for grp in steps for x in grp] + ["local_base"]:
    files = sorted(glob.glob(os.path.join(T1, "gens", arm, "*.png")))[:250]
    if not files: continue
    body = "B" if "B_" in arm or arm.startswith("nomarkB") else "A"
    m = np.array([float((train[body] @ thumb(f)).max()) for f in files])
    rec = {"n": len(files), "max_ncc_mean": float(m.mean()), "max_ncc_p95": float(np.percentile(m, 95)), "frac_above_0.5": float((m > 0.5).mean())}
    if arm != "local_base" and len(base_files) >= len(files):
        rec["mean_abs_diff_to_local_base_gray"] = float(np.mean([np.abs(np.asarray(Image.open(f).convert("L"), np.float32) - np.asarray(Image.open(g).convert("L"), np.float32)).mean()
                                                                for f, g in zip(files, base_files)]))
    mem[arm] = rec
out["memorization"] = mem
json.dump(out, open(os.path.join(T1, "dose_stats.json"), "w"), indent=1); print(json.dumps(out, indent=1))
