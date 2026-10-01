"""Defect D9: recompute the paired gates for every secondary pair with the study's own definition.

g4_prep.py and g6_prep.py formed the per-image own-minus-other contrast correctly and then combined the
two bodies with a minus: R_real = 0.5*(mean_A - mean_B) and an AUC that asked whether body A's images
outscored body B's. Both are the wrong comparison - the canonical statistic, used everywhere else in the
study (see real_contrast() in g3_estimator_scale.py), is 0.5*(mean_A + mean_B), and the held-out AUC
separates own-fingerprint scores from other-fingerprint scores on the same images.

This recomputes both readings from the stored H_{A,B}.npz and K files, prints the old and the corrected
values side by side, and rewrites the gates block of each manifest in place. No new data is read and no
generation is touched: the same held-out photographs are being re-summarised.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

OUT = os.path.join(EINV.V2, "out")
PAIRS = [("fp_p10", "Huawei P10 Plus (G4)", ["E2"]),
         ("fp_5c", "Apple iPhone 5c (G6)", ["E2", "FLAT"]),
         ("fp_p20", "Huawei P20 (primary smartphone)", ["E2"]),
         ("fp_kodak", "Kodak M1063 (five-body arm)", ["E2"])]


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def gates_for(d, tag):
    """Return (R_real, AUC, per-body contrast) with the study's definition."""
    K = {r: np.load(os.path.join(d, f"K_{r}_{tag}.npy")) for r in ("A", "B")}
    own_all, oth_all, per = [], [], {}
    for role, other in (("A", "B"), ("B", "A")):
        z = np.load(os.path.join(d, f"H_{role}.npz")); Y, W = z["Y"], z["W"]
        own = [ncc(W[i], Y[i] * K[role]) for i in range(len(Y))]
        oth = [ncc(W[i], Y[i] * K[other]) for i in range(len(Y))]
        per[role] = float(np.mean(own) - np.mean(oth))
        own_all += own; oth_all += oth
    own_all, oth_all = np.array(own_all), np.array(oth_all)
    R = 0.5 * (per["A"] + per["B"])
    auc = float(np.mean([(p > n) + 0.5 * (p == n) for p in own_all for n in oth_all]))
    paired = float(np.mean(own_all > oth_all))
    return R, auc, per, paired


for sub, label, tags in PAIRS:
    d = os.path.join(OUT, sub)
    mp = os.path.join(d, "manifest.json")
    if not os.path.exists(mp) or not os.path.exists(os.path.join(d, "H_A.npz")):
        print(f"-- {label}: no paired held-out block, skipped"); continue
    man = json.load(open(mp)); g = dict(man.get("gates", {}))
    print(f"\n== {label} [{sub}]")
    for tag in tags:
        if not os.path.exists(os.path.join(d, f"K_A_{tag}.npy")): continue
        R, auc, per, paired = gates_for(d, tag)
        sfx = "" if tag == "E2" else "_" + tag
        oldR = g.get("R_real" + sfx, g.get(f"R_real_{tag}"))
        oldA = g.get("AUC_held_out" + sfx, g.get(f"AUC_held_out_{tag}"))
        print(f"   {tag:4s} R_real {R:+.6f}   (as filed {oldR:+.6f})")
        print(f"        AUC    {auc:.4f}      (as filed {oldA:.4f})   paired own>other {paired:.3f}")
        print(f"        per body: A {per['A']:+.6f}   B {per['B']:+.6f}")
        g[f"R_real{sfx}"] = R; g[f"AUC_held_out{sfx}"] = auc
        g[f"per_body_contrast{sfx}"] = per; g[f"paired_own_gt_other{sfx}"] = paired
    g["_corrected"] = ("D9: R_real and AUC recomputed with the canonical paired definition "
                       "0.5*(A+B) and own-vs-other scoring; superseded values kept below")
    g["_superseded"] = {k: v for k, v in man.get("gates", {}).items() if not k.startswith("_")}
    man["gates"] = g
    json.dump(man, open(mp, "w"), indent=1)
    print("   manifest gates rewritten")
