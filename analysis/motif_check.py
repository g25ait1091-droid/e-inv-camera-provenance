# %% M — is the Dresden split motif-disjoint, and does it matter?
#
# The reviewer's concern: images of the same motif appearing in both E2 (fingerprint
# estimation) and H (query) would inflate R_real, and since lambda_U = U / R_real an
# inflated denominator makes the limit look tighter than it is. The published audit used a
# maximum embedding similarity of 0.789 against a threshold of 0.95 that was never
# calibrated against known same-motif and different-motif pairs.
#
# This cell calibrates that threshold from the data itself and then acts on it.
#
# HOW THE CALIBRATION WORKS. Dresden captured several exposures of each motif in sequence,
# so similarity as a function of index gap carries the motif structure directly: pairs a few
# indices apart are usually the same motif, pairs far apart are usually not. The gap curve
# therefore gives both distributions without needing motif labels, and the threshold falls
# out of where they separate.
#
# WHAT IT COSTS. Thumbnails for both devices, a few fingerprint re-estimations and one VAE
# pass over a subset of H. Roughly 25 minutes, no training, nothing overwritten.
#
# Run inside the pilot notebook after its config and forensic-core cells.
import os, json, glob, itertools, csv as _csv
import numpy as np
from PIL import Image
from scipy import stats as sps

OUT = os.path.join(ROOT, "motif_check")
os.makedirs(OUT, exist_ok=True)
PAIR = ("A", "B")
N_TH = 64                      # thumbnail edge
GAP_FAR = 150                  # index gap treated as certainly a different motif

def _thumb(fp, n=N_TH):
    with Image.open(fp) as im:
        a = np.asarray(im.convert("L").resize((n, n), Image.BILINEAR), np.float32).ravel()
    a = a - a.mean(); s = a.std()
    return a / (s if s > 1e-6 else 1.0)      # contrast-normalised: flat scenes stop saturating

# ---- 1. every image of each device, in acquisition order -------------------------------
MAN = {}
for r in _csv.DictReader(open(os.path.join(ROOT, "manifest", "manifest.csv"))):
    MAN.setdefault((r["role"], r["split"]), []).append(r["path"])

allimg = {}
for role in PAIR:
    paths = sorted({p for (rr, _), v in MAN.items() if rr == role for p in v})
    allimg[role] = paths
    print(f"[M] {role}: {len(paths)} images across all splits")

TH = {role: np.stack([_thumb(p) for p in allimg[role]]) for role in PAIR}
print("[M] thumbnails loaded\n")

# ---- 2. similarity as a function of index gap: the motif structure ----------------------
print("[M] similarity vs index gap (this is the motif block size)")
print(f"    {'gap':>5}{'mean sim':>11}{'p90':>9}   interpretation")
gapstats = {}
for role in PAIR:
    T = TH[role]; n = len(T); d = T.shape[1]
    for gap in (1, 2, 3, 5, 8, 12, 20, 40, 80, GAP_FAR):
        if gap >= n: continue
        s = np.array([float(T[i] @ T[i+gap] / d) for i in range(n-gap)])
        gapstats.setdefault(gap, []).append(s)
for gap in sorted(gapstats):
    s = np.concatenate(gapstats[gap])
    note = ""
    if gap == 1: note = "<- adjacent, usually the same motif"
    if gap == GAP_FAR: note = "<- far apart, almost certainly different motifs"
    print(f"    {gap:5d}{s.mean():11.4f}{np.percentile(s,90):9.4f}   {note}")

near = np.concatenate(gapstats[1])            # same-motif proxy
far  = np.concatenate(gapstats[GAP_FAR])      # different-motif proxy
# threshold at the crossing point that best separates the two empirical distributions
grid = np.linspace(min(far.min(), near.min()), max(far.max(), near.max()), 400)
err  = [(near < t).mean() + (far >= t).mean() for t in grid]
THRESH = float(grid[int(np.argmin(err))])
sep = (far >= THRESH).mean(); miss = (near < THRESH).mean()
print(f"\n[M] calibrated threshold {THRESH:.4f}")
print(f"    false 'same motif' among far pairs : {100*sep:.2f}%")
print(f"    missed 'same motif' among adjacent : {100*miss:.2f}%")
print(f"    (the published audit used 0.95 on an uncalibrated scale)")
if sep > 0.05 or miss > 0.35:
    print("    *** the two distributions overlap heavily; treat the motif verdict below as")
    print("        indicative rather than definitive.")

# ---- 3. do E2 and H share motifs? -------------------------------------------------------
idx = {role: {p: i for i, p in enumerate(allimg[role])} for role in PAIR}
report = {"threshold": THRESH, "near_mean": float(near.mean()),
          "far_mean": float(far.mean()), "devices": {}}
overlap_any = False
print("\n[M] cross-split motif overlap")
for role in PAIR:
    T = TH[role]; d = T.shape[1]
    ent = {}
    for sp in ("E1", "E2", "T", "H"):
        ent[sp] = [idx[role][p] for p in MAN[(role, sp)]]
    dev = {}
    for a, b in itertools.combinations(("E1", "E2", "T", "H"), 2):
        A_ = T[ent[a]]; B_ = T[ent[b]]
        S = (A_ @ B_.T) / d
        share = int((S >= THRESH).sum())
        pairs_a = int((S >= THRESH).any(axis=1).sum())
        dev[f"{a}-{b}"] = {"pairs": share, "images_in_first": pairs_a,
                           "max_sim": float(S.max())}
        flag = ""
        if a == "E2" and b == "H" and share:
            flag = "   <- THIS IS THE ONE THAT MATTERS"; overlap_any = True
        elif share: flag = "   (harmless: neither is the query set)"
        print(f"    {role} {a:2s}-{b:2s}: {share:5d} pairs above threshold, "
              f"{pairs_a:3d} of {len(ent[a])} images, max sim {S.max():.4f}{flag}")
    report["devices"][role] = dev

# ---- 4. if E2 and H share motifs, recompute the denominator without them ---------------
if not overlap_any:
    print("\n" + "=" * 74)
    print("NO E2/H MOTIF OVERLAP at the calibrated threshold. The fingerprint set and the")
    print("query set are motif-disjoint, so R_real is not inflated by shared content and the")
    print("normalised limits stand as published. Report the calibrated threshold and this")
    print("result in place of the uncalibrated 0.95 audit.")
    print("=" * 74)
else:
    print("\n[M] E2/H overlap found — rebuilding a motif-clean denominator (no retraining)")
    clean = {}
    for role in PAIR:
        T = TH[role]; d = T.shape[1]
        e2 = [idx[role][p] for p in MAN[(role, "E2")]]
        h  = [idx[role][p] for p in MAN[(role, "H")]]
        S = (T[e2] @ T[h].T) / d
        keep_h  = [h[j]  for j in range(len(h))  if not (S[:, j] >= THRESH).any()]
        keep_e2 = [e2[i] for i in range(len(e2)) if not (S[i, :] >= THRESH).any()]
        clean[role] = {"H": keep_h, "E2": keep_e2}
        print(f"    {role}: H {len(h)} -> {len(keep_h)} images, "
              f"E2 {len(e2)} -> {len(keep_e2)} images")
    if min(len(clean[r]["H"]) for r in PAIR) < 12 or \
       min(len(clean[r]["E2"]) for r in PAIR) < 40:
        print("    *** too little left to re-estimate reliably; report the overlap as a")
        print("        limitation instead of a corrected number.")
    else:
        K = {}
        for role in PAIR:
            files = [allimg[role][i] for i in clean[role]["E2"]]
            K[role] = estK(files)
            print(f"    {role}: fingerprint re-estimated from {len(files)} motif-clean images")
        per = []
        for role in PAIR:
            oth = PAIR[1] if role == PAIR[0] else PAIR[0]
            v = []
            for i in clean[role]["H"]:
                Y = load_lum(allimg[role][i]); W = wres(Y)
                v.append(ncc(W, Y * K[role]) - ncc(W, Y * K[oth]))
            per.append(float(np.mean(v)))
            print(f"    {role}: motif-clean contrast {per[-1]:.5e} over {len(v)} images")
        R_clean = float(np.mean(per))
        R_PUB = 3.56703416571125e-02
        U_DEV = 8.880180005344921e-05
        rel = (R_clean - R_PUB) / R_PUB
        report["R_real_published"] = R_PUB
        report["R_real_motif_clean"] = R_clean
        report["relative_change"] = rel
        report["lambda_U_published_pct"] = 100 * U_DEV / R_PUB
        report["lambda_U_motif_clean_pct"] = 100 * U_DEV / R_clean
        print("\n" + "=" * 74)
        print(f"  R_real published    {R_PUB:.5e}")
        print(f"  R_real motif-clean  {R_clean:.5e}   ({rel:+.1%})")
        print(f"  lambda_U published    {100*U_DEV/R_PUB:.4f}%")
        print(f"  lambda_U motif-clean  {100*U_DEV/R_clean:.4f}%")
        print()
        if abs(rel) < 0.05:
            print("  The denominator barely moves. Motif overlap is present but immaterial;")
            print("  report both numbers and keep the published limit.")
        elif rel < 0:
            print("  The clean denominator is SMALLER, so the published limit was OPTIMISTIC.")
            print("  Report the motif-clean value as primary and explain why.")
        else:
            print("  The clean denominator is LARGER, so the published limit was CONSERVATIVE.")
            print("  Report both; the published number remains a valid upper limit.")
        print("=" * 74)

json.dump(report, open(os.path.join(OUT, "motif_check.json"), "w"), indent=2)
print(f"\n[M] written to {os.path.join(OUT, 'motif_check.json')}")
print("\nNote: theta is unaffected either way. It is measured on GENERATIONS against E2")
print("fingerprints, and both devices photographed the same motifs, so residual scene")
print("structure correlates with K_A and K_B equally and cancels in the paired contrast.")
print("Only the denominator can be inflated by motif overlap, and only R_real feeds it.")
