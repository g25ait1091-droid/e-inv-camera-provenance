"""Fig. S4 (fig:s04estimator): the primary D200 limit under other fingerprint estimates.
Supplement S08 (sec:primary), OUTLINE §6 row S08: "G3 limits by estimate, H2 replicate limits, iPhone scatter".
Double column, 6.99 in wide (fv_style.DC), figure* unscaled; it fits the supplement's 7.14-in text width.

Panels
  (a) The 24 primary D200 adapters rescored under five fingerprint estimates (G3, 7,500 generated images).
      Per estimate: lambda_hat = theta_sym / R_real(own) with +-1 Welch SE, the one-sided 99 % symmetric limit
      (black tick) and the nominal max-arm limit (grey tick), all in % of that estimate's own R_real.
      Source: out/g3_estimator_scale.json -> estimates.{E2,E1,P,H1,H2}.{theta_sym, welch_se, R_real_own_scale,
      sym_limit99_pct, maxarm_U_pct}.  Photograph counts beneath the tick labels: numbers.json text of
      nLimGthree{Etwo,Eone,Pooled,HalfOne,HalfTwo}Photos.
  (b) The 24 bootstrap replicates of the E2 estimate (H2, 200 fixed images per adapter): per replicate
      lambda_hat +-1 SE and the one-sided 99 % symmetric limit, in % of that replicate's own R_real; dashed line
      at the estimation-inflated symmetric limit.
      Source: out/h2_estimation_error.json -> replicates[*].{replicate, theta_sym_pct, se_pct, limit99_pct},
      inflated_limit_pct.  Printed value: numbers.json text of nLimBootInflated.
  (c) iPhone 5c pair (VISION D05 / D14): per-adapter own-body contrast under the flat-field estimate against the
      natural-image estimate, each in % of that estimate's own R_real; seeds 0-11 per arm, in seed order.
      Source: out/g6_p5c.json -> estimators.{E2,FLAT}.{per_adapter_A, per_adapter_B, R_real};
      estimator_dependence.per_adapter_corr.  Printed value: numbers.json text of nGenGsixCorr.

Nothing plotted is typed in: every value is read from the files above and cross-checked against the
numbers.json `value` of the macro that the text quotes for it.  Run:
    python src/fv/figS_04_estimator.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_style import plt, C, DC, BODY_A, BODY_B, MARK_A, MARK_B, SYM, SYM_MARK, panel_label, save  # noqa: E402

OUT = (EINV.V2 + "/out")
NUMS = (EINV.V2 + "/paper/fv/numbers.json")
FIG_W = DC                     # journal double-column width, inches
FS = 8                         # every in-axes annotation, same size as the tick labels
FIG_H = 2.30

nums = json.load(open(NUMS, encoding="utf-8"))


def txt(k):
    """Printed text of a macro, made safe for matplotlib (LaTeX minus -> Unicode minus)."""
    return nums[k]["text"].replace("$-$", "\u2212")


def val(k):
    return float(nums[k]["value"])


def check(x, k, rel=1e-9):
    assert np.isclose(x, val(k), rtol=rel, atol=0), (k, x, val(k))


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ data
g3 = load("g3_estimator_scale.json")["estimates"]
EST = [  # (key in file, tick label, photographs macro, macro prefix)
    ("E2", r"$\mathcal{E}_2$", "nLimGthreeEtwoPhotos", "nLimGthreeEtwo"),
    ("E1", r"$\mathcal{E}_1$", "nLimGthreeEonePhotos", "nLimGthreeEone"),
    ("P", r"$\mathcal{E}_1{\cup}\mathcal{E}_2$", "nLimGthreePooledPhotos", "nLimGthreePooled"),
    ("H1", r"$\mathcal{E}_2^{(1)}$", "nLimGthreeHalfOnePhotos", "nLimGthreeHalfOne"),
    ("H2", r"$\mathcal{E}_2^{(2)}$", "nLimGthreeHalfTwoPhotos", "nLimGthreeHalfTwo"),
]
a_lam, a_se, a_sym, a_max = [], [], [], []
for key, _, _, pre in EST:
    e = g3[key]
    R = e["R_real_own_scale"]
    a_lam.append(100 * e["theta_sym"] / R)
    a_se.append(100 * e["welch_se"] / R)
    a_sym.append(e["sym_limit99_pct"])
    a_max.append(e["maxarm_U_pct"])
    check(e["sym_limit99_pct"], pre + "Sym"); check(e["maxarm_U_pct"], pre + "Max")
    check(e["theta_sym"], pre + "Theta"); check(R, pre + "Rreal")
    # the file's limit is theta + t_{0.99,df} SE on the same scale: confirm the plotted pieces agree
    from scipy import stats
    assert np.isclose(100 * (e["theta_sym"] + stats.t.ppf(0.99, e["welch_df"]) * e["welch_se"]) / R,
                      e["sym_limit99_pct"], rtol=1e-6)

h2 = load("h2_estimation_error.json")
reps = sorted(h2["replicates"], key=lambda r: int(r["replicate"]))
b_idx = np.array([int(r["replicate"]) for r in reps])
b_lam = np.array([r["theta_sym_pct"] for r in reps])
b_se = np.array([r["se_pct"] for r in reps])
b_lim = np.array([r["limit99_pct"] for r in reps])
b_infl = h2["inflated_limit_pct"]
assert len(reps) == int(val("nLimBootReps"))
check(b_infl, "nLimBootInflated"); check(b_lim.mean(), "nLimBootLimitMean")
check(b_lim.min(), "nLimBootLimitMin"); check(b_lim.max(), "nLimBootLimitMax")
assert sum(r["one_sided_p"] < 0.01 for r in reps) == int(val("nLimBootReject"))

g6 = load("g6_p5c.json")
E2, FL = g6["estimators"]["E2"], g6["estimators"]["FLAT"]
c_x = {b: 100 * np.asarray(E2[f"per_adapter_{b}"]) / E2["R_real"] for b in "AB"}   # seeds 0..11
c_y = {b: 100 * np.asarray(FL[f"per_adapter_{b}"]) / FL["R_real"] for b in "AB"}
r_file = g6["estimator_dependence"]["per_adapter_corr"]
r_now = np.corrcoef(np.r_[c_x["A"], c_x["B"]], np.r_[c_y["A"], c_y["B"]])[0, 1]
assert np.isclose(r_now, r_file, rtol=1e-9); check(r_file, "nGenGsixCorr")
assert int((c_x["A"] > 0).sum()) == int(val("nMainGsixNatPosA")) and int((c_x["B"] > 0).sum()) == int(val("nMainGsixNatPosB"))
assert int((c_y["A"] > 0).sum()) == int(val("nMainGsixFlatPosA")) and int((c_y["B"] > 0).sum()) == int(val("nMainGsixFlatPosB"))
assert len(c_x["A"]) == len(c_x["B"]) == int(val("nGenGsixAdapters"))

# ------------------------------------------------------------------ geometry (inches)
L, B_, T_ = 0.55, 0.50, 0.14        # left margin of (a), bottom margin, top margin
WA, WB, WC = 1.58, 1.80, 1.45       # axes widths
G_AB, G_BC = 0.95, 0.55             # gaps: (a)'s line-end labels plus (b)'s y label; (c)'s y label
H = FIG_H - B_ - T_
fig = plt.figure(figsize=(FIG_W, FIG_H))
xa = L
xb = xa + WA + G_AB
xc = xb + WB + G_BC
assert xc + WC <= FIG_W - 0.10, xc + WC      # room for (c)'s last x tick label
axA = fig.add_axes([xa / FIG_W, B_ / FIG_H, WA / FIG_W, H / FIG_H])
axB = fig.add_axes([xb / FIG_W, B_ / FIG_H, WB / FIG_W, H / FIG_H])
axC = fig.add_axes([xc / FIG_W, B_ / FIG_H, WC / FIG_W, H / FIG_H])

TICK_HW = 0.22                      # half-width of a limit tick in (a), category units
MS = 4.0
ZERO = dict(color=C["light"], lw=0.6, zorder=0)


def lam_se(ax, x, lam, se, ms=MS):
    ax.errorbar(x, lam, yerr=se, fmt=SYM_MARK, ms=ms, mfc=SYM, mec=SYM, ecolor=SYM, elinewidth=0.8,
                capsize=0, zorder=3)


# ---- (a) five estimates
x = np.arange(len(EST))
axA.axhline(0, **ZERO)
for i in x:
    axA.plot([i - TICK_HW, i + TICK_HW], [a_max[i]] * 2, color=C["grey"], lw=1.3, solid_capstyle="butt", zorder=2)
    axA.plot([i - TICK_HW, i + TICK_HW], [a_sym[i]] * 2, color=SYM, lw=1.3, solid_capstyle="butt", zorder=2)
lam_se(axA, x, a_lam, a_se)
axA.set_xlim(-0.5, len(EST) - 0.5)
axA.set_ylim(-0.1, 0.7)
axA.set_yticks(np.arange(-0.1, 0.71, 0.1))
axA.set_yticklabels(["\u22120.1", "0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"])
axA.set_xticks(x)
axA.set_xticklabels([lab for _, lab, _, _ in EST], fontsize=8)
axA.tick_params(axis="x", length=0, pad=3)
for i, (_, _, ph, _) in zip(x, EST):     # photographs per body, on one baseline beneath the names
    axA.annotate(txt(ph), xy=(i, 0), xycoords=("data", "axes fraction"), xytext=(0, -15.5),
                 textcoords="offset points", ha="center", va="top", fontsize=FS, color=C["ink"])
axA.set_ylabel(r"Transfer (% of own $R_{\mathrm{real}}$)")
# labels at the line ends, right of the last category
lx = len(EST) - 0.5 + 0.08
kw = dict(fontsize=FS, va="center", ha="left", clip_on=False)
axA.text(lx, a_max[-1], "max-arm\nlimit", color=C["grey"], linespacing=1.0, **kw)
axA.text(lx, a_sym[-1], "symmetric\nlimit", color=SYM, linespacing=1.0, **kw)
axA.text(lx, a_lam[-1] - 0.005, r"$\hat{\lambda}\pm$SE", color=SYM, **kw)
panel_label(axA, "(a)", x=-0.19)

# ---- (b) 24 bootstrap replicates
axB.axhline(0, **ZERO)
axB.axhline(b_infl, color=C["grey"], ls=(0, (3, 2)), lw=0.6, zorder=1)
axB.text(b_idx[-1] + 0.6, b_infl + 0.006, f"inflated limit\n{txt('nLimBootInflated')}%", ha="right",
         va="bottom", fontsize=FS, color=C["grey"], linespacing=1.0, multialignment="right")
for i, lim in zip(b_idx, b_lim):
    axB.plot([i - 0.36, i + 0.36], [lim] * 2, color=SYM, lw=1.1, solid_capstyle="butt", zorder=2)
lam_se(axB, b_idx, b_lam, b_se, ms=3.2)
axB.set_xlim(-0.8, b_idx[-1] + 0.8)
axB.set_ylim(-0.21, 0.2)
axB.set_yticks(np.arange(-0.2, 0.201, 0.1))
axB.set_yticklabels(["\u22120.2", "\u22120.1", "0", "0.1", "0.2"])
axB.set_xticks(np.arange(0, b_idx[-1] + 1, 5))
axB.set_xticks(b_idx, minor=True)
axB.set_xlabel(r"Bootstrap replicate of $\mathcal{E}_2$")
axB.set_ylabel(r"Transfer (% of replicate's $R_{\mathrm{real}}$)")
panel_label(axB, "(b)", x=-0.24)

# ---- (c) iPhone 5c: flat field against natural image, per adapter
axC.axhline(0, **ZERO); axC.axvline(0, **ZERO)
for b, col, mk in (("A", BODY_A, MARK_A), ("B", BODY_B, MARK_B)):
    axC.plot(c_x[b], c_y[b], ls="none", marker=mk, ms=MS, mfc=col, mec=col, zorder=3)
axC.set_xlim(-0.5, 0.6)
axC.set_ylim(-0.3, 0.2)
axC.set_xticks(np.arange(-0.4, 0.61, 0.2))
axC.set_xticklabels(["\u22120.4", "\u22120.2", "0", "0.2", "0.4", "0.6"])
axC.set_yticks(np.arange(-0.3, 0.21, 0.1))
axC.set_yticklabels(["\u22120.3", "\u22120.2", "\u22120.1", "0", "0.1", "0.2"])
axC.set_xlabel(r"Natural-image $\theta_{x,j}$ (% of $R_{\mathrm{real}}$)")
axC.set_ylabel(r"Flat-field $\theta_{x,j}$ (% of $R_{\mathrm{real}}$)")
axC.text(0.97, 0.03, f"$r$ = {txt('nGenGsixCorr')}", transform=axC.transAxes, ha="right", va="bottom", fontsize=FS)
axC.text(np.min(c_x["A"]), np.max(c_y["A"]) + 0.02, "arm A", color=BODY_A, ha="left", va="bottom", fontsize=FS)
axC.text(np.max(c_x["B"]), np.min(c_y["B"]) - 0.02, "arm B", color=BODY_B, ha="right", va="top", fontsize=FS)
panel_label(axC, "(c)", x=-0.30)

plt.rcParams["savefig.bbox"] = None          # keep the figure exactly at the text width
path = save(fig, "figS_04_estimator")

# ------------------------------------------------------------------ report
print(path, f"{FIG_W:.3f} x {FIG_H:.2f} in")
for (k, *_), l, s, sy, mx in zip(EST, a_lam, a_se, a_sym, a_max):
    print(f"(a) {k:3s} lam {l:+.4f}  se {s:.4f}  sym {sy:.4f}  max-arm {mx:.4f}")
print(f"(b) replicates {len(reps)}: lam {b_lam.min():+.4f}..{b_lam.max():+.4f}  limit {b_lim.min():+.4f}.."
      f"{b_lim.max():+.4f}  inflated {b_infl:.4f}")
print(f"(c) r {r_file:.4f}; x range {min(c_x['A'].min(), c_x['B'].min()):+.3f}..{max(c_x['A'].max(), c_x['B'].max()):+.3f}"
      f"; y range {min(c_y['A'].min(), c_y['B'].min()):+.3f}..{max(c_y['A'].max(), c_y['B'].max()):+.3f}")
print("(c) A_s0", c_x["A"][0], c_y["A"][0], " B_s11", c_x["B"][11], c_y["B"][11])
