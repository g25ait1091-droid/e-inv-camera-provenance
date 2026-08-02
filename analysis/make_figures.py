#!/usr/bin/env python3
"""Figures for the E-INV IEEE Access manuscript.

Every value is transcribed from the archived result JSONs (verified against the
experiment records; see docs/ADVERSARIAL_REVIEW.md Part 1). No figure shows data
that does not appear in the paper.

IEEE Access: single column 3.5 in, double column 7.16 in.
Palette is colour-blind safe (blue / orange / teal) and stays legible in
grayscale because the hues also differ in lightness.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
from overlapcheck import check

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.6,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.6,
    "axes.linewidth": 0.7, "axes.edgecolor": "#3a3a3a",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType, never Type 3
    "lines.linewidth": 1.2, "figure.dpi": 400, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    "legend.frameon": True, "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc", "legend.borderpad": 0.4,
})

BLUE, ORANGE, TEAL = "#2E5C8A", "#D97B29", "#3D8B7D"
RED, GREY, INK, PALE = "#A8443C", "#8A8A8A", "#2b2b2b", "#DCE5EE"
SC, DC = 3.5, 7.16
BOX = dict(boxstyle="round,pad=0.28", fc="white", ec="#d0d0d0", lw=0.5, alpha=0.94)

R_REAL, R_VAE, U_DEV = 3.56703416571125e-02, 1.30592770366e-02, 8.880180005344921e-05
A_MEANS = np.array([8.850071894e-06, -6.325359893e-06, 2.2739481612e-06,
                    -2.264346126e-05, 7.531645252e-06, 1.03266094242e-04])
B_MEANS = np.array([-3.41837958e-05, 1.3627266574e-05, 4.9241766654e-05,
                    4.0683344753e-05, -1.6195523868e-05, 3.3124104778e-05])
U_A, U_B = 8.880180005344921e-05, 6.923789808639606e-05

# =========================================================== Fig 1  pipeline
fig, ax = plt.subplots(figsize=(DC, 2.45))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
stages = [
    (2,  "Real image", r"$R_{\mathrm{real}}=3.57\times10^{-2}$" + "\nsame-model AUC 1.000", BLUE),
    (26, "Latent autoencoder", r"$R_{\mathrm{VAE}}=1.31\times10^{-2}$" + "\n" + r"$\eta=0.366$,  AUC 0.981", BLUE),
    (50, "LoRA objective", r"measurable from $\alpha\approx3$" + "\nnot device- or\nalignment-specific", ORANGE),
    (74, "Text-to-image\ngeneration", r"$\lambda_U\leq0.32\%$" + "\n" + r"$\tau_U\leq0.91\%$", BLUE),
]
W, H, YB = 22, 19, 50
for x, title, val, col in stages:
    ax.add_patch(FancyBboxPatch((x, YB), W, H, boxstyle="round,pad=0.6",
                                fc=PALE if col == BLUE else "#FBEBDC",
                                ec=col, lw=1.1, zorder=2))
    ax.text(x + W/2, YB + H*0.5, title, ha="center", va="center",
            fontsize=7.4, color=INK, weight="bold", zorder=3, linespacing=1.3)
    ax.text(x + W/2, YB - 6, val, ha="center", va="top", fontsize=6.6,
            color=INK, zorder=3, linespacing=1.5)
for x in (24, 48, 72):
    ax.add_patch(FancyArrowPatch((x, YB + H/2), (x + 2.0, YB + H/2), arrowstyle="-|>",
                                 mutation_scale=9, lw=1.1, color="#666666", zorder=1))
ax.plot([2, 48], [88, 88], color=BLUE, lw=2.4, solid_capstyle="round")
ax.text(25, 92, "device-specific identity", ha="center", fontsize=7.0, color=BLUE)
ax.plot([50, 72], [88, 88], color=ORANGE, lw=2.4, solid_capstyle="round")
ax.text(61, 92, "pattern energy", ha="center", fontsize=7.0, color=ORANGE)
ax.plot([74, 96], [88, 88], color=BLUE, lw=2.4, solid_capstyle="round")
ax.text(85, 92, "device-specific identity", ha="center", fontsize=7.0, color=BLUE)
fig.savefig("fig1_pipeline.pdf"); plt.close(fig)

# ================================================ Fig 2  stage localisation
fig, ax = plt.subplots(figsize=(SC, 2.75), constrained_layout=True)
vals = [R_REAL, R_VAE, U_DEV]
bars = ax.bar(range(3), vals, color=[BLUE, TEAL, ORANGE], edgecolor="white",
              lw=0.8, width=0.6, zorder=3)
ax.set_yscale("log"); ax.set_ylim(2e-5, 4.5e-1)
ax.set_xticks(range(3))
ax.set_xticklabels(["Real\nimages", "After\nautoencoder", "Generated\n(upper limit)"])
ax.set_xlim(-0.62, 2.62)
ax.set_ylabel("device-specific paired contrast")
for b, v, t in zip(bars, vals, [r"$3.57\times10^{-2}$", r"$1.31\times10^{-2}$",
                                r"$8.88\times10^{-5}$"]):
    ax.text(b.get_x() + b.get_width()/2, v*1.45, t, ha="center", fontsize=6.6,
            color=INK, zorder=4)
ax.plot([0.0, 1.0], [1.05e-1, 1.05e-1], color=TEAL, lw=1.0, zorder=4)
ax.plot([0.0, 0.0], [7.5e-2, 1.05e-1], color=TEAL, lw=1.0, zorder=4)
ax.plot([1.0, 1.0], [2.8e-2, 1.05e-1], color=TEAL, lw=1.0, zorder=4)
ax.text(0.5, 1.62e-1, r"$\eta=0.366$ survives", ha="center", fontsize=6.8,
        color=TEAL, bbox=BOX, zorder=6)
ax.text(2.0, 3.0e-3, "0.680\\% of the\nround-trip contrast", ha="center",
        fontsize=6.8, color=ORANGE, bbox=BOX, zorder=6)
ax.grid(axis="y", ls=":", lw=0.5, color="#bbbbbb", zorder=0); ax.set_axisbelow(True)
check(fig, ax, "Fig2 stage localisation"); fig.savefig("fig2_stage_localisation.pdf"); plt.close(fig)

# =========================================== Fig 3  adapter contrasts + limits
fig, ax = plt.subplots(figsize=(SC, 2.75), constrained_layout=True)
x = np.arange(6)
ax.axhspan(-1.0, U_A, color=PALE, zorder=0)
ax.axhline(0, color="#999999", lw=0.8, zorder=1)
ax.axhline(U_A, color=BLUE, ls="--", lw=1.1, zorder=3)
ax.axhline(U_B, color=ORANGE, ls="--", lw=1.1, zorder=3)
ax.plot(x - 0.12, A_MEANS, "o", ms=4.8, color=BLUE, mec="white", mew=0.8,
        zorder=5, label="arm A")
ax.plot(x + 0.12, B_MEANS, "s", ms=4.4, color=ORANGE, mec="white", mew=0.8,
        zorder=5, label="arm B")
ax.set_ylim(-7.0e-5, 1.62e-4); ax.set_xlim(-0.6, 5.6)
ax.text(-0.5, U_A + 6.0e-6, r"$U_A = 8.88\times10^{-5}$", fontsize=6.5,
        color=BLUE, zorder=6)
ax.text(2.85, U_B - 1.55e-5, r"$U_B = 6.92\times10^{-5}$", fontsize=6.5,
        color=ORANGE, zorder=6)
ax.set_xticks(x); ax.set_xticklabels([f"s{i}" for i in range(6)])
ax.set_xlabel("adapter training seed")
ax.set_ylabel(r"$\theta$   (paired device contrast)")
ax.legend(loc="lower left", ncol=2, handletextpad=0.35, columnspacing=1.1)
ax.grid(axis="y", ls=":", lw=0.5, color="#bbbbbb", zorder=2)
check(fig, ax, "Fig3 adapter bounds"); fig.savefig("fig3_adapter_bounds.pdf"); plt.close(fig)

# ================================================== Fig 4  full-pipeline calib
al = np.array([0.0, 0.01, 0.02, 0.05, 0.10, 0.25])
mu = np.array([-6.6188e-05, 7.5227e-05, 2.1720e-04, 6.4109e-04, 1.3484e-03, 3.4691e-03])
se = np.array([3.190e-05, 3.196e-05, 3.211e-05, 3.321e-05, 3.684e-05, 5.609e-05])
b0, b1 = -6.598e-05, 1.4142e-02
fig, ax = plt.subplots(figsize=(SC, 2.75), constrained_layout=True)
g = np.linspace(0, 0.265, 200)
ax.plot(g, b0 + b1*g, "-", color=GREY, lw=1.1, zorder=2,
        label=r"fit  $-6.60\!\times\!10^{-5}\!+\!1.414\!\times\!10^{-2}\alpha$")
ax.errorbar(al, mu, yerr=2.576*se, fmt="o", ms=4.4, color=BLUE, mec="white", mew=0.7,
            ecolor=BLUE, elinewidth=1.0, capsize=2.2, zorder=4,
            label="measured, 99\\% CI")
ax.axhline(U_DEV, color=ORANGE, ls="--", lw=1.1, zorder=3)
ax.set_yscale("symlog", linthresh=2e-4, linscale=0.55)
ax.set_ylim(-2.4e-4, 9.5e-3); ax.set_xlim(-0.014, 0.285)
ax.text(0.198, 1.05e-4, r"$U_{\mathrm{device}}$", fontsize=6.8, color=ORANGE, zorder=6)
ax.plot([0.02], [2.1720e-04], "o", ms=11, mfc="none", mec=RED, mew=1.2, zorder=5)
ax.annotate(r"$\alpha=0.02$:  $t=6.76$", xy=(0.0225, 1.75e-4),
            xytext=(0.058, 9.0e-5), fontsize=6.4, color=RED, bbox=BOX, zorder=7,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color=RED,
                            shrinkA=3, shrinkB=5))
ax.set_xlabel(r"nominal injection coefficient $\alpha$")
ax.set_ylabel("measured paired contrast")
ax.legend(loc="lower right")
ax.grid(ls=":", lw=0.5, color="#bbbbbb"); ax.set_axisbelow(True)
check(fig, ax, "Fig4 calibration"); fig.savefig("fig4_calibration.pdf"); plt.close(fig)

# ====================================== Fig 5  additive decomposition, held out
pred = np.array([1.7503041728874e-04, -5.08799672046e-06, -1.7127678303011e-04,
                 -4.262306438426e-05, 4.395742684609e-05, 1.641782918339e-04,
                 -6.06562557551304e-05, -1.484407887232904e-04,
                 -1.7916208248198e-05, 6.283496089271e-05])
obs = np.array([2.38523828717058e-04, -1.8773640109810e-05, -1.0718390619085e-04,
                -4.955961699735e-05, 8.91924040228e-05, 1.822040796695e-04,
                -5.363168176e-07, -1.7285995764400e-04, -8.475834481345e-05,
                5.02460149055e-05])
fig, ax = plt.subplots(figsize=(SC, 2.9), constrained_layout=True)
lim = 3.0e-4
ax.axhline(0, color="#dddddd", lw=0.7, zorder=0); ax.axvline(0, color="#dddddd", lw=0.7, zorder=0)
ax.plot([-lim, lim], [-lim, lim], ls="--", lw=0.9, color=GREY, zorder=1, label=r"$y=x$")
ax.plot(pred[:5], obs[:5], "o", ms=4.8, color=BLUE, mec="white", mew=0.8, zorder=3,
        label="fit s0 $\\rightarrow$ predict s1")
ax.plot(pred[5:], obs[5:], "^", ms=5.0, color=TEAL, mec="white", mew=0.8, zorder=3,
        label="fit s1 $\\rightarrow$ predict s0")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim*1.42)
ax.set_xlabel(r"predicted $\hat\theta_d = b_d-\overline{b}_{d'\neq d}$   (held-out seed)")
ax.set_ylabel(r"observed $\theta_d$")
ax.text(-2.82e-4, 4.05e-4,
        r"cross-validated $R^2 = 0.870$" + "\n"
        r"device level $r = +0.979$  ($p=0.0036$)" + "\n"
        r"exact permutation $p = 0.0167$",
        fontsize=6.6, va="top", ha="left", bbox=BOX, zorder=6, linespacing=1.55)
ax.legend(loc="lower right", handletextpad=0.35)
ax.grid(ls=":", lw=0.5, color="#cccccc"); ax.set_axisbelow(True)
check(fig, ax, "Fig5 additive"); fig.savefig("fig5_additive.pdf"); plt.close(fig)

# ================================================== Fig 6  replication summary
rows = [("Primary (SD 3.5)", 6, 0.32, "seeds"),
        ("Primary, matched $n$", 3, 0.699, "seeds"),
        ("FLUX.1-dev", 3, 0.777, "seeds"),
        ("Full fine-tuning", 3, 0.747, "sym"),
        ("Kodak, 5 devices", 5, 1.74, "devices"),
        ("Low/mid band", 6, 9.72, "sym")]
fig, ax = plt.subplots(figsize=(SC, 3.35), constrained_layout=True)
y = np.arange(len(rows))[::-1]
for yy, (lab, n, v, unit) in zip(y, rows):
    col = {"seeds": BLUE, "devices": ORANGE, "sym": TEAL}[unit]
    mk  = {"seeds": "o",  "devices": "D",    "sym": "s"}[unit]
    ax.plot([0.24, v], [yy, yy], color=col, lw=1.8, alpha=0.40, zorder=2,
            solid_capstyle="round")
    ax.plot(v, yy, mk, ms=6.2 if mk != "s" else 5.8, color=col,
            mec="white", mew=1.0, zorder=3)
    ax.text(58, yy, f"{v:.2f}\\%", fontsize=6.9, va="center", ha="right",
            color=col, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels([f"{r[0]}\n$n={r[1]}$" for r in rows], fontsize=6.6, linespacing=1.45)
ax.set_xscale("log"); ax.set_xlim(0.24, 62)
ax.set_ylim(-0.95, len(rows) - 0.35)
ax.set_xlabel(r"upper limit $\lambda_U$  (\% of device contrast)")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=5.6, mec="white",
                          label="over adapter seeds"),
                   Line2D([], [], marker="D", ls="", color=ORANGE, ms=5.2, mec="white",
                          label="over physical devices"),
                   Line2D([], [], marker="s", ls="", color=TEAL, ms=5.0, mec="white",
                          label="symmetric-interaction bound")],
          loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2,
          handletextpad=0.35, columnspacing=1.0, title="generalises over",
          title_fontsize=6.4)
ax.grid(axis="x", ls=":", lw=0.5, color="#bbbbbb"); ax.set_axisbelow(True)
check(fig, ax, "Fig6 replication"); fig.savefig("fig6_replication.pdf"); plt.close(fig)

# ============================ Fig 7  main effects vs the paired contrast
fig, ax = plt.subplots(figsize=(SC, 2.75), constrained_layout=True)
KA = np.array([7.47187381886e-05, 1.28947288974e-04, 1.8076668250287e-04])
KB = np.array([7.311918480133e-05, 1.385090347857e-04, 2.3799077210533e-04])
diff = np.abs(KA - KB)
xg = np.arange(3); w = 0.3
ax.bar(xg - w/2, KA, w, color=BLUE, edgecolor="white", lw=0.7, zorder=3,
       label=r"$\rho \rightarrow \hat K_A$")
ax.bar(xg + w/2, KB, w, color=TEAL, edgecolor="white", lw=0.7, zorder=3,
       label=r"$\rho \rightarrow \hat K_B$")
ax.plot(xg, diff, "o-", ms=5.2, lw=1.5, color=ORANGE, mec="white", mew=0.9, zorder=4,
        label="paired difference")
ax.set_xticks(xg); ax.set_xticklabels(["A-gens", "B-gens", "D-gens"])
ax.set_xlim(-0.55, 2.55)
ax.set_ylabel("mean correlation with fingerprint")
ax.set_yscale("log"); ax.set_ylim(2.2e-7, 1.6e-2)
ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3])
ax.legend(loc="upper center", ncol=3, handletextpad=0.3, columnspacing=0.9,
          borderpad=0.35)
ax.grid(axis="y", ls=":", lw=0.5, color="#bbbbbb"); ax.set_axisbelow(True)
check(fig, ax, "Fig7 main effects"); fig.savefig("fig7_main_effects.pdf"); plt.close(fig)

print("wrote fig1..fig7")
