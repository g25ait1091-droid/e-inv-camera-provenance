"""Figure 8 — what passes through personalization, on log-log axes: output contrast in generations
against the pattern's stored input contrast (never-injected offset removed), for the non-repeating
+/-1 fields, a non-repeating top-octave field, two repeating tiles (36 px, off grid; 32 px, on the
latent grid), the published DiffusionShield watermark, and the natural fingerprint. Reads
out/t1/summary.json, summary_derived.json, wm_summary.json, periodic_summary_final.json and the frozen
ledger; writes paper/fig8_ladder.pdf. Style matches make_figures_v2.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman"], "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.2, "axes.linewidth": 0.6})
BLUE, ORANGE, TEAL, RED, GREY, INK, GOLD, GREEN = "#2E5C8A", "#C8641E", "#0B7A6E", "#A8432F", "#8A8A8A", "#1F2A37", "#8A6D1F", "#3B7D23"
T1 = os.path.join(EINV.V2, 'out', 't1'); OUT = os.path.join(EINV.V2, 'paper', 'fig8_ladder.pdf')
S = json.load(open(os.path.join(T1, "summary.json")))["arms"]; D = json.load(open(os.path.join(T1, "summary_derived.json")))
WM = json.load(open(os.path.join(T1, "wm_summary.json"))); PER = json.load(open(os.path.join(T1, "periodic_summary_final.json")))["fields"]
led = json.load(open(os.path.join(EINV.REPO, 'analysis', 'FINAL_LEDGER.json')))
R_real = led["denominators"]["R_real"]; U_dev = led["primary"]["U_device"]; th_sym = led["primary"]["theta_sym"]
off = D["offsets"]["M_minus_Mp_never_injected"]["mean"]; off_se = D["offsets"]["M_minus_Mp_never_injected"]["se_pooled_images"]
offL = D["offsets"]["ML_minus_Mp_never_injected"]["mean"]
wm_off = WM["cluster"]["never_injected_offset"]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 3.45), gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.45})
# ---- (a) output vs input contrast, log-log, offset removed ----
xs = np.logspace(np.log10(0.015), np.log10(1.5), 50)
ax.plot(xs, 5e-4 * xs, ls="-", lw=0.8, color=GREY, label=r"transmission $5\times10^{-4}$", zorder=1)
ax.plot(xs, 4.0e-2 * xs, ls="-.", lw=0.8, color=GREEN, label=r"transmission $4.0\times10^{-2}$", zorder=1)
ax.plot(xs, 1.5e-3 * xs, ls="--", lw=0.8, color=RED, label=r"natural-fingerprint upper limit $\lambda_U$", zorder=1)
ax.axhspan(1e-7, 2 * off_se, color=GREY, alpha=0.18, lw=0, label="within 2 SE of the never-injected arms")
marks = [("mark_rand_a3_s0", "contrast_M", BLUE, "o"), ("mark_rand_a3_s1", "contrast_M", BLUE, "o"), ("mark_rand_a3_s2", "contrast_M", BLUE, "o"),
         ("mark_rand_a12_s0", "contrast_M", BLUE, "s"), ("mark_lowmid_a12_s0", "contrast_ML", TEAL, "D")]
done = set()
for arm, key, col, mk in marks:
    e = S[arm]; lab = {"o": r"non-repeating $\pm1$ field, $\alpha=3$ (3 seeds)", "s": r"non-repeating $\pm1$ field, $\alpha=12$", "D": r"non-repeating band-limited field, $\alpha=12$"}[mk]
    o = offL if key == "contrast_ML" else off; y = e[key] - o
    ax.errorbar(e["R_mark"], y, yerr=e[key + "_se"], fmt=mk, ms=4.3, color=col, mfc=col if e["decoy_rank_of_true_M"] == 1 else "white",
                mew=0.9, capsize=2, lw=0.8, label=None if lab in done else lab, zorder=3); done.add(lab)
b0 = PER["band0"]; y0 = max(b0["excess"], 3e-6)
ax.errorbar(b0["R"], y0, yerr=[[min(b0["excess_se"], 0.9 * y0)], [b0["excess_se"]]], fmt="o", ms=4.3, color=GREY, mfc="white", mew=0.9, capsize=2, lw=0.8,
            label="non-repeating top-octave field (not detected)", zorder=3)
# Entry 56: tiles from periodic2_summary (per32/per36 three adapters each; per24/28/40/48 one each)
P2 = json.load(open(os.path.join(T1, "periodic2_summary.json")))["fields"]
TILES = [("per28", ORANGE, "v", "tiles off the latent grid (28, 36 px)"), ("per36", ORANGE, "v", None),
         ("per24", GREEN, "D", "tiles on the latent grid (24, 40, 48 px)"), ("per40", GREEN, "D", None), ("per48", GREEN, "D", None),
         ("per32", GREEN, "*", "tile repeating every 32 px (3 adapters)")]
for key, col, mk, lab in TILES:
    p = P2[key]; y = p["lambda_mean_pct"] / 100 * p["R"]; ye = p["lambda_se_pct"] / 100 * p["R"]
    filled = all(a["decoy_rank"] == 1 for a in p["adapters"])
    ax.errorbar(p["R"], y, yerr=ye, fmt=mk, ms=6 if mk == "*" else 4.2, color=col, mfc=col if filled else "white",
                mew=0.9, capsize=2, lw=0.8, label=lab, zorder=4)
for i, arm in enumerate(("wm_ds_s0", "wm_ds_s1", "wm_ds_s2")):
    e = WM["arms"][arm]
    ax.errorbar(WM["R_wm"], e["contrast"] - wm_off, yerr=e["contrast_se"], fmt="P", ms=5.2, color=GOLD, mfc=GOLD, mew=0.9, capsize=2, lw=0.8,
                label="DiffusionShield, released strength (3 seeds)" if i == 0 else None, zorder=3)
ax.errorbar(R_real, th_sym, yerr=[[0], [U_dev - th_sym]], fmt="^", ms=5.2, color=RED, capsize=2.5, lw=0.9,
            label=r"natural PRNU, 12 adapters: $\theta_{\rm sym}$, bar to $U_{\rm device}$", zorder=5)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.015, 1.5); ax.set_ylim(2e-6, 4e-1)
ax.set_xlabel("stored input contrast in the training images  ($R_{\\rm mark}$; $R_{\\rm real}$ for PRNU)")
ax.set_ylabel("output contrast in generations, offset removed")
ax.text(0.016, 2.6e-6, "filled marker: true field ranks 1st of 31 decoys", fontsize=6.0, color=INK, ha="left", va="bottom")
# legend below both panels, so no line or marker in (a) is hidden behind it
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=3, frameon=False, handlelength=1.8,
           columnspacing=1.4, labelspacing=0.35, fontsize=6.3)
ax.set_title("(a) what passes: spatial structure dominates", fontsize=7.8, loc="left", color=INK)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
# ---- (b) decoy ranks ----
rows = [("base (never)", S["base"]["decoy_rank_of_true_M"], GREY), ("$A$ arm (never)", S["A_raw_s0_r16"]["decoy_rank_of_true_M"], GREY),
        ("$B$ arm (never)", S["B_raw_s0_r16"]["decoy_rank_of_true_M"], GREY), ("$\\pm1$, $\\alpha=1$ (rounds away)", S["mark_rand_a1_s0"]["decoy_rank_of_true_M"], GREY),
        ("$\\pm1$, $\\alpha=3$, s0", S["mark_rand_a3_s0"]["decoy_rank_of_true_M"], BLUE), ("$\\pm1$, $\\alpha=3$, s1", S["mark_rand_a3_s1"]["decoy_rank_of_true_M"], BLUE),
        ("$\\pm1$, $\\alpha=3$, s2", S["mark_rand_a3_s2"]["decoy_rank_of_true_M"], BLUE), ("$\\pm1$, $\\alpha=12$", S["mark_rand_a12_s0"]["decoy_rank_of_true_M"], BLUE),
        ("band-limited, $\\alpha=12$", S["mark_lowmid_a12_s0"]["decoy_rank_of_true_M"], TEAL), ("top-octave field", PER["band0"]["decoy_rank_true"], GREY),
        ]
P2r = json.load(open(os.path.join(T1, "periodic2_summary.json")))["fields"]
for key, lab, col in (("per28", "tile, 28 px", ORANGE), ("per36", "tile, 36 px (worst of 3)", ORANGE), ("per24", "tile, 24 px", GREEN),
                      ("per40", "tile, 40 px", GREEN), ("per48", "tile, 48 px", GREEN), ("per32", "tile, 32 px (worst of 3)", GREEN)):
    rows.append((lab, max(a["decoy_rank"] for a in P2r[key]["adapters"]), col))
rows += [(f"DiffShield, s{i}", WM["arms"][f"wm_ds_s{i}"]["decoy_rank_of_true_W"], GOLD) for i in range(3)]
labels, ranks, cols = zip(*rows)
y = np.arange(len(ranks))[::-1]
bx.barh(y, ranks, color=cols, height=0.62, alpha=0.9)
bx.axvline(16, color=RED, lw=0.8, ls=":"); bx.text(16.4, y[0] + 0.55, "chance median", fontsize=6.0, color=RED, va="bottom")
for yi, r in zip(y, ranks): bx.text(r + 0.4, yi, str(r), va="center", fontsize=6.0, color=INK)
bx.set_yticks(y); bx.set_yticklabels(labels, fontsize=5.8); bx.set_xlim(0, 31.5); bx.set_xlabel("rank of the true field among 31 (1 = best)")
bx.set_title("(b) alignment specificity", fontsize=7.8, loc="left", color=INK)
for s_ in ("top", "right"): bx.spines[s_].set_visible(False)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02); print("wrote", OUT)
