"""Figure 9 — the frequency response of personalization for non-repeating patterns: autoencoder-only
transmission (Entry 37 A1) and full-pipeline transmission (A2) per octave band, with the fingerprint's
energy distribution and the repeating tiles and watermark marked for comparison. Undetected bands are
drawn as 2-SE upper limits. Reads out/t1/band_summary.json, band_fields.json, periodic2_summary.json, band2_summary.json,
wm_summary.json; writes paper/fig9_bands.pdf. Style matches make_figures_v2.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman"], "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.4, "axes.linewidth": 0.6})
BLUE, ORANGE, TEAL, RED, GREY, INK, GOLD, GREEN = "#2E5C8A", "#C8641E", "#0B7A6E", "#A8432F", "#8A8A8A", "#1F2A37", "#8A6D1F", "#3B7D23"
T1 = os.path.join(EINV.V2, 'out', 't1'); OUT = os.path.join(EINV.V2, 'paper', 'fig9_bands.pdf')
B = json.load(open(os.path.join(T1, "band_summary.json"))); F = json.load(open(os.path.join(T1, "band_fields.json")))
P2 = json.load(open(os.path.join(T1, "periodic2_summary.json")))["fields"]; WM = json.load(open(os.path.join(T1, "wm_summary.json")))
B2 = json.load(open(os.path.join(T1, "band2_summary.json")))["bands"]
curve = [dict(c) for c in B["curve"]]; nb = len(curve)
# Entry 56: the two finest octaves from three adapters each (adapter-level SE)
for b in (0, 1):
    r = B2[f"band{b}"]; curve[b]["T_full"] = r["T_mean_pct"] / 100; curve[b]["T_full_se"] = r["T_se_adapter_pct"] / 100
    curve[b]["detected_3se"] = curve[b]["T_full"] > 3 * curve[b]["T_full_se"]
labels = ["2–4", "4–8", "8–16", "16–32", "32–64", "64–128"]
x = np.arange(nb)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.7), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.38})
# (a) transmission per band, log scale
Tv = [c["T_vae"] for c in curve]
ax.plot(x, Tv, "-o", color=BLUE, ms=4, lw=1.0, label="autoencoder alone")
for c, xi in zip(curve, x):
    T, se = c["T_full"], c["T_full_se"]
    if c["detected_3se"]:
        ax.errorbar(xi, T, yerr=2 * se, fmt="s", ms=4.5, color=RED, capsize=2, lw=0.9, zorder=3)
    else:
        up = max(T, 0) + 2 * se
        ax.errorbar(xi, up, yerr=[[up * 0.45], [0]], fmt="_", ms=8, color=RED, lw=0.9, uplims=True, capsize=2, zorder=3)
ax.plot([], [], "s", color=RED, ms=4.5, label="full pipeline (detected, ±2 SE; finest two: 3 adapters)")
ax.plot([], [], marker="$\\downarrow$", ls="none", color=RED, ms=7, label="full pipeline, not detected (2-SE upper limit)")
lam = lambda k: P2[k]["lambda_mean_pct"] / 100
ax.axhline(lam("per32"), color=GREEN, lw=0.8, ls="--")
ax.text(nb - 0.55, lam("per32") * 1.12, "tile every 32 px (3 adapters)", color=GREEN, fontsize=6.2, ha="right", va="bottom")
on = [lam(k) for k in ("per24", "per40", "per48")]; off = [lam(k) for k in ("per28", "per36")]
ax.axhspan(min(on), max(on), color=GREEN, alpha=0.15, lw=0)
ax.text(nb - 0.55, max(on) * 1.08, "tiles on the 8-px grid (24, 40, 48 px)", color=GREEN, fontsize=6.2, ha="right", va="bottom")
ax.axhspan(min(off), max(off), color=ORANGE, alpha=0.18, lw=0)
ax.text(nb - 0.55, max(off) * 1.1, "off-grid tiles (28, 36 px)", color=ORANGE, fontsize=6.2, ha="right", va="bottom")
ax.axhline(WM["cluster"]["lambda_wm_offset_corrected_pct"] / 100, color=GOLD, lw=0.8, ls=":")
ax.text(-0.3, WM["cluster"]["lambda_wm_offset_corrected_pct"] / 100 * 0.93, "DiffusionShield", color=GOLD, fontsize=6.2, ha="left", va="top")
ax.set_yscale("log"); ax.set_ylim(3e-5, 2.0); ax.set_xlim(-0.4, nb - 0.4)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlabel("pattern period of the octave band (pixels)")
ax.set_ylabel("transmission (output / stored input contrast)")
ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="none", handlelength=1.4, borderpad=0.3, labelspacing=0.25)
ax.set_title("(a) non-repeating patterns, by frequency band", fontsize=7.8, loc="left", color=INK)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
# (b) where the fingerprint's energy lies
e = F["energy_fraction_K_A_E2"]
bx.bar(x, [100 * v for v in e], color=GREY, width=0.62)
for xi, v in zip(x, e): bx.text(xi, 100 * v + 1.2, f"{100 * v:.1f}" if v >= 0.001 else "<0.1", ha="center", fontsize=6.2, color=INK)
bx.set_xticks(x); bx.set_xticklabels(labels); bx.set_xlabel("pattern period of the octave band (pixels)")
bx.set_ylabel("share of the fingerprint's energy (%)"); bx.set_ylim(0, 70)
bx.set_title("(b) where the camera fingerprint's energy lies", fontsize=7.8, loc="left", color=INK)
for s_ in ("top", "right"): bx.spines[s_].set_visible(False)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02); print("wrote", OUT)
