"""Fig. 4 (fig:map): the transmission map.  Double column, 6.99 x 2.9 in (OUTLINE section 4, Fig. 4).

(a) Transmission of designed patterns through personalization (2000 steps, local stack, body-A training crops
    of the Nikon D200 pair), % of each pattern's own stored contrast, log axis.  Five groups, left to right:
      1  non-repeating octaves 2-4, 4-8, 8-16 px      band2_summary.json (bands 0-1), band_summary.json (curve[2])
      2  non-repeating fields: random field with the fingerprint's spectrum (additive 1 and 4 gray,
         multiplicative 4 gray) and body B's fingerprint estimate K_B^E2 injected at alpha = 12 and 48
                                                                                  kfield_summary.json
      3  tiles off the 8-px latent grid, 28 and 36 px                          periodic2_summary.json
      4  tiles on the 8-px latent grid, 24/32/40/48 px                         periodic2_summary.json
      5  DiffusionShield (three adapters)                                      wm_summary.json
    Per-adapter values are small hollow markers (seed order, left to right); the mean +-1 SE is filled.
    A dashed band marks the transmission expected of a non-repeating field with each tile's spectrum
    (Entry 56 method via num_n2.band_response/predict, the code behind nMapSpecPredMin/Max), and one
    dashed segment spans group 2 at the prediction fixed before those arms for K_B's spectrum, which the
    random fields share (kfield_summary.json prediction_K_spectrum_pct = nKnownPred, Entries 61/69/72).
(b) What was stored: luminance of training crop 0000 of six injected sets minus that of the unmarked crop
    (none_a0/0000.png), a 96-px window on the 8-px grid, with 8-px tick marks on the top and left edges.

Every plotted value is read from the result files; nothing is typed in.  The script asserts that every
plotted mean equals its printed macro in numbers.json.  Run:
    python src/fv/fig_04_map.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import json
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_style import plt, DC, C, BODY_B, MARK_B, panel_label, save  # noqa: E402
import num_n2  # noqa: E402  (band_response / predict: the spectrum-matched non-repeating prediction)

from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

T1 = (EINV.V2 + "/out/t1")
TRAIN = T1 + "/train_png"
MANIFEST = (EINV.V2 + "/out/fp/manifest.json")
NUMJSON = (EINV.V2 + "/paper/fv/numbers.json")
W_IN, H_IN = DC, 2.9
YMIN, YMAX = 1e-3, 10.0
CROP = 96                    # px shown in (b)
Y0 = X0 = 464                # window origin: a multiple of 8, near the crop centre (464 = 58 x 8)
GRAY_RANGE, K_RANGE = 8.0, 3.0   # display scales in (b), gray levels (design constants of the figure)

plt.rcParams["savefig.bbox"] = "standard"      # keep the canvas exactly DC wide


def load(name):
    with open(os.path.join(T1, name), encoding="utf-8") as f:
        return json.load(f)


def seed(arm):
    return int(re.search(r"_s(\d+)$", arm).group(1))


def lum(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


# ------------------------------------------------------------------------------------------------ data
def collect(num):
    """Return the plotted items, each with its adapters (seed order), mean and SE, all in %."""
    b2 = load("band2_summary.json")["bands"]
    b1 = load("band_summary.json")
    kf = load("kfield_summary.json")["fields"]
    p2 = load("periodic2_summary.json")["fields"]
    wm = load("wm_summary.json")
    items = []

    # (1) non-repeating octaves: bands 0-1 have three adapters, band 2 one
    for b, lab in ((0, "2\u20134"), (1, "4\u20138")):
        d = b2[f"band{b}"]
        ad = sorted(d["adapters"], key=lambda a: seed(a["arm"]))
        items.append(dict(g=0, lab=lab, kind="nonrep", ad=[a["T_pct"] for a in ad],
                          m=d["T_mean_pct"], se=d["T_se_adapter_pct"]))
    c2 = [c for c in b1["curve"] if c["band"] == 2][0]
    n2 = [a for a in b1["arms"] if re.fullmatch(r"band2_s\d+", a)]
    assert len(n2) == 1
    items.append(dict(g=0, lab="8\u201316", kind="nonrep", ad=[100 * c2["T_full"]],
                      m=100 * c2["T_full"], se=100 * c2["T_full_se"]))

    # (2) non-repeating fields: random field with K's spectrum, then K itself (body B's E2 estimate)
    for key, lab, kind, sub in (("gkadd_a1", "1", "nonrep", "f"), ("gkadd_a4", "4", "nonrep", "f"),
                                ("gkmul_a4", "4\n(mult.)", "nonrep", "f"),
                                ("kinj_a12", "12", "fp", "k"), ("kinj_a48", "48", "fp", "k")):
        d = kf[key]
        ad = sorted(d["adapters"], key=lambda a: seed(a["arm"]))
        items.append(dict(g=1, sub=sub, lab=lab, kind=kind, ad=[a["T_pct"] for a in ad],
                          m=d["T_mean_pct"], se=d["T_se_pct"]))

    # (3) off-grid and (4) on-grid tiles
    for g, pers in ((2, (28, 36)), (3, (24, 32, 40, 48))):
        for p in pers:
            d = p2[f"per{p}"]
            assert d["on_latent_grid_8px"] == (g == 3)
            ad = sorted(d["adapters"], key=lambda a: seed(a["arm"]))
            items.append(dict(g=g, lab=f"{p}", kind="on" if g == 3 else "off",
                              ad=[a["lambda_pct"] for a in ad], m=d["lambda_mean_pct"], se=d["lambda_se_pct"]))

    # (5) DiffusionShield: per-arm (contrast - never-injected offset) / stored contrast
    cl, R = wm["cluster"], wm["R_wm"]
    arms = sorted([a for a in wm["arms"] if a.startswith("wm_ds_s")], key=seed)
    per = [100 * (wm["arms"][a]["contrast"] - cl["never_injected_offset"]) / R for a in arms]
    assert np.allclose([wm["arms"][a]["contrast"] for a in arms], cl["per_seed"])
    m = cl["lambda_wm_offset_corrected_pct"]
    assert abs(np.mean(per) - m) < 1e-9
    se = 100 * cl["sd"] / np.sqrt(len(arms)) / R
    items.append(dict(g=4, lab="Diffusion-\nShield", kind="on", ad=per, m=m, se=se))
    # the prediction for K_B's spectrum (shared by the random fields), fixed before group 2 was trained; in %
    kpred = load("kfield_summary.json")["prediction_K_spectrum_pct"]

    # spectrum-matched non-repeating prediction for each tile (as in num_n2 / Entry 56)
    T, SE = num_n2.band_response()
    Fe = dict(load("periodic_fields.json"))
    Fe.update(load("periodic2_fields.json"))
    pred = {f"{p}": 100 * num_n2.predict(Fe[f"per{p}"]["energy_fraction_by_band_b0_b5"], T, SE)[0]
            for p in (24, 28, 32, 36, 40, 48)}
    preds = list(pred.values())
    # the same prediction for DiffusionShield's luminance spectrum (num_n2, nMapSpecPredWm)
    pred["Diffusion-\nShield"] = 100 * num_n2.predict(
        load("band_fields.json")["energy_fraction_DiffusionShield_lum"], T, SE)[0]
    for it in items:
        if it["g"] in (2, 3, 4):               # tiles and DiffusionShield only (K's "48" is an alpha, not a period)
            it["pred"] = pred[it["lab"]]

    # ---- plotted values must agree with the printed macros
    chk = [("nMapBandFinest", items[0]["m"], 2), ("nMapBandFourEight", items[1]["m"], 3),
           ("nMapBandEightSixteen", items[2]["m"], 3), ("nKnownFieldOne", items[3]["m"], 3),
           ("nKnownFieldFour", items[4]["m"], 3), ("nKnownFieldMul", items[5]["m"], 3),
           ("nKnownAlphaTwelve", items[6]["m"], 3), ("nKnownAlphaFortyEight", items[7]["m"], 3),
           ("nMapTileTwentyEight", items[8]["m"], 3), ("nMapTileThirtySix", items[9]["m"], 3),
           ("nMapTileTwentyFour", items[10]["m"], 4), ("nMapTileThirtyTwo", items[11]["m"], 4),
           ("nMapTileForty", items[12]["m"], 4), ("nMapTileFortyEight", items[13]["m"], 4),
           ("nMapWm", items[14]["m"], 3), ("nMapSpecPredMin", min(preds), 2), ("nMapSpecPredMax", max(preds), 2),
           ("nMapSpecPredWm", items[14]["pred"], 2),
           ("nMapBandFinestSeedOne", items[0]["ad"][1], 2), ("nMapTileThirtyTwoSeedZero", items[11]["ad"][0], 3),
           ("nKnownAlphaTwelveSeedTwo", items[6]["ad"][2], 3), ("nKnownPred", kpred, 3)]
    for name, v, sigf in chk:
        assert f"{float(num[name]['text']):.{sigf}g}" == f"{v:.{sigf}g}", (name, num[name]["text"], v)
    return items, min(preds), max(preds), kpred


# ------------------------------------------------------------------------------------------------ figure
def main():
    with open(NUMJSON, encoding="utf-8") as f:
        num = json.load(f)
    items, pmin, pmax, kpred = collect(num)

    fig = plt.figure(figsize=(W_IN, H_IN))
    ink, grey, light = C["ink"], C["grey"], C["light"]

    # ---------------- (a) dot plot, left 70 %
    La, Ba, Ta = 0.50, 0.68, 0.14          # inches: left, bottom, top margins of (a)
    XB0 = 4.78                             # inches: left edge of (b)
    wa = XB0 - La - 0.16
    ha = H_IN - Ba - Ta
    ax = fig.add_axes([La / W_IN, Ba / H_IN, wa / W_IN, ha / H_IN])
    ax.set_yscale("log")
    ax.set_ylim(YMIN, YMAX)

    GAP, SUBGAP = 0.75, 0.35                # extra space between groups / between field and K, item units
    xs, x, prev = [], 0.0, None
    for it in items:
        if prev is not None:
            x += (1.15 if it["g"] == 0 else 1.0) + (GAP if it["g"] != prev["g"] else 0.0) \
                + (SUBGAP if it.get("sub") and prev.get("sub") and it["sub"] != prev["sub"] else 0.0)
        xs.append(x)
        prev = it
    xs = np.array(xs)
    ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)

    style = dict(nonrep=(grey, "o"), off=(grey, "o"), on=(ink, "o"), fp=(BODY_B, MARK_B))
    for it, x0 in zip(items, xs):
        col, mk = style[it["kind"]]
        n = len(it["ad"])
        xm = x0 + (0.16 if n > 1 else 0.0)
        # per-adapter values (hollow), seed order left to right
        if n > 1:
            xa = x0 - 0.20 + 0.1 * np.arange(n) - (0.05 if n == 2 else 0.0)
            for xi, v in zip(xa, it["ad"]):
                if v <= YMIN:
                    ax.annotate("", xy=(xi, YMIN), xytext=(xi, YMIN * 2.2),
                                arrowprops=dict(arrowstyle="-|>", color=col, lw=0.5, mutation_scale=4))
                else:
                    ax.plot(xi, v, ls="none", marker=mk, ms=2.6, mfc="white", mec=col, mew=0.55, zorder=3)
        # mean +- 1 SE (filled); a lower end at or below the floor runs to the axis
        lo, hi = it["m"] - it["se"], it["m"] + it["se"]
        ax.vlines(xm, max(lo, YMIN), hi, color=col, lw=0.7, zorder=3, clip_on=False)
        cw = 0.07
        ax.hlines(hi, xm - cw, xm + cw, color=col, lw=0.7, zorder=3)
        if lo > YMIN:
            ax.hlines(lo, xm - cw, xm + cw, color=col, lw=0.7, zorder=3)
        ax.plot(xm, it["m"], ls="none", marker=mk, ms=3.8, mfc=col, mec=col, mew=0.5, zorder=4)

    # spectrum-matched non-repeating prediction: one thin dashed segment per tile and for DiffusionShield,
    # each at that pattern's own prediction; label once, at the right end of the last tile's segment
    last = None
    for it, x0 in zip(items, xs):
        if "pred" in it:
            ax.hlines(it["pred"], x0 - 0.46, x0 + 0.46, color=grey, lw=0.6, ls=(0, (3, 2)), zorder=1)
            if it["g"] == 3:
                last = (x0 + 0.46, it["pred"])
    ax.text(last[0], last[1] / 1.25, "non-repeating\nprediction", ha="right", va="top", fontsize=8,
            color=grey, linespacing=1.0)
    # the random fields and K_B share one spectrum, hence one prediction: a single segment across group 2,
    # labelled just above its right end (the group's markers all lie below it)
    g1 = xs[[i for i, it in enumerate(items) if it["g"] == 1]]
    ax.hlines(kpred, g1.min() - 0.46, g1.max() + 0.46, color=grey, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(g1.max() + 0.46, kpred * 1.2, "non-repeating\nprediction", ha="right", va="bottom", fontsize=8,
            color=grey, linespacing=1.0)

    # group separators
    groups = sorted(set(it["g"] for it in items))
    gx = {g: xs[[i for i, it in enumerate(items) if it["g"] == g]] for g in groups}
    for g in groups[:-1]:
        xsep = 0.5 * (gx[g].max() + gx[g + 1].min())
        ax.axvline(xsep, color=light, lw=0.4, zorder=0)

    # tick labels (tier 1), what was varied (tier 2), and the category (tier 3, with a thin rule)
    ax.set_xticks(xs)
    ax.set_xticklabels([it["lab"] for it in items], fontsize=8, linespacing=0.95, va="top")
    ax.tick_params(axis="x", length=0, pad=2.5)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    sub = lambda s: xs[[i for i, it in enumerate(items) if it.get("sub") == s]]  # noqa: E731
    tier2 = [(gx[0].mean(), "octave (px)"), (sub("f").mean(), "random field (gray)"),
             (sub("k").mean(), "$\\hat{K}_{\\mathrm{B}}$ ($\\alpha$)"),
             (gx[2].mean(), "tile (px)"), (gx[3].mean(), "tile (px)")]
    Y2, Y3 = -0.285 / ha, -0.455 / ha          # tiers 2 and 3, inches below the axis
    for xc, s in tier2:
        ax.text(xc, Y2 - 0.09 / ha, s, transform=tr, ha="center", va="baseline", fontsize=8, color=ink)
    cats = [((gx[0].min(), gx[1].max()), "non-repeating", grey),
            ((gx[2].min(), gx[2].max()), "off 8-px grid", grey),
            ((gx[3].min(), gx[4].max()), "on 8-px grid", ink)]
    x_l, x_r = ax.get_xlim()
    seps = [x_l] + [0.5 * (gx[g].max() + gx[g + 1].min()) for g in groups[:-1]] + [x_r]
    for (a0, a1), s, col in cats:
        e0 = max(v for v in seps if v < a0) + 0.12          # rule runs between the enclosing separators
        e1 = min(v for v in seps if v > a1) - 0.12
        ax.plot([e0, e1], [Y3, Y3], transform=tr, color=col, lw=0.5, clip_on=False, solid_capstyle="butt")
        ax.text(0.5 * (e0 + e1), Y3 - 0.03 / ha, s, transform=tr, ha="center", va="top", fontsize=8, color=col)

    ax.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1, 10]))
    ax.set_yticklabels(["0.001", "0.01", "0.1", "1", "10"])
    ax.yaxis.set_minor_locator(FixedLocator([k * 10.0 ** e for e in range(-3, 1) for k in range(2, 10)]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Transmission (% of stored contrast)", labelpad=3)

    # ---------------- (b) stored patterns, right 30 %: two rows of three crops, spanning the height of (a)
    none = lum(os.path.join(TRAIN, "none_a0", "0000.png"))
    crops = [("per32_a4", "32-px tile", GRAY_RANGE), ("per36_a4", "36-px tile", GRAY_RANGE),
             ("dswm_a1", "DiffusionShield", GRAY_RANGE),
             ("band0_a4", "2\u20134-px octave", GRAY_RANGE),
             ("kinj_a12", "$\\hat{K}_{\\mathrm{B}}$, $\\alpha = 12$", K_RANGE),
             ("gkadd_a4", "random field", GRAY_RANGE)]
    xb0 = XB0
    gap_in = 0.05
    cw_in = (W_IN - 0.01 - xb0 - 2 * gap_in) / 3
    lab_h = 0.20                                             # room for the label beneath a crop
    top = Ba + ha                                            # align the first row with the top of (a)
    row_top = [top, Ba + lab_h + cw_in]                      # second row: its label ends at the axis of (a)
    axes_b = []
    for i, (d, lab, rng) in enumerate(crops):
        r, c = divmod(i, 3)
        left = xb0 + c * (cw_in + gap_in)
        bottom = row_top[r] - cw_in
        axb = fig.add_axes([left / W_IN, bottom / H_IN, cw_in / W_IN, cw_in / H_IN])
        diff = lum(os.path.join(TRAIN, d, "0000.png")) - none
        win = diff[Y0:Y0 + CROP, X0:X0 + CROP]
        axb.imshow(win, cmap="gray", vmin=-rng, vmax=rng, interpolation="nearest")
        # 8-px latent-grid ticks on the top and left edges, pointing outward; no labels
        tk = np.arange(0, CROP + 1, 8) - 0.5
        axb.set_xticks(tk); axb.set_yticks(tk)
        axb.xaxis.tick_top()
        axb.tick_params(length=1.4, width=0.3, color=ink, labeltop=False, labelleft=False,
                        labelbottom=False, direction="out", pad=0)
        axb.set_xlim(-0.5, CROP - 0.5); axb.set_ylim(CROP - 0.5, -0.5)
        for s in axb.spines.values():
            s.set_visible(True); s.set_linewidth(0.4); s.set_color(ink)
        axb.set_xlabel(lab, fontsize=8, labelpad=2)
        axes_b.append(axb)

    # panel labels at the same height, just above the top of (a)
    ylab = (Ba + ha + 0.02) / H_IN
    fig.text(0.02 / W_IN, ylab, "(a)", ha="left", va="bottom", fontsize=9)
    fig.text((xb0 - 0.10) / W_IN, ylab, "(b)", ha="left", va="bottom", fontsize=9)

    with open(MANIFEST, encoding="utf-8") as f:
        src = json.load(f)["A"]["T"][0]
    print("crop 0000 of body A's T split:", src)
    return save(fig, "fig04_map")


if __name__ == "__main__":
    print(main())
