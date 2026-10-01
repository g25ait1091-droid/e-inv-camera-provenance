"""Fig. S9 (fig:s09shift): the shifted-template control (OUTLINE §6 S14 "Fig. S9").
Double column, 6.99 in wide, three panels sharing one row per arm.

Rows: the nine local-stack arms of C5/H3 (base model; body A primary adapter seed 0 and unmarked adapters
seeds 0-2; body B the same), then one pooled row. Every arm is scored against body A's E2 estimate
(c5_shift_grid.json -> fingerprint), circularly shifted.

(a) Profile over residue classes (C5, 150 generations per arm): the mean shifted-template score in each of
    the 64 classes (d_y mod 8, d_x mod 8) of the 769 displacements.  Grey dots: the 63 off-grid classes;
    coloured marker: the on-grid class (0, 0).  Source: arm_results.<arm>.mod8_class_means_rows_dy_cols_dx.
(b) On-grid excess (C5): mean over the 12 on-grid displacements minus the mean over the other 757, +-1 SE
    across displacements.  Pooled row: pooled_adapted (the eight adapted arms).
    Source: arm_results.<arm>.excess_on8_vs_off, pooled_adapted.
(c) Grid-periodic prediction against observation on two disjoint samples of 60 generations per arm
    (H3: images 1-60; H3b: images 61-120).  Filled marker: observed excess, sample 1; open: sample 2;
    +-1 SE as stored (observed_excess_se); grey vertical tick: the prediction for that sample.
    Pooled row: mean over the nine arms, SE = sqrt(sum SE^2)/9 (h3b pooled_on_grid; recomputed for H3 and
    asserted against the numbers.json macros).
    Source: h3_shift_model.json / h3b_shift_model.json -> arms.<arm>.{observed,predicted}_excess.

Every plotted value is read from the result files; every printed number is a numbers.json `text`.
Run:  python src/fv/figS_09_shift.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fv_style import (plt, DC, C, BODY_A, BODY_B, MARK_A, MARK_B, SYM, SYM_MARK,  # noqa: E402
                      panel_label, save)
from matplotlib.ticker import MultipleLocator  # noqa: E402

V2 = EINV.V2
OUT = os.path.join(V2, "out")
NUMJSON = os.path.join(V2, "paper", "fv", "numbers.json")
SCALE = 1e4                      # axes in units of 1e-4 (NCC)
KEY_Y = 1.025                    # axes-fraction baseline of the keys and panel labels above the rows

plt.rcParams["savefig.bbox"] = "standard"      # made at final size: the canvas is exactly DC wide


def load(name):
    with open(os.path.join(OUT, name) if not os.path.isabs(name) else name, encoding="utf-8") as f:
        return json.load(f)


def check(num, macro, value, rel=1e-9):
    v = num[macro]["value"]
    assert abs(v - value) <= rel * max(abs(v), abs(value), 1e-300), (macro, v, value)


def arm_meta(arm):
    """(body, kind, seed) from the arm name; base -> (None, 'base', None)."""
    if arm == "local_base":
        return None, "base", None
    m = re.fullmatch(r"local_([AB])_raw_s(\d+)", arm)
    if m:
        return m.group(1), "primary", int(m.group(2))
    m = re.fullmatch(r"nomark(B?)_s(\d+)", arm)
    if m:
        return ("B" if m.group(1) else "A"), "unmarked", int(m.group(2))
    raise ValueError(arm)


def ordered_arms(arms):
    """Base first; then body A, then body B; within a body primary before unmarked, each by integer seed."""
    def key(a):
        body, kind, seed = arm_meta(a)
        if kind == "base":
            return (0, 0, 0, 0)
        return (1, 0 if body == "A" else 1, 0 if kind == "primary" else 1, seed)
    return sorted(arms, key=key)


def style(arm):
    body, kind, _ = arm_meta(arm)
    if kind == "base":
        return C["grey"], "o"
    return (BODY_A, MARK_A) if body == "A" else (BODY_B, MARK_B)


def label(arm):
    body, kind, seed = arm_meta(arm)
    if kind == "base":
        return "Base model"
    return f"{body}, {kind}, seed {seed}"


def main():
    num = load(NUMJSON)
    c5 = load("c5_shift_grid.json")
    h3 = load("h3_shift_model.json")
    h3b = load("h3b_shift_model.json")

    arms = ordered_arms(c5["arms"])
    assert set(arms) == set(h3["arms"]) == set(h3b["arms"]) and len(arms) == 9
    adapted = [a for a in arms if a != "local_base"]

    # ------------------------------------------------------------------ consistency with the macros
    d = np.asarray(c5["design"]["displacements_dy_dx"])
    on8 = (d[:, 0] % 8 == 0) & (d[:, 1] % 8 == 0)
    cls_count = np.bincount((d[:, 0] % 8) * 8 + d[:, 1] % 8, minlength=64)
    check(num, "nShiftDisplacements", len(d))
    check(num, "nShiftOnGrid", int(on8.sum()))
    check(num, "nShiftImgPerArm", c5["n_images_per_arm"])
    check(num, "nShiftArms", len(arms))
    base = c5["arm_results"]["local_base"]["excess_on8_vs_off"]
    check(num, "nShiftBaseExcess", base["excess"])
    check(num, "nShiftBaseSE", base["se"])
    check(num, "nShiftBaseZ", base["z"])
    zs = [c5["arm_results"][a]["excess_on8_vs_off"]["z"] for a in adapted]
    exs = [c5["arm_results"][a]["excess_on8_vs_off"]["excess"] for a in adapted]
    check(num, "nShiftAdaptedZLow", min(zs)); check(num, "nShiftAdaptedZHigh", max(zs))
    check(num, "nShiftAdaptedExcessLow", min(exs)); check(num, "nShiftAdaptedExcessHigh", max(exs))
    check(num, "nShiftPooledExcess", c5["pooled_adapted"]["excess"])
    check(num, "nShiftPooledZ", c5["pooled_adapted"]["z"])
    assert abs(np.mean(exs) - c5["pooled_adapted"]["excess"]) < 1e-12      # pooled = mean of the 8 arms
    check(num, "nShiftHthreeImages", h3["images_per_arm"])
    assert h3b["images_per_arm"] == h3["images_per_arm"] and h3b["image_offset"] == h3["images_per_arm"]

    def pooled(h):
        o = np.array([h["arms"][a]["observed_excess"] for a in arms])
        p = np.array([h["arms"][a]["predicted_excess"] for a in arms])
        s = np.array([h["arms"][a]["observed_excess_se"] for a in arms])
        return float(o.mean()), float(p.mean()), float(np.sqrt((s ** 2).sum()) / len(s))
    p1, p2 = pooled(h3), pooled(h3b)
    check(num, "nShiftHthreeObs", p1[0]); check(num, "nShiftHthreePred", p1[1]); check(num, "nShiftHthreeSE", p1[2])
    pb = h3b["pooled_on_grid"]
    for mine, key, mac in ((p2[0], "observed_excess", "nShiftHthreebObs"), (p2[1], "predicted_excess", "nShiftHthreebPred"),
                           (p2[2], "se", "nShiftHthreebSE")):
        assert abs(mine - pb[key]) < 1e-15, key
        check(num, mac, pb[key])

    # ------------------------------------------------------------------ layout
    n = len(arms)
    y_arm = {a: float(i) for i, a in enumerate(arms)}
    y_pool = n + 0.35
    H_IN = 2.62
    L, Rm, T, B = 1.02, 0.06, 0.27, 0.40
    gap = 0.22
    widths = np.array([2.05, 1.72, 1.92])
    widths = widths * (DC - L - Rm - 2 * gap) / widths.sum()
    fig = plt.figure(figsize=(DC, H_IN))
    axs, x0 = [], L
    for w in widths:
        axs.append(fig.add_axes([x0 / DC, B / H_IN, w / DC, (H_IN - T - B) / H_IN]))
        x0 += w + gap
    axa, axb, axc = axs

    ylim = (y_pool + 0.6, -0.6)
    for ax in axs:
        ax.set_ylim(*ylim)
        ax.axvline(0, color=C["light"], lw=0.6, zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axa.set_yticks([y_arm[a] for a in arms] + [y_pool])
    axa.set_yticklabels([label(a) for a in arms] + ["Pooled"])
    for ax in (axb, axc):
        ax.set_yticks([])
    # light separators between the groups (base | A | B | pooled)
    groups = [0.5, 0.5 + sum(1 for a in arms if arm_meta(a)[0] == "A")]
    for ax in axs:
        for yy in groups + [n - 0.5 + 0.02]:
            ax.axhline(yy, color=C["light"], lw=0.4, ls=(0, (1, 2)), zorder=0)

    # ------------------------------------------------------------------ (a) residue-class profile
    rng = np.random.default_rng(0)          # vertical jitter of the off-grid dots only (display)
    for a in arms:
        m = np.asarray(c5["arm_results"][a]["mod8_class_means_rows_dy_cols_dx"], dtype=float)
        assert m.shape == (8, 8)
        assert abs(m[0, 0] - c5["arm_results"][a]["excess_on8_vs_off"]["mean_on8"]) < 1e-15
        off = np.delete(m.ravel(), 0)
        y = y_arm[a]
        axa.plot(off * SCALE, y + rng.uniform(-0.22, 0.22, off.size), ls="none", marker="o", ms=1.7,
                 mfc=C["light"], mec="none", zorder=2)
        col, mk = style(a)
        axa.plot([m[0, 0] * SCALE], [y], ls="none", marker=mk, ms=4.0, mfc=col, mec="white", mew=0.4, zorder=4)
    axa.set_xlim(-2.0, 2.0)
    axa.xaxis.set_major_locator(MultipleLocator(1.0))
    axa.xaxis.set_minor_locator(MultipleLocator(0.5))
    axa.set_xlabel(r"Class-mean score ($\times10^{-4}$)", labelpad=2)
    # direct labels, written once above the first row
    on_grid = [c5["arm_results"][a]["mod8_class_means_rows_dy_cols_dx"][0][0] * SCALE for a in arms]
    axa.text(-0.55, KEY_Y, "other 63 classes", transform=axa.get_xaxis_transform(), ha="center",
             va="bottom", fontsize=7, color=C["grey"])
    axa.text(float(np.median(on_grid)), KEY_Y, "class (0, 0)", transform=axa.get_xaxis_transform(),
             ha="center", va="bottom", fontsize=7, color=C["ink"])

    # ------------------------------------------------------------------ (b) on-grid excess, C5
    for a in arms:
        r = c5["arm_results"][a]["excess_on8_vs_off"]
        col, mk = style(a)
        axb.errorbar([r["excess"] * SCALE], [y_arm[a]], xerr=[r["se"] * SCALE], fmt=mk, ms=4.0, mfc=col,
                     mec="white", mew=0.4, ecolor=col, elinewidth=0.8, capsize=0, zorder=4)
    pa = c5["pooled_adapted"]
    axb.errorbar([pa["excess"] * SCALE], [y_pool], xerr=[pa["se"] * SCALE], fmt=SYM_MARK, ms=4.0, mfc=SYM,
                 mec="white", mew=0.4, ecolor=SYM, elinewidth=0.8, capsize=0, zorder=4)
    axb.set_xlim(-0.5, 2.0)
    axb.xaxis.set_major_locator(MultipleLocator(0.5))
    axb.xaxis.set_minor_locator(MultipleLocator(0.25))
    axb.set_xlabel(r"On-grid excess ($\times10^{-4}$)", labelpad=2)
    zx = 1.98
    axb.text(zx, y_arm["local_base"], f"$z$ = {num['nShiftBaseZ']['text']}", ha="right", va="center",
             fontsize=7, color=C["ink"])
    axb.text(zx, y_pool, f"$z$ = {num['nShiftPooledZ']['text']}", ha="right", va="center", fontsize=7,
             color=C["ink"])

    # ------------------------------------------------------------------ (c) H3 / H3b
    dy = 0.2
    tick_h = 0.34
    for a in arms + ["__pooled__"]:
        if a == "__pooled__":
            y, col, mk = y_pool, SYM, SYM_MARK
            rows = [(p1[0], p1[2], p1[1]), (p2[0], p2[2], p2[1])]
        else:
            y = y_arm[a]
            col, mk = style(a)
            rows = [(h["arms"][a]["observed_excess"], h["arms"][a]["observed_excess_se"],
                     h["arms"][a]["predicted_excess"]) for h in (h3, h3b)]
        for k, (obs, se, pred) in enumerate(rows):
            yy = y - dy if k == 0 else y + dy
            axc.plot([pred * SCALE] * 2, [yy - tick_h / 2, yy + tick_h / 2], color=C["grey"], lw=1.0,
                     solid_capstyle="butt", zorder=3)
            axc.errorbar([obs * SCALE], [yy], xerr=[se * SCALE], fmt=mk, ms=3.4 if k else 3.6,
                         mfc=col if k == 0 else "white", mec=col, mew=0.8 if k else 0.4,
                         ecolor=col, elinewidth=0.7, capsize=0, zorder=4)
    axc.set_xlim(-3.0, 4.0)
    axc.xaxis.set_major_locator(MultipleLocator(1.0))
    axc.xaxis.set_minor_locator(MultipleLocator(0.5))
    axc.set_xlabel(r"On-grid excess ($\times10^{-4}$)", labelpad=2)
    # key, written once above the first row (axes-fraction y)
    tr = axc.get_xaxis_transform()
    kx = [-2.1, 0.2, 2.45]
    th = 0.045
    axc.plot([kx[0]], [KEY_Y + th / 2], ls="none", marker="o", ms=3.6, mfc=C["ink"], mec="white", mew=0.4,
             transform=tr, clip_on=False)
    axc.text(kx[0] + 0.17, KEY_Y, "sample 1", transform=tr, va="bottom", fontsize=7, color=C["ink"])
    axc.plot([kx[1]], [KEY_Y + th / 2], ls="none", marker="o", ms=3.4, mfc="white", mec=C["ink"], mew=0.8,
             transform=tr, clip_on=False)
    axc.text(kx[1] + 0.17, KEY_Y, "sample 2", transform=tr, va="bottom", fontsize=7, color=C["ink"])
    axc.plot([kx[2]] * 2, [KEY_Y + th / 2 - 0.028, KEY_Y + th / 2 + 0.028], color=C["grey"], lw=1.0,
             transform=tr, clip_on=False, solid_capstyle="butt")
    axc.text(kx[2] + 0.14, KEY_Y, "prediction", transform=tr, va="bottom", fontsize=7, color=C["grey"])

    # panel labels "(a)" 9 pt regular, upper left, above the plotting area (clear of the key in (c))
    for ax, s in zip(axs, ("(a)", "(b)", "(c)")):
        panel_label(ax, s, x=-0.03, y=KEY_Y - 0.004)

    out = save(fig, "figS_09_shift")

    # ------------------------------------------------------------------ spot checks for the record
    print("class counts per mod-8 class: min", cls_count.min(), "max", cls_count.max(),
          "class with 13:", [divmod(int(i), 8) for i in np.where(cls_count == 13)[0]])
    for a in arms:
        m = np.asarray(c5["arm_results"][a]["mod8_class_means_rows_dy_cols_dx"]).ravel()
        r = c5["arm_results"][a]["excess_on8_vs_off"]
        print(f"{a:16s} (0,0) {m[0]:+.3e} rank {int((m > m[0]).sum()) + 1}/64 | C5 excess {r['excess']:+.3e} "
              f"se {r['se']:.2e} | H3 obs {h3['arms'][a]['observed_excess']:+.3e} pred "
              f"{h3['arms'][a]['predicted_excess']:+.3e} | H3b obs {h3b['arms'][a]['observed_excess']:+.3e} "
              f"pred {h3b['arms'][a]['predicted_excess']:+.3e}")
    print("pooled C5", pa["excess"], pa["se"], "| H3", p1, "| H3b", p2)
    return out


if __name__ == "__main__":
    print(main())
