"""Fig. S8 (fig:s08weights): cosine between adapter updates dW = BA, both doses (OUTLINE §6 S13 "Fig. S8").
Double column, 6.99 in wide, three panels:
    (a) 12 x 12 cosine matrix at 2000 steps   (out/h5_weight_signature.json -> doses.2000.cosine_matrix)
    (b) 12 x 12 cosine matrix at 16000 steps  (doses.16000.cosine_matrix)
    (c) the 66 pair cosines per dose, grouped as the registered statistic D groups them (same body /
        different bodies), with the group means doses.*.mean_cos_same_body / mean_cos_diff_body.
Adapters are ordered body A then body B, each by integer seed. The diagonal (1 by construction) is left
blank. Colour is log-scaled (grey sequential) because the off-diagonal values span more than a decade.
In (c) the cross-body pairs whose two adapters share a training seed are drawn as open markers; the grouping
is read from the adapter names in the file, and is descriptive (it is not part of the registered test).

Every plotted value is read from the result file; the printed D and p are numbers.json `text`.  Run:
    python src/fv/figS_08_weights.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import itertools
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fv_style import plt, DC, C, BODY_A, BODY_B, MARK_A, MARK_B, SYM, panel_label, save  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

V2 = EINV.V2
SRC = os.path.join(V2, "out", "h5_weight_signature.json")
NUMJSON = os.path.join(V2, "paper", "fv", "numbers.json")
H_IN = 2.45
VMIN, VMAX = 0.008, 0.7                      # colour range, chosen to enclose every off-diagonal value

plt.rcParams["savefig.bbox"] = "standard"    # made at final size: the canvas is exactly DC wide


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def parse(name):
    """nomark_s3 -> ('A', 3); nomarkB_s3 -> ('B', 3); dose16k_A_s3 -> ('A', 3); dose16k_B_s3 -> ('B', 3)."""
    m = re.fullmatch(r"(nomark|nomarkB|dose16k_A|dose16k_B)_s(\d+)", name)
    assert m, name
    body = {"nomark": "A", "nomarkB": "B", "dose16k_A": "A", "dose16k_B": "B"}[m.group(1)]
    return body, int(m.group(2))


def ordered(dose):
    """Matrix and labels re-ordered body A then B, each by integer seed (the file order is checked, not trusted)."""
    names = dose["adapters"]
    M = np.asarray(dose["cosine_matrix"], dtype=float)
    assert M.shape == (len(names), len(names)) and np.allclose(M, M.T) and np.allclose(np.diag(M), 1.0)
    keys = [parse(n) for n in names]
    idx = sorted(range(len(names)), key=lambda i: (keys[i][0], keys[i][1]))
    return M[np.ix_(idx, idx)], [keys[i] for i in idx]


def check(num, macro, value):
    v = num[macro]["value"]
    assert abs(v - value) <= 1e-12 * max(1.0, abs(v)), (macro, v, value)


def matrix_panel(ax, M, keys):
    n = len(keys)
    A = np.ma.masked_where(np.eye(n, dtype=bool), M)
    cmap = plt.get_cmap("Greys").copy()
    cmap.set_bad("white")
    im = ax.imshow(A, cmap=cmap, norm=LogNorm(VMIN, VMAX), interpolation="nearest")
    nA = sum(1 for b, _ in keys if b == "A")
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4); s.set_color(C["ink"])
    ax.axhline(nA - 0.5, color=C["ink"], lw=0.5)
    ax.axvline(nA - 0.5, color=C["ink"], lw=0.5)
    ticks = list(range(n))
    labs = [str(s) for _, s in keys]
    ax.set_xticks(ticks); ax.set_xticklabels(labs)
    ax.set_yticks(ticks); ax.set_yticklabels(labs)
    ax.tick_params(length=1.5, pad=1.5, labelsize=8)
    ax.xaxis.set_ticks_position("bottom")
    # body brackets, in the body colour, outside the tick labels
    for b, col in (("A", BODY_A), ("B", BODY_B)):
        ii = [i for i, (bb, _) in enumerate(keys) if bb == b]
        c = 0.5 * (ii[0] + ii[-1])
        ax.annotate(f"body {b}", xy=(c, 0), xycoords=("data", "axes fraction"), xytext=(0, -10),
                    textcoords="offset points", ha="center", va="top", fontsize=8, color=col,
                    annotation_clip=False)
        ax.annotate(f"body {b}", xy=(0, c), xycoords=("axes fraction", "data"), xytext=(-15, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8, color=col,
                    rotation=90, annotation_clip=False)
    return im


def main():
    num = load(NUMJSON)
    h5 = load(SRC)
    doses = [("2000", "Two"), ("16000", "Sixteen")]

    mats = {}
    for d, tag in doses:
        v = h5["doses"][d]
        M, keys = ordered(v)
        mats[d] = (M, keys)
        # the figure's inputs reproduce the file's own summaries and the macros
        n = len(keys)
        same, cross = [], []
        for i, j in itertools.combinations(range(n), 2):
            (same if keys[i][0] == keys[j][0] else cross).append(M[i, j])
        assert len(same) + len(cross) == v["n_pairs"]
        assert abs(np.mean(same) - v["mean_cos_same_body"]) < 1e-12
        assert abs(np.mean(cross) - v["mean_cos_diff_body"]) < 1e-12
        assert abs(np.mean(same) - np.mean(cross) - v["D"]) < 1e-12
        check(num, f"nWt{tag}SameBody", v["mean_cos_same_body"])
        check(num, f"nWt{tag}CrossBody", v["mean_cos_diff_body"])
        check(num, f"nWt{tag}D", v["D"])
        check(num, f"nWt{tag}P", v["perm_p_one_sided"])
        off = M[~np.eye(n, dtype=bool)]
        assert off.min() >= VMIN and off.max() <= VMAX, (off.min(), off.max())
    check(num, "nWtPerms", h5["doses"]["2000"]["n_permutations"])
    check(num, "nWtAdapters", len(h5["doses"]["2000"]["adapters"]))

    fig = plt.figure(figsize=(DC, H_IN))
    # layout in inches, converted to figure fractions (the canvas is exactly DC x H_IN)
    L, MW, GAP, CBG, CBW, CBR, CL = 0.40, 1.80, 0.40, 0.07, 0.07, 0.50, 0.50
    BOT, TOP = 0.47, 0.13
    MH = H_IN - BOT - TOP
    MW = min(MW, MH)
    fx, fy = (lambda v: v / DC), (lambda v: v / H_IN)
    xa = L
    xb = xa + MW + GAP
    xcb = xb + MW + CBG
    xc = xcb + CBW + CBR + CL
    wc = DC - 0.04 - xc
    axa = fig.add_axes([fx(xa), fy(BOT), fx(MW), fy(MW)])
    axb = fig.add_axes([fx(xb), fy(BOT), fx(MW), fy(MW)])
    cax = fig.add_axes([fx(xcb), fy(BOT), fx(CBW), fy(MW)])
    axc = fig.add_axes([fx(xc), fy(BOT), fx(wc), fy(MW)])

    im = None
    for ax, (d, _), lab in ((axa, doses[0], "(a)"), (axb, doses[1], "(b)")):
        M, keys = mats[d]
        im = matrix_panel(ax, M, keys)
        ax.set_xlabel(f"adapter seed, {d} steps", labelpad=11)
        panel_label(ax, lab, x=-0.03, y=1.01)

    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.4)
    cb.ax.yaxis.set_major_locator(FixedLocator([0.01, 0.03, 0.1, 0.3]))
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
    cb.ax.yaxis.set_minor_locator(NullLocator())
    cb.ax.tick_params(length=1.5, pad=1.5, labelsize=8, width=0.4)
    cb.set_label(r"cosine of $\Delta W$", labelpad=2)

    # (c) pair cosines by registered group, both doses
    rng = np.random.default_rng(8)                    # jitter only
    xpos = {("2000", "same"): 0, ("2000", "cross"): 1, ("16000", "same"): 2.5, ("16000", "cross"): 3.5}
    for d, tag in doses:
        M, keys = mats[d]
        v = h5["doses"][d]
        n = len(keys)
        for i, j in itertools.combinations(range(n), 2):
            (bi, si), (bj, sj) = keys[i], keys[j]
            if bi == bj:
                # body A pairs left of centre, body B pairs right, so neither hides the other
                x = xpos[(d, "same")] + (-0.16 if bi == "A" else 0.16)
                col, mk = (BODY_A, MARK_A) if bi == "A" else (BODY_B, MARK_B)
                fc, ms = col, 2.6
            else:
                x = xpos[(d, "cross")]
                col, mk = C["grey"], "^"
                # cross-body pairs that share a seed: open (white-filled, larger so the opening shows)
                fc, ms = ("white", 3.4) if si == sj else (col, 2.6)
            ax_x = x + rng.uniform(-0.08, 0.08)
            axc.plot(ax_x, M[i, j], mk, ms=ms, mfc=fc, mec=col, mew=0.5, zorder=3)
        for grp, key in (("same", "mean_cos_same_body"), ("cross", "mean_cos_diff_body")):
            x = xpos[(d, grp)]
            axc.plot([x - 0.38, x + 0.38], [v[key]] * 2, color=SYM, lw=1.0, zorder=2)
        xm = 0.5 * (xpos[(d, "same")] + xpos[(d, "cross")])
        axc.text(xm, 0.83, f"$D$ = {num[f'nWt{tag}D']['text'].replace('$-$', chr(8722))}\n"
                           f"$p$ = {num[f'nWt{tag}P']['text']}",
                 ha="center", va="bottom", fontsize=8, color=C["ink"], linespacing=1.15)
        axc.text(xm, -0.115, f"{d} steps", transform=axc.get_xaxis_transform(), ha="center", va="top",
                 fontsize=8)
    axc.set_yscale("log")
    axc.set_ylim(VMIN, 1.3)
    axc.yaxis.set_major_locator(FixedLocator([0.01, 0.03, 0.1, 0.3, 1]))
    axc.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
    axc.yaxis.set_minor_formatter(NullFormatter())
    axc.set_xlim(-0.5, 4.0)
    axc.set_xticks([xpos[k] for k in sorted(xpos, key=xpos.get)])
    axc.set_xticklabels(["same", "cross"] * 2, fontsize=8)
    axc.tick_params(axis="x", length=0, pad=2)
    axc.set_ylabel(r"cosine of $\Delta W$", labelpad=2)
    panel_label(axc, "(c)", x=-0.03, y=1.01)

    return save(fig, "figS_08_weights")


if __name__ == "__main__":
    print(main())
