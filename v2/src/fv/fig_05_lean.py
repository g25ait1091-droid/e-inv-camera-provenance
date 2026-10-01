"""Fig. 5 (fig:lean): the shared lean, and why the paired design is needed.
Double column, 6.99 x 2.5 in, three panels (OUTLINE §4 "Fig. 5").

(a) iPhone 5c pair (VISION D05 / D14): own-body contrast theta_{x,j} of each adapter, % of the R_real of
    the estimate that scores it; columns A-natural, B-natural, A-flat, B-flat.
    Source: out/g6_p5c.json -> estimators.{E2,FLAT}.{per_adapter_A, per_adapter_B, R_real}.
(b) Huawei P20 pair (Daxing 1104 / 1103): the same, natural-image estimate only.
    Source: out/g4b_p20.json -> per_adapter_A, per_adapter_B, gates.R_real.
(c) Shared lean |theta_A - theta_B| / 2, % of each detector's or estimate's own real contrast (log axis).
    Detectors (D200): NCC, PCE, low/mid from out/t2_summary.json and Noiseprint from
    out/t2_noiseprint_summary.json (primary seeds 0-2 per arm, 500 generations each); learned CNN from
    out/t2_learned_ext.json (seeds 0-11 per arm, 250 generations each).
    NCC across pairs: D200 primary from out/c6_coverage.json (checked against FINAL_LEDGER.json);
    iPhone natural / flat and P20 from the files of (a) and (b).

Every plotted value is read from the result files; every printed number is the `text` of a numbers.json
macro, and each plotted value is checked against that macro's `value`.  Adapters are in integer seed order
(the per_adapter lists are written s0..s11 by g6_measure.py / g4b_measure.py); within a column the seed
sets the horizontal offset, seed 0 at the left.  Run:
    python src/fv/fig_05_lean.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_style import plt, DC, C, BODY_A, BODY_B, MARK_A, MARK_B, save  # noqa: E402

from matplotlib.ticker import FixedLocator, NullFormatter, LogLocator  # noqa: E402

OUT = (EINV.V2 + "/out")
LEDGER = (EINV.REPO + "/analysis/FINAL_LEDGER.json")
NUMJSON = (EINV.V2 + "/paper/fv/numbers.json")
H_IN = 2.5

plt.rcParams["savefig.bbox"] = "standard"      # made at final size: the canvas is exactly DC wide


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tex_plain(s):
    """numbers.json text -> plain string for matplotlib (only the forms used here occur)."""
    return s.replace("$-$", "\u2212").replace("{,}", ",")


def main():
    num = load(NUMJSON)
    g6 = load(os.path.join(OUT, "g6_p5c.json"))
    p20 = load(os.path.join(OUT, "g4b_p20.json"))
    t2 = load(os.path.join(OUT, "t2_summary.json"))
    npr = load(os.path.join(OUT, "t2_noiseprint_summary.json"))
    cnn = load(os.path.join(OUT, "t2_learned_ext.json"))
    c6 = load(os.path.join(OUT, "c6_coverage.json"))
    led = load(LEDGER)

    def chk(macro, v, rel=1e-6):
        ref = float(num[macro]["value"])
        assert abs(v - ref) <= rel * max(abs(ref), 1e-30), (macro, v, ref)

    # ------------------------------------------------------------------ (a), (b) per-adapter values, %
    cols_a = []   # (arm, estimate, values %, n positive)
    for est in ("E2", "FLAT"):
        e = g6["estimators"][est]
        R = e["R_real"]
        for arm in ("A", "B"):
            v = 100.0 * np.asarray(e[f"per_adapter_{arm}"], float) / R
            cols_a.append((arm, est, v))
    assert g6["estimators"]["FLAT"]["R_real"] == g6["gates"]["R_real_FLAT"]
    assert g6["estimators"]["E2"]["R_real"] == g6["gates"]["R_real"]
    R20 = p20["gates"]["R_real"]
    chk("nEstGfourbRreal", R20)
    assert R20 == p20["R_real"]
    cols_b = [(arm, "E2", 100.0 * np.asarray(p20[f"per_adapter_{arm}"], float) / R20) for arm in ("A", "B")]

    # counts of positive adapters: plotted text is the macro text, checked against the data
    cnt_macro = {("a", "A", "E2"): "nMainGsixNatPosA", ("a", "B", "E2"): "nMainGsixNatPosB",
                 ("a", "A", "FLAT"): "nMainGsixFlatPosA", ("a", "B", "FLAT"): "nMainGsixFlatPosB",
                 ("b", "A", "E2"): "nMainPtwentyPosA", ("b", "B", "E2"): "nMainPtwentyPosB"}
    n_adapt = {"a": num["nDataIPhoneAdaptersPerArm"]["text"], "b": num["nDataPTwentyPairAdaptersPerArm"]["text"]}
    for pan, cols in (("a", cols_a), ("b", cols_b)):
        for arm, est, v in cols:
            assert len(v) == int(n_adapt[pan]), (pan, len(v))
            m = cnt_macro[(pan, arm, est)]
            assert int(num[m]["text"]) == int((v > 0).sum()), (m, (v > 0).sum())

    # ------------------------------------------------------------------ (c) shared lean, % of own contrast
    def t2_share(det):
        d = t2[det]
        arms = d["arms"]
        tA = np.mean([arms[f"A_raw_s{s}_r16"]["paired_contrast"] for s in range(3)])
        tB = np.mean([arms[f"B_raw_s{s}_r16"]["paired_contrast"] for s in range(3)])
        return 100.0 * abs(0.5 * (tA - tB)) / d["real_paired_contrast"]

    # adapter / image counts behind the detector rows (stated in the caption through macros)
    for det in ("ncc", "pce", "lowmid"):
        for s in range(3):
            for arm in ("A", "B"):
                assert t2[det]["arms"][f"{arm}_raw_s{s}_r16"]["n"] == int(num["nDataPrimaryGensPerAdapter"]["text"])
    assert len(cnn["A_means"]) == len(cnn["B_means"]) == int(num["nDetLearnedSeedsTotal"]["text"])
    assert cnn["G_per_arm"] == int(num["nDetLearnedImgPerArm"]["text"])
    det_rows = [
        ("NCC (seeds 0–2)", t2_share("ncc"), "nMainShareNcc", "filled"),   # same score as the D200 row below,
                                                                              # on 3 of its 12 adapters per arm
        ("PCE", t2_share("pce"), "nMainSharePce", "filled"),
        ("Low/mid band", t2_share("lowmid"), "nMainShareLowmid", "filled"),
        ("Noiseprint", float(npr["additive_over_R_pct"]), "nMainShareNoiseprint", "filled"),
        ("Learned CNN", float(cnn["additive_over_R_pct"]), "nMainShareCnn", "filled"),
    ]
    # D200 primary, 12 per arm: c6_coverage.json, checked against the ledger's per-adapter values
    RD = led["denominators"]["R_real"]
    la = np.asarray(led["primary"]["per_adapter_A"]); lb = np.asarray(led["primary"]["per_adapter_B"])
    assert len(la) == len(lb) == int(num["nDataPrimaryAdaptersPerArm"]["text"])
    assert abs(0.5 * (la.mean() - lb.mean()) - c6["inputs"]["m_observed_additive_part"]) < 1e-12
    d200 = 100.0 * abs(c6["inputs"]["m_observed_additive_part"]) / RD
    e2, fl = g6["estimators"]["E2"], g6["estimators"]["FLAT"]
    ip_nat = 100.0 * abs(0.5 * (e2["mean_A"] - e2["mean_B"])) / e2["R_real"]
    ip_flat = 100.0 * abs(0.5 * (fl["mean_A"] - fl["mean_B"])) / fl["R_real"]
    p20_add = 100.0 * abs(p20["additive_part"]) / R20
    assert abs(0.5 * (p20["mean_A"] - p20["mean_B"]) - p20["additive_part"]) < 1e-15
    pair_rows = [
        ("Nikon D200 (seeds 0–11)", d200, "nMainPrimaryAdditivePct", "filled"),
        ("iPhone 5c", ip_nat, "nMainGsixNatAdditivePct", "filled"),
        ("iPhone 5c, flat field", ip_flat, "nMainGsixFlatAdditivePct", "hollow"),
        ("Huawei P20", p20_add, "nMainPtwentyAdditivePct", "filled"),
    ]
    for _, v, m, _ in det_rows + pair_rows:
        chk(m, v if m != "nMainPrimaryAdditivePct" else -v)

    # ================================================================== layout (inches)
    fig = plt.figure(figsize=(DC, H_IN))
    B_, T_ = 0.56, 0.17                   # bottom: tick labels + estimate row + axis label; top: panel labels
    h = H_IN - B_ - T_
    xa, wa = 0.47, 1.58                   # (a)
    xb, wb = xa + wa + 0.12, 0.80         # (b) shares y with (a)
    xc = 4.30                             # (c) axis left edge (row labels sit to its left)
    wc = DC - xc - 0.20
    ax_a = fig.add_axes([xa / DC, B_ / H_IN, wa / DC, h / H_IN])
    ax_b = fig.add_axes([xb / DC, B_ / H_IN, wb / DC, h / H_IN], sharey=ax_a)
    ax_c = fig.add_axes([xc / DC, B_ / H_IN, wc / DC, h / H_IN])

    ink, grey = C["ink"], C["grey"]
    col = {"A": BODY_A, "B": BODY_B}
    mk = {"A": MARK_A, "B": MARK_B}
    ms = {"A": 3.3, "B": 3.0}             # squares drawn slightly smaller so both read the same size
    dx = (np.arange(12) - 5.5) * 0.036    # seed 0 ... 11, left to right within a column

    all_v = np.concatenate([c[2] for c in cols_a + cols_b])
    ylo, yhi = -0.55, 0.72
    assert all_v.min() > ylo and all_v.max() < yhi - 0.12, (all_v.min(), all_v.max())

    def strip(ax, cols, xpos, pan):
        ax.axhline(0.0, color=grey, lw=0.6, ls=(0, (3, 2)), zorder=1)
        for x0, (arm, est, v) in zip(xpos, cols):
            hollow = est == "FLAT"
            ax.plot(x0 + dx, v, ls="none", marker=mk[arm], ms=ms[arm],
                    mfc="white" if hollow else col[arm], mec=col[arm], mew=0.7 if hollow else 0.4,
                    zorder=3, clip_on=False)
            ax.hlines(v.mean(), x0 - 0.30, x0 + 0.30, color=ink, lw=0.9, zorder=4)
            ax.text(x0, yhi, f"{num[cnt_macro[(pan, arm, est)]]['text']}/{n_adapt[pan]}", ha="center", va="top",
                    fontsize=7.5, color=ink)
        ax.set_xticks(xpos)
        ax.set_xticklabels([c[0] for c in cols])
        ax.tick_params(axis="x", length=0, pad=2)
        ax.spines["bottom"].set_visible(False)

    # ---------------------------------------------------------------- (a)
    xpa = [0.0, 1.0, 2.35, 3.35]
    strip(ax_a, cols_a, xpa, "a")
    ax_a.set_xlim(-0.55, 3.9)
    ax_a.set_ylim(ylo, yhi)
    ax_a.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    ax_a.set_yticklabels(["\u22120.4", "\u22120.2", "0", "0.2", "0.4", "0.6"])
    ax_a.set_ylabel(r"Own-body contrast $\theta_{x,j}$ (% of $R_{\mathrm{real}}$)", labelpad=2)
    for xm, lab in ((0.5, "natural"), (2.85, "flat field")):
        ax_a.text(xm, -0.115, lab, transform=ax_a.get_xaxis_transform(), ha="center", va="top",
                  fontsize=8)
    ax_a.set_xlabel("iPhone 5c pair", labelpad=13)

    # ---------------------------------------------------------------- (b)
    xpb = [0.0, 1.0]
    strip(ax_b, cols_b, xpb, "b")
    ax_b.set_xlim(-0.55, 1.55)
    plt.setp(ax_b.get_yticklabels(), visible=False)
    ax_b.text(0.5, -0.115, "natural", transform=ax_b.get_xaxis_transform(), ha="center", va="top",
              fontsize=8)
    ax_b.set_xlabel("Huawei P20 pair", labelpad=13)

    # ---------------------------------------------------------------- (c)
    ax = ax_c
    ax.set_xscale("log")
    ax.set_xlim(0.005, 300)
    # rows, top to bottom; a blank row separates the two groups
    rows = [("hdr", "Detectors, D200 pair")] + [("row", r) for r in det_rows] + \
           [("hdr", "Camera pairs, NCC")] + [("row", r) for r in pair_rows]
    n = len(rows)
    ypos = np.arange(n)[::-1].astype(float)
    ypos[1 + len(det_rows):] -= 0.35            # a little air before the second group
    ax.set_ylim(ypos.min() - 0.7, ypos.max() + 0.5)
    ax.axvline(100.0, color=grey, lw=0.6, ls=(0, (3, 2)), zorder=1)
    # label in the header row, which carries no guide line
    ax.text(100.0 / 1.25, ypos.max(), "100%", ha="right", va="center", fontsize=7.5, color=grey)
    ticks, labs, guides = [], [], []
    for y, (kind, r) in zip(ypos, rows):
        if kind == "hdr":
            ax.text(-0.03, y, r, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=8, style="italic", color=ink)
            continue
        name, v, m, fill = r
        ticks.append(y); labs.append(name)
        ax.plot([v], [y], ls="none", marker="o", ms=3.3, mfc="white" if fill == "hollow" else ink,
                mec=ink, mew=0.7 if fill == "hollow" else 0.4, zorder=3)
        txt = tex_plain(num[m]["text"]).lstrip("\u2212")      # magnitude (the D200 value is negative)
        t = ax.text(v * 1.28, y, txt, ha="left", va="center", fontsize=7.5, color=ink, clip_on=False)
        guides.append((y, v, t))
    # Row guides span the whole axis (a log axis has no zero, so a stem from the left edge would encode an
    # arbitrary length); they are interrupted around the dot and its value label.
    fig.canvas.draw()
    inv = ax.transData.inverted()
    x0, x1 = ax.get_xlim()
    for y, v, t in guides:
        right = inv.transform(t.get_window_extent().get_points())[1, 0] * 1.25
        ax.hlines(y, x0, v / 1.35, color=C["light"], lw=0.4, ls=(0, (1, 1.5)), zorder=1)
        if right < x1:
            ax.hlines(y, right, x1, color=C["light"], lw=0.4, ls=(0, (1, 1.5)), zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 1.0, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Shared lean $|\bar{\theta}_A-\bar{\theta}_B|/2$ (% of own real contrast)", labelpad=2)

    # ---------------------------------------------------------------- panel labels, upper left of each panel
    ytop = (H_IN - 0.02) / H_IN
    for s, xin in (("(a)", 0.02), ("(b)", xb - 0.02), ("(c)", 3.02)):
        fig.text(xin / DC, ytop, s, ha="left", va="top", fontsize=9)

    out = save(fig, "fig05_lean")

    # ---------------------------------------------------------------- spot-check print
    for pan, cols in (("a", cols_a), ("b", cols_b)):
        for arm, est, v in cols:
            print(f"({pan}) {arm} {est}: mean {v.mean():+.4f}%  min {v.min():+.4f}  max {v.max():+.4f}  "
                  f"s0 {v[0]:+.4f}  s11 {v[-1]:+.4f}  pos {(v > 0).sum()}/12")
    for name, v, m, _ in det_rows + pair_rows:
        print(f"(c) {name:24s} {v:10.4f}%   macro {m} = {num[m]['text']}")
    print("saved", out)


if __name__ == "__main__":
    main()
