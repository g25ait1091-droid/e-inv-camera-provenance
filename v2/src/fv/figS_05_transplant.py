"""Fig. S5 (fig:s05transplant): the residual-transplant positive control, increment against scale s.
Single column, 3.35 x 3.55 in, two stacked panels sharing the s axis (OUTLINE §6 S09 "Fig. S5").

Design (RESULTS.md Entry 67, C7).  The un-normalised per-channel wavelet residual N of each of body A's 40
held-out photographs (Nikon_D200_1) is added to base-model generations of the local stack (generation i gets
residual i mod 40) at scale s and rounded once: Z_s = round(clip(Z + s N, 0, 255)).  For each detector the
increment is contrast(s) - contrast(0) on the same image, where contrast = value(->A) - value(->B); it is
tested with a paired t (df n - 1).  Detection means t > 3.

(a) increment t against s, one line per detector; dashed line at t = 3.
(b) increment as a percentage of that detector's own real paired contrast R
    (results.<det>.R_real_paired_contrast), with the two-sided 99 % t interval stored in the file
    (increment_ci99, mean +- t_{0.995, n-1} SE) converted to the same percentage.

Source: out/c7_transplant.json -> results.{ncc,pce,lowmid,noiseprint}.per_s[*].{s, increment_t, increment,
increment_ci99, increment_pct_R_real_paired}.  s = 0 is the reference (increment 0 by construction) and is
not drawn.  Every plotted value is read from the file; the script checks the file's own percentages, the
smallest detected s, and every numbers.json nDetTrans* macro against the values it plots.  Nothing is printed
inside the axes except detector names and the t = 3 reference label (a design constant).  Run:
    python src/fv/figS_05_transplant.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_style import plt, SC, C, save  # noqa: E402

from matplotlib.ticker import FixedLocator, MultipleLocator  # noqa: E402

SRC = (EINV.V2 + "/out/c7_transplant.json")
NUMJSON = (EINV.V2 + "/paper/fv/numbers.json")
H_IN = 3.55

plt.rcParams["savefig.bbox"] = "standard"      # made at final size: the canvas is exactly SC wide

# detector key, printed name, marker, filled?, macro stem
DETS = [
    ("lowmid", "Low/mid", "^", False, "Lowmid"),
    ("ncc", "NCC", "o", True, "Ncc"),
    ("noiseprint", "Noiseprint", "s", False, "Noiseprint"),
    ("pce", "PCE", "v", True, "Pce"),
]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    d = load(SRC)
    num = load(NUMJSON)
    res = d["results"]
    s_design = [float(x) for x in d["design"]["s"]]
    assert s_design[0] == 0.0

    def chk(macro, v):
        """The printed macro text must be the plotted value rounded as printed."""
        txt = num[macro]["text"].replace("$-$", "-")
        dec = len(txt.split(".")[1]) if "." in txt else 0
        assert abs(round(v, dec) - float(txt)) < 1e-9, (macro, v, txt)
        ref = float(num[macro]["value"])
        assert abs(v - ref) <= 1e-9 * max(abs(ref), 1e-12), (macro, v, ref)

    data = {}
    for key, name, mk, filled, stem in DETS:
        r = res[key]
        per = r["per_s"]
        s = np.array([p["s"] for p in per], float)
        assert list(s) == s_design, (key, s)
        per = per[1:]                                   # s = 0 is the reference
        s = s[1:]
        R = float(r["R_real_paired_contrast"])
        n = int(r["n_images"])
        t = np.array([p["increment_t"] for p in per], float)
        inc = np.array([p["increment"] for p in per], float)
        lo = np.array([p["increment_ci99"][0] for p in per], float)
        hi = np.array([p["increment_ci99"][1] for p in per], float)
        pct = 100.0 * inc / R
        # the file's own percentage and df agree with the recomputation
        assert np.allclose(pct, [p["increment_pct_R_real_paired"] for p in per], rtol=1e-9, atol=0)
        assert all(int(p["df"]) == n - 1 for p in per)
        assert np.all(lo <= inc) and np.all(inc <= hi)
        # smallest detected scale recomputed from t > 3 (every larger s also detected)
        det = s[t > 3]
        s_min = float(det.min())
        assert np.all(t[s >= s_min] > 3), key
        assert s_min == float(r["smallest_s_detected_t_gt_3"]), key
        chk(f"nDetTrans{stem}S", s_min)
        chk(f"nDetTrans{stem}TAtS", float(t[s == s_min][0]))
        chk(f"nDetTrans{stem}TOne", float(t[s == 1.0][0]))
        chk(f"nDetTrans{stem}PctOne", float(pct[s == 1.0][0]))
        chk(f"nDetTrans{stem}N", float(n))
        data[key] = dict(s=s, t=t, pct=pct, lo=100.0 * lo / R, hi=100.0 * hi / R, n=n)
    # stored change and residual RMS (caption macros) come from the same file
    ncc_per = res["ncc"]["per_s"]
    for sv, m in ((0.1, "nDetTransStoredRmsPointOne"), (0.25, "nDetTransStoredRmsQuarter"),
                  (0.5, "nDetTransStoredRmsHalf"), (1.0, "nDetTransStoredRmsOne")):
        chk(m, float([p for p in ncc_per if p["s"] == sv][0]["stored_change_rms_gray"]))
    chk("nDetTransResidRms", float(np.mean(d["design"]["residual_rms_gray_mean_per_channel"])))
    # denominators R (caption macros): the file's R equals the detector panel's real paired contrast
    for key, _, _, _, stem in DETS:
        chk(f"nDet{stem}Real", float(res[key]["R_real_paired_contrast"]))
    # 40 residuals (one per held-out body-A photograph), generation i <- residual i mod 40
    assert "(40)" in d["design"]["real_source"] and int(num["nDataSplitH"]["text"]) == 40
    assert data["ncc"]["n"] == data["pce"]["n"] == data["lowmid"]["n"]

    # ================================================================== layout (inches)
    fig = plt.figure(figsize=(SC, H_IN))
    L_, R_ = 0.44, 0.60                     # right margin holds the line-end labels
    B_, T_, G_ = 0.36, 0.14, 0.20           # bottom (ticks + label), top (panel label), gap between panels
    w = SC - L_ - R_
    h = (H_IN - B_ - T_ - G_) / 2.0
    ax_t = fig.add_axes([L_ / SC, (B_ + h + G_) / H_IN, w / SC, h / H_IN])
    ax_p = fig.add_axes([L_ / SC, B_ / H_IN, w / SC, h / H_IN], sharex=ax_t)

    ink, grey, light = C["ink"], C["grey"], C["light"]
    # multiplicative s offsets in both panels (log axis): equal visual spacing at every s
    dodge = {"lowmid": 10 ** -0.021, "ncc": 10 ** -0.007, "noiseprint": 10 ** 0.007, "pce": 10 ** 0.021}

    def draw(ax, ykey, errors):
        ax.axhline(0.0, color=light, lw=0.5, zorder=1)
        for key, name, mk, filled, _ in DETS:
            v = data[key]
            x = v["s"] * dodge[key]
            if errors:
                ax.vlines(x, v["lo"], v["hi"], color=ink, lw=0.6, zorder=2)
            ax.plot(x, v[ykey], color=ink, lw=0.7, zorder=3, marker=mk,
                    ms=3.6 if mk in "^v" else 3.2, mfc=ink if filled else "white", mec=ink,
                    mew=0.4 if filled else 0.7, clip_on=False)

    def end_labels(ax, ykey, nudge):
        for key, name, *_ in DETS:
            v = data[key]
            y = v[ykey][-1] + nudge.get(key, 0.0)
            ax.text(1.0 * 10 ** 0.06, y, name, ha="left", va="center", fontsize=8, color=ink,
                    clip_on=False)

    # ---------------------------------------------------------------- (a) increment t
    draw(ax_t, "t", errors=False)
    ax_t.axhline(3.0, color=grey, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax_t.text(0.97, 2.4, r"$t = 3$", ha="right", va="top", fontsize=7.5, color=grey)
    ax_t.set_ylim(-3, 29)
    ax_t.yaxis.set_major_locator(FixedLocator([0, 5, 10, 15, 20, 25]))
    ax_t.set_ylabel(r"Increment, paired $t$", labelpad=3)
    plt.setp(ax_t.get_xticklabels(), visible=False)
    end_labels(ax_t, "t", {"noiseprint": 1.1, "pce": -1.1})

    # ---------------------------------------------------------------- (b) increment, % of own R
    draw(ax_p, "pct", errors=True)
    ax_p.set_ylim(-3, 43)
    ax_p.yaxis.set_major_locator(FixedLocator([0, 10, 20, 30, 40]))
    ax_p.set_ylabel(r"Increment (% of own $R$)", labelpad=3)
    end_labels(ax_p, "pct", {"ncc": 1.4, "noiseprint": -1.4})

    # s spans one decade (0.1 to 1): log axis, so the four design scales sit about evenly apart
    ax_p.set_xscale("log")
    ax_p.set_xlim(0.1 / 10 ** 0.06, 1.0 * 10 ** 0.04)
    ax_p.xaxis.set_major_locator(FixedLocator(s_design[1:]))
    ax_p.xaxis.set_minor_locator(FixedLocator([]))
    ax_p.set_xticklabels(["0.1", "0.25", "0.5", "1"])
    ax_p.set_xlabel(r"Transplant scale $s$", labelpad=2)

    # ---------------------------------------------------------------- panel labels, upper left
    for ax, lab in ((ax_t, "(a)"), (ax_p, "(b)")):
        pos = ax.get_position()
        fig.text(0.03 / SC, pos.y1 + 0.02 / H_IN, lab, ha="left", va="bottom", fontsize=9)

    out = save(fig, "figS_05_transplant")

    # ---------------------------------------------------------------- spot-check print
    for key, name, *_ in DETS:
        v = data[key]
        for i, sv in enumerate(v["s"]):
            print(f"{name:10s} n={v['n']:3d} s={sv:4.2f}  t={v['t'][i]:+8.3f}  pct={v['pct'][i]:+9.4f}%  "
                  f"ci99=[{v['lo'][i]:+.4f}, {v['hi'][i]:+.4f}]")
    print("saved", out)


if __name__ == "__main__":
    main()
