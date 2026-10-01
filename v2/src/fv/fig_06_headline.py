"""Fig. 6 (fig:headline): the evidence behind the headline limit.
Single column, 3.35 x 4.2 in, two stacked panels (OUTLINE §4 "Fig. 6").

(a) Own-body contrast theta_{x,j} of each of the 12 + 12 primary D200 adapters (SD-3.5 Medium, LoRA rank 16,
    2000 steps, archive stack, 500 generations each), as % of R_real (E2 estimate), in integer seed order
    within each arm; arm means as short black bars; per-arm one-sided 99.5 % limits, nominal (dashed) and
    calibrated with c* (solid; the bound of record, RESULTS Entry 117), each in its arm colour.  Arm B's lines
    carry the max-arm limits.
    Sources: FINAL_LEDGER.json -> primary.{per_adapter_A, per_adapter_B, U_A, U_B, U_device},
    denominators.R_real; out/fv_seedbank_calib.json -> limits.bound_of_record.{U_A, U_B, U_device, c, R_real}
    (c* = 1.31 with the seed-bank term); out/h6_calibrated_limit.json -> primary_d200 (c = 1.25, checked only).
    The G3 local re-scoring is never plotted here (FIG_ASSETS §9.4).
(b) Worst-cell coverage of a nominal one-sided 99 % limit, symmetric (open black diamond: the paper-wide
    theta_sym marker, replacing OUTLINE's open circle so that a diamond never means max-arm) and max-arm
    (filled square), with neither / estimation only / both error components simulated
    (out/h4_coverage2.json -> worst_case.{c6_reproduction, estimation_only, headline}); the max-arm limit
    with c applied, both components (out/h6_calibrated_limit.json -> scenarios["both components
    (headline)"].worst_coverage_at_c; filled square in its own column), and with the seed-bank term added at
    c* (out/fv_seedbank_calib.json -> crn_run_4000.variants.point.worst_coverage_at_c_star; its own column).

Every plotted value is read from the result files and checked against its numbers.json macro; every printed
number is the macro's `text`.  Run:
    python src/fv/fig_06_headline.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_style import plt, SC, C, BODY_A, BODY_B, MARK_A, MARK_B, SYM, SYM_MARK, panel_label, save  # noqa: E402

from matplotlib.ticker import FixedLocator, MultipleLocator, FuncFormatter  # noqa: E402

OUT = (EINV.V2 + "/out")
LEDGER = (EINV.REPO + "/analysis/FINAL_LEDGER.json")
NUMJSON = (EINV.V2 + "/paper/fv/numbers.json")
H_IN = 4.2
N_PRESPEC = 6            # seeds 0-5 pre-specified (filled), 6-11 extension (open): OUTLINE §4.0 design constant

plt.rcParams["savefig.bbox"] = "standard"      # made at final size: the canvas is exactly SC wide


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tex_plain(s):
    """numbers.json text -> plain string for matplotlib (only the forms used here occur)."""
    return s.replace("$-$", "\u2212").replace("{,}", ",")


def check(num, macro, value, rel=1e-9):
    v = num[macro]["value"]
    assert abs(v - value) <= rel * max(abs(v), abs(value), 1e-300), (macro, v, value)


def minus_fmt(x, _pos):
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.1f}"
    return s.replace("-", "\u2212")


def main():
    num = load(NUMJSON)
    led = load(LEDGER)
    h6 = load(os.path.join(OUT, "h6_calibrated_limit.json"))
    h4 = load(os.path.join(OUT, "h4_coverage2.json"))
    sb = load(os.path.join(OUT, "fv_seedbank_calib.json"))      # RESULTS Entry 117 item 1: the bound of record

    # ------------------------------------------------------------------ (a) data
    p = led["primary"]
    R = float(led["denominators"]["R_real"])
    k = int(p["k_per_arm"])
    thA = np.array(p["per_adapter_A"], dtype=float)      # seeds 0..k-1 in integer order (ledger note)
    thB = np.array(p["per_adapter_B"], dtype=float)
    assert "seeds 0-11 in numeric order" in p["_per_adapter_note"]
    assert len(thA) == len(thB) == k == int(num["nLimAdaptersPerArm"]["text"])
    h6c = h6["primary_d200"]                    # c = 1.25 (H6), checked, not drawn
    cal = sb["limits"]["bound_of_record"]       # c* = 1.31 with the seed-bank term: drawn as "calibrated"
    assert sb["rule"]["branch"] == "changes" and sb["rule"]["c_star"] == cal["c"]
    assert abs(cal["R_real"] - R) < 1e-15 and abs(h6c["R_real"] - R) < 1e-15, "R_real differs"
    c = float(cal["c"])
    c_h6 = float(h6c["c"])

    lamA, lamB = 100 * thA / R, 100 * thB / R
    meanA, meanB = lamA.mean(), lamB.mean()
    nomA, nomB = 100 * p["U_A"] / R, 100 * p["U_B"] / R
    calA, calB = 100 * cal["U_A"] / R, 100 * cal["U_B"] / R

    # recompute the per-arm limits from the per-adapter values (eq. maxarm) and check the files
    tcrit = float(p["t_crit_0995"])
    for th, U_nom, U_cal, U_h6 in ((thA, p["U_A"], cal["U_A"], h6c["U_A"]), (thB, p["U_B"], cal["U_B"], h6c["U_B"])):
        hw = tcrit * th.std(ddof=1) / np.sqrt(k)
        assert abs(th.mean() + hw - U_nom) < 2e-9, (th.mean() + hw, U_nom)
        assert abs(th.mean() + c * hw - U_cal) < 2e-9, (th.mean() + c * hw, U_cal)
        assert abs(th.mean() + c_h6 * hw - U_h6) < 2e-9, (th.mean() + c_h6 * hw, U_h6)
    # the max-arm limit is arm B's (nominal and both calibrations); the labels below are on arm B's lines
    assert p["U_device"] == p["U_B"] and cal["U_device"] == cal["U_B"] and h6c["U_device"] == h6c["U_B"]
    check(num, "nLimNom", nomB, 1e-3)
    check(num, "nLimRec", calB, 1e-3)
    check(num, "nLimRecFourDigit", calB, 1e-6)
    check(num, "nLimCal", 100 * h6c["U_B"] / R, 1e-6)
    check(num, "nLimUAPct", nomA, 1e-3)
    check(num, "nLimRecUAPct", calA, 1e-3)
    check(num, "nLimRecC", c)
    check(num, "nLimCalC", c_h6)
    check(num, "nLimThetaAMean", thA.mean(), 1e-6)
    check(num, "nLimThetaBMean", thB.mean(), 1e-6)

    # ------------------------------------------------------------------ (b) data
    wc = h4["worst_case"]
    cov = [  # (x label, symmetric, max-arm, macros)
        ("Neither", wc["c6_reproduction"]["min_coverage_sym"], wc["c6_reproduction"]["min_coverage_maxarm"],
         ("nLimCovNeitherSym", "nLimCovNeitherMax")),
        ("Estimation\nonly", wc["estimation_only"]["min_coverage_sym"], wc["estimation_only"]["min_coverage_maxarm"],
         ("nLimCovEstSym", "nLimCovEstMax")),
        ("Both", wc["headline"]["min_coverage_sym"], wc["headline"]["min_coverage_maxarm"],
         ("nLimCovSym", "nLimCovMax")),
    ]
    for _, s, m, (ms, mm) in cov:
        check(num, ms, s)
        check(num, mm, m)
    sc = h6["scenarios"]["both components (headline)"]
    assert sc["c"] == c_h6
    cov_c = float(sc["worst_coverage_at_c"])
    check(num, "nLimCalCovAtC", cov_c)
    pt = sb["crn_run_4000"]["variants"]["point"]
    assert pt["c_star"] == c and pt["n_rep"] == h6["n_rep"]
    cov_rec = float(pt["worst_coverage_at_c_star"])
    check(num, "nLimRecCovAtRecC", cov_rec)
    assert wc["headline"]["n_cells"] == int(num["nLimCovCells"]["text"])
    assert h4["n_rep"] == h6["n_rep"] == int(num["nLimCalReps"]["text"])

    # printed annotations: numbers.json text only
    t_cal = tex_plain(num["nLimRec"]["text"])
    t_nom = tex_plain(num["nLimNom"]["text"])
    t_c = tex_plain(num["nLimRecC"]["text"])
    t_ch6 = tex_plain(num["nLimCalC"]["text"])

    # ------------------------------------------------------------------ layout (inches -> fractions)
    fig = plt.figure(figsize=(SC, H_IN))
    L, Rm = 0.50, 0.50                  # right margin carries the line-end labels of (a) and (b)
    T, B = 0.13, 0.64                   # three-line tick labels in (b) (Entry 117 column) need more room
    h_a, h_b, gap = 2.05, 0.98, 0.40
    assert abs(T + h_a + gap + h_b + B - H_IN) < 1e-9
    w = SC - L - Rm
    ax_a = fig.add_axes([L / SC, (B + h_b + gap) / H_IN, w / SC, h_a / H_IN])
    ax_b = fig.add_axes([L / SC, B / H_IN, w / SC, h_b / H_IN])
    ink, grey = C["ink"], C["grey"]

    # ================================================================== (a)
    ax = ax_a
    gap_x = 2.0
    xA = np.arange(k, dtype=float)
    xB = xA + k + gap_x
    half = 0.45                          # limit lines span the arm's seeds +- this
    ax.axhline(0, color=C["light"], lw=0.5, zorder=0)

    for x, lam, col, mk, nom, cl, mean in ((xA, lamA, BODY_A, MARK_A, nomA, calA, meanA),
                                           (xB, lamB, BODY_B, MARK_B, nomB, calB, meanB)):
        x0, x1 = x[0] - half, x[-1] + half
        ax.hlines(nom, x0, x1, color=col, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.hlines(cl, x0, x1, color=col, lw=0.8, ls="-", zorder=1)
        pre = np.arange(k) < N_PRESPEC
        ms = 3.8 if mk == "o" else 3.4
        ax.plot(x[pre], lam[pre], ls="none", marker=mk, ms=ms, mfc=col, mec=col, mew=0.7, zorder=3)
        ax.plot(x[~pre], lam[~pre], ls="none", marker=mk, ms=ms, mfc="white", mec=col, mew=0.8, zorder=3)
        xc = 0.5 * (x[0] + x[-1])
        ax.hlines(mean, xc - 1.4, xc + 1.4, color=SYM, lw=1.3, zorder=2)

    # max-arm limits: labels at the right-hand ends of arm B's lines (numbers.json text)
    xl = xB[-1] + half + 0.35
    ax.text(xl, calB, f"calibrated\n{t_cal}%", ha="left", va="bottom", fontsize=7.5, color=BODY_B,
            linespacing=1.05, clip_on=False)
    ax.text(xl, nomB, f"{t_nom}%\nnominal", ha="left", va="top", fontsize=7.5, color=BODY_B,
            linespacing=1.05, clip_on=False)

    # arm names, top of each arm
    ytop = 0.345
    ax.text(xA[0] - half, ytop, "Arm A", ha="left", va="top", fontsize=8, color=BODY_A)
    ax.text(xB[0] - half, ytop, "Arm B", ha="left", va="top", fontsize=8, color=BODY_B)

    ax.set_xlim(xA[0] - 0.9, xB[-1] + 0.9)
    ax.set_ylim(-0.32, 0.35)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FuncFormatter(minus_fmt))
    maj = [0, 2, 4, 6, 8, 10]
    ax.xaxis.set_major_locator(FixedLocator(list(xA[maj]) + list(xB[maj])))
    ax.set_xticklabels([str(s) for s in maj] * 2)
    ax.xaxis.set_minor_locator(FixedLocator([v for v in np.r_[xA, xB] if v not in set(np.r_[xA[maj], xB[maj]])]))
    ax.set_xlabel("Adapter seed", labelpad=2)
    ax.set_ylabel(r"Own-body contrast (% of $R_{\mathrm{real}}$)", labelpad=2)
    panel_label(ax, "(a)", x=-0.13, y=1.0)

    # ================================================================== (b)
    ax = ax_b
    dx = 0.13
    xs = np.arange(len(cov) + 2, dtype=float)
    # symmetric construction: open black diamond (theta_sym's marker paper-wide); max-arm: filled black square.
    # The calibrated max-arm points are the same construction, so they keep the filled square, each in its own
    # column: c = 1.25 with the two measured components, and c* with the seed-bank term added as well.
    for i, (_, s, m, _) in enumerate(cov):
        ax.plot(xs[i] - dx, s, ls="none", marker=SYM_MARK, ms=3.8, mfc="white", mec=SYM, mew=0.8, zorder=3)
        ax.plot(xs[i] + dx, m, ls="none", marker="s", ms=3.6, mfc=SYM, mec=SYM, mew=0.6, zorder=3)
    ax.plot(xs[-2], cov_c, ls="none", marker="s", ms=3.6, mfc=SYM, mec=SYM, mew=0.6, zorder=4)
    ax.plot(xs[-1], cov_rec, ls="none", marker="s", ms=3.6, mfc=SYM, mec=SYM, mew=0.6, zorder=4)

    ax.axhline(0.99, color=grey, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(xs[-1] + 0.5 + 0.12, 0.99, "0.99", ha="left", va="center", fontsize=7.5, color=grey,
            clip_on=False)
    # direct labels at the "Both" category
    i = 2
    ax.text(xs[i] - dx + 0.14, cov[i][1], "symmetric", ha="left", va="center", fontsize=7.5, color=ink)
    ax.text(xs[i] + dx - 0.14, cov[i][2], "max-arm", ha="right", va="center", fontsize=7.5, color=ink)

    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
    ax.set_ylim(0.75, 1.005)
    ax.yaxis.set_major_locator(FixedLocator([0.75, 0.80, 0.85, 0.90, 0.95, 1.00]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.2f}"))
    ax.xaxis.set_major_locator(FixedLocator(xs))
    labs = ["Neither", "Estim.\nonly", "Both"]
    assert [lab.split("\n")[0][:5] for lab, *_ in cov] == ["Neith", "Estim", "Both"]
    ax.set_xticklabels(labs + [f"Both,\n$c$ = {t_ch6}", f"+ seed\nbank,\n$c$ = {t_c}"], fontsize=7.5,
                       linespacing=1.0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlabel("Error components simulated", labelpad=3)
    ax.set_ylabel("Coverage", labelpad=2)          # worst-cell coverage; the caption says so
    panel_label(ax, "(b)", x=-0.13, y=1.0)

    out = save(fig, "fig06_headline")

    # spot checks printed for the record
    print("R_real", R, "c", c)
    print("arm A %:", np.round(lamA, 4).tolist())
    print("arm B %:", np.round(lamB, 4).tolist())
    print("means", round(meanA, 4), round(meanB, 4), "nominal", round(nomA, 4), round(nomB, 4),
          "calibrated", round(calA, 4), round(calB, 4))
    print("coverage", [(lab.replace("\n", " "), s, m) for lab, s, m, _ in cov], "at c", cov_c)
    return out


if __name__ == "__main__":
    print(main())
