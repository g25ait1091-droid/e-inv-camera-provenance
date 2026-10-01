"""Fig. S3 (fig:s03coverage): worst-cell coverage of the max-arm limit against the half-width multiplier c.
Single column, 3.35 x 2.25 in, one panel (OUTLINE §6 S07 "Fig. S3").

Worst-cell coverage (minimum over the 27 scenario cells, 4,000 replications each) of the max-arm one-sided
limit U = max_x [ mean_x + c * t_{0.995,k-1} * s_x / sqrt(k) ], k = 12 adapters per arm, for every c on the
registered grid 1.00, 1.01, ..., 2.50, in the three scenarios of the calibration run:
    estimation and training set (measured)   -> the curve that sets c   (black, solid)
    estimation only                          -> (grey, solid)
    neither                                  -> regenerated and checked, not drawn (1 at every c)
The smallest c whose worst-cell coverage reaches 0.99 is marked with the black diamond.
Added 1 Oct 2026 (RESULTS Entry 117 item 1): the same scenario with the seed-bank term (dashed black), read from
the stored curve of out/fv_seedbank_calib.json (crn_run_4000.variants.point), with its c* = 1.31 marked by an
open diamond; that run's zero-term curve is asserted equal to the regenerated H6 curve.

Data. out/h6_calibrated_limit.json stores the grid only as [start, stop, step] and, per scenario, the worst-cell
coverage at c = 1 and at the chosen c; the curve itself was not written to the file. It is therefore
regenerated here by calling the registered code unchanged (src/h6_calibrated_limit.py: simulate() and
coverage_by_c(), seed 106061, the same draws for every c; never its main(), which would rewrite the result
file). The script then asserts, before drawing, that the regenerated curve reproduces every value the file
stores (grid ends and step, n_rep, seed, both SDs, c, worst coverage at c and at 1, for all three scenarios)
and the numbers.json macros. Any drift in the inputs stops the figure.

Every printed number is a numbers.json `text`.  Run:
    python src/fv/figS_03_coverage.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import json
import os
import sys

V2 = EINV.V2
os.environ["EINV_V2"] = V2            # einv_paths default would point at <repo>/v2/workspace
os.environ.pop("H4_REP", None)        # the registered 4,000 replications
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, EINV.SRC)
from fv_style import plt, SC, C, SYM, SYM_MARK, save  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator  # noqa: E402

import h4_coverage2 as H4            # noqa: E402  (import has no side effects beyond reading the ledger)
import h6_calibrated_limit as H6     # noqa: E402

OUT = os.path.join(V2, "out")
NUMJSON = os.path.join(V2, "paper", "fv", "numbers.json")
H_IN = 2.25
TARGET = 0.99                          # nominal level of the one-sided limit (design constant)

plt.rcParams["savefig.bbox"] = "standard"      # made at final size: the canvas is exactly SC wide


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check(num, macro, value, rel=1e-9):
    v = num[macro]["value"]
    assert abs(v - value) <= rel * max(abs(v), abs(value), 1e-300), (macro, v, value)


def main():
    num = load(NUMJSON)
    h6 = load(os.path.join(OUT, "h6_calibrated_limit.json"))

    # ------------------------------------------------------------------ registered settings = file
    g0, g1, gstep = h6["grid"]
    grid = H6.GRID
    assert abs(grid[0] - g0) < 1e-12 and abs(grid[-1] - g1) < 1e-12
    assert np.allclose(np.diff(grid), gstep)
    assert H6.SEED == h6["seed"] and H4.NREP == h6["n_rep"]
    sd_est, _ = H4.estimation_sd()
    sd_train, _meta = H4.training_sd()
    assert sd_est == h6["estimation_sd"] and sd_train == h6["training_sd"], (sd_est, sd_train)
    check(num, "nLimEstSD", sd_est)
    check(num, "nLimTrainSD", sd_train)
    check(num, "nLimCalReps", h6["n_rep"])

    # ------------------------------------------------------------------ regenerate the curves
    scen = [  # (file key, sd_est, sd_train, label, colour, linestyle, lw)
        ("both components (headline)", sd_est, sd_train, "estimation and training set", SYM, "-", 1.0),
        ("estimation only", sd_est, 0.0, "estimation only", C["grey"], "-", 1.0),
        ("neither (C6 reproduction)", 0.0, 0.0, "neither", C["light"], (0, (3, 1.6)), 1.0),
    ]
    curves = {}
    for key, se_, st_, *_ in scen:
        cells = H6.simulate(se_, st_)
        assert len(cells) == int(num["nLimCovCells"]["text"])
        cov = H6.coverage_by_c(cells)
        worst = cov.min(0)
        ok = np.where(worst >= TARGET)[0]
        c_min = float(grid[ok[0]])
        rec = h6["scenarios"][key]
        assert c_min == rec["c"], (key, c_min, rec["c"])
        assert float(worst[ok[0]]) == rec["worst_coverage_at_c"], key
        assert float(worst[0]) == rec["worst_coverage_at_1"], key
        curves[key] = (worst, c_min, int(np.argmin(cov[:, 0])), cells)

    head = h6["scenarios"]["both components (headline)"]
    c_star = float(head["c"])
    cov_star = float(head["worst_coverage_at_c"])
    check(num, "nLimCalC", c_star)
    check(num, "nLimCalCovAtC", cov_star)
    check(num, "nLimCalCovAtOne", head["worst_coverage_at_1"])
    check(num, "nLimCalCovEstOnly", h6["scenarios"]["estimation only"]["worst_coverage_at_1"])
    check(num, "nLimCalCestOnly", h6["scenarios"]["estimation only"]["c"])
    assert h6["primary_d200"]["c"] == c_star
    t_c = num["nLimCalC"]["text"]

    # ------------------------------------------------------------------ the seed-bank run (RESULTS Entry 117 item 1)
    # Stored curve of the registered 4,000-replication common-random-number run with the seed-bank term at its
    # point estimate (out/fv_seedbank_calib.json -> crn_run_4000.variants.point.worst_coverage_curve). With the
    # term set to zero that run reproduced H6 exactly (zero_term_run), which is asserted against the curve
    # regenerated above.
    sb = load(os.path.join(OUT, "fv_seedbank_calib.json"))
    zt = sb["zero_term_run"]["worst_coverage_curve"]
    assert np.allclose(zt["c"], grid) and np.array_equal(np.asarray(zt["worst_coverage"]),
                                                         curves["both components (headline)"][0])
    pt = sb["crn_run_4000"]["variants"]["point"]
    sb_c = np.asarray(pt["worst_coverage_curve"]["c"], dtype=float)
    sb_w = np.asarray(pt["worst_coverage_curve"]["worst_coverage"], dtype=float)
    assert np.allclose(sb_c, grid)
    c_rec = float(pt["c_star"])
    j_rec = int(np.argmin(np.abs(grid - c_rec)))
    assert sb_w[j_rec] >= TARGET and np.all(sb_w[:j_rec] < TARGET)
    check(num, "nLimRecC", c_rec)
    check(num, "nLimRecCovAtRecC", sb_w[j_rec])
    check(num, "nLimRecCovAtOne", sb_w[0])
    check(num, "nLimRecCovAtCalC", sb_w[int(np.argmin(np.abs(grid - c_star)))])
    t_crec = num["nLimRecC"]["text"]

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(SC, H_IN))
    L, Rm, T, B = 0.50, 0.14, 0.08, 0.38
    ax = fig.add_axes([L / SC, B / H_IN, (SC - L - Rm) / SC, (H_IN - T - B) / H_IN])

    ax.axhline(TARGET, color=C["grey"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(grid[-1], TARGET - 0.0003, f"nominal {TARGET:.2f}", ha="right", va="top", fontsize=7.5,
            color=C["grey"])

    # 'neither' is regenerated and asserted above but not drawn: it is 1 at every c (coverage cannot fall as
    # c grows on fixed draws), and a dashed line at 1.000 read as a second reference line beside the nominal
    # one. The caption states it.
    assert np.all(curves["neither (C6 reproduction)"][0] == 1.0)
    for key, _se, _st, lab, col, ls, lw in reversed(scen[:2]):
        worst = curves[key][0]
        ax.plot(grid, worst, color=col, ls=ls, lw=lw, zorder=2 if col != SYM else 3,
                solid_capstyle="butt", clip_on=False)

    ax.plot(grid, sb_w, color=SYM, ls=(0, (3, 1.6)), lw=0.9, zorder=3, clip_on=False)

    ax.plot([c_star], [cov_star], ls="none", marker=SYM_MARK, ms=4.0, mfc=SYM, mec="white", mew=0.5,
            zorder=5, clip_on=False)
    ax.text(c_star - 0.03, cov_star + 0.0004, f"$c$ = {t_c}", ha="right", va="bottom", fontsize=7.5,
            color=SYM)
    ax.plot([c_rec], [sb_w[j_rec]], ls="none", marker=SYM_MARK, ms=4.0, mfc="white", mec=SYM, mew=0.8,
            zorder=6, clip_on=False)
    ax.text(c_rec + 0.035, sb_w[j_rec] - 0.0006, f"$c$ = {t_crec}", ha="left", va="top", fontsize=7.5,
            color=SYM)

    # direct labels beside each curve (the right ends coincide near 1, so labels sit where curves separate)
    w_head = curves["both components (headline)"][0]
    w_est = curves["estimation only"][0]
    j = int(np.searchsorted(grid, 1.52))
    ax.text(grid[j], w_head[j] - 0.0011, "estimation and training set", ha="left", va="top", fontsize=7.5,
            color=SYM)
    ax.text(1.50, 0.9862, "dashed: estimation and training set\nwith the seed-bank term", ha="left", va="center",
            fontsize=7.5, color=SYM, linespacing=1.1)
    j2 = int(np.searchsorted(grid, 1.04))
    ax.text(grid[j2], w_est[j2] - 0.0006, "estimation only", ha="left", va="top", fontsize=7.5,
            color=C["grey"])

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0.977, 1.0015)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.2f}"))
    ax.yaxis.set_major_locator(FixedLocator([0.98, 0.985, 0.99, 0.995, 1.0]))
    ax.yaxis.set_minor_locator(MultipleLocator(0.0025))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.3f}"))
    ax.set_xlabel("Half-width multiplier $c$", labelpad=2)
    ax.set_ylabel("Worst-cell coverage", labelpad=3)

    out = save(fig, "figS_03_coverage")

    # spot checks for the record
    for key, *_ in scen:
        w, cm, jw, cells = curves[key]
        cw = cells[jw]
        print(f"{key:28s} c_min={cm:.2f}  worst@1={w[0]:.5f}  worst@c_min={w[np.searchsorted(grid, cm)]:.5f}"
              f"  worst@1.50={w[np.searchsorted(grid, 1.5)]:.5f}  worst@2.50={w[-1]:.5f}"
              f"  worst cell@1: {cw['spread']}, m={cw['m']}, i={cw['i']}")
    return out


if __name__ == "__main__":
    print(main())
