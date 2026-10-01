"""Fig. S10 (fig:s10power; graphic figs/fig07_power.pdf): what the limit means for attribution.
Single column, 3.35 x 2.6 in, one panel.

True-positive rate at 1 % false-positive rate against the number of generated images per decision G, for the
examiner of the power model: it holds every candidate camera's fingerprint and main effect, scores fresh
generations of one personalized model and Bonferroni-corrects over the M - 1 wrong candidates. The curves are
verify_v2.tpr() (the repository verifier, imported, not copied) with the inputs of out/t3_power_v4.json -> inputs
(U_device, SE_img_500), the calibrated limit of record of out/fv_seedbank_calib.json (c* = 1.31, RESULTS Entry
117; its stored rates at G = 500 and 5000 are the grey circles), and the persistent term sigma_mu as
estimated from the paired contrast in out/fv_sigma.json (RESULTS Entry 114):

  solid          M = 2, 5, 50 at the nominal limit with sigma_mu = 0 (the paired estimate; a bound for every value);
  grey band      M = 2 at the nominal limit, sigma_mu from 0 to the upper end of its exact 95 % interval; the lower
                 edge (thin line) is the curve at that upper end;
  dashed         M = 2, 5, 50 at ten times the nominal limit, sigma_mu = 0;
  thin grey      M = 2 at the calibrated limit of record, sigma_mu = 0;
  open circles   the values stored in the result files at G = 500 and 5000 (t3_power_v4.json rows_sigma_mu_zero;
                 fv_sigma.json power.*), to show agreement;
  right margin   a short tick at the unlimited-image TPR of the band's lower edge.

Every plotted value is read from the result files; nothing is typed in.  Run:
    python src/fv/fig_07_power.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import contextlib
import importlib.util
import io
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_style import plt, SC, C, save  # noqa: E402

from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator  # noqa: E402

OUT = (EINV.V2 + "/out")
REPO = EINV.REPO
H_IN = 2.6                       # figure height, inches (OUTLINE Fig. 7)
G_MIN, G_MAX = 10, 1e5           # x range (OUTLINE Fig. 7)
MS = (2, 5, 50)                  # candidate-set sizes (design constants of the power table)

AX = [0.125, 0.155, 0.80, 0.815]    # axes box in figure fractions

plt.rcParams["savefig.bbox"] = "standard"   # keep the canvas exactly SC wide


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def import_verifier():
    """verify_v2 runs its checks at import and calls sys.exit; capture its output, catch the exit, keep tpr()."""
    spec = importlib.util.spec_from_file_location("verify_v2", REPO + "/verify_v2.py")
    mod = importlib.util.module_from_spec(spec)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
        except Exception:   # tpr() is defined before most checks; keep it if it exists
            pass
    assert hasattr(mod, "tpr"), "verify_v2.tpr not defined"
    return mod


def main():
    vv = import_verifier()
    pw = load(OUT + "/t3_power_v4.json")
    inp = pw["inputs"]
    U_nom, se500 = inp["U_device"], inp["SE_img_500"]
    U_h6 = load(OUT + "/h6_calibrated_limit.json")["primary_d200"]["U_device"]
    SB = load(OUT + "/fv_seedbank_calib.json")                 # RESULTS Entry 117 item 1: the bound of record
    U_cal = SB["limits"]["bound_of_record"]["U_device"]        # drawn as "calibrated" (c* = 1.31)
    rec = SB["derived_at_new_limit"]
    assert abs(rec["U_new"] - U_cal) < 1e-15 and abs(rec["inputs"]["SE_img_500"] - se500) < 1e-15
    S = load(OUT + "/fv_sigma.json")
    assert S["entry"] == "RESULTS.md Entry 114"
    sp = S["power"]
    assert abs(sp["inputs"]["U_nominal"] - U_nom) < 1e-15 and abs(sp["inputs"]["U_calibrated"] - U_h6) < 1e-15
    s_hat = S["primary_crossed_model"]["sigma_u_estimate"]                   # 0, on the boundary
    s_hi = S["primary_crossed_model"]["sigma_u_interval_exact"]["two_sided_95"][1]
    assert s_hat == 0.0

    rows_ten = {r["M_candidates"]: r for r in pw["rows"] if r["transfer"] == "10x the upper limit"}
    rows_zero = {r["M_candidates"]: r for r in pw["rows_sigma_mu_zero"]}
    ten = rows_ten[2]["theta_true"] / U_nom                       # the file's multiplier (10)
    assert abs(ten - 10) < 1e-9

    G = np.logspace(math.log10(G_MIN), math.log10(G_MAX), 400)

    def tpr1(U, M, sigma, g):
        if sigma == 0.0 and math.isinf(g):
            return 1.0
        vv.U_DEVICE = U                                             # tpr() reads the module-level U_DEVICE
        return vv.tpr(g, sigma_mu=sigma, se500=se500, M=M)

    def curve(U, M, sigma):
        return np.array([tpr1(U, M, sigma, g) for g in G])

    # agreement of the imported model with the stored values (the markers must sit on the curves)
    for M in MS:
        for g in (500, 5000):
            assert abs(tpr1(U_nom, M, 0.0, g) - rows_zero[M][f"TPR_G{g}"]) < 1e-9
            assert abs(tpr1(U_nom, M, s_hi, g) - sp["nominal"]["paired_high_95"][f"M{M}"][f"TPR_G{g}"]) < 1e-9
    zc = rec["at_new_limit"]["calibrated_rows"]["zero"]["M2"]
    for g in (500, 5000):
        assert abs(tpr1(U_h6, 2, 0.0, g) - sp["calibrated"]["zero"]["M2"][f"TPR_G{g}"]) < 1e-9
        assert abs(tpr1(U_cal, 2, 0.0, g) - zc[f"TPR_G{g}"]) < 1e-9
    for M in MS:                                                    # ten times the limit: exact counts in the file
        g50 = sp["ten_times_nominal_G_for_TPR50"]["zero"][f"M{M}"]
        assert abs(tpr1(ten * U_nom, M, 0.0, g50) - 0.5) < 1e-9

    ink, grey, light = C["ink"], C["grey"], C["light"]
    fig = plt.figure(figsize=(SC, H_IN))
    ax = fig.add_axes(AX)
    mk = dict(marker="o", ms=3.0, mfc="white", mew=0.6, ls="none", zorder=5, clip_on=False)

    # grey band: M = 2, nominal limit, sigma_mu from 0 to the interval's upper end
    y0 = curve(U_nom, 2, 0.0)
    y_hi = curve(U_nom, 2, s_hi)
    ax.fill_between(G, y_hi, y0, color=light, alpha=0.40, lw=0, zorder=1)
    ax.plot(G, y_hi, color=ink, lw=0.5, zorder=2)
    h = sp["nominal"]["paired_high_95"]["M2"]
    ax.plot([500, 5000], [h["TPR_G500"], h["TPR_G5000"]], mec=ink, **mk)

    # thin grey: M = 2 at the calibrated limit (the bound of record), sigma_mu = 0
    ax.plot(G, curve(U_cal, 2, 0.0), color=grey, lw=0.6, zorder=2)
    ax.plot([500, 5000], [zc["TPR_G500"], zc["TPR_G5000"]], mec=grey, **mk)

    # dashed: ten times the nominal limit, sigma_mu = 0
    for M in MS:
        ax.plot(G, curve(ten * U_nom, M, 0.0), color=ink, lw=0.9, ls=(0, (4, 2)), zorder=3)

    # solid: nominal limit, sigma_mu = 0
    for M in MS:
        ax.plot(G, curve(U_nom, M, 0.0), color=ink, lw=0.9, zorder=3)
        ax.plot([500, 5000], [rows_zero[M]["TPR_G500"], rows_zero[M]["TPR_G5000"]], mec=ink, **mk)

    ax.set_xscale("log")
    ax.set_xlim(G_MIN, G_MAX)
    ax.set_ylim(0, 1.015)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.set_xlabel("Generated images per decision, $G$", labelpad=2)
    ax.set_ylabel("TPR at 1% FPR", labelpad=3)

    fs = 7.5
    # right margin: unlimited-image TPR of the band's lower edge
    trans = ax.get_yaxis_transform()
    x0, x1 = 1.015, 1.055
    ginf_hi = h["TPR_G_inf"]
    ax.plot([x0, x1], [ginf_hi] * 2, color=ink, lw=0.8, transform=trans, clip_on=False)

    # numerals M just right of each curve at TPR 0.3, for both groups; group names
    def x_at(U, M, y):
        yc = curve(U, M, 0.0)
        return float(np.exp(np.interp(y, yc, np.log(G))))
    for U, yl in ((ten * U_nom, 0.30), (U_nom, 0.30)):
        for M in MS:
            ax.text(x_at(U, M, yl) * 1.08, yl, f"{M}", ha="left", va="center", fontsize=fs, color=ink)
    ax.text(260, 0.80, "10× nominal limit", ha="left", va="center", fontsize=fs, color=ink)
    ax.text(9.5e4, 0.55, "nominal limit", ha="right", va="center", fontsize=fs, color=ink)
    ax.text(x_at(U_cal, 2, 0.93) / 1.12, 0.93, "calibrated", ha="right", va="center", fontsize=fs,
            color=grey)
    ax.text(9.5e4, ginf_hi - 0.025, "$\\sigma_\\mu$ at its\n95% upper limit", ha="right", va="top",
            fontsize=fs, color=ink, linespacing=1.1)

    save(fig, "fig07_power")

    # report the values used, for the spot-check
    print("U_nom", U_nom, "U_cal (record)", U_cal, "U_H6", U_h6, "sigma_hi", s_hi, "SE500", se500)
    for M in MS:
        print("M", M, "sigma 0:", rows_zero[M]["TPR_G500"], rows_zero[M]["TPR_G5000"],
              "| sigma_hi:", sp["nominal"]["paired_high_95"][f"M{M}"]["TPR_G500"],
              sp["nominal"]["paired_high_95"][f"M{M}"]["TPR_G_inf"],
              "| 10x G50:", sp["ten_times_nominal_G_for_TPR50"]["zero"][f"M{M}"])
    print("calibrated M=2 sigma 0", zc["TPR_G500"], zc["TPR_G5000"], zc["G_for_TPR50"])


if __name__ == "__main__":
    main()
