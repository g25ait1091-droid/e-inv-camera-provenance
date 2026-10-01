"""Fig. 3 (fig:bands): the autoencoder is a sharp low-pass filter, and the fingerprint's energy sits
where it cuts.  Single column, 3.35 x 3.3 in, two stacked panels sharing the octave axis.

(a) Transmission per octave band, % of the pattern's stored band contrast (log axis):
    autoencoder alone (T_vae, band_vae.json) and full pipeline (band2_summary.json for bands 0-1,
    three adapters; band_summary.json curve[2..5], one adapter each).
(b) Share of energy per octave of the natural-image estimate of body A's fingerprint (K_A, E2) and of
    the DiffusionShield luminance pattern (band_fields.json).

Every plotted value is read from the result files; nothing is typed in.  Run:
    python src/fv/fig_03_bands.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_style import plt, SC, C, BODY_A, MARK_A, panel_label, save  # noqa: E402

from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator  # noqa: E402

OUT = (EINV.V2 + "/out/t1")
NUMJSON = (EINV.V2 + "/paper/fv/numbers.json")
Z99 = 2.33          # one-sided 99 % normal quantile, used for the unresolved bands (OUTLINE Fig. 3 spec)
H_IN = 3.3          # figure height, inches (OUTLINE Fig. 3)

# The figure is made at final size: keep the canvas exactly SC wide (no tight bounding box).
plt.rcParams["savefig.bbox"] = "standard"


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    vae = load("band_vae.json")
    b2 = load("band2_summary.json")
    b1 = load("band_summary.json")
    flds = load("band_fields.json")
    with open(NUMJSON, encoding="utf-8") as f:
        num = json.load(f)

    edges = np.array(vae["edges_cycles_per_px"], dtype=float)          # band b: [f_lo, f_hi] cycles/px
    nb = len(edges)
    assert np.allclose(edges, np.array(b1["edges_cycles_per_px"])), "band edges differ between files"
    assert np.allclose(edges, np.array(flds["edges_cycles_per_px"])), "band edges differ between files"
    per_lo = 1.0 / edges[:, 1]                                          # period range in px: 2-4 ... 64-128
    per_hi = 1.0 / edges[:, 0]
    x = np.log2(np.sqrt(per_lo * per_hi))                              # log spacing, one unit per octave
    xlabels = [f"{p:.0f}\u2013{q:.0f}" for p, q in zip(per_lo, per_hi)]

    # ---- autoencoder alone, % of stored band contrast
    t_vae = np.array([100.0 * vae["T_vae"][f"band{b}"] for b in range(nb)])

    # ---- full pipeline
    curve = {c["band"]: c for c in b1["curve"]}
    full = {}   # band -> dict(mean, se, n, upper99 or None, resolved)
    for b in range(nb):
        key = f"band{b}"
        if key in b2["bands"]:                                          # three-adapter replicate (bands 0-1)
            d = b2["bands"][key]
            full[b] = dict(mean=d["T_mean_pct"], se=d["T_se_adapter_pct"], n=d["n_adapters"],
                           upper=d["T_upper99_pct"], resolved=True)
        else:                                                           # one adapter (bands 2-5)
            c = curve[b]
            n = len([a for a in b1["arms"] if re.fullmatch(rf"band{b}_s\d+", a)])
            full[b] = dict(mean=100.0 * c["T_full"], se=100.0 * c["T_full_se"], n=n, upper=None,
                           resolved=bool(c["detected_3se"]))
    # generations per adapter (stated in the caption): every band and no-pattern arm must agree
    rows2 = np.load(os.path.join(OUT, "band2_rows.npz"))
    n_gen = {k: rows2[k].shape[0] for k in rows2.files}
    n_gen.update({a: v["n"] for a, v in b1["arms"].items()})
    assert len(set(n_gen.values())) == 1, n_gen
    # the caption prints this count with \nMapGensPerAdapter (the map designs, band arms included; the
    # same macro Table 2 uses for the band rows); the unmarked offset arms above share the count
    assert str(next(iter(n_gen.values()))) == num["nMapGensPerAdapter"]["text"], n_gen
    # adapter counts shown in the figure must agree with numbers.json
    assert str(full[0]["n"]) == num["nMapBandAdaptersFinest"]["text"]
    assert f"{full[0]['upper']:.3f}" == num["nMapBandFinestUpper"]["text"]
    assert f"{full[1]['mean']:.3f}" == num["nMapBandFourEight"]["text"]
    assert f"{full[2]['mean']:.3f}" == num["nMapBandEightSixteen"]["text"]
    assert sum(full[b]["n"] for b in range(nb)) == int(num["nMapBandAdapters"]["text"])

    # ---- energy share per octave, %
    e_k = 100.0 * np.array(flds["energy_fraction_K_A_E2"], dtype=float)
    e_w = 100.0 * np.array(flds["energy_fraction_DiffusionShield_lum"], dtype=float)
    assert f"{e_k[0]:.0f}" == num["nAEEnergyTopA"]["text"]

    # ---------------------------------------------------------------- layout (inches -> figure fraction)
    fig = plt.figure(figsize=(SC, H_IN))
    L, R = 0.62, 0.06                     # left margin holds tick labels + y label; right margin small
    B, T = 0.40, 0.14                     # bottom holds x tick labels + x label; top holds panel label
    gap = 0.26
    h_a, h_b = 1.68, None
    h_b = H_IN - B - T - gap - h_a
    w = SC - L - R
    ax_a = fig.add_axes([L / SC, (B + h_b + gap) / H_IN, w / SC, h_a / H_IN])
    ax_b = fig.add_axes([L / SC, B / H_IN, w / SC, h_b / H_IN], sharex=ax_a)

    ink = C["ink"]
    grey = C["grey"]

    # ================================================================ (a) transmission
    ax = ax_a
    ymin, ymax = 1e-3, 200.0
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)

    # autoencoder alone: open grey circles joined by a thin line
    ax.plot(x, t_vae, color=grey, lw=0.7, zorder=2)
    ax.plot(x, t_vae, ls="none", marker="o", ms=3.6, mfc="white", mec=grey, mew=0.8, zorder=3)
    ax.text(x[-1], t_vae[-1] * 1.35, "autoencoder", ha="right", va="bottom", fontsize=7.5, color=grey)

    # full pipeline: resolved bands, black filled markers +- 1 SE (lower end clipped at the axis floor)
    res = [b for b in range(nb) if full[b]["resolved"] or full[b]["upper"] is not None]
    unres = [b for b in range(nb) if b not in res]
    xr = np.array([x[b] for b in res])
    m = np.array([full[b]["mean"] for b in res])
    se = np.array([full[b]["se"] for b in res])
    lo = np.maximum(m - se, ymin * 1.0001)
    hi = m + se
    # the caption says the lower end of the 2-4 px bar is below zero and cut at the axis, and names no
    # other cut bar: hold the data to that
    clipped = [b for b, v in zip(res, m - se) if v <= ymin]
    assert clipped == [0] and full[0]["mean"] - full[0]["se"] < 0, (clipped, full[0])
    ax.vlines(xr, lo, hi, color=ink, lw=0.7, zorder=3)
    capw = 0.07
    for xi, l0, h0, lo_clipped in zip(xr, lo, hi, (m - se) <= ymin):
        ax.hlines(h0, xi - capw, xi + capw, color=ink, lw=0.7, zorder=3)
        if not lo_clipped:
            ax.hlines(l0, xi - capw, xi + capw, color=ink, lw=0.7, zorder=3)
    ax.plot(xr, m, ls="none", marker="o", ms=3.6, mfc=ink, mec=ink, mew=0.6, zorder=4)
    ax.plot(xr, m, color=ink, lw=0.6, zorder=2)

    # band 0: one-sided 99 % upper limit as a short horizontal tick with a downward arrow
    b0 = full[0]
    ax.hlines(b0["upper"], x[0] - 0.16, x[0] + 0.16, color=ink, lw=0.9, zorder=4)
    ax.annotate("", xy=(x[0], b0["upper"] / 1.9), xytext=(x[0], b0["upper"]),
                arrowprops=dict(arrowstyle="-|>", color=ink, lw=0.6, mutation_scale=5,
                                shrinkA=0, shrinkB=0), zorder=4)

    # unresolved bands: open triangle at T + 2.33 SE (upper bound only)
    xu = np.array([x[b] for b in unres])
    up = np.array([full[b]["mean"] + Z99 * full[b]["se"] for b in unres])
    ax.plot(xu, up, ls="none", marker="v", ms=4.0, mfc="white", mec=ink, mew=0.7, zorder=4)
    ax.text(xu.mean(), up.min() / 1.7, "unresolved", ha="center", va="top", fontsize=7.5, color=ink)

    # label for the full-pipeline series, placed beside its line
    ax.text(x[1] + 0.12, full[1]["mean"] / 2.3, "full pipeline", ha="left", va="top", fontsize=7.5,
            color=ink)

    # adapters per point, above each full-pipeline marker (from the result files)
    for b in range(nb):
        f = full[b]
        top = f["upper"] if f["upper"] is not None else (
            f["mean"] + f["se"] if b in res else f["mean"] + Z99 * f["se"])
        ax.text(x[b], top * 1.45, f"{f['n']}", ha="center", va="bottom", fontsize=7, color=ink)

    ax.set_ylabel("Transmission (% of stored\nband contrast)", fontsize=8.5, labelpad=2)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 10, 100])
    ax.set_yticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "1", "10", "100"])
    plt.setp(ax.get_xticklabels(), visible=False)
    panel_label(ax, "(a)", x=(0.20 - L) / w, y=1.0)

    # ================================================================ (b) energy share
    ax = ax_b
    ax.plot(x, e_k, color=BODY_A, lw=0.7, zorder=2)
    ax.plot(x, e_k, ls="none", marker=MARK_A, ms=3.6, mfc=BODY_A, mec=BODY_A, zorder=3)
    ax.plot(x, e_w, color=grey, lw=0.7, zorder=2)
    ax.plot(x, e_w, ls="none", marker="s", ms=3.2, mfc=grey, mec=grey, zorder=3)
    # the two series coincide near zero at the coarse end, so a frameless key sits in the empty corner
    hk, = ax.plot([], [], color=BODY_A, lw=0.7, marker=MARK_A, ms=3.6, mfc=BODY_A, mec=BODY_A)
    hw, = ax.plot([], [], color=grey, lw=0.7, marker="s", ms=3.2, mfc=grey, mec=grey)
    ax.legend([hk, hw], [r"$\hat{K}_A$ (body A)", "DiffusionShield"], loc="upper right",
              fontsize=7.5, handlelength=1.8, borderaxespad=0.1, labelspacing=0.3)
    ax.set_ylim(0, 60)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("Energy share (% of total)", fontsize=8.5, labelpad=2)
    ax.set_xlabel("Octave band, period (px)", fontsize=8.5, labelpad=2)
    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.set_xticklabels(xlabels)
    ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)
    panel_label(ax, "(b)", x=(0.20 - L) / w, y=1.0)

    # align y labels of both panels
    fig.align_ylabels([ax_a, ax_b])

    out = save(fig, "fig03_bands")
    # report the plotted values for spot checks
    print("x labels:", xlabels)
    print("T_vae %:", np.round(t_vae, 2))
    for b in range(nb):
        f = full[b]
        print(f"band{b}: mean {f['mean']:.5f} se {f['se']:.5f} n {f['n']} upper99 {f['upper']} "
              f"resolved {f['resolved']} T+2.33SE {f['mean'] + Z99 * f['se']:.4f}")
    print("energy K_A %:", np.round(e_k, 3), " DS %:", np.round(e_w, 3))
    print("corners K_A %:", round(100 - e_k.sum(), 2), " DS %:", round(100 - e_w.sum(), 2))
    print("generations per adapter:", sorted(set(n_gen.values())), "arms:", sorted(n_gen))
    print("saved", out)


if __name__ == "__main__":
    main()
