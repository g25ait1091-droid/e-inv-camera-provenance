"""Fig. S6 (fig:s06replications): per-adapter own-body contrasts of the replication designs.
Double column, 6.99 in wide, two rows (OUTLINE §6 S10 "Fig. S6").

Top row, Nikon D200 pair (Dresden), one panel per design; in each, arm A (vermillion circles) and arm B
(blue squares) with one marker per adapter in integer seed order, the arm mean as a short black bar, and the
symmetric estimate theta_sym = (mean_A + mean_B)/2 as a black diamond with +-1 SE.  Values are % of the D200
R_real (E2 estimate, FINAL_LEDGER denominators.R_real).
  (a) second training set, local stack, 3/arm   out/g5_alt_training.json alt.per_adapter_{A,B}, welch_se;
                                                seed labels from out/t1/summary_alt.json (natural_paired_KA_minus_KB)
  (b) second environment, local stack, 6/arm    out/g2_pooled_six.json per_adapter_{A,B}, welch_se;
                                                seeds 0-2 labelled by out/t1/dose_stats.json doses["2000"],
                                                seeds 3-5 by out/t1/summary_nomarkrep.json and out/g_chain12.json G2
  (c) FLUX.1-dev, archive stack, 6/arm          out/flux_seed_ext_summary.json arms.{A,B}.adapters[].{seed,theta},
                                                symmetric.welch_se
  (d) full fine-tuning, archive stack, 3/arm    paper/fv/work/drive_snapshot/E_FULLFT_results.json (copy of Drive
                                                E_FULLFT) arms.{A,B}.means (seed order 0,1,2), symmetric.{per_seed,
                                                theta_sym, sd}; SE = sd/sqrt(3) (the paired-by-seed statistic)
  Open markers in (b) and (c): seeds 3-5, the adapters added by the later registered extension.
Bottom row, five-body groups (archive stack), 2 adapters per body; x = body, sorted by device number:
  (e) Kodak M1063, five Dresden bodies          work/drive_snapshot/C4_multidev.json variants.raw.{per_adapter,
                                                device_means, grand_mean, U_device_level, t_crit}, R_real
  (f) Huawei P20, Daxing 1101-1105              work/drive_snapshot/D6_results.json reps.K.{per_adapter,
                                                per_device, grand_mean, U, t_crit, R_real}
  grey circles = adapters (seed 0 left, seed 1 right), black bar = body mean, black diamond = mean of the
  five body means +-1 SE (SE = SD of the body means / sqrt(5)); values are % of the group's own R_real
  (point estimate; the device-level limit of Table S15 divides by its lower 99 % bound instead).

Every plotted value is read from the result files and cross-checked against the numbers.json macro of the
same quantity (theta_sym, t, limits reconstructed from the plotted values).  No number is printed in the
figure.  Run:
    python src/fv/figS_06_replications.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import json
import os
import re
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_style import plt, DC, C, BODY_A, BODY_B, MARK_A, MARK_B, SYM, SYM_MARK, panel_label, save  # noqa: E402

from matplotlib.ticker import MultipleLocator, FuncFormatter  # noqa: E402

OUT = (EINV.V2 + "/out")
FV = (EINV.V2 + "/paper/fv")
LEDGER = (EINV.REPO + "/analysis/FINAL_LEDGER.json")
NUMJSON = FV + "/numbers.json"
SNAP = FV + "/work/drive_snapshot"
KODAK_MANIFEST = (EINV.V2 + "/data/manifests/kodak/manifest.json")

T_IN, ROW_IN, MID_IN, B_IN = 0.16, 1.40, 0.72, 0.52     # top margin, row height, gap between rows, bottom
H_IN = T_IN + 2 * ROW_IN + MID_IN + B_IN
N_FIRST = 3          # seeds 0-2 first run; 3-5 the registered extension (open markers) in (b) and (c)

plt.rcParams["savefig.bbox"] = "standard"      # made at final size: the canvas is exactly DC wide


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def close(a, b, rel=1e-6, what=""):
    assert abs(a - b) <= rel * max(abs(a), abs(b), 1e-300), (what, a, b)


def check_text(num, macro, value):
    """value, rounded as the macro prints it, equals the macro's printed text."""
    t = num[macro]["text"].replace("$-$", "-").replace("{,}", "")
    m = re.match(r"^\$?(-?[0-9.]+)(?:\\times10\^\{(-?\d+)\})?\$?$", t)
    assert m, (macro, t)
    mant = m.group(1)
    dec = len(mant.split(".")[1]) if "." in mant else 0
    if m.group(2):
        e = int(m.group(2))
        assert round(value / 10 ** e, dec) == float(mant), (macro, t, value)
    else:
        assert round(value, dec) == float(mant), (macro, t, value)


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se = 0.5 * np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    return se, df


def minus_fmt(x, _pos):
    if abs(x) < 1e-12:
        return "0"
    return f"{x:.1f}".replace("-", "\u2212")


def main():
    num = load(NUMJSON)
    led = load(LEDGER)
    R = led["denominators"]["R_real"]
    close(R, num["nLimRrealFull"]["value"], 1e-12, "R_real")
    pct = lambda v, r=R: 100.0 * np.asarray(v, float) / r  # noqa: E731

    designs = []    # dict(A=(seeds, vals%), B=(...), ts%, se%, open_from)

    # ------------------------------------------------------------ (a) second training set (G5)
    g5 = load(OUT + "/g5_alt_training.json")
    alt = g5["alt"]
    close(g5["R_real"], R, 1e-12, "G5 R_real")
    sa = load(OUT + "/t1/summary_alt.json")["arms"]
    arms = {}
    for x, sgn in (("A", 1.0), ("B", -1.0)):
        vals = alt[f"per_adapter_{x}"]
        seeds = []
        for v in vals:        # locate each value's seed tag in the labelled per-arm summary
            hit = [int(k.rsplit("_s", 1)[1]) for k, r in sa.items()
                   if k.startswith(f"alt_{x}_s") and abs(sgn * r["natural_paired_KA_minus_KB"] - v) < 1e-15]
            assert len(hit) == 1, (x, v, hit)
            seeds.append(hit[0])
        o = np.argsort(seeds)
        arms[x] = (list(np.asarray(seeds)[o]), pct(np.asarray(vals)[o]))
    ts = 0.5 * (np.mean(alt["per_adapter_A"]) + np.mean(alt["per_adapter_B"]))
    close(ts, alt["theta_sym"], 1e-9, "G5 theta")
    se, df = welch(alt["per_adapter_A"], alt["per_adapter_B"])
    close(se, alt["welch_se"], 1e-9, "G5 se"); close(df, alt["welch_df"], 1e-9, "G5 df")
    check_text(num, "nGenGfiveLam", 100 * ts / R)
    close(100 * (ts + stats.t.ppf(0.99, df) * se) / R, num["nGenGfiveLimit"]["value"], 1e-6, "G5 limit")
    assert len(arms["A"][0]) == int(num["nGenGfiveAdapters"]["text"]) == int(num["nDataSecondSetAdaptersPerArm"]["text"])
    designs.append(dict(key="G5", xlabel="Second training set", arms=arms, ts=100 * ts / R, se=100 * se / R,
                        open_from=99))

    # ------------------------------------------------------------ (b) second environment (G2)
    g2 = load(OUT + "/g2_pooled_six.json")
    d2k = load(OUT + "/t1/dose_stats.json")["doses"]["2000"]
    rep = load(OUT + "/t1/summary_nomarkrep.json")["arms"]
    ch = load(OUT + "/g_chain12.json")["G2"]
    arms = {}
    for x, sgn, pre in (("A", 1.0, "nomark_s"), ("B", -1.0, "nomarkB_s")):
        vals = np.asarray(g2[f"per_adapter_{x}"], float)
        # seeds 0-2: the dose-axis 2000 arms, with tags
        tags = d2k[f"{x}_arms"]
        assert tags == [f"{pre}{s}" for s in (0, 1, 2)], tags
        assert np.allclose(vals[:3], d2k[f"{x}_own"], rtol=0, atol=1e-18)
        # seeds 3-5: the registered extension, tags from its summary
        for j, s in enumerate((3, 4, 5)):
            close(sgn * rep[f"{pre}{s}"]["natural_paired_KA_minus_KB"], vals[3 + j], 1e-12, f"G2 {x} s{s}")
        assert np.allclose(vals[3:], ch[f"per_adapter_{x}"], rtol=0, atol=1e-18)
        arms[x] = ([0, 1, 2, 3, 4, 5], pct(vals))
    ts = 0.5 * (np.mean(g2["per_adapter_A"]) + np.mean(g2["per_adapter_B"]))
    close(ts, g2["theta_sym"], 1e-9, "G2 theta")
    se, df = welch(g2["per_adapter_A"], g2["per_adapter_B"])
    close(se, g2["welch_se"], 1e-9, "G2 se"); close(df, g2["welch_df"], 1e-9, "G2 df")
    check_text(num, "nGenGtwoLam", 100 * ts / R)
    check_text(num, "nGenGtwoLimit", 100 * (ts + stats.t.ppf(0.99, df) * se) / R)
    check_text(num, "nGenGtwoNewLam", 100 * 0.5 * (np.mean(ch["per_adapter_A"]) + np.mean(ch["per_adapter_B"])) / R)
    assert len(arms["A"][0]) == int(num["nGenGtwoAdapters"]["text"])
    designs.append(dict(key="G2", xlabel="Local-stack retraining", arms=arms, ts=100 * ts / R, se=100 * se / R,
                        open_from=N_FIRST))

    # ------------------------------------------------------------ (c) FLUX.1-dev
    fx = load(OUT + "/flux_seed_ext_summary.json")
    close(fx["R_real_ledger"], R, 1e-12, "FLUX R_real")
    arms = {}
    for x in ("A", "B"):
        ad = sorted(fx["arms"][x]["adapters"], key=lambda a: int(a["seed"]))
        assert [int(a["seed"]) for a in ad] == list(range(6))
        assert all(a["tag"] == f"{x}_raw_s{a['seed']}_flux" for a in ad)
        vals = np.array([a["theta"] for a in ad])
        close(vals.mean(), fx["arms"][x]["mean"], 1e-9, "FLUX mean " + x)
        arms[x] = ([int(a["seed"]) for a in ad], pct(vals))
    sym = fx["symmetric"]
    ts = 0.5 * (fx["arms"]["A"]["mean"] + fx["arms"]["B"]["mean"])
    close(ts, sym["theta_sym"], 1e-9, "FLUX theta")
    se, df = welch([a["theta"] for a in fx["arms"]["A"]["adapters"]], [a["theta"] for a in fx["arms"]["B"]["adapters"]])
    close(se, sym["welch_se"], 1e-9, "FLUX se"); close(df, sym["welch_df"], 1e-9, "FLUX df")
    # the file's own R_real is the 6-digit literal; the plotted % uses the ledger value (identical to 6 digits)
    check_text(num, "nGenFluxLam", 100 * ts / R)
    check_text(num, "nGenFluxSym", 100 * (ts + stats.t.ppf(0.99, df) * se) / R)
    assert len(arms["A"][0]) == int(num["nGenFluxAdapters"]["text"])
    designs.append(dict(key="FLUX", xlabel="FLUX.1-dev", arms=arms, ts=100 * ts / R, se=100 * se / R,
                        open_from=N_FIRST))

    # ------------------------------------------------------------ (d) full fine-tuning
    ft = load(SNAP + "/E_FULLFT_results.json")
    arms = {}
    for x in ("A", "B"):
        vals = np.asarray(ft["arms"][x]["means"], float)
        assert vals.size == ft["arms"][x]["n"] == int(num["nGenFullAdapters"]["text"])
        close(vals.mean(), ft["arms"][x]["theta"], 1e-9, "FT theta " + x)
        arms[x] = ([0, 1, 2], pct(vals))          # stored in seed order s0, s1, s2 (A_ft_s0..s2)
    per_seed = 0.5 * (np.asarray(ft["arms"]["A"]["means"]) + np.asarray(ft["arms"]["B"]["means"]))
    assert np.allclose(per_seed, ft["symmetric"]["per_seed"], rtol=1e-9, atol=0)
    ts = per_seed.mean()
    close(ts, ft["symmetric"]["theta_sym"], 1e-9, "FT theta_sym")
    se = per_seed.std(ddof=1) / np.sqrt(per_seed.size)
    close(per_seed.std(ddof=1), ft["symmetric"]["sd"], 1e-9, "FT sd")
    check_text(num, "nGenFullT", ts / se)
    check_text(num, "nGenFullSym", 100 * (ts + stats.t.ppf(0.995, per_seed.size - 1) * se) / R)
    close(ft["symmetric"]["U"], ts + stats.t.ppf(0.995, per_seed.size - 1) * se, 1e-6, "FT U")
    check_text(num, "nGenFullMaxArm", 100 * max(ft["arms"]["A"]["U"], ft["arms"]["B"]["U"]) / R)
    designs.append(dict(key="FT", xlabel="Full fine-tuning", arms=arms, ts=100 * ts / R, se=100 * se / R,
                        open_from=99))

    # ------------------------------------------------------------ (e) Kodak five bodies
    c4 = load(SNAP + "/C4_multidev.json")
    raw = c4["variants"]["raw"]
    man = load(KODAK_MANIFEST)["picks"]
    Rk = c4["R_real"]
    kod = []
    for dev, blk in man.items():
        did = int(blk["id"].rsplit("_", 1)[1])         # Dresden device number of Kodak_M1063_<n>
        ad = sorted([a for a in raw["per_adapter"] if a["dev"] == dev], key=lambda a: int(a["seed"]))
        assert [int(a["seed"]) for a in ad] == [0, 1]
        v = np.array([a["theta"] for a in ad])
        close(v.mean(), raw["device_means"][dev], 1e-9, "Kodak mean " + dev)
        kod.append((did, dev, v))
    kod.sort(key=lambda t: t[0])
    means = np.array([t[2].mean() for t in kod])
    close(means.mean(), raw["grand_mean"], 1e-9, "Kodak grand")
    check_text(num, "nGenKodakGrandMean", means.mean())
    se_k = means.std(ddof=1) / np.sqrt(means.size)
    close(raw["t_crit"], stats.t.ppf(0.995, means.size - 1), 1e-6, "Kodak t_crit")
    close(means.mean() + raw["t_crit"] * se_k, raw["U_device_level"], 1e-6, "Kodak U")
    check_text(num, "nGenKodakLimit", 100 * raw["U_device_level"] / c4["R_real_lower99"])
    assert len(kod) == int(num["nGenKodakBodies"]["text"])
    assert sum(np.all(np.sign(t[2]) == np.sign(t[2][0])) for t in kod) == int(num["nGenKodakSignAgree"]["text"])
    five = [dict(key="Kodak", xlabel="Kodak M1063 body (Dresden device)",
                 bodies=[(str(t[0]), pct(t[2], Rk)) for t in kod],
                 ts=100 * means.mean() / Rk, se=100 * se_k / Rk, R=Rk)]

    # ------------------------------------------------------------ (f) P20 five bodies
    d6 = load(SNAP + "/D6_results.json")
    K = d6["reps"]["K"]
    Rp = K["R_real"]
    p20 = []
    for dev in sorted(d6["devices"], key=int):
        ad = sorted([a for a in K["per_adapter"] if str(a["dev"]) == dev], key=lambda a: int(a["seed"]))
        assert [int(a["seed"]) for a in ad] == [0, 1]
        v = np.array([a["theta"] for a in ad])
        close(v.mean(), K["per_device"][dev], 1e-9, "P20 mean " + dev)
        p20.append((dev, v))
    means = np.array([t[1].mean() for t in p20])
    close(means.mean(), K["grand_mean"], 1e-9, "P20 grand")
    se_p = means.std(ddof=1) / np.sqrt(means.size)
    close(means.mean() + K["t_crit"] * se_p, K["U"], 1e-6, "P20 U")
    check_text(num, "nGenPtwentyLimit", 100 * K["U"] / K["R_real_lower99"])
    close(K["signflip_p"], led["smartphone"]["signflip_p"], 1e-3, "P20 p vs ledger")
    # the caption states one generation count for both groups (nDataKodakGensPerAdapter)
    assert num["nDataKodakGensPerAdapter"]["text"] == num["nDataPTwentyGensPerAdapter"]["text"]
    assert num["nDataKodakAdaptersPerArm"]["text"] == num["nDataPTwentyAdaptersPerArm"]["text"] == "2"
    five.append(dict(key="P20", xlabel="Huawei P20 body (Daxing device)",
                     bodies=[(t[0], pct(t[1], Rp)) for t in p20],
                     ts=100 * means.mean() / Rp, se=100 * se_p / Rp, R=Rp))

    # ================================================================ layout (inches)
    fig = plt.figure(figsize=(DC, H_IN))
    L_, R_, GAP = 0.47, 0.04, 0.16
    T_, ROWH, MID, B_ = T_IN, ROW_IN, MID_IN, B_IN
    w_top = (DC - L_ - R_ - 3 * GAP) / 4
    y_top = H_IN - T_ - ROWH
    y_bot = B_
    assert abs(y_top - MID - ROWH - y_bot) < 1e-9, "row heights do not add up"
    axes_top = [fig.add_axes([(L_ + k * (w_top + GAP)) / DC, y_top / H_IN, w_top / DC, ROWH / H_IN])
                for k in range(4)]
    w_bot = 2 * w_top + GAP
    axes_bot = [fig.add_axes([(L_ + k * (w_bot + GAP)) / DC, y_bot / H_IN, w_bot / DC, ROWH / H_IN],
                             sharey=axes_top[0]) for k in range(2)]
    for ax in axes_top[1:]:
        ax.sharey(axes_top[0])

    ink, grey = C["ink"], C["grey"]
    col = {"A": BODY_A, "B": BODY_B}
    mk = {"A": MARK_A, "B": MARK_B}
    msz = {"A": 3.4, "B": 3.1}

    def zero(ax):
        ax.axhline(0.0, color=grey, ls=(0, (3, 2)), lw=0.6, zorder=1)

    def diamond(ax, xc, v, se):
        ax.errorbar(xc, v, yerr=se, fmt=SYM_MARK, ms=4.0, mew=0.7, mec=SYM, mfc=SYM, ecolor=SYM,
                    elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)

    def xstep(ax, inches):
        lo, hi = ax.get_xlim()
        return inches * (hi - lo) / (ax.get_position().width * DC)

    # ---------------------------------------------------------------- top row: D200 designs
    XC = {"A": 0.0, "B": 1.0}
    for ax, d in zip(axes_top, designs):
        ax.set_xlim(-0.55, 2.45)
        zero(ax)
        step = xstep(ax, 0.052)
        for x in ("A", "B"):
            seeds, vals = d["arms"][x]
            n = len(vals)
            dx = (np.arange(n) - (n - 1) / 2.0) * step
            for s, v, e in zip(seeds, vals, dx):
                filled = s < d["open_from"]
                ax.plot(XC[x] + e, v, ls="none", marker=mk[x], ms=msz[x], mew=0.7, mec=col[x],
                        mfc=col[x] if filled else "white", zorder=3)
            half = max(0.5 * (dx.max() - dx.min()) + 0.5 * step, 0.16)
            ax.plot([XC[x] - half, XC[x] + half], [vals.mean()] * 2, color=SYM, lw=1.2,
                    solid_capstyle="butt", zorder=2)
        diamond(ax, 2.0, d["ts"], d["se"])
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["A", "B", r"$\theta_{\mathrm{sym}}$"])
        ax.get_xticklabels()[0].set_color(BODY_A)
        ax.get_xticklabels()[1].set_color(BODY_B)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.set_xlabel(d["xlabel"], labelpad=3)

    # ---------------------------------------------------------------- bottom row: five-body groups
    for ax, d in zip(axes_bot, five):
        nb = len(d["bodies"])
        xm = nb + 0.35
        ax.set_xlim(-0.6, xm + 0.6)
        zero(ax)
        step = xstep(ax, 0.075)
        for k, (lab, vals) in enumerate(d["bodies"]):
            dx = (np.arange(vals.size) - (vals.size - 1) / 2.0) * step
            ax.plot(k + dx, vals, ls="none", marker="o", ms=3.4, mew=0.7, mec=grey, mfc=grey, zorder=3)
            ax.plot([k - 0.24, k + 0.24], [vals.mean()] * 2, color=SYM, lw=1.2, solid_capstyle="butt", zorder=2)
        diamond(ax, xm, d["ts"], d["se"])
        ax.set_xticks(list(range(nb)) + [xm])
        ax.set_xticklabels([lab for lab, _ in d["bodies"]] + ["Mean"])
        ax.tick_params(axis="x", length=0, pad=3)
        ax.set_xlabel(d["xlabel"], labelpad=3)

    # ---------------------------------------------------------------- shared y
    allv = np.concatenate([d["arms"][x][1] for d in designs for x in "AB"]
                          + [v for d in five for _, v in d["bodies"]])
    lo, hi = allv.min(), allv.max()
    y0, y1 = lo - 0.05, hi + 0.05
    ax0 = axes_top[0]
    ax0.set_ylim(y0, y1)
    ax0.yaxis.set_major_locator(MultipleLocator(0.2 if (y1 - y0) < 1.8 else 0.4))
    ax0.yaxis.set_major_formatter(FuncFormatter(minus_fmt))
    for ax in axes_top[1:] + axes_bot[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)
    ylab = r"Own-body contrast (% of $R_{\mathrm{real}}$)"
    axes_top[0].set_ylabel(ylab, labelpad=2)
    axes_bot[0].set_ylabel(ylab, labelpad=2)

    for ax, s in zip(axes_top + axes_bot, "abcdef"):
        panel_label(ax, f"({s})", x=-0.02 if ax in (axes_top[0], axes_bot[0]) else 0.0)

    path = save(fig, "figS_06_replications")

    # ---------------------------------------------------------------- report (spot checks)
    print("saved", path, "range", round(lo, 4), round(hi, 4))
    for d in designs:
        print(d["key"], "A", [f"{v:+.4f}" for v in d["arms"]["A"][1]], "B", [f"{v:+.4f}" for v in d["arms"]["B"][1]],
              "ts", f"{d['ts']:+.4f}", "se", f"{d['se']:.4f}")
    for d in five:
        print(d["key"], "R", f"{d['R']:.5f}", [(lab, [f"{x:+.4f}" for x in v]) for lab, v in d["bodies"]],
              "mean", f"{d['ts']:+.4f}", "se", f"{d['se']:.4f}")


if __name__ == "__main__":
    main()
