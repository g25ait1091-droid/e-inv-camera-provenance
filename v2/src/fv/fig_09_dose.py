"""Fig. 9 (fig:dose): heavy adaptation -- the fingerprint passes, and follows its sign.
Double column, 6.99 x 2.6 in, two panels sharing the y axis (OUTLINE §4 "Fig. 9").

(a) Own-body contrast theta_{x,j} of every adapter, % of R_real (D200 pair, E2 estimate), at 2000 (local
    stack, 3 per body), 8000 (2 per body) and 16000 (6 per body) training steps; theta_sym +- 1 SE as black
    diamonds; the open diamond at 16000 is theta_sym of the registered replication (seeds 3-5) alone; the
    grey tick at 2000 and 8000 is the symmetric one-sided 99 % limit.
    Source: out/t1/dose_stats.json -> doses.{2000,8000,16000}.{A_arms,B_arms,A_own,B_own,theta_sym,
    adapter_SE,lambda_U_plugin_pct}; doses.16000_new.{theta_sym, adapter_SE}.
    doses.16000 already holds seeds 0-5 (the 16000_new arms included), so no separate replication column.
(b) 16000 steps, normal (6 per body) against inverted (8 per body) training; per-adapter points; body
    means (short bars) joined within body; theta_sym +- 1 SE per condition, printed via \nDoseSixteenLam and
    \nDoseInvLam.
    Source: out/g1_within_body.json -> normal_16k_own.{A,B}, inverted_own.{A,B}; seed tags from
    out/g1_ext.json -> per_adapter_{A,B}; theta_sym and SE of the inverted condition from
    g1_ext.json all_eight.{theta_sym, welch_se}.

Nothing is typed in: every plotted value is read from these files and checked against numbers.json;
every printed number is the `text` of a numbers.json macro.  Adapters are sorted by integer seed and placed
left to right within a column.  Filled markers: pre-specified seeds (normal 0-2, inverted 0-2); open:
registered replication (normal 16000 seeds 3-5) and inversion extension (seeds 3-7).
Run:  python src/fv/fig_09_dose.py
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
from fv_style import (plt, DC, C, BODY_A, BODY_B, MARK_A, MARK_B, SYM, SYM_MARK,  # noqa: E402
                      panel_label, save)

OUT = (EINV.V2 + "/out")
LEDGER = (EINV.REPO + "/analysis/FINAL_LEDGER.json")
NUMJSON = (EINV.V2 + "/paper/fv/numbers.json")
H_IN = 2.6
FS_NOTE = 8          # every in-axes annotation at tick size (no mixed 7 / 7.5 pt text)

plt.rcParams["savefig.bbox"] = "standard"      # made at final size: the canvas is exactly DC wide


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tex_plain(s):
    """numbers.json text -> plain string for matplotlib (only simple decimals occur here)."""
    s = s.replace("$-$", "\u2212").replace("{,}", ",")
    assert "$" not in s and "\\" not in s, s
    return s


def seed_of(name):
    m = re.search(r"_s(\d+)$", name)
    assert m, name
    return int(m.group(1))


def main():
    num = load(NUMJSON)
    ds = load(os.path.join(OUT, "t1", "dose_stats.json"))["doses"]
    wb = load(os.path.join(OUT, "g1_within_body.json"))
    ext = load(os.path.join(OUT, "g1_ext.json"))
    led = load(LEDGER)

    def chk(macro, v, rel=5e-3):
        """plotted value agrees with the macro's recorded value (to its printed precision)."""
        ref = float(num[macro]["value"])
        assert abs(v - ref) <= rel * max(abs(ref), 1e-30), (macro, v, ref)

    # ---------------------------------------------------------------- denominator R_real
    R = led["denominators"]["R_real"]
    for d in ("2000", "8000", "16000", "16000_new"):          # every lambda in the file uses this R_real
        r = ds[d]["theta_sym"] / (ds[d]["lambda_sym_pct"] / 100.0)
        assert abs(r - R) < 1e-9 * R + 1e-12, (d, r, R)
    r_inv = ext["all_eight"]["theta_sym"] / (ext["all_eight"]["theta_sym_pct"] / 100.0)
    assert abs(r_inv - R) < 1e-6 * R, (r_inv, R)
    chk("nLimRrealFull", R, 1e-5)
    pct = lambda v: 100.0 * np.asarray(v, float) / R  # noqa: E731

    # ---------------------------------------------------------------- (a) dose axis
    doses = []
    # adapter counts: the caption prints the Table 2 / Table 5 macros (folder counts); the dose_stats
    # macros must agree with them, and both with the plotted arms
    for d, nm_ad, nm_cap, nm_lim in (
            ("2000", "nDoseTwoThousandAdapters", "nDoseTwoThousandAdaptersPerBody", "nDoseTwoThousandLimit"),
            ("8000", "nDoseEightThousandAdapters", "nDataEightKAdaptersPerArm", "nDoseEightThousandLimit"),
            ("16000", "nDoseSixteenAdapters", "nDataSixteenKAdaptersPerArm", None)):
        e = ds[d]
        arms = {}
        assert num[nm_ad]["text"] == num[nm_cap]["text"], (nm_ad, nm_cap)
        for x in ("A", "B"):
            names, vals = e[f"{x}_arms"], e[f"{x}_own"]
            assert len(names) == len(vals) == int(num[nm_cap]["text"])
            order = sorted(range(len(names)), key=lambda i: seed_of(names[i]))
            seeds = [seed_of(names[i]) for i in order]
            arms[x] = (seeds, pct([vals[i] for i in order]))
            # theta_x is the mean of the per-adapter own contrasts
            assert abs(np.mean(vals) - e[f"theta_{x}"]) < 1e-15
        ts = pct(e["theta_sym"]); se = pct(e["adapter_SE"])
        assert abs(0.5 * (e["theta_A"] + e["theta_B"]) - e["theta_sym"]) < 1e-15
        lim = e["lambda_U_plugin_pct"] if nm_lim else None
        if nm_lim:
            chk(nm_lim, lim)
        doses.append((d, arms, float(ts), float(se), lim))
    chk("nDoseTwoThousandLam", doses[0][2]); chk("nDoseEightThousandLam", doses[1][2])
    chk("nDoseSixteenLam", doses[2][2]); chk("nDoseSixteenSE", ds["16000"]["adapter_SE"])
    # the 16000 column holds seeds 0-5, the registered replication (16000_new, seeds 3-5) among them
    new = ds["16000_new"]
    assert set(new["A_arms"]) <= set(ds["16000"]["A_arms"]) and set(new["B_arms"]) <= set(ds["16000"]["B_arms"])
    assert sorted(seed_of(n) for n in new["A_arms"]) == [3, 4, 5]
    rep_ts, rep_se = float(pct(new["theta_sym"])), float(pct(new["adapter_SE"]))
    chk("nDoseRepLam", rep_ts)

    # ---------------------------------------------------------------- (b) normal vs inverted at 16000
    cond = {}
    for x in ("A", "B"):
        # normal: g1_within_body lists equal dose_stats 16000 in its arm order (seeds 0-5)
        nrm = np.asarray(wb["normal_16k_own"][x], float)
        assert np.allclose(nrm, ds["16000"][f"{x}_own"], rtol=0, atol=1e-18)
        nseeds = [seed_of(n) for n in ds["16000"][f"{x}_arms"]]
        o = np.argsort(nseeds)
        cond[(x, "normal")] = (list(np.asarray(nseeds)[o]), pct(nrm[o]))
        # inverted: seed tags from g1_ext per_adapter_{A,B}; values must match g1_within_body
        tag = ext[f"per_adapter_{x}"]
        items = sorted(((int(k[1:]), v) for k, v in tag.items()), key=lambda t: t[0])
        inv = np.asarray(wb["inverted_own"][x], float)
        assert np.allclose(sorted(inv), sorted(v for _, v in items), rtol=0, atol=1e-18)
        assert len(items) == int(num["nDoseInvAdapters"]["text"]) == int(num["nDataInvertedAdaptersPerArm"]["text"])
        cond[(x, "inverted")] = ([s for s, _ in items], pct([v for _, v in items]))
    # within-body differences (plotted as the slope of the joining lines) agree with the macros
    chk("nDoseWithinALam", cond[("A", "inverted")][1].mean() - cond[("A", "normal")][1].mean())
    chk("nDoseWithinBLam", cond[("B", "inverted")][1].mean() - cond[("B", "normal")][1].mean())
    inv_ts = float(pct(ext["all_eight"]["theta_sym"])); inv_se = float(pct(ext["all_eight"]["welch_se"]))
    chk("nDoseInvLam", inv_ts); chk("nDoseInvSE", ext["all_eight"]["welch_se"])
    assert abs(0.5 * (cond[("A", "inverted")][1].mean() + cond[("B", "inverted")][1].mean()) - inv_ts) < 1e-9
    nrm_ts, nrm_se = doses[2][2], doses[2][3]

    # printed text (macros)
    txt_norm = "+" + tex_plain(num["nDoseSixteenLam"]["text"])
    txt_inv = tex_plain(num["nDoseInvLam"]["text"])
    txt_rep = "seeds 3\u20135"

    # ================================================================ layout (inches)
    fig = plt.figure(figsize=(DC, H_IN))
    B_, T_ = 0.50, 0.17
    h = H_IN - B_ - T_
    xa, wa = 0.50, 2.60
    xb = xa + wa + 0.30
    wb_ = DC - xb - 0.62                 # room at the right for the printed theta_sym values
    ax_a = fig.add_axes([xa / DC, B_ / H_IN, wa / DC, h / H_IN])
    ax_b = fig.add_axes([xb / DC, B_ / H_IN, wb_ / DC, h / H_IN], sharey=ax_a)

    ink, grey = C["ink"], C["grey"]
    col = {"A": BODY_A, "B": BODY_B}
    mk = {"A": MARK_A, "B": MARK_B}
    msz = {"A": 3.4, "B": 3.1}
    STEP_IN = 0.042                      # horizontal spacing of adapters within a column, inches (seed order)

    def column(ax, xc, x, seeds, vals, open_from):
        n = len(vals)
        lo_, hi_ = ax.get_xlim()
        step = STEP_IN * (hi_ - lo_) / (ax.get_position().width * DC)
        dx = (np.arange(n) - (n - 1) / 2.0) * step
        for s, v, d in zip(seeds, vals, dx):
            filled = s < open_from
            ax.plot(xc + d, v, ls="none", marker=mk[x], ms=msz[x], mew=0.7, mec=col[x],
                    mfc=col[x] if filled else "white", zorder=3, clip_on=False)
        return dx

    def diamond(ax, xc, v, se, filled=True, ms=4.0):
        ax.errorbar(xc, v, yerr=se, fmt=SYM_MARK, ms=ms, mew=0.7, mec=SYM, mfc=SYM if filled else "white",
                    ecolor=SYM, elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4, clip_on=False)

    # ---------------------------------------------------------------- (a)
    ax = ax_a
    ax.set_xlim(-0.6, 2.88)
    ax.axhline(0.0, color=grey, ls=(0, (3, 2)), lw=0.6, zorder=1)
    # A and B columns kept apart: at 16000 (6 per body) an offset of -0.28 made A's seed-5 circle abut
    # B's seed-0 square
    OFF = {"A": -0.36, "B": 0.0}
    XS = 0.26
    open_from = {"2000": 99, "8000": 99, "16000": 3}
    for k, (d, arms, ts, se, lim) in enumerate(doses):
        for x in ("A", "B"):
            column(ax, k + OFF[x], x, arms[x][0], arms[x][1], open_from[d])
        diamond(ax, k + XS, ts, se)
        if lim is not None:
            ax.plot([k + XS - 0.07, k + XS + 0.07], [lim, lim], color=grey, lw=0.9, solid_capstyle="butt",
                    zorder=2)
    # the replication alone (seeds 3-5), small open diamond beside the pooled one
    diamond(ax, 2 + XS + 0.13, rep_ts, rep_se, filled=False, ms=3.4)
    ax.text(2 + XS + 0.13 + 0.08, rep_ts, txt_rep, ha="left", va="center", fontsize=FS_NOTE,
            color=ink)
    # label the limit tick once, at 8000
    # (to the tick's left: on its right the label crowded the 16000-step arm-A markers)
    ax.text(1 + XS - 0.10, doses[1][4], "99% limit", ha="right", va="center", fontsize=FS_NOTE, color=grey)
    ax.set_xticks([k + 0.5 * (OFF["A"] + XS) for k in range(3)])   # label centred under each group
    ax.set_xticklabels([d for d, *_ in doses])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlabel("Training steps", labelpad=2)
    ax.set_ylabel(r"Own-body contrast (% of $R_{\mathrm{real}}$)", labelpad=2)

    # ---------------------------------------------------------------- (b)
    ax = ax_b
    WG, BG = 1.3, 1.45                    # within-body and between-group spacing, axis units
    GX = {("A", "normal"): 0.0, ("A", "inverted"): WG, ("B", "normal"): WG + BG, ("B", "inverted"): 2 * WG + BG}
    SX = {"normal": 2 * WG + 2 * BG, "inverted": 2 * WG + 2 * BG + 1.3}
    ax.set_xlim(-0.5, SX["inverted"] + 0.3)
    ax.axhline(0.0, color=grey, ls=(0, (3, 2)), lw=0.6, zorder=1)
    for x in ("A", "B"):
        means = []
        for c in ("normal", "inverted"):
            seeds, vals = cond[(x, c)]
            xc = GX[(x, c)]
            dx = column(ax, xc, x, seeds, vals, 3)
            m = vals.mean()
            half = (dx.max() - dx.min()) / 2 + 0.08
            ax.plot([xc - half, xc + half], [m, m], color=col[x], lw=1.3, solid_capstyle="butt", zorder=2)
            means.append((xc + half, xc - half, m))
        # thin line joining the two body means: the within-body drop
        ax.plot([means[0][0], means[1][1]], [means[0][2], means[1][2]], color=col[x], lw=0.7, zorder=2)
    diamond(ax, SX["normal"], nrm_ts, nrm_se)
    diamond(ax, SX["inverted"], inv_ts, inv_se)
    # descriptive line joining the two theta_sym values (defined in the caption)
    ax.plot([SX["normal"], SX["inverted"]], [nrm_ts, inv_ts], color=SYM, lw=0.7, zorder=2)
    ax.text(SX["normal"] - 0.22, nrm_ts, txt_norm + "%", ha="right", va="center", fontsize=FS_NOTE, color=ink)
    ax.text(SX["inverted"] + 0.22, inv_ts, txt_inv + "%", ha="left", va="center", fontsize=FS_NOTE, color=ink,
            clip_on=False)
    ticks = [GX[("A", "normal")], GX[("A", "inverted")], GX[("B", "normal")], GX[("B", "inverted")],
             SX["normal"], SX["inverted"]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["normal", "inverted"] * 3)
    ax.tick_params(axis="x", length=0, pad=3)
    plt.setp(ax.get_yticklabels(), visible=False)
    for xm, lab, cc in ((0.5 * WG, "Body A", BODY_A), (1.5 * WG + BG, "Body B", BODY_B),
                        (0.5 * (SX["normal"] + SX["inverted"]), r"$\theta_{\mathrm{sym}}$", ink)):
        ax.text(xm, -0.105, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5,
                color=cc)

    # shared y range from the data
    allv = np.concatenate([v for (_, v) in cond.values()] + [a[x][1] for _, a, *_ in doses for x in "AB"])
    lo, hi = allv.min(), max(allv.max(), doses[1][4])
    # pad so that no marker sits on the bottom spine (lowest adapter is -0.70 %)
    ax_a.set_ylim(np.floor(lo * 10) / 10 - 0.06, np.ceil(hi * 10) / 10 + 0.02)
    # ticks cover the whole axis (the top adapter, +0.54 %, sat above the last labelled tick)
    ax_a.set_yticks(np.arange(np.ceil(lo * 5) / 5, ax_a.get_ylim()[1] + 1e-9, 0.2))
    ax_a.set_yticklabels([("%.1f" % t).replace("-", "\u2212") if abs(t) > 1e-9 else "0"
                          for t in ax_a.get_yticks()])

    for a, s in ((ax_a, "(a)"), (ax_b, "(b)")):
        dxl = 0.42 if a is ax_a else 0.12
        fig.text((a.get_position().x0 * DC - dxl) / DC, 1 - 0.02 / H_IN, s, ha="left", va="top", fontsize=9)

    out = save(fig, "fig09_dose")
    # report plotted values for spot checks
    print("R_real", R)
    for d, arms, ts, se, lim in doses:
        print(d, "A", np.round(arms["A"][1], 4), "seeds", arms["A"][0])
        print(d, "B", np.round(arms["B"][1], 4), "seeds", arms["B"][0])
        print(d, "theta_sym %.4f se %.4f lim %s" % (ts, se, lim))
    print("rep theta %.4f se %.4f" % (rep_ts, rep_se))
    for k, v in cond.items():
        print(k, v[0], np.round(v[1], 4), "mean %.4f" % v[1].mean())
    print("inv theta %.4f se %.4f" % (inv_ts, inv_se))
    print(out)


if __name__ == "__main__":
    main()
