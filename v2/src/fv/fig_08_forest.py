"""Fig. 8 (fig:forest): the standard-dose estimates across designs, and the pooled result.
Single column, 3.35 x 4.2 in, one forest panel (OUTLINE §4 "Fig. 8").

Each row is a paired (symmetric) estimate lambda_hat = theta_sym / R_real (%), on the pair's own R_real and
under the named fingerprint estimate, drawn with +-1 SE and a thin vertical tick at the one-sided 99 % upper
value: lambda_hat + t_{0.99, Welch df} * SE for the design rows, lambda_hat + z_{0.99} * SE for the pooled
rows.  A tick beyond the right edge of the axis is drawn as a short arrow at the edge.

Rows and sources (top to bottom):
  D200 (Dresden)
    1  primary, archive stack            out/m1_pooled.json per_design["D200 primary"]      (df from FINAL_LEDGER per-adapter)
    2  second training set (G5)          per_design["D200 second training set (G5)"]       (df: out/g5_alt_training.json alt.welch_df)
    3  second environment (G2)           per_design["D200 second environment (G2)"]        (df: out/g2_pooled_six.json welch_df)
    -  (no row: the dose series' 2000-step arm is seeds 0-2 of row 3, the same adapters and images; it is
        shown in fig:dose (a) and Table 5 only.  The script asserts that identity.)
    5  FLUX.1-dev                       per_design["D200 FLUX.1-dev"]                      (df: out/flux_seed_ext_summary.json)
    6  five captions (primary s0-s2)     out/t5/summary_derived.json diverse_bank           (SE = adapter_level_SE / R_real)
  iPhone 5c (VISION)
    7  natural image                     per_design["iPhone 5c (E2)"]                       (df: out/g6_p5c.json)
    8  flat field (hollow)               per_design["iPhone 5c (FLAT)"]
    9  five captions, natural image      out/g6_p5c_div.json estimators.E2.symmetric, R_real
   10  five captions, flat field (hollow) estimators.FLAT
  P20 (Daxing)
   11  natural image                     per_design["P20 (G4b)"]                            (df: out/g4b_p20.json)
  Pooled
   12  three pairs, random effects       primary.random                (black diamond)
   13  three pairs, fixed effect         primary.fixed                 (grey diamond)
   14  flat-field sensitivity            sensitivity_flat_field.fixed  (grey hollow diamond; quoted = fixed)
   15  D200 combined (rows 1,2,3,5)      secondary.d200_combination.fixed (grey diamond; I^2 printed)
Pooled encoding (editor's change to the OUTLINE table, for one meaning per style): black = random effects,
grey = fixed effect, hollow = flat-field estimate, as in the design rows.  OUTLINE row 4 (the 2000-step
dose arm) is dropped as a duplicate of row 3's first three adapters (row numbers above keep the OUTLINE's).
Rows 1, 7 and 11 (the inputs of row 12) carry a small bullet.

Every plotted value is read from the result files and cross-checked (per_design against its own source
file; the pooled z-ticks against upper99_pct; values against numbers.json).  Every printed number (adapter
counts, I^2) is a numbers.json `text`.  Run:
    python src/fv/fig_08_forest.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_style import plt, SC, C, SYM, SYM_MARK, save  # noqa: E402

from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

OUT = (EINV.V2 + "/out")
LEDGER = (EINV.REPO + "/analysis/FINAL_LEDGER.json")
NUMJSON = (EINV.V2 + "/paper/fv/numbers.json")
H_IN = 4.2
XMIN, XMAX = -0.25, 0.35          # x range, % of R_real (OUTLINE Fig. 8)
Q = 0.99                           # one-sided level of every tick (design constant)

AX = [0.505, 0.085, 0.475, 0.905]  # axes box in figure fractions (left, bottom, width, height)

plt.rcParams["savefig.bbox"] = "standard"   # made at final size: the canvas is exactly SC wide


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tex_plain(s):
    return s.replace("$-$", "\u2212").replace("{,}", ",")


def close(a, b, rel=1e-6, what=""):
    assert abs(a - b) <= rel * max(abs(a), abs(b), 1e-300), (what, a, b)


def welch_df(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    return (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))


def main():
    num = load(NUMJSON)
    txt = lambda m: tex_plain(num[m]["text"])       # noqa: E731
    m1 = load(OUT + "/m1_pooled.json")
    pdz = m1["per_design"]
    led = load(LEDGER)
    R_d200 = led["denominators"]["R_real"]

    # the caption prints one images-per-adapter count for all D200 rows and one for the iPhone 5c and P20 rows
    d200_gens = ("nDataPrimaryGensPerAdapter", "nDataSecondSetGensPerAdapter", "nDataSecondEnvGensPerAdapter",
                 "nDataFluxGensPerAdapter", "nDataPromptBankGensPerAdapter")
    assert len({txt(m) for m in d200_gens}) == 1, [(m, txt(m)) for m in d200_gens]
    phone_gens = ("nDataIPhoneGensPerAdapter", "nDataPromptsGensPerAdapter", "nDataPTwentyPairGensPerAdapter")
    assert len({txt(m) for m in phone_gens}) == 1, [(m, txt(m)) for m in phone_gens]

    rows = []   # dict(kind, label, lam, se, up, style, bullet)

    def design(label, lam, se, df, style, bullet=False, key=None):
        up = lam + stats.t.ppf(Q, df) * se
        rows.append(dict(kind="row", label=label, lam=lam, se=se, up=up, df=df, style=style,
                         bullet=bullet, key=key))

    def pooled(label, blk, style, key=None):
        lam, se = blk["lambda_pct"], blk["se_pct"]
        up = lam + stats.norm.ppf(Q) * se
        close(up, blk["upper99_pct"], 1e-9, "pooled upper " + label)
        rows.append(dict(kind="row", label=label, lam=lam, se=se, up=up, df=np.inf, style=style,
                         bullet=False, key=key))

    def group(label):
        rows.append(dict(kind="group", label=label))

    # ---------------- D200 (Dresden)
    group("Nikon D200 (Dresden)")
    p = led["primary"]
    d = pdz["D200 primary"]
    close(d["theta"], p["theta_sym"], 1e-4, "primary theta")
    df_p = welch_df(p["per_adapter_A"], p["per_adapter_B"])
    se_chk = 0.5 * np.sqrt(np.var(p["per_adapter_A"], ddof=1) / 12 + np.var(p["per_adapter_B"], ddof=1) / 12)
    close(se_chk, d["se"], 1e-3, "primary se")
    assert str(d["n"]) == txt("nDataPrimaryAdaptersPerArm")
    close(d["lambda_pct"], num["nLimPoolPrimaryLambda"]["value"], 1e-9, "primary lam")
    design(f"Primary, archive stack ({txt('nDataPrimaryAdaptersPerArm')})", d["lambda_pct"], d["se_pct"],
           df_p, "fill", bullet=True, key="D200 primary")

    g5 = load(OUT + "/g5_alt_training.json")["alt"]
    d = pdz["D200 second training set (G5)"]
    close(d["theta"], g5["theta_sym"], 1e-9, "G5 theta"); close(d["se"], g5["welch_se"], 1e-9, "G5 se")
    assert str(d["n"]) == txt("nGenGfiveAdapters")
    close(d["lambda_pct"], num["nGenGfiveLam"]["value"], 1e-9, "G5 lam")
    design(f"Second training set ({txt('nGenGfiveAdapters')})", d["lambda_pct"], d["se_pct"],
           g5["welch_df"], "fill", key="G5")

    g2 = load(OUT + "/g2_pooled_six.json")
    d = pdz["D200 second environment (G2)"]
    close(d["theta"], g2["theta_sym"], 1e-9, "G2 theta"); close(d["se"], g2["welch_se"], 1e-9, "G2 se")
    assert str(d["n"]) == txt("nGenGtwoAdapters")
    design(f"Local-stack retraining ({txt('nGenGtwoAdapters')})", d["lambda_pct"], d["se_pct"],
           g2["welch_df"], "fill", key="G2")
    close(rows[-1]["up"], g2["limit99_pct"], 1e-5, "G2 tick vs limit99_pct")

    # The dose series' 2000-step arm is not a row: its adapters are the first three of the row above (same
    # local-stack adapters and images), so a row would plot the same data twice.  Its point is in fig:dose (a)
    # and Table 5.
    ds = load(OUT + "/t1/dose_stats.json")["doses"]["2000"]
    assert np.allclose(ds["A_own"], g2["per_adapter_A"][:3]) and np.allclose(ds["B_own"], g2["per_adapter_B"][:3])

    fx = load(OUT + "/flux_seed_ext_summary.json")["symmetric"]
    d = pdz["D200 FLUX.1-dev"]
    close(d["theta"], fx["theta_sym"], 1e-9, "FLUX theta"); close(d["se"], fx["welch_se"], 1e-9, "FLUX se")
    assert str(d["n"]) == txt("nGenFluxAdapters")
    design(f"FLUX.1-dev ({txt('nGenFluxAdapters')})", d["lambda_pct"], d["se_pct"], fx["welch_df"], "fill",
           key="FLUX")
    close(rows[-1]["up"], fx["lambda_sym_U_pct"], 1e-5, "FLUX tick vs file")

    cb = load(OUT + "/t5/summary_derived.json")["diverse_bank"]
    R6 = cb["theta_sym"] / (cb["lambda_sym_pct"] / 100.0)
    close(R6, R_d200, 1e-6, "captions R_real")
    close(cb["lambda_sym_pct"], num["nGenCaptionsLam"]["value"], 1e-9, "captions lam")
    close(cb["adapter_level_SE"], num["nGenCaptionsSE"]["value"], 1e-9, "captions SE")
    design(f"Five captions ({txt('nGenCaptionsAdapters')})", cb["lambda_sym_pct"],
           100.0 * cb["adapter_level_SE"] / R_d200, cb["welch_df"], "fill", key="captions")

    # ---------------- iPhone 5c (VISION)
    group("iPhone 5c (VISION)")
    g6 = load(OUT + "/g6_p5c.json")["estimators"]
    for est, lab, sty, bul in (("E2", "Natural image", "fill", True), ("FLAT", "Flat field", "hollow", False)):
        d = pdz[f"iPhone 5c ({est})"]
        s = g6[est]["symmetric"]
        close(d["theta"], s["theta_sym"], 1e-9, "G6 theta " + est); close(d["se"], s["welch_se"], 1e-9, "G6 se")
        close(d["R"], g6[est]["R_real"], 1e-9, "G6 R " + est)
        assert str(d["n"]) == txt("nGenGsixAdapters")
        design(f"{lab} ({txt('nGenGsixAdapters')})", d["lambda_pct"], d["se_pct"], s["welch_df"], sty,
               bullet=bul, key="G6 " + est)
        close(rows[-1]["up"], s["lambda_sym_pct"], 1e-6, "G6 tick vs file " + est)
    pv = load(OUT + "/g6_p5c_div.json")["estimators"]
    for est, lab, sty in (("E2", "Five captions, natural", "fill"), ("FLAT", "Five captions, flat field", "hollow")):
        s, R = pv[est]["symmetric"], pv[est]["R_real"]
        assert str(pv[est]["n_adapters_per_arm"]) == txt("nPoneAdapters")
        design(f"{lab} ({txt('nPoneAdapters')})", 100.0 * s["theta_sym"] / R, 100.0 * s["welch_se"] / R,
               s["welch_df"], sty, key="P1 " + est)
        close(rows[-1]["up"], s["lambda_sym_pct"], 1e-6, "P1 tick vs file " + est)
    close(rows[-1]["lam"], num["nPoneFlatLam"]["value"], 0.02, "P1 flat lam")

    # ---------------- P20 (Daxing)
    group("Huawei P20 (Daxing)")
    gb = load(OUT + "/g4b_p20.json")["symmetric"]
    d = pdz["P20 (G4b)"]
    close(d["theta"], gb["theta_sym"], 1e-9, "P20 theta"); close(d["se"], gb["welch_se"], 1e-9, "P20 se")
    assert str(d["n"]) == txt("nGenGfourbAdapters")
    design(f"Natural image ({txt('nGenGfourbAdapters')})", d["lambda_pct"], d["se_pct"], gb["welch_df"], "fill",
           bullet=True, key="P20")
    close(rows[-1]["up"], gb["lambda_sym_pct"], 1e-6, "P20 tick vs file")

    # ---------------- pooled
    group("Pooled")
    assert m1["primary"]["designs"] == ["D200 primary", "iPhone 5c (E2)", "P20 (G4b)"]
    assert m1["sensitivity_flat_field"]["designs"] == ["D200 primary", "iPhone 5c (FLAT)", "P20 (G4b)"]
    assert m1["sensitivity_flat_field"]["quoted"] == "fixed"
    assert m1["secondary"]["d200_combination"]["quoted"] == "fixed"
    k = txt("nLimPoolPairs"); assert str(m1["primary"]["k"]) == k
    close(m1["primary"]["random"]["lambda_pct"], num["nLimPoolLambda"]["value"], 1e-9, "pool lam")
    pooled(f"{k} pairs, random effects", m1["primary"]["random"], "pool")
    pooled(f"{k} pairs, fixed effect", m1["primary"]["fixed"], "pool_grey")
    # one encoding for the pooled rows: black = random effects (pre-specified), grey = fixed effect,
    # hollow = flat-field estimate (as in the design rows)
    pooled(f"{k} pairs, flat field, fixed", m1["sensitivity_flat_field"]["fixed"], "pool_grey_hollow")
    dc = m1["secondary"]["d200_combination"]
    i2 = txt("nLimPoolDtwoHundredItwo")
    assert f"{100 * dc['I2']:.0f}" == i2
    assert dc["designs"] == ["D200 primary", "D200 second environment (G2)", "D200 second training set (G5)",
                             "D200 FLUX.1-dev"]      # rows 1, 3, 2, 5 (named in the caption)
    pooled(f"D200 designs, fixed ($I^2={i2}\\%$)", dc["fixed"], "pool_grey")

    # ---------------- layout: y positions (top = 0, going down), extra gap before each group
    y, ys = 0.0, []
    for i, r in enumerate(rows):
        if r["kind"] == "group" and i > 0:
            y += 0.55
        ys.append(y)
        y += 1.0
    ys = np.array(ys)

    fig = plt.figure(figsize=(SC, H_IN))
    ax = fig.add_axes(AX)
    ink, grey = C["ink"], C["grey"]
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(ys[-1] + 0.7, -0.6)

    ax.axvline(0.0, color=grey, lw=0.6, ls=(0, (3, 2)), zorder=1)

    fs = 8.0                                    # row labels at the tick size (one text size in the panel)
    trans_lab = ax.get_yaxis_transform()        # x in axes units, y in data units
    ax_w_in = AX[2] * SC
    x_group = -(AX[0] * SC - 0.04) / ax_w_in    # left edge of the figure (+0.04 in), in axes units
    x_row = x_group + 0.11 / ax_w_in            # rows indented 0.11 in
    x_bul = x_group + 0.055 / ax_w_in           # pooled-input bullet, centred in the indent
    th = 0.25                                   # half-height of the 99 % tick, in row units

    for r, yy in zip(rows, ys):
        if r["kind"] == "group":
            ax.text(x_group, yy, r["label"], transform=trans_lab, ha="left", va="center", fontsize=fs,
                    color=ink, style="italic", clip_on=False)
            continue
        st = r["style"]
        col = grey if st in ("pool_grey", "pool_grey_hollow") else SYM
        hollow = st in ("hollow", "pool_hollow", "pool_grey_hollow")
        lam, se, up = r["lam"], r["se"], r["up"]
        if st.startswith("pool"):
            # pooled rows: the diamond itself spans lambda_hat +- 1 SE (meta-analysis convention)
            hd = 0.27
            ax.fill([lam - se, lam, lam + se, lam], [yy, yy - hd, yy, yy + hd], closed=True,
                    fc="white" if hollow else col, ec=col, lw=0.8, joinstyle="miter", zorder=4)
        else:
            ax.plot([lam - se, lam + se], [yy, yy], color=col, lw=0.9, solid_capstyle="butt", zorder=3)
            ax.plot([lam], [yy], ls="none", marker=SYM_MARK, ms=3.8, mec=col, mew=0.8,
                    mfc="white" if hollow else col, zorder=4)
        if up <= XMAX:
            ax.plot([up, up], [yy - th, yy + th], color=col, lw=0.6, zorder=3)   # thin: secondary to the estimate
        else:   # tick beyond the axis: short arrow at the edge
            ax.annotate("", xy=(XMAX + 0.004, yy), xytext=(XMAX - 0.035, yy), xycoords="data",
                        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", lw=0.7,
                                        color=col, shrinkA=0, shrinkB=0), annotation_clip=False, zorder=3)
        ax.text(x_row, yy, r["label"], transform=trans_lab, ha="left", va="center", fontsize=fs, color=ink,
                clip_on=False)
        if r["bullet"]:
            ax.text(x_bul, yy, "\u2022", transform=trans_lab, ha="center", va="center", fontsize=fs,
                    color=ink, clip_on=False)

    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(FixedLocator([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _p: "0" if abs(x) < 1e-12 else f"{x:.1f}".replace("-", "\u2212")))
    ax.set_xlabel(r"$\hat{\lambda}=\theta_{\mathrm{sym}}/R_{\mathrm{real}}$ (%)", labelpad=2, fontsize=fs)

    save(fig, "fig08_forest")

    for r in rows:
        if r["kind"] == "row":
            print(f"{r['label']:<42s} lam={r['lam']:+.5f}  se={r['se']:.5f}  df={r['df']:.2f}  up99={r['up']:.4f}"
                  + ("  [beyond axis]" if r["up"] > XMAX else ""))


if __name__ == "__main__":
    main()
