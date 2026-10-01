"""Fig. S2 (fig:s02patterns): what every designed-pattern arm stored in the training crops.

For each designed arm of the transmission map (OUTLINE section 6, S06; Table S8 rows), the luminance of
training crop 0000 with the pattern minus the luminance of the same crop without it (none_a0/0000.png),
in the central 128 x 128-px window of the 1024-px crop, on one common grey scale (+-8 gray levels).

Rows (panels):
  (a) non-repeating octave bands band0..band5          out/t1/train_png/band{b}_a4
      periods from out/t1/band_fields.json -> edges_cycles_per_px
  (b) non-repeating fields: random field with K_B's spectrum (additive 1 and 4 gray, multiplicative 4 gray),
      then K_B^E2 itself at alpha 12, 48 and 3 (dithered)     out/t1/train_png/{gkadd_a1,gkadd_a4,gkmul_a4,
      kinj_a12,kinj_a48,kinjd_a3}; kind / amp / dither from out/t1/kfield_materialise.json
  (c) random tiles by period, then DiffusionShield    out/t1/train_png/per{P}_a4, dswm_a1;
      period and 8-px-grid membership from out/t1/periodic_fields.json + periodic2_fields.json
Arms whose names encode a design amplitude are labelled with that design constant; no result number is
printed in the figure.  The script prints, per arm, the RMS of the shown window and of the whole crop 0000
(RGB, as the materialisation files define stored_change_rms_gray) beside the 50-crop mean stored in the
materialisation files, as a check that the crops shown are the crops that were trained on.

Run:  python src/fv/figS_02_stored_patterns.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_style import plt, C, DC, save  # noqa: E402

T1 = (EINV.V2 + "/out/t1")
TRAIN = T1 + "/train_png"
MANIFEST = (EINV.V2 + "/out/fp/manifest.json")
CROP = 128                     # px shown (release policy: at most 128 px)
Y0 = X0 = 448                  # window origin: 56 x 8, so the window is centred on the 1024-px crop
GRAY = 8.0                     # display half-range, gray levels (design constant of the figure)

# ---- geometry (inches). Journal double-column width (fv_style.DC = 6.99 in), figure* unscaled; it also fits
# the supplement's wider IEEEtran text block.
FIG_W = DC
NCOL = 7
LM = 0.24                      # left margin: panel labels
GAP = 0.06                     # between crops
TILE = (FIG_W - LM - (NCOL - 1) * GAP) / NCOL
FS = 8                         # every label and tick label in the figure, pt
LAB_H = 0.33                   # below each crop: its label (up to two lines at 8 pt)
ROW_GAP = 0.06                 # between the label block and the next row
TOP = 0.03
FIG_H = TOP + 3 * TILE + 3 * LAB_H + 2 * ROW_GAP

plt.rcParams["savefig.bbox"] = "standard"      # keep the canvas exactly the text width


def load(name):
    with open(os.path.join(T1, name), encoding="utf-8") as f:
        return json.load(f)


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def lum(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def fmt_period(p):
    return f"{int(round(p))}"


def arms():
    """Rows of (folder, label, 50-crop stored RMS from the materialisation file or None)."""
    bf = load("band_fields.json")["edges_cycles_per_px"]
    bm = load("band_materialise.json")
    km = load("kfield_materialise.json")
    pf = dict(load("periodic_fields.json"))
    pf.update(load("periodic2_fields.json"))
    pm = dict(load("periodic_materialise.json"))
    pm.update({k: v for k, v in load("periodic2_materialise.json").items() if "stored_change_rms_gray" in v})
    tm = json.load(open(os.path.join(TRAIN, "materialise.json"), encoding="utf-8"))

    # (a) octave bands, finest first: period range 1/f_hi .. 1/f_lo
    row_a = []
    for b, (f_lo, f_hi) in enumerate(bf):
        lab = f"{fmt_period(1 / f_hi)}\u2013{fmt_period(1 / f_lo)} px"
        row_a.append((f"band{b}_a4", lab, bm[f"band{b}"]["stored_change_rms_gray"]))

    # (b) random field with K's spectrum, then K itself
    row_b = []
    for key in ("gkadd_a1", "gkadd_a4", "gkmul_a4", "kinj_a12", "kinj_a48", "kinjd_a3"):
        d = km[key]
        if d["kind"] == "gadd":
            lab = f"random field,\nadded, {d['amp']:g} gray"
        elif d["kind"] == "gmul":
            lab = f"random field,\nmultiplied, {key.rsplit('_a', 1)[1]} gray"   # amp there is a factor
        else:
            lab = "$\\hat{K}_{\\mathrm{B}}$, $\\alpha = " + f"{d['amp']:g}$" + (",\ndithered" if d["dither"] else "")
        row_b.append((key, lab, d["stored_change_rms_gray"]))

    # (c) tiles by period, then DiffusionShield
    row_c = []
    for p in sorted(int(k[3:]) for k in pf):
        d = pf[f"per{p}"]
        assert d["period_px"] == p
        on = d.get("on_latent_grid_8px", p % 8 == 0)
        assert on == (p % 8 == 0)
        row_c.append((f"per{p}_a4", f"{p}-px tile,\n{'on' if on else 'off'} 8-px grid",
                      pm[f"per{p}"]["stored_change_rms_gray"]))
    row_c.append(("dswm_a1", "DiffusionShield", tm["dswm_a1"]["stored_change_rms_gray"]))
    return [row_a, row_b, row_c]


def main():
    rows = arms()
    none = rgb(os.path.join(TRAIN, "none_a0", "0000.png"))
    ink = C["ink"]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    print(f"figure {FIG_W:.3f} x {FIG_H:.3f} in; tile {TILE:.3f} in")
    print(f"{'arm':10s} {'win lum RMS':>11s} {'crop RGB RMS':>12s} {'50-crop RMS':>11s} {'|win|>8 %':>9s}")
    first_ax = []
    for r, row in enumerate(rows):
        row_top = FIG_H - TOP - r * (TILE + LAB_H + ROW_GAP)
        for c, (arm, lab, rms50) in enumerate(row):
            a = rgb(os.path.join(TRAIN, arm, "0000.png"))
            d_rgb = a - none
            diff = lum(d_rgb)
            win = diff[Y0:Y0 + CROP, X0:X0 + CROP]
            print(f"{arm:10s} {np.sqrt(np.mean(win ** 2)):11.3f} {np.sqrt(np.mean(d_rgb ** 2)):12.3f} "
                  f"{rms50:11.3f} {100 * np.mean(np.abs(win) > GRAY):9.2f}")
            left = LM + c * (TILE + GAP)
            ax = fig.add_axes([left / FIG_W, (row_top - TILE) / FIG_H, TILE / FIG_W, TILE / FIG_H])
            ax.imshow(win, cmap="gray", vmin=-GRAY, vmax=GRAY, interpolation="nearest")
            tk = np.arange(0, CROP + 1, 8) - 0.5                    # 8-px latent grid, top and left edges
            ax.set_xticks(tk); ax.set_yticks(tk)
            ax.xaxis.tick_top()
            ax.tick_params(length=1.4, width=0.3, color=ink, labeltop=False, labelleft=False,
                           labelbottom=False, direction="out", pad=0)
            ax.set_xlim(-0.5, CROP - 0.5); ax.set_ylim(CROP - 0.5, -0.5)
            for s in ax.spines.values():
                s.set_visible(True); s.set_linewidth(0.4); s.set_color(ink)
            ax.set_xlabel(lab, fontsize=FS, labelpad=2, linespacing=1.0, va="top")
            if c == 0:
                first_ax.append((ax, row_top))

        if r == 0:
            # grey scale for every crop, in the free seventh cell of row (a)
            left = LM + 6 * (TILE + GAP)
            bw, bh = 0.075, 0.80 * TILE
            cax = fig.add_axes([(left + 0.06) / FIG_W, (row_top - 0.5 * TILE - 0.5 * bh) / FIG_H,
                                bw / FIG_W, bh / FIG_H])
            grad = np.linspace(GRAY, -GRAY, 256)[:, None]
            cax.imshow(grad, cmap="gray", vmin=-GRAY, vmax=GRAY, aspect="auto",
                       extent=(0, 1, -GRAY, GRAY), interpolation="bilinear")
            cax.set_xticks([])
            cax.yaxis.tick_right(); cax.yaxis.set_label_position("right")
            cax.set_yticks([-GRAY, 0, GRAY])
            cax.set_yticklabels([f"\u2212{GRAY:g}", "0", f"+{GRAY:g}"])
            cax.tick_params(axis="y", length=2.0, width=0.4, pad=1.5, labelsize=FS)
            for s in cax.spines.values():
                s.set_visible(True); s.set_linewidth(0.4); s.set_color(ink)
            cax.set_ylabel("Gray levels", fontsize=FS, labelpad=2)

    for (ax, row_top), s in zip(first_ax, ("(a)", "(b)", "(c)")):
        fig.text(0.02 / FIG_W, row_top / FIG_H, s, ha="left", va="top", fontsize=9)

    # layout check: no crop label may run into its neighbour or off the canvas
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    boxes = sorted((ax.xaxis.label.get_window_extent(rend) for ax in fig.axes if ax.get_xlabel()),
                   key=lambda b: (round(b.y1), b.x0))
    for b0, b1 in zip(boxes, boxes[1:]):
        if abs(b0.y1 - b1.y1) < 1 and b1.x0 - b0.x1 < 2 * fig.dpi / 72:
            raise SystemExit(f"crop labels closer than 2 pt: {b0} {b1}")
    W = fig.get_figwidth() * fig.dpi
    assert all(b.x0 >= 0 and b.x1 <= W for b in boxes), "label off the canvas"

    with open(MANIFEST, encoding="utf-8") as f:
        m = json.load(f)
    print("crop 0000 of body A's T split:", m["A"]["device"], m["A"]["T"][0])
    return save(fig, "figS_02_stored_patterns")


if __name__ == "__main__":
    print(main())
