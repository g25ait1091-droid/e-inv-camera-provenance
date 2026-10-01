"""Fig. S1 (fig:s01inputs): the personalization photographs of every pair.

For each of the three paired designs (Nikon D200, Dresden; Apple iPhone 5c, VISION; Huawei P20, Daxing)
and each body, one row: the native frame of the first training photograph with its 1024-px centre-crop box,
then every tenth crop of the 50-crop training split T (indices 0, 10, 20, 30, 40, in split order).

Sources (read, never typed):
  splits and file names   out/fp/manifest.json (A.T, B.T), out/fp_5c/manifest.json (roles.*.T),
                          out/fp_p20b/manifest.json (roles.*.T)
  training crops          D200 A: out/t1/train_png/colab_a0 (archive-stack training set);
                          D200 B: $EINV_MYDRIVE/inv_channel/E_INV_P0_v3/train_png/B_raw (archive-stack training set;
                                  fallback out/t1/train_png/noneB_a0, the local decode of the same photographs);
                          iPhone 5c: out/t1/train_png/p5c{A,B}_a0; P20: out/t1/train_png/p20b{A,B}_a0
  native frames           $EINV_MYDRIVE/forensic_datasets/{dresden,vision}, $EINV_DAXING (paths from the manifests)
  printed device IDs      numbers.json "text" of nDataDTwoHundredBody{A,B}, nDataIPhoneBody{A,B},
                          nDataPTwentyPairBody{A,B}
The script checks that every shown crop i is the centred 1024-px window of native photograph i of T (the local
decodes match exactly; the archive PNGs of the D200 differ by the JPEG-decoder difference only, mean < 0.5 gray)
and prints the check.
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
from fv_style import plt, C, DC, BODY_A, BODY_B, image_panel, save  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUT = (EINV.V2 + "/out")
TP = OUT + "/t1/train_png"
NUMS = (EINV.V2 + "/paper/fv/numbers.json")
CROP = 1024                                 # design constant: centred 1024 x 1024 training crop
SHOW = (0, 10, 20, 30, 40)                  # every tenth crop of the 50-crop T split
DPI_IMG = 300                               # embedded image resolution at printed size

# ---- geometry (inches). Journal double-column width (fv_style.DC = 6.99 in), placed unscaled; it fits the
# supplement's 7.14-in text width.
FIG_W = DC
LAB_W = 0.40                                # left margin for the rotated row labels
GAP = 0.04                                  # between tiles
GAP_N = 0.10                                # between native frame and crops
GAP_PAIR = 0.21                             # between pairs (holds the panel label)
HEAD = 0.17                                 # column-header line above the first row
EDGE = 0.02                                 # right and bottom margin so the outer tile frames are not clipped
ASPECT_D200 = 3872 / 2592                   # widest native frame sets the native column
TILE = (FIG_W - LAB_W - GAP_N - 4 * GAP - EDGE) / (5 + ASPECT_D200)
NAT_W = ASPECT_D200 * TILE

nums = json.load(open(NUMS, encoding="utf-8"))
def txt(k): return nums[k]["text"]


def manifest_rows():
    m = json.load(open(OUT + "/fp/manifest.json"))
    m5 = json.load(open(OUT + "/fp_5c/manifest.json"))
    m20 = json.load(open(OUT + "/fp_p20b/manifest.json"))
    vis = (EINV.DATASETS + "/vision/dataset/{}/images/nat/")
    dax = (EINV.DAXING + "/image/1101-1104/{}/90/")
    b_arch = (EINV.MYDRIVE + "/inv_channel/E_INV_P0_v3/train_png/B_raw")
    b_dir = b_arch if os.path.isdir(b_arch) else TP + "/noneB_a0"
    rows = [
        # (panel, body, model label, device label, crop dir, native paths, colour)
        ("a", "A", "D200", "Nikon_D200_" + txt("nDataDTwoHundredBodyA"), TP + "/colab_a0", m["A"]["T"], BODY_A),
        ("a", "B", "D200", "Nikon_D200_" + txt("nDataDTwoHundredBodyB"), b_dir, m["B"]["T"], BODY_B),
        ("b", "A", "iPhone 5c", txt("nDataIPhoneBodyA"), TP + "/p5cA_a0",
         [vis.format(m5["roles"]["A"]["device"]) + f for f in m5["roles"]["A"]["T"]], BODY_A),
        ("b", "B", "iPhone 5c", txt("nDataIPhoneBodyB"), TP + "/p5cB_a0",
         [vis.format(m5["roles"]["B"]["device"]) + f for f in m5["roles"]["B"]["T"]], BODY_B),
        ("c", "A", "P20", txt("nDataPTwentyPairBodyA"), TP + "/p20bA_a0",
         [dax.format(m20["roles"]["A"]["device"]) + f for f in m20["roles"]["A"]["T"]], BODY_A),
        ("c", "B", "P20", txt("nDataPTwentyPairBodyB"), TP + "/p20bB_a0",
         [dax.format(m20["roles"]["B"]["device"]) + f for f in m20["roles"]["B"]["T"]], BODY_B),
    ]
    # the device labels must name the manifest's devices
    assert m["A"]["device"].endswith(txt("nDataDTwoHundredBodyA")) and m["B"]["device"].endswith(txt("nDataDTwoHundredBodyB"))
    assert m5["roles"]["A"]["device"].startswith(txt("nDataIPhoneBodyA")) and m5["roles"]["B"]["device"].startswith(txt("nDataIPhoneBodyB"))
    assert m20["roles"]["A"]["device"] == txt("nDataPTwentyPairBodyA") and m20["roles"]["B"]["device"] == txt("nDataPTwentyPairBodyB")
    return rows, b_dir


def shrink(im, width_px):
    h = int(round(im.height * width_px / im.width))
    return np.asarray(im.resize((width_px, h), Image.LANCZOS))


def centre_box(W, H):
    x0, y0 = (W - CROP) // 2, (H - CROP) // 2
    return x0, y0


def main():
    rows, b_dir = manifest_rows()
    n_rows = len(rows)
    fig_h = HEAD + n_rows * TILE + 3 * GAP + 2 * GAP_PAIR + EDGE
    fig = plt.figure(figsize=(FIG_W, fig_h))
    fx = lambda x: x / FIG_W
    fy = lambda y: y / fig_h
    tile_px = int(round(TILE * DPI_IMG))
    nat_px = int(round(NAT_W * DPI_IMG))
    provenance = []

    y_top = fig_h - HEAD                    # top edge of the current row, inches from the bottom
    prev_panel = None
    for r, (panel, body, model, dev, cdir, natives, col) in enumerate(rows):
        if prev_panel is not None:
            y_top -= GAP_PAIR if panel != prev_panel else GAP
        y0 = y_top - TILE

        # native frame of crop 0 with its crop box
        nat_path = natives[SHOW[0]]
        with Image.open(nat_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            x_c, y_c = centre_box(W, H)
            crop_native = np.asarray(im.crop((x_c, y_c, x_c + CROP, y_c + CROP)), np.float32)
            w_in = TILE * W / H             # displayed width at row height TILE
            small = shrink(im, int(round(w_in * DPI_IMG)))
        x_nat = LAB_W + (NAT_W - w_in) / 2
        ax = fig.add_axes([fx(x_nat), fy(y0), fx(w_in), fy(TILE)])
        image_panel(ax, small)
        ax.images[0].set_interpolation("none")
        s = small.shape[1] / W
        ax.add_patch(Rectangle((x_c * s - 0.5, y_c * s - 0.5), CROP * s, CROP * s, fill=False,
                               edgecolor=col, lw=1.0))
        ax.set_xlim(-0.5, small.shape[1] - 0.5); ax.set_ylim(small.shape[0] - 0.5, -0.5)

        # every tenth training crop
        for k, i in enumerate(SHOW):
            fp = os.path.join(cdir, f"{i:04d}.png")
            with Image.open(fp) as im:
                im = im.convert("RGB")
                arr = np.asarray(im, np.float32)
                small_c = shrink(im, tile_px)
            # every shown crop must be the centred window of native photograph i of T
            if k == 0:
                win = crop_native
                Wi, Hi, xi, yi = W, H, x_c, y_c
            else:
                with Image.open(natives[i]) as nim:
                    nim = nim.convert("RGB")
                    Wi, Hi = nim.size
                    xi, yi = centre_box(Wi, Hi)
                    win = np.asarray(nim.crop((xi, yi, xi + CROP, yi + CROP)), np.float32)
            mad = float(np.abs(arr - win).mean())
            assert mad < 1.0, (model, body, i, mad)   # decoder difference only (D200 archive decode)
            provenance.append((model, body, i, os.path.basename(natives[i]), Wi, Hi, xi, yi, mad))
            x = LAB_W + NAT_W + GAP_N + k * (TILE + GAP)
            ax = fig.add_axes([fx(x), fy(y0), fx(TILE), fy(TILE)])
            image_panel(ax, small_c)
            ax.images[0].set_interpolation("none")

        # row label: model and body, device ID
        fig.text(fx(LAB_W - 0.22), fy(y0 + TILE / 2), f"{model} {body}", rotation=90,
                 ha="center", va="center", fontsize=8, color=C["ink"])
        fig.text(fx(LAB_W - 0.09), fy(y0 + TILE / 2), dev, rotation=90,
                 ha="center", va="center", fontsize=8, color=C["grey"])

        # panel label in the line above the first row of each pair
        if panel != prev_panel:
            fig.text(0.0, fy(y_top + 0.03), f"({panel})", ha="left", va="bottom", fontsize=8, color=C["ink"])
        prev_panel = panel
        y_top = y0

    # column headers (once, above the first row)
    yh = fy(fig_h - HEAD + 0.03)
    fig.text(fx(LAB_W + NAT_W / 2), yh, "native frame of crop 0", ha="center", va="bottom", fontsize=8,
             color=C["ink"])
    for k, i in enumerate(SHOW):
        x = LAB_W + NAT_W + GAP_N + k * (TILE + GAP) + TILE / 2
        fig.text(fx(x), yh, f"crop {i}", ha="center", va="bottom", fontsize=8, color=C["ink"])

    plt.rcParams["savefig.bbox"] = None     # keep the figure exactly at the text width
    out = save(fig, "figS_01_inputs")
    print("saved", out, f"{FIG_W:.3f} x {fig_h:.3f} in; tile {TILE:.3f} in, {tile_px} px")
    print("B crops from", b_dir)
    for p in provenance:
        print("  %-9s %s crop %2d  %-26s native %dx%d  box origin (%d,%d)  |crop - native window| mean %.3f gray" % p)


if __name__ == "__main__":
    main()
