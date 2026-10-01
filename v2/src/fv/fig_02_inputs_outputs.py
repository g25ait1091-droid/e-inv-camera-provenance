"""Fig. 2 (fig:io) -- what goes in and what comes out.

Four rows of six 1.05-in tiles on the 6.99-in text width (OUTLINE.md section 4, "Fig. 2"):
  (a) inputs: training crop of D200 A, its native frame with the 1024-px crop box, training crops of
      D200 B, P20 1104 and iPhone 5c D05, and the first D05 flat field (centre 1024 px);
  (b) residuals and fingerprints, one 128-px centre window: W(Y) and Y*K_A of held-out A photograph 0,
      K_A^E2, K_B^E2, and two patterns as stored in training crops (inverted fingerprint, 32-px tile);
  (c) generations at j = 0 (seed 770000): base, D200-A and D200-B adapters (street caption), D200-A adapter
      (single caption), 16000-step A adapter, 16000-step inverted A adapter;
  (d) the DINOv2-closest 16000-step generation beside its nearest training crop, then W(Z) of (c1), (c2),
      (c5), (c6) in the same window and scaling rule as row (b).

Nothing is typed in: file names and device IDs come from the split manifests, the closest pair from
out/t1/dino_memorization.json (top16k[0]) and its printed cosine from numbers.json (nMemMaxCos). The
residual W is the study's own wavelet residual, taken verbatim from src/fingerprints.py (the function is
extracted with ast, so that module's top-level split construction is not run).

Tile ids: the rows carry the letters (a)-(d) at the left and the columns the numbers 1-6 above row (a), so
tile (c4) is row (c), column 4; the caption and the text cite tiles by these ids.

Grey maps: residual and fingerprint tiles are shown with a symmetric grey map, each clipped at +-3 SD of
the tile itself (OUTLINE 4.0); the two stored patterns use fixed ranges of +-3 and +-8 gray levels.

Writes paper/fv/figs/fig02_inputs_outputs.pdf (+ .png preview) and prints the provenance of every tile.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import ast, io, json, os, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_style import plt, DC, C, BODY_A, save  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = EINV.V2
OUT, T1 = f"{ROOT}/out", f"{ROOT}/out/t1"
TRAIN, GENS, GENS5 = f"{T1}/train_png", f"{T1}/gens", f"{OUT}/t5/gens"
GENS_V1 = (EINV.DATA + "/gens")
B_RAW = (EINV.MYDRIVE + "/inv_channel/E_INV_P0_v3/train_png/B_raw")
B_FALLBACK = f"{TRAIN}/noneB_a0"
VISION = (EINV.DATASETS + "/vision/dataset")
NUMBERS = f"{ROOT}/paper/fv/numbers.json"

TILE_IN, DPI_IMG = 1.05, 600             # printed tile size and target image resolution
PX = int(round(TILE_IN * DPI_IMG))       # 630 px per photograph tile
WIN = 128                                # residual / fingerprint window (release policy: <= 128 px)
MEAS = 1024
CLIP_SD = 3.0                            # residual / fingerprint tiles: +-3 SD of the tile
INV_SCALE, TILE_SCALE = 3.0, 8.0          # stored patterns: fixed +-3 and +-8 gray levels


# ---------- the study's residual, extracted verbatim from src/fingerprints.py ----------
def _load_residual():
    src = open(f"{ROOT}/src/fingerprints.py", encoding="utf-8").read()
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Tuple) and "SIG0" in [e.id for e in t.elts]
                                                  for t in node.targets):
            keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in ("_conv2", "wavelet_residual"):
            keep.append(node)
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "fingerprints.py", "exec"), ns)
    assert ns["MEAS"] == MEAS
    return ns["wavelet_residual"]


wavelet_residual = _load_residual()


def lum(rgb):
    rgb = np.asarray(rgb, np.float32)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def rgb_of(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def centre_crop(a, n=MEAS):
    h, w = a.shape[:2]
    return a[(h - n) // 2:(h - n) // 2 + n, (w - n) // 2:(w - n) // 2 + n]


def window(a):
    c = (a.shape[0] - WIN) // 2
    return a[c:c + WIN, c:c + WIN]


def shrink(a, width_px):
    im = Image.fromarray(np.asarray(a, np.uint8))
    h = int(round(im.height * width_px / im.width))
    return np.asarray(im.resize((width_px, h), Image.LANCZOS))


def sd_display(r):
    """Symmetric grey map of a signed field: zero mid-grey, clipped at +-3 SD of this tile."""
    r = np.asarray(r, np.float64)
    r = r - r.mean()
    return np.clip(r / (CLIP_SD * r.std()), -1, 1)


def grey_tile(r):
    """128-px signed field in [-1, 1] -> nearest-neighbour upsampled so the PDF keeps the pixels sharp."""
    k = int(np.ceil(PX / WIN))
    return np.kron(r, np.ones((k, k), np.float32))


# ---------- provenance ----------
nums = json.load(open(NUMBERS, encoding="utf-8"))
cos_text = nums["nMemMaxCos"]["text"]
man = json.load(open(f"{OUT}/fp/manifest.json"))
man5 = json.load(open(f"{OUT}/fp_5c/manifest.json"))
man20 = json.load(open(f"{OUT}/fp_p20b/manifest.json"))
dino = json.load(open(f"{T1}/dino_memorization.json"))
top = max(dino["top16k"], key=lambda r: r["cos"])
assert top is dino["top16k"][0]
assert f"{top['cos']:.3f}" == cos_text, (top["cos"], cos_text)

devA, devB = man["A"]["device"], man["B"]["device"]            # Nikon_D200_1, Nikon_D200_0
phone = man5["roles"]["A"]["device"]                            # D05_Apple_iPhone5c
phone_id = phone.split("_")[0]
assert phone_id == nums["nDataIPhoneBodyA"]["text"]
p20 = str(man20["roles"]["A"]["device"])                        # 1104
assert p20 == nums["nDataPTwentyPairBodyA"]["text"]
assert devA.endswith("_" + nums["nDataDTwoHundredBodyA"]["text"])

b_path = f"{B_RAW}/0000.png" if os.path.exists(f"{B_RAW}/0000.png") else f"{B_FALLBACK}/0000.png"
flat_dir = f"{VISION}/{phone}/images/flat"
flat_path = f"{flat_dir}/{sorted(os.listdir(flat_dir))[0]}"
close_gen, close_crop = top["generation"].replace("\\", "/"), top["training_crop"].replace("\\", "/")
close_j = int(os.path.splitext(os.path.basename(close_gen))[0])
close_k = int(os.path.splitext(os.path.basename(close_crop))[0])

prov = {
    "a1": (f"{TRAIN}/colab_a0/0000.png", f"Dresden {devA}, T[0] = {os.path.basename(man['A']['T'][0])}"),
    "a2": (man["A"]["T"][0], "native frame of a1"),
    "a3": (b_path, f"Dresden {devB}, T[0] = {os.path.basename(man['B']['T'][0])}"),
    "a4": (f"{TRAIN}/p20bA_a0/0000.png", f"Daxing {p20}, T[0] = {man20['roles']['A']['T'][0]}"),
    "a5": (f"{TRAIN}/p5cA_a0/0000.png", f"VISION {phone}, T[0] = {man5['roles']['A']['T'][0]}"),
    "a6": (flat_path, f"VISION {phone}, first flat field by name"),
    "b1": (f"{OUT}/fp/H_A.npz", f"W[0], H[0] = {os.path.basename(man['A']['H'][0])}"),
    "b2": (f"{OUT}/fp/H_A.npz + K_A_E2.npy", "Y[0] * K_A_E2"),
    "b3": (f"{OUT}/fp/K_A_E2.npy", ""), "b4": (f"{OUT}/fp/K_B_E2.npy", ""),
    "b5": (f"{TRAIN}/invA_a0/0000.png - none_a0/0000.png", "luminance"),
    "b6": (f"{TRAIN}/per32_a4/0000.png - none_a0/0000.png", "luminance"),
    "c1": (f"{GENS5}/base/00000.png", "five-caption bank, j=0 (street)"),
    "c2": (f"{GENS5}/A_raw_s0_r16/00000.png", "five-caption bank, j=0 (street)"),
    "c3": (f"{GENS5}/B_raw_s0_r16/00000.png", "five-caption bank, j=0 (street)"),
    "c4": (f"{GENS_V1}/A_raw_s0_r16/00000.png", "single caption, j=0"),
    "c5": (f"{GENS}/dose16k_A_s0/00000.png", "single caption, j=0"),
    "c6": (f"{GENS}/inv16k_A_s0/00000.png", "single caption, j=0"),
    "d1": (close_gen, f"DINOv2 top16k[0], cos {top['cos']:.4f}"),
    "d2": (close_crop, f"T[{close_k}] = {os.path.basename(man['A']['T'][close_k])}"),
}
for k, (p, note) in prov.items():
    print(f"{k}: {p}  [{note}]")
for k, (p, _) in prov.items():
    if k[0] in "acd":
        assert os.path.exists(p), p

# ---------- data checks ----------
H = np.load(f"{OUT}/fp/H_A.npz")
Y0, W0 = H["Y"][0].astype(np.float32), H["W"][0].astype(np.float32)
KA, KB = np.load(f"{OUT}/fp/K_A_E2.npy"), np.load(f"{OUT}/fp/K_B_E2.npy")
Y0_chk = lum(centre_crop(rgb_of(man["A"]["H"][0])))
print(f"check H_A Y[0] vs centre crop of {os.path.basename(man['A']['H'][0])}: max |diff| = "
      f"{np.abs(Y0_chk - Y0).max():.3g}")
print(f"check H_A W[0] vs wavelet_residual(Y[0]): max |diff| = {np.abs(wavelet_residual(Y0) - W0).max():.3g}")
a1 = rgb_of(prov["a1"][0]).astype(np.float32)
nat = rgb_of(man["A"]["T"][0])
print(f"check a1 vs centre crop of native frame: mean |diff| = {np.abs(a1 - centre_crop(nat)).mean():.3f} gray")
d2 = rgb_of(close_crop).astype(np.float32)
print(f"check d2 vs centre crop of T[{close_k}]: mean |diff| = "
      f"{np.abs(d2 - centre_crop(rgb_of(man['A']['T'][close_k]))).mean():.3f} gray")

# ---------- tiles ----------
row_a = [
    shrink(rgb_of(prov["a1"][0]), PX),
    shrink(nat, PX),
    shrink(rgb_of(prov["a3"][0]), PX),
    shrink(rgb_of(prov["a4"][0]), PX),
    shrink(rgb_of(prov["a5"][0]), PX),
    shrink(centre_crop(rgb_of(flat_path)), PX),
]
none0 = lum(rgb_of(f"{TRAIN}/none_a0/0000.png"))
inv_stored = lum(rgb_of(f"{TRAIN}/invA_a0/0000.png")) - none0
tile_stored = lum(rgb_of(f"{TRAIN}/per32_a4/0000.png")) - none0
print(f"stored inversion, 128-px window: RMS {window(inv_stored).std():.2f} gray, "
      f"{np.mean(np.abs(window(inv_stored)) > INV_SCALE):.3f} of pixels beyond +-{INV_SCALE:.0f}; corr with "
      f"-Y*K_A_E2 {np.corrcoef(window(inv_stored).ravel(), -window(none0 * KA).ravel())[0, 1]:.3f}")
print(f"stored 32-px tile, 128-px window: RMS {window(tile_stored).std():.2f} gray, "
      f"{np.mean(np.abs(window(tile_stored)) > TILE_SCALE):.3f} of pixels beyond +-{TILE_SCALE:.0f}")
row_b = [
    sd_display(window(W0)),
    sd_display(window(Y0 * KA)),
    sd_display(window(KA)),
    sd_display(window(KB)),
    np.clip(window(inv_stored) / INV_SCALE, -1, 1),
    np.clip(window(tile_stored) / TILE_SCALE, -1, 1),
]
gen_paths = [prov[f"c{i}"][0] for i in range(1, 7)]
gens = [rgb_of(p) for p in gen_paths]
assert all(g.shape[:2] == (MEAS, MEAS) for g in gens)
row_c = [shrink(g, PX) for g in gens]
RES_OF = [0, 1, 4, 5]                      # W(Z) of tiles c1, c2, c5, c6
row_d_img = [shrink(rgb_of(close_gen), PX), shrink(rgb_of(close_crop), PX)]
row_d_res = [sd_display(window(wavelet_residual(lum(gens[i])))) for i in RES_OF]
for i, r in zip(RES_OF, row_d_res):
    print(f"W(Z) of c{i + 1}: {np.mean(np.abs(r) >= 1):.3f} of window pixels at the +-3 SD clip")

# ---------- labels (two lines beneath each tile) ----------
res_name = {0: "base model", 1: "D200 A adapter", 4: "D200 A, 16000 steps", 5: "D200 A, inverted"}
L = {
    "a": ["D200 A\ntraining crop", "D200 A\nnative frame", "D200 B\ntraining crop",
          f"P20 {p20} (A)\ntraining crop", f"iPhone 5c {phone_id} (A)\ntraining crop",
          f"iPhone 5c {phone_id} (A)\nflat field"],
    "b": [r"$W(Y)$" + "\nheld-out photograph", r"$Y\odot\hat{K}_A$" + "\nsame photograph",
          r"$\hat{K}_A$" + "\nbody A", r"$\hat{K}_B$" + "\nbody B",
          "inverted fingerprint\nstored in crop", "32-px tile\nstored in crop"],
    "c": ["base model\nstreet caption", "D200 A adapter\nstreet caption", "D200 B adapter\nstreet caption",
          "D200 A adapter\nsingle caption", "D200 A, 16000 steps\nsingle caption",
          "D200 A, 16000 steps\nfingerprint inverted"],
    "d": [f"D200 A, 16000 steps\nimage $j={close_j}$", f"nearest training crop\ncos {cos_text}, not a copy"]
         + [f"$W(Z)$ of (c{i + 1})\n{res_name[i]}" for i in RES_OF],
}

# ---------- layout (inches, created at final size) ----------
W_FIG = DC
LEFT = 0.21                                  # room for the row labels (a)-(d)
GAP = (W_FIG - LEFT - 6 * TILE_IN) / 5
LAB_FS = 8
BASE1, BASE2 = 0.148, 0.274                  # baselines of the two label lines below the tile bottom (>= 2 pt clear of the frame)
ROW_GAP = 0.105                              # from the second baseline to the next row's tile top
PITCH = TILE_IN + BASE2 + ROW_GAP
HEAD = 0.17                                  # column-number line above row (a)
HEAD_BASE = 0.060                            # its baseline, above the tile tops of row (a)
TOP = HEAD
H_FIG = TOP + 4 * PITCH - ROW_GAP + 0.035    # room for descenders under the last row
fig = plt.figure(figsize=(W_FIG, H_FIG))
r_names = "abcd"
labels_drawn = []                            # (row, text artist) for the overlap check


def add_tile(r, c, img, label, grey=False, height=None):
    h = height or TILE_IN
    x0 = LEFT + c * (TILE_IN + GAP)
    ytop = TOP + r * PITCH + (TILE_IN - h) / 2
    ax = fig.add_axes([x0 / W_FIG, 1 - (ytop + h) / H_FIG, TILE_IN / W_FIG, h / H_FIG])
    if grey:
        ax.imshow(img, cmap="gray", vmin=-1, vmax=1, interpolation="none")
    else:
        ax.imshow(img, interpolation="none")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4); s.set_color(C["ink"])
    ybot = TOP + r * PITCH + TILE_IN          # each label line on its own fixed baseline
    for line, base in zip(label.split("\n"), (BASE1, BASE2)):
        t = fig.text((x0 + TILE_IN / 2) / W_FIG, 1 - (ybot + base) / H_FIG, line, ha="center",
                     va="baseline", fontsize=LAB_FS, color=C["ink"])
        labels_drawn.append((r, base, t))
    if c == 0:                               # row label "(a)"... at the upper left, 9 pt regular
        fig.text(0.0, 1 - (TOP + r * PITCH) / H_FIG, f"({r_names[r]})", ha="left", va="top",
                 fontsize=9, color=C["ink"])
    return ax


col_nums = []                                # column numbers 1-6 above row (a): tile (c4) = row (c), column 4
for c in range(6):
    x0 = LEFT + c * (TILE_IN + GAP)
    col_nums.append(fig.text((x0 + TILE_IN / 2) / W_FIG, 1 - (TOP - HEAD_BASE) / H_FIG, str(c + 1),
                             ha="center", va="baseline", fontsize=9, color=C["ink"]))


for c, img in enumerate(row_a):
    if c == 1:                               # native frame, letterboxed in the tile, crop box at 1 pt
        h = TILE_IN * img.shape[0] / img.shape[1]
        ax = add_tile(0, c, img, L["a"][c], height=h)
        s = img.shape[1] / nat.shape[1]
        x = (nat.shape[1] - MEAS) // 2 * s - 0.5
        y = (nat.shape[0] - MEAS) // 2 * s - 0.5
        ax.add_patch(Rectangle((x, y), MEAS * s, MEAS * s, fill=False, lw=1.0, ec=BODY_A))
    else:
        add_tile(0, c, img, L["a"][c])
for c, img in enumerate(row_b):
    add_tile(1, c, grey_tile(img), L["b"][c], grey=True)
for c, img in enumerate(row_c):
    add_tile(2, c, img, L["c"][c])
for c, img in enumerate(row_d_img):
    add_tile(3, c, img, L["d"][c])
for c, img in enumerate(row_d_res):
    add_tile(3, c + 2, grey_tile(img), L["d"][c + 2], grey=True)

# ---------- legibility check: horizontal clearance between neighbouring labels, clearance to the tile ----------
fig.canvas.draw()
rend = fig.canvas.get_renderer()
min_gap = np.inf
for r in range(4):
    for base in (BASE1, BASE2):
        bb = sorted((t.get_window_extent(rend) for rr, b, t in labels_drawn if rr == r and b == base),
                    key=lambda e: e.x0)
        for e0, e1 in zip(bb, bb[1:]):
            min_gap = min(min_gap, (e1.x0 - e0.x1) / fig.dpi)
        for e in bb:
            assert e.x0 >= 0 and e.x1 <= W_FIG * fig.dpi, "label outside the figure"
top_clear = min(((1 - (TOP + rr * PITCH + TILE_IN) / H_FIG) * H_FIG * fig.dpi - t.get_window_extent(rend).y1)
                / fig.dpi for rr, b, t in labels_drawn if b == BASE1)
print(f"smallest horizontal gap between neighbouring labels: {min_gap:.3f} in; "
      f"smallest clearance from a tile to its label: {top_clear:.3f} in")
head_clear = min((c.get_window_extent(rend).y0 - (1 - TOP / H_FIG) * H_FIG * fig.dpi) / fig.dpi for c in col_nums)
head_room = min((H_FIG * fig.dpi - c.get_window_extent(rend).y1) / fig.dpi for c in col_nums)
print(f"column numbers: {head_clear:.3f} in above the tiles of row (a), {head_room:.3f} in below the figure top")
assert min_gap > 0.03 and top_clear > 0.025 and head_clear > 0.025 and head_room >= 0

plt.rcParams["savefig.bbox"] = None          # keep the page width exactly DC
pdf = save(fig, "fig02_inputs_outputs")


# ---------- recompress the photographic tiles in the PDF as JPEG (grey residual tiles stay lossless) ----------
def jpeg_photos(path, quality=92):
    import pymupdf
    doc = pymupdf.open(path)
    page = doc[0]
    n = 0
    for info in page.get_images(full=True):
        xref = info[0]
        pix = pymupdf.Pixmap(doc, xref)
        if pix.n - pix.alpha != 3:           # only RGB photographs
            continue
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples if not pix.alpha
                             else pymupdf.Pixmap(pix, 0).samples)
        a = np.asarray(im)
        if np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2]):
            continue                         # grey residual / fingerprint tile: keep lossless
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=quality, subsampling=0, optimize=True)
        page.replace_image(xref, stream=buf.getvalue()); n += 1
    tmp = path + ".tmp"
    doc.save(tmp, garbage=4, deflate=True); doc.close()
    os.replace(tmp, path)
    return n


before = os.path.getsize(pdf)
n_jpeg = jpeg_photos(pdf)
import pymupdf  # noqa: E402
with pymupdf.open(pdf) as d:
    wpt, hpt = d[0].rect.width, d[0].rect.height
print(f"wrote {pdf}: {before / 1e6:.2f} MB -> {os.path.getsize(pdf) / 1e6:.2f} MB ({n_jpeg} photographs as JPEG q92); "
      f"page {wpt / 72:.3f} x {hpt / 72:.3f} in; tile {TILE_IN} in, gap {GAP:.3f} in; photo tiles {PX} px "
      f"({DPI_IMG} dpi); residual windows {WIN} px")
