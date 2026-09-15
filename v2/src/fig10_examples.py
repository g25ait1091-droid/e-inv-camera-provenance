"""Figure — what goes in and what comes out. (a) body A's training photographs (first three crops);
(b) the base model and a primary personalized adapter at the same seed (first two seeds of the paired
bank, no selection); (c) the designed patterns, a 96x96 corner of each field magnified; (d) the most
memorized 16000-step generation beside its nearest training crop, chosen by the thumbnail correlation
used in dose_stats.py over every 16000-step adapter present. Reads out/t1; writes
paper/fig_examples.pdf and prints the numbers the caption quotes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, glob, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman"],
                            "mathtext.fontset": "stix", "font.size": 7})
INK = "#1F2A37"
T1 = os.path.join(EINV.V2, 'out', 't1'); OUT = os.path.join(EINV.V2, 'paper', 'fig_examples.pdf')
SEEDS = (0, 1)

def rgb(path, size=600):
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS))
def detail(path, s=384):
    """Detail-level signature: 384^2 grayscale minus its Gaussian blur (sigma 3 px), unit norm. The
    256^2 thumbnail correlation of dose_stats.py mostly measures layout (bright sky over a dark band),
    so a hazy, detail-free generation can outscore a genuine near copy; panel (d) is chosen on detail."""
    from scipy import ndimage
    g = np.asarray(Image.open(path).convert("L").resize((s, s), Image.BOX), np.float32)
    d = (g - ndimage.gaussian_filter(g, 3.0)).ravel(); d -= d.mean()
    return d / (np.linalg.norm(d) + 1e-12)
def gray(path):
    return np.asarray(Image.open(path).convert("L"), np.float32)
def thumb(path):
    a = np.asarray(Image.open(path).convert("L").resize((256, 256), Image.BOX), np.float32).ravel(); a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-12)

# (a) inputs
train_A = sorted(glob.glob(os.path.join(T1, "train_png", "none_a0", "*.png")))
# (b) base vs personalized at matched seeds. FIG10_B=prompt draws them from the five-caption bank
# (out/t5: caption j mod 5 with seed j; seed 0 = street, seed 1 = room interior) instead of the single
# training caption, whose token also names a rifle.
if os.environ.get("FIG10_B") == "prompt":
    T5 = os.path.join(EINV.V2, 'out', 't5', 'gens'); OUT = OUT.replace(".pdf", "_promptbank.pdf")
    pairs = [(os.path.join(T5, "base", f"{i:05d}.png"), os.path.join(T5, "A_raw_s0_r16", f"{i:05d}.png")) for i in SEEDS]
else:
    pairs = [(os.path.join(T1, "gens", "local_base", f"{i:05d}.png"), os.path.join(T1, "gens", "local_A_raw_s0", f"{i:05d}.png")) for i in SEEDS]
for i, (b, p) in zip(SEEDS, pairs):
    print(f"seed {i}: mean |personalized - base| = {np.abs(gray(p) - gray(b)).mean():.1f} gray levels")
# (d) most memorized 16000-step generation
train = {"A": sorted(glob.glob(os.path.join(T1, "train_png", "none_a0", "*.png"))),
         "B": sorted(glob.glob(os.path.join(T1, "train_png", "noneB_a0", "*.png")))}
tstack = {b: np.stack([thumb(f) for f in fs]) for b, fs in train.items()}
dstack = {b: np.stack([detail(f) for f in fs]) for b, fs in train.items()}
cands = []
for arm in sorted(d for d in os.listdir(os.path.join(T1, "gens")) if d.startswith("dose16k")):
    body = "B" if "_B_" in arm else "A"
    for g in sorted(glob.glob(os.path.join(T1, "gens", arm, "*.png")))[:250]:
        cd = dstack[body] @ detail(g); k = int(cd.argmax())
        cands.append((float(cd[k]), float(tstack[body][k] @ thumb(g)), g, train[body][k], arm))
cands.sort(key=lambda x: -x[0])
print(f"memorization: {len(cands)} generations; top by pixel-aligned detail correlation (not used for the panel):")
for c in cands[:3]:
    print(f"  detail r {c[0]:.3f}  thumbnail r {c[1]:.3f}  {c[4]} {os.path.basename(c[2])} vs {os.path.basename(c[3])}")
# Panel (d) is chosen by DINOv2 cosine, the near-copy measure of the study's copy audit (src/dino_memorization.py)
DJ = json.load(open(os.path.join(T1, "dino_memorization.json"))); top = DJ["top16k"][0]
best = (top["cos"], None, top["generation"], top["training_crop"], top["arm"])
print(f"panel (d): DINOv2 cosine {top['cos']:.3f}  {top['arm']} {os.path.basename(top['generation'])} vs {os.path.basename(top['training_crop'])}")
# (c) designed patterns
FIELDS = [(r"$\pm1$ field", "M_rand.npy"), ("top-octave field", "F_band0.npy"), ("tile, 36-px period", "P36.npy"),
          ("tile, 32-px period", "P32.npy"), ("DiffusionShield", "W_ds_lum.npy")]
def patch(fname, s=96):
    f = np.load(os.path.join(T1, "fields", fname)).astype(np.float32)
    if f.ndim == 3: f = f.mean(axis=2)
    p = f[:s, :s] - f[:s, :s].mean(); return p / (np.abs(p).max() + 1e-12)

fig = plt.figure(figsize=(7.16, 2.62))
gs = GridSpec(2, 7, figure=fig, wspace=0.07, hspace=0.42, left=0.005, right=0.995, top=0.86, bottom=0.03)
def show(r, c, img, title, cmap=None):
    ax = fig.add_subplot(gs[r, c])
    ax.imshow(img, cmap=cmap, vmin=None if cmap is None else -1, vmax=None if cmap is None else 1, interpolation="nearest" if cmap else "lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.4); s.set_color("#9AA5B1")
    ax.set_title(title, fontsize=6.2, color=INK, pad=2)
    return ax
for c in range(3):
    show(0, c, rgb(train_A[c]), f"training photograph {c + 1}")
for j, (i, (b, p)) in enumerate(zip(SEEDS, pairs)):
    show(0, 3 + 2 * j, rgb(b), f"seed {i}: base model")
    show(0, 4 + 2 * j, rgb(p), f"seed {i}: personalized")
PERIOD = {"P36.npy": 36, "P32.npy": 32, "W_ds_lum.npy": 32}
for c, (lab, fn) in enumerate(FIELDS):
    ax = show(1, c, patch(fn), lab, cmap="gray")
    if fn in PERIOD:  # mark the tile boundaries so the repetition is visible
        for v in range(PERIOD[fn], 96, PERIOD[fn]):
            ax.axvline(v - 0.5, color="#C8641E", lw=0.7); ax.axhline(v - 0.5, color="#C8641E", lw=0.7)
show(1, 5, rgb(best[2]), "generation, 16000 steps")
show(1, 6, rgb(best[3]), f"nearest crop (cos {best[0]:.2f})")
# group labels
def group(c0, c1, row, text):
    b0 = gs[row, c0].get_position(fig); b1 = gs[row, c1].get_position(fig)
    y = b0.y1 + 0.085
    fig.text(b0.x0, y, text, fontsize=7.2, color=INK, fontweight="bold", ha="left", va="bottom")
    fig.add_artist(plt.Line2D([b0.x0, b1.x1], [y - 0.012, y - 0.012], lw=0.5, color="#9AA5B1"))
group(0, 2, 0, "(a) input: body $A$'s training photographs")
group(3, 6, 0, "(b) output: base model and personalized model, same seed")
group(0, 4, 1, r"(c) designed patterns, $96\times96$-pixel corner magnified")
group(5, 6, 1, "(d) closest to a training crop, body $B$" if "_B_" in best[4] else "(d) closest to a training crop, body $A$")
fig.savefig(OUT, dpi=600); print("wrote", OUT)
