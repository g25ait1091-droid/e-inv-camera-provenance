"""Figure 1 (v2): the end-to-end pipeline Reviewer 1 asked for, drawn explicitly.
Top row: the six steps. Bottom row: the comparator and the three measurement endpoints, each
attached by an arrow to the step it observes. Full text width, embedded Type-42 fonts.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})

OUT = os.path.join(EINV.V2, 'paper', 'fig1_pipeline_v2')
W, H = 7.16, 3.15
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
INK, GREY, LIGHT, TEAL, RED = "#0D1B2A", "#55616F", "#EEF2F5", "#0E8A7D", "#B5452F"

def box(x, y, w, h, title, body, fc=LIGHT, ec=INK, lw=0.8, tcol=INK, ts=6.9, bs=6.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=lw))
    ax.text(x + w/2, y + h - 0.10, title, ha="center", va="top", fontsize=ts, fontweight="bold", color=tcol)
    ax.text(x + w/2, y + h - 0.30, body, ha="center", va="top", fontsize=bs, color=GREY, linespacing=1.22)

def arrow(p0, p1, col=INK, lw=0.9):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=lw, color=col, shrinkA=1, shrinkB=1))

# ---- top row: six steps ----
n, m, gap = 6, 0.14, 0.10
bw = (W - 2*m - (n-1)*gap) / n; bh = 0.92; y0 = 1.62
xs = [m + i*(bw + gap) for i in range(n)]
steps = [
 ("1 · Camera body",  "one physical body $A$\n(Nikon D200, body 1)\ncarries fingerprint $K_A$"),
 ("2 · Photographs",  "50 training crops\n$1024^2$, native pixels\none fixed caption"),
 ("3 · Base model",   "SD-3.5-medium:\nlatent autoencoder\n+ transformer"),
 ("4 · Personalize",  "rank-16 LoRA\n2000 steps\n12 seeds per arm"),
 ("5 · Text prompt",  "the same caption\npaired seed bank\nno image input"),
 ("6 · Generations",  "500 per adapter\n$1024^2$ PNG\n7,500 in base study"),
]
for x, (t, b) in zip(xs, steps): box(x, y0, bw, bh, t, b)
for i in range(n-1): arrow((xs[i] + bw + 0.005, y0 + bh/2), (xs[i+1] - 0.005, y0 + bh/2))

# ---- bottom row: comparator + three endpoints, evenly spaced ----
k, ew, eh, ey = 4, (W - 2*m - 3*gap) / 4, 0.98, 0.16
exs = [m + i*(ew + gap) for i in range(k)]
box(exs[0], ey, ew, eh, "Comparator: body $B$", "second body of the same model\nfingerprint $K_B$ from disjoint images\nnever used in training\n"
    "$\\rho(\\to K_A)-\\rho(\\to K_B)$ cancels all\nthey share: model, pipeline, generator", fc="white", ec=RED, lw=1.0, tcol=RED, bs=5.7)
ends = [
 ("Endpoint 1 · autoencoder", "real photographs in,\nreconstructed out: is device\ncontrast still there?\n$\\eta = 0.366$, AUC $1.000\\to0.981$", 2),
 ("Endpoint 2 · objective",   "inject fixed-pattern energy\nat amplitude $\\alpha$: does the\ntraining loss respond, to what?\nresponds — not to identity", 3),
 ("Endpoint 3 · output",      "paired contrast per adapter\nover 500 images; 12 adapters\nper arm; simultaneous limit\n$\\lambda_U \\leq 0.15\\,\\%$ of real contrast", 5),
]
for (t, b, step), x in zip(ends, exs[1:]):
    box(x, ey, ew, eh, t, b, fc="white", ec=TEAL, lw=1.0, tcol=TEAL, bs=5.7)
    arrow((xs[step] + bw/2, y0 - 0.005), (x + ew/2, ey + eh + 0.005), col=TEAL)
arrow((exs[0] + ew, ey + 0.18), (exs[3], ey + 0.18), col=RED, lw=0.7)

# ---- banner ----
ax.text(m, H - 0.10, "Does the fingerprint of body $A$, present in every training photograph, appear in images generated from text alone?",
        ha="left", va="top", fontsize=7.0, fontweight="bold", color=INK)
ax.text(m, H - 0.27, "Each endpoint is measured on its own, so a null says where the signal is lost rather than only that it is lost.",
        ha="left", va="top", fontsize=6.2, color=GREY)

fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=200); print("wrote", OUT)
