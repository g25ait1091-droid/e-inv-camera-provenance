"""Fig. S7 (fig:s07closeness): closeness of generations to the training crops, by dose (DINOv2).

(a) For every examined arm, the nearest-crop DINOv2 cosine of each of its first 250 generations (grey
    points, horizontal jitter for legibility only) and the arm mean (body marker). Arms in dose order:
    base model, one adapter per body at 2000 and 8000 steps (seed 0), the twelve 16000-step adapters
    (seeds in integer order, A then B). Dashed line: the study's copy threshold.
(b) The four closest 16000-step pairs with distinct training crops: each generation beside its nearest
    training crop. The pairs are ringed and numbered in (a).

Sources (read, never typed):
  per-arm summaries, copy threshold, top pairs   out/t1/dino_memorization.json (arms.*, copy_threshold_study, top16k)
  per-generation cosines                         recomputed from the stored embeddings
                                                 out/t1/dino_memorization_emb.npz (<arm>, train_A, train_B) exactly as
                                                 src/dino_memorization.py does (max over the body's 50 crops of E @ T^T);
                                                 the script asserts the recomputed n, mean, p95, max and share >= 0.90
                                                 equal the stored summaries, and each shown pair's cosine equals top16k.
  training-crop provenance                       out/fp/manifest.json (A.T, B.T, device)
  printed text                                   numbers.json "text" of nMemCopyThreshold
The shown crops are the local decodes (train_png/none_a0, noneB_a0) that the embeddings were computed on;
the script checks each is the centred 1024-px window of its native Dresden frame.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_style import plt, DC, C, BODY_A, BODY_B, MARK_A, MARK_B, refline, image_panel, save  # noqa: E402

OUT = (EINV.V2 + "/out")
T1 = OUT + "/t1"
NUMS = (EINV.V2 + "/paper/fv/numbers.json")
CROP = 1024                   # design constant: centred 1024 x 1024 training crop
NPAIR = 4                     # pairs shown in (b)
PX = 400                      # embedded pixels per tile (about 520 dpi at printed size)

plt.rcParams["savefig.bbox"] = None          # keep the canvas exactly the supplement text width

# ---- geometry (inches). Journal double-column width (6.99 in), the same as every other double-column figure;
# placed unscaled (it fits the supplement's 516 pt text width).
FIG_W = DC
LM, RM = 0.42, 0.02          # left margin (y label), right margin
TOP = 0.14                   # above panel (a) (panel label)
AX_H = 1.62                  # panel (a) axes height
XLAB = 0.36                  # tick labels + dose labels under (a)
GAP_AB = 0.13                # between (a) and (b) (holds panel label (b))
G_IN, G_PAIR = 0.035, 0.16   # tile gaps: within a pair, between pairs
TILE = (FIG_W - LM - RM - NPAIR * G_IN - (NPAIR - 1) * G_PAIR) / (2 * NPAIR)
LAB_B = 0.27                 # two-line labels beneath the tiles
FIG_H = TOP + AX_H + XLAB + GAP_AB + TILE + LAB_B

nums = json.load(open(NUMS, encoding="utf-8"))
def txt(k): return nums[k]["text"]


def seed_of(arm):
    return int(arm.rsplit("_s", 1)[1])


def load():
    d = json.load(open(T1 + "/dino_memorization.json"))
    z = np.load(T1 + "/dino_memorization_emb.npz")
    arms = d["arms"]
    # dose order; within the 16000-step dose A then B, seeds in integer order
    d16 = lambda b: sorted([a for a in arms if a.startswith(f"dose16k_{b}_")], key=seed_of)
    order = [("base", ["local_base"]), ("2000", ["nomark_s0", "nomarkB_s0"]),
             ("8000", ["dose8k_A_s0", "dose8k_B_s0"]), ("16000", d16("A") + d16("B"))]
    listed = [a for _, g in order for a in g]
    assert sorted(listed) == sorted(arms), "every stored arm is plotted once"
    TE = {"A": z["train_A"], "B": z["train_B"]}
    cos, arg = {}, {}
    for a in listed:
        s = z[a] @ TE[arms[a]["body"]].T
        m, k = s.max(1), s.argmax(1)
        st = arms[a]
        assert len(m) == st["n"]
        for key, v in (("max_cos_mean", m.mean()), ("max_cos_p95", np.percentile(m, 95)),
                       ("max_cos_max", m.max()), ("frac_ge_0.90", (m >= 0.90).mean()),
                       ("frac_ge_0.85", (m >= 0.85).mean())):
            assert abs(v - st[key]) < 1e-5, (a, key, v, st[key])
        # generation index j <-> embedding row: first 250 files in name order, named 00000..00249
        files = sorted(glob.glob(f"{T1}/gens/{a}/*.png"))[:st["n"]]
        assert [int(os.path.basename(f)[:-4]) for f in files] == list(range(st["n"])), a
        cos[a], arg[a] = m, k
    return d, order, cos, arg


def pick_pairs(d, cos, arg):
    """The NPAIR highest-cosine pairs of top16k with distinct training crops, in descending cosine."""
    top = sorted(d["top16k"], key=lambda r: -r["cos"])
    assert top[0]["cos"] == max(v["max_cos_max"] for v in d["arms"].values())
    chosen, seen = [], set()
    for r in top:
        crop = r["training_crop"].replace("\\", "/")
        if crop in seen:
            continue
        seen.add(crop)
        gen = r["generation"].replace("\\", "/")
        j = int(os.path.basename(gen)[:-4])
        k = int(os.path.basename(crop)[:-4])
        a = r["arm"]
        assert abs(cos[a][j] - r["cos"]) < 1e-5 and arg[a][j] == k, (a, j, k)
        chosen.append(dict(rank=top.index(r) + 1, arm=a, body=d["arms"][a]["body"], seed=seed_of(a),
                           j=j, k=k, cos=r["cos"], gen=gen, crop=crop))
        if len(chosen) == NPAIR:
            break
    assert len(chosen) == NPAIR
    return chosen


def rgb(p, n=PX):
    return np.asarray(Image.open(p).convert("RGB").resize((n, n), Image.LANCZOS))


def check_crops(chosen):
    man = json.load(open(OUT + "/fp/manifest.json"))
    for c in chosen:
        native = man[c["body"]]["T"][c["k"]]
        im = Image.open(native)
        w, h = im.size
        box = ((w - CROP) // 2, (h - CROP) // 2, (w - CROP) // 2 + CROP, (h - CROP) // 2 + CROP)
        ref = np.asarray(im.convert("RGB").crop(box), dtype=np.float32)
        got = np.asarray(Image.open(c["crop"]).convert("RGB"), dtype=np.float32)
        c["native"] = os.path.basename(native)
        c["device"] = man[c["body"]]["device"]
        print(f"pair {chosen.index(c) + 1} (top16k rank {c['rank']}): {c['arm']} j={c['j']} cos={c['cos']:.4f} "
              f"vs {c['body']} T[{c['k']}] = {c['device']} {c['native']}; crop vs native centre window "
              f"mean |diff| = {np.abs(got - ref).mean():.3f} gray")


def main():
    d, order, cos, arg = load()
    chosen = pick_pairs(d, cos, arg)
    check_crops(chosen)
    arms = d["arms"]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    # ---------------- (a) distributions
    ax = fig.add_axes([LM / FIG_W, (FIG_H - TOP - AX_H) / FIG_H, (FIG_W - LM - RM) / FIG_W, AX_H / FIG_H])
    rng = np.random.default_rng(0)
    xs, ticks, ticklab, groups, x = {}, [], [], [], 0.0
    for gname, garms in order:
        x0 = x
        for a in garms:
            xs[a] = x; ticks.append(x)
            ticklab.append("" if a == "local_base" else f"{arms[a]['body']}{seed_of(a)}")
            x += 1.0
        groups.append((gname, x0, x - 1.0))
        x += 0.6
    jit = {}
    for a, x0 in xs.items():
        m = cos[a]
        jit[a] = x0 + rng.uniform(-0.30, 0.30, len(m))
        ax.scatter(jit[a], m, s=1.1, marker="o", lw=0, color=C["light"], alpha=0.9, zorder=2, rasterized=True)
    for a, x0 in xs.items():
        m = arms[a]["max_cos_mean"]
        if a == "local_base":
            col, mk, filled = C["grey"], "o", True
        else:
            b = arms[a]["body"]
            col, mk = (BODY_A, MARK_A) if b == "A" else (BODY_B, MARK_B)
            filled = seed_of(a) <= 2
        ax.plot([x0], [m], mk, ms=4.2, mec=col, mew=0.9, mfc=col if filled else "white", zorder=4)
    pts = [(jit[c["arm"]][c["j"]], cos[c["arm"]][c["j"]]) for c in chosen]
    for i, (px, py) in enumerate(pts, 1):
        ax.plot([px], [py], "o", ms=5.0, mfc="none", mec=C["black"], mew=0.6, zorder=5)
        # numeral to the right, unless another ringed point sits just to the right at a similar height
        crowded = any(0 < qx - px < 0.6 and abs(qy - py) < 0.08 for qx, qy in pts)
        ax.annotate(str(i), (px, py), xytext=(-3.6 if crowded else 3.6, 0), textcoords="offset points",
                    fontsize=7.5, ha="right" if crowded else "left", va="center", color=C["black"], zorder=6)
    refline(ax, d["copy_threshold_study"], "copy threshold " + txt("nMemCopyThreshold"), text_x=0.995, fs=7.5)
    assert abs(d["copy_threshold_study"] - nums["nMemCopyThreshold"]["value"]) < 1e-12
    ax.set_xlim(-0.7, x - 0.6 + 0.1)
    assert min(v.min() for v in cos.values()) >= 0.0
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklab, fontsize=8)
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.set_ylabel("Nearest-crop cosine (DINOv2)")
    for gname, a0, a1 in groups:
        lab = "base model" if gname == "base" else f"{gname} steps"
        ax.text((a0 + a1) / 2, -0.155, lab, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8)
        if a1 > a0:
            ax.plot([a0 - 0.3, a1 + 0.3], [-0.135, -0.135], transform=ax.get_xaxis_transform(),
                    color=C["ink"], lw=0.5, clip_on=False)
    fig.text(0.004, 1 - 0.02 / FIG_H, "(a)", ha="left", va="top", fontsize=9)

    # ---------------- (b) closest pairs
    yb = LAB_B / FIG_H
    left = LM
    ax0 = None
    # pair cosine printed beneath the generation, 3 decimals as in numbers.json; pair 1 must equal nMemMaxCos
    assert f"{chosen[0]['cos']:.3f}" == txt("nMemMaxCos"), (chosen[0]["cos"], txt("nMemMaxCos"))
    for i, c in enumerate(chosen, 1):
        for t, (p, lab) in enumerate([
                (c["gen"], f"{i}: {c['body']}{c['seed']}, $j={c['j']}$\ncosine {c['cos']:.3f}"),
                (c["crop"], f"nearest crop\n{c['body']}, $\\mathcal{{T}}$[{c['k']}]")]):
            a = fig.add_axes([left / FIG_W, yb, TILE / FIG_W, TILE / FIG_H])
            image_panel(a, rgb(p), label=lab)
            a.xaxis.label.set_linespacing(1.05)
            ax0 = ax0 or a
            left += TILE + (G_IN if t == 0 else G_PAIR)
    fig.text(0.004, (LAB_B + TILE) / FIG_H, "(b)", ha="left", va="top", fontsize=9)

    out = save(fig, "figS_07_closeness")
    print(f"figure {FIG_W:.3f} x {FIG_H:.3f} in; tile {TILE:.3f} in -> {out}")
    for a in xs:
        print(f"  {a:14s} n={arms[a]['n']} mean={arms[a]['max_cos_mean']:.4f} max={arms[a]['max_cos_max']:.4f}")


if __name__ == "__main__":
    main()
