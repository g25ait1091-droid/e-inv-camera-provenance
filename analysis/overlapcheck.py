"""Detect text/legend collisions with plotted data, in display coordinates."""
import numpy as np
from matplotlib.text import Text, Annotation
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

def _inter(a, b):
    x0, x1 = max(a.x0, b.x0), min(a.x1, b.x1)
    y0, y1 = max(a.y0, b.y0), min(a.y1, b.y1)
    return max(0.0, x1-x0) * max(0.0, y1-y0)

def check(fig, ax, name, pad=1.0):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hits = []

    labels = []
    for t in ax.texts:
        if t.get_text().strip():
            labels.append(("text:'%s'" % t.get_text().split("\n")[0][:26], t.get_window_extent(r)))
    lg = ax.get_legend()
    if lg is not None:
        labels.append(("legend", lg.get_window_extent(r)))

    # arrow paths: sample the drawn patch and treat the points as data
    arrows = []
    owner = []
    for t in list(ax.texts):
        ap = getattr(t, "arrow_patch", None)
        if ap is None: continue
        owner.append(t)
        try:
            p = ap.get_path().transformed(ap.get_transform())
            v = p.vertices
            if len(v) >= 2:
                seg = []
                for k in range(len(v)-1):
                    for f in np.linspace(0, 1, 24):
                        seg.append(v[k]*(1-f) + v[k+1]*f)
                arrows.append(np.array(seg))
        except Exception:
            pass

    data = []
    arrow_of = {}
    for i, a in enumerate(arrows):
        nm = f"arrow{i}"
        data.append((nm, a))
        if i < len(owner):
            arrow_of[nm] = "text:'%s'" % owner[i].get_text().split("\n")[0][:26]
    for p in ax.patches:
        if isinstance(p, Rectangle) and p.get_width() > 0:
            bb = p.get_window_extent(r)
            axbb = ax.get_window_extent(r)
            if bb.width > 0.92*axbb.width:      # background span, not data
                continue
            if bb.height > 2 and bb.width > 2:
                data.append(("bar", bb))
    for ln in ax.lines:
        xy = ln.get_xydata()
        if len(xy) == 0 or ln.get_linestyle() in ("--", ":", "-.") and ln.get_marker() in ("", "None"):
            continue
        pts = ax.transData.transform(xy)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts):
            data.append(("marker/line", pts))

    for lname, lb in labels:
        for dname, d in data:
            if isinstance(d, np.ndarray):
                if arrow_of.get(dname) == lname:      # a label's own leader
                    continue
                inside = ((d[:,0] > lb.x0-pad) & (d[:,0] < lb.x1+pad) &
                          (d[:,1] > lb.y0-pad) & (d[:,1] < lb.y1+pad))
                if inside.any():
                    hits.append(f"{lname} covers {int(inside.sum())} {dname} point(s)")
            else:
                if _inter(lb, d) > 4:
                    hits.append(f"{lname} overlaps a {dname}")

    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if _inter(labels[i][1], labels[j][1]) > 4:
                hits.append(f"{labels[i][0]} overlaps {labels[j][0]}")

    hits = sorted(set(hits))
    print(f"  {name:30s} {'OK' if not hits else 'COLLISIONS:'}")
    for h in hits: print(f"      - {h}")
    if hits:
        inv = ax.transData.inverted()
        for lname, lb in labels:
            (x0,y0),(x1,y1) = inv.transform([[lb.x0,lb.y0],[lb.x1,lb.y1]])
            print(f"        {lname[:34]:34s} data x {x0:+.3f}..{x1:+.3f}  y {y0:+.3e}..{y1:+.3e}")
    return hits
