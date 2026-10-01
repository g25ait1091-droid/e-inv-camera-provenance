"""G1 (RESULTS.md Entry 76) - training crops with each body's own fingerprint divided out.

For body X the 50 training crops become round(clip(Y / (1 + K_X_E1))), applied per colour channel and
rounded once. K_X_E1 is the E1 estimate, disjoint from the E2 estimates that measure the generations, so
suppression cannot be tuned on the measuring template. Writes out/t1/train_png/{supA_a0,supB_a0} and
out/t1/suppress.json, which records for each body the clipped-pixel fraction, the RMS change in gray
levels, and the residual fingerprint contrast R_sup of the stored crops against the uninjected crops -
the quantity the Entry 76 reading is downgraded on if suppression leaves more than 25 % standing.
"""
import os, sys, json, glob
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2; T1 = os.path.join(V2, "out", "t1"); FP = os.path.join(V2, "out", "fp")
BODY = {"A": ("Nikon_D200_1", "none_a0"), "B": ("Nikon_D200_0", "noneB_a0")}

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def lum(a): return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)

def main():
    from fingerprints import wavelet_residual
    rep = {"entry": "RESULTS.md Entry 76 (G1)", "rule": "Y / (1 + K_own_E1), clipped, rounded once", "bodies": {}}
    for role, (dev, src_dir) in BODY.items():
        K = np.load(os.path.join(FP, f"K_{role}_E1.npy")).astype(np.float32)
        Km = {r: np.load(os.path.join(FP, f"K_{r}_E2.npy")).astype(np.float32) for r in ("A", "B")}
        other = "B" if role == "A" else "A"
        src = sorted(glob.glob(os.path.join(T1, "train_png", src_dir, "*.png")))
        assert len(src) == 50, f"{src_dir}: {len(src)} crops"
        dst = os.path.join(T1, "train_png", f"sup{role}_a0"); os.makedirs(dst, exist_ok=True)
        clip, rms, c_before, c_after = [], [], [], []
        for i, f in enumerate(src):
            rgb = np.asarray(Image.open(f).convert("RGB"), np.float32)
            Ys = rgb / (1.0 + K[..., None])
            clip.append(float(((Ys < 0) | (Ys > 255)).mean()))
            sup = np.clip(np.rint(Ys), 0, 255).astype(np.uint8)
            Image.fromarray(sup).save(os.path.join(dst, f"{i:04d}.png"), compress_level=1)
            rms.append(float(np.sqrt(((sup.astype(np.float32) - rgb) ** 2).mean())))
            for arr, acc in ((rgb, c_before), (sup.astype(np.float32), c_after)):
                Y = lum(arr); W = wavelet_residual(Y)
                acc.append(ncc(W, Y * Km[role]) - ncc(W, Y * Km[other]))
        b, a = float(np.mean(c_before)), float(np.mean(c_after))
        rep["bodies"][role] = {"device": dev, "source": src_dir, "n": len(src),
                               "clipped_pixel_fraction": float(np.mean(clip)),
                               "change_rms_gray": float(np.mean(rms)),
                               "contrast_uninjected": b, "contrast_suppressed": a,
                               "R_sup_fraction_remaining": a / b if b else None,
                               "suppression_pct": 100 * (1 - a / b) if b else None}
        print(role, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rep["bodies"][role].items()}, flush=True)
    worst = max(abs(v["R_sup_fraction_remaining"]) for v in rep["bodies"].values())
    rep["reading_downgraded_to_descriptive"] = bool(worst > 0.25)
    json.dump(rep, open(os.path.join(T1, "suppress.json"), "w"), indent=1)
    print(f"[g1] worst residual {worst:.3f} of the uninjected contrast; "
          f"{'DOWNGRADED' if worst > 0.25 else 'within the registered 25 % allowance'}", flush=True)

if __name__ == "__main__":
    main()
