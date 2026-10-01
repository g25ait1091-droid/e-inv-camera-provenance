"""G1 (RESULTS.md Entry 77) - training crops with each body's own fingerprint inverted.

For body X the 50 training crops become round(clip(Y (1 - ALPHA K_X_E1) + u)), u ~ U[-0.5, 0.5) drawn per
pixel before the single rounding, applied per colour channel. Dividing the fingerprint out (Entry 76) moved
pixels by less than a twentieth of a quantisation step and rounding erased it; inverting at ALPHA = 6 with
dither both survives quantisation and asks a sharper question - does the 16000-step signal follow the
fingerprint's sign? K_X_E1 is the E1 estimate, disjoint from the E2 estimates that measure the generations.
Writes out/t1/train_png/{invA_a0,invB_a0} and out/t1/invert.json with, per body, the clipped-pixel
fraction, the RMS change in gray levels, and the stored contrast R_inv against the uninjected crops.
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
    ALPHA = 6.0
    rng = np.random.default_rng(20260919)
    rep = {"entry": "RESULTS.md Entry 77 (G1, inversion)", "alpha": ALPHA,
           "rule": "Y (1 - alpha K_own_E1) + dither, clipped, rounded once", "bodies": {}}
    for role, (dev, src_dir) in BODY.items():
        K = np.load(os.path.join(FP, f"K_{role}_E1.npy")).astype(np.float32)
        Km = {r: np.load(os.path.join(FP, f"K_{r}_E2.npy")).astype(np.float32) for r in ("A", "B")}
        other = "B" if role == "A" else "A"
        src = sorted(glob.glob(os.path.join(T1, "train_png", src_dir, "*.png")))
        assert len(src) == 50, f"{src_dir}: {len(src)} crops"
        dst = os.path.join(T1, "train_png", f"inv{role}_a0"); os.makedirs(dst, exist_ok=True)
        clip, rms, c_before, c_after = [], [], [], []
        for i, f in enumerate(src):
            rgb = np.asarray(Image.open(f).convert("RGB"), np.float32)
            Ys = rgb * (1.0 - ALPHA * K[..., None]) + rng.uniform(-0.5, 0.5, rgb.shape)
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
                               "R_inv_fraction_of_uninjected": a / b if b else None}
        print(role, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rep["bodies"][role].items()}, flush=True)
    json.dump(rep, open(os.path.join(T1, "invert.json"), "w"), indent=1)
    print("[g1] inverted crops written; stored contrast as a fraction of uninjected: "
          + ", ".join(f"{r}={v['R_inv_fraction_of_uninjected']:+.3f}" for r, v in rep["bodies"].items()), flush=True)

if __name__ == "__main__":
    main()
