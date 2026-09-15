"""Full-split fingerprints for the primary pair, using the study's own forensic core.

Reproduces S0 discovery and split construction deterministically from the Dresden directory
(E1 = 80, E2 = 140, T = 50, H = 40, guard 10, contiguous by sorted filename), then estimates
K from E1 and from E2 separately for A and B and caches the H-split luminance crops and
residuals. Output: out/fp/K_{A,B}_{E1,E2}.npy, out/fp/H_{A,B}.npz, out/fp/manifest.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, re, glob, json, time
import numpy as np, pywt, torch, torch.nn.functional as tF
from PIL import Image

OUT = os.path.join(EINV.V2, 'out', 'fp'); os.makedirs(OUT, exist_ok=True)
BASE = os.path.join(EINV.DATASETS, 'dresden', 'Dresden_Exp')
MEAS, SIG0, WAVELET, LEVELS = 1024, 2.0, "db8", 4
N_E1, N_E2, N_T, N_H, GUARD = 80, 140, 50, 40, 10
DEV = {"A": "Nikon_D200_1", "B": "Nikon_D200_0"}

def load_lum_crop(fp):
    with Image.open(fp) as im:
        im = im.convert("RGB"); W, H = im.size
        a = np.asarray(im.crop(((W-MEAS)//2, (H-MEAS)//2, (W-MEAS)//2+MEAS, (H-MEAS)//2+MEAS)), np.float32)
    return (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(np.float32)

def _conv2(x, k):
    return tF.conv2d(torch.from_numpy(np.ascontiguousarray(x))[None,None],
                     torch.from_numpy(k)[None,None], padding=k.shape[0]//2)[0,0].numpy()

def wavelet_residual(img):
    s0 = np.float32(SIG0)**2
    co = pywt.wavedec2(img, WAVELET, level=LEVELS, mode="periodization"); out = [co[0]*0.0]
    for (cH, cV, cD) in co[1:]:
        band = []
        for c in (cH, cV, cD):
            c = c.astype(np.float32); vmin = None
            for w in (3, 5, 7, 9):
                v = np.maximum(_conv2(c*c, np.ones((w, w), np.float32)/(w*w)) - s0, 0.0)
                vmin = v if vmin is None else np.minimum(vmin, v)
            band.append(c*(s0/(vmin+s0)))
        out.append(tuple(band))
    n = pywt.waverec2(out, WAVELET, mode="periodization")[:img.shape[0], :img.shape[1]]
    r = n - n.mean(); return (r/float(r.std())).astype(np.float32)

def zero_mean(K):
    K = K - K.mean(axis=1, keepdims=True); return K - K.mean(axis=0, keepdims=True)

def wiener_dft(K):
    F = np.fft.fft2(K.astype(np.float32)); mag = (np.abs(F)/K.shape[0]).astype(np.float32)
    sigma = float(np.median(mag))/0.6745; s2 = np.float32(sigma**2); vmin = None
    for w in (3, 5, 7, 9):
        k = np.ones((w, w), np.float32)/(w*w); mu = _conv2(mag, k)
        v = np.maximum(_conv2(mag*mag, k) - mu*mu, 0.0); vmin = v if vmin is None else np.minimum(vmin, v)
    return np.real(np.fft.ifft2(F*(s2/(vmin+s2)).astype(np.float32))).astype(np.float32)

def estimate_K(files):
    num = den = None
    for fp in files:
        Y = load_lum_crop(fp); W = wavelet_residual(Y)
        num = W*Y if num is None else num + W*Y; den = Y*Y if den is None else den + Y*Y
    return wiener_dft(zero_mean((num/np.maximum(den, 1e-6)).astype(np.float32)))

FN = re.compile(r"^(?P<m>.+)_(?P<i>\d+)_(?P<s>\d+)\.(jpe?g|JPE?G)$")
dv = {}
for fp in glob.glob(os.path.join(BASE, "**", "*.*"), recursive=True):
    m = FN.match(os.path.basename(fp))
    if m: dv.setdefault(f"{m['m']}_{m['i']}", []).append(fp)
for v in dv.values(): v.sort()

def splits(files):
    i = 0; sp = {}
    for name, n in (("E1", N_E1), ("E2", N_E2), ("T", N_T), ("H", N_H)):
        sp[name] = files[i:i+n]; i += n + GUARD
    return sp

if __name__ == "__main__":
    man = {}
    t0 = time.time()
    for role, did in DEV.items():
        sp = splits(dv[did]); man[role] = {"device": did, **{k: v for k, v in sp.items()}}
        for es in ("E1", "E2"):
            p = os.path.join(OUT, f"K_{role}_{es}.npy")
            if not os.path.exists(p):
                np.save(p, estimate_K(sp[es])); print(f"[fp] K_{role}_{es} from {len(sp[es])} images  ({time.time()-t0:.0f}s)", flush=True)
        p = os.path.join(OUT, f"H_{role}.npz")
        if not os.path.exists(p):
            Y = np.stack([load_lum_crop(f) for f in sp["H"]]); W = np.stack([wavelet_residual(y) for y in Y])
            np.savez_compressed(p, Y=Y, W=W); print(f"[fp] H_{role}: {len(sp['H'])} crops + residuals cached", flush=True)
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)

    KA1, KA2, KB1, KB2 = (np.load(os.path.join(OUT, f"K_{r}_{e}.npy")) for r, e in (("A","E1"),("A","E2"),("B","E1"),("B","E2")))
    def ncc(a, b):
        a = a - a.mean(); b = b - b.mean(); return float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    gates = {"kappa_model_E2": ncc(KA2, KB2), "splithalf_A": ncc(KA1, KA2), "splithalf_B": ncc(KB1, KB2)}
    json.dump(gates, open(os.path.join(OUT, "gates.json"), "w"), indent=1)
    print("[fp] gates:", gates, flush=True)
    print("[fp] done in", round(time.time()-t0), "s")
