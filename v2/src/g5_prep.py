"""G5 preparation (RESULTS.md Entry 78) - a second, disjoint training set for each D200 body.

Every one of the 24 primary adapters saw the same 50 photographs of its body, so the primary limit
generalizes over adapter seeds and not over training sets. This draws a second training set of 50 from the
images that no split uses: outside E1, E2, H and the primary T. Body A has 70 such images and body B 62.
The disjointness is asserted, not assumed. Writes out/t1/train_png/{altA_a0,altB_a0} and
out/t1/alt_trainset.json.
"""
import os, sys, json
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

T1 = os.path.join(EINV.V2, "out", "t1"); N = 50
BODY = {"A": "Nikon_D200_1", "B": "Nikon_D200_0"}

def main():
    from fingerprints import dv, splits
    import t1_ladder as L
    rep = {"entry": "RESULTS.md Entry 78 (G5)", "n_per_body": N, "bodies": {}}
    for role, dev in BODY.items():
        files = dv[dev]; sp = splits(files)
        used = set(sp["E1"]) | set(sp["E2"]) | set(sp["H"]) | set(sp["T"])
        pool = [f for f in files if f not in used]
        assert len(pool) >= N, f"{dev}: {len(pool)} spare, need {N}"
        alt = pool[:N]
        assert not (set(alt) & used), "alternate training set overlaps a used split"
        d = os.path.join(T1, "train_png", f"alt{role}_a0"); os.makedirs(d, exist_ok=True)
        for i, f in enumerate(alt):
            Image.fromarray(np.clip(np.rint(L.load_rgb_crop(f)), 0, 255).astype(np.uint8)).save(
                os.path.join(d, f"{i:04d}.png"), compress_level=1)
        rep["bodies"][role] = {"device": dev, "spare_pool": len(pool), "chosen": N,
                               "files": [os.path.basename(f) for f in alt],
                               "disjoint_from_primary_T": True}
        print(f"[g5] {role} = {dev}: {len(pool)} spare, wrote {N} crops, disjoint from every used split", flush=True)
    json.dump(rep, open(os.path.join(T1, "alt_trainset.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
