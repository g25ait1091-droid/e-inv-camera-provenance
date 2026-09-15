"""Near-copy measure with DINOv2, the embedding the study's copy-detection audit uses (flag at cosine >= 0.90).

The 256^2 thumbnail correlation of dose_stats.py mostly measures layout, and a pixel-aligned detail
correlation misses copies that are slightly shifted, so neither identifies near copies reliably. DINOv2
(facebook/dinov2-base, CLS token, cached locally) is robust to small shifts. For every generation of the
16000-step adapters, and for 2000-step and base-model generations as a reference, the maximum cosine to
the 50 training crops of its body. CPU only. Writes out/t1/dino_memorization.json and _emb.npz.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import glob, json, numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

T1 = os.path.join(EINV.V2, 'out', 't1'); G = 250
torch.set_num_threads(6)
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base", local_files_only=True)
model = AutoModel.from_pretrained("facebook/dinov2-base", local_files_only=True).eval()

@torch.no_grad()
def emb(paths, bs=16):
    out = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        z = model(**proc(images=ims, return_tensors="pt")).pooler_output
        out.append(torch.nn.functional.normalize(z, dim=-1).numpy())
    return np.concatenate(out)

train = {"A": sorted(glob.glob(os.path.join(T1, "train_png", "none_a0", "*.png"))),
         "B": sorted(glob.glob(os.path.join(T1, "train_png", "noneB_a0", "*.png")))}
TE = {b: emb(fs) for b, fs in train.items()}
arms = sorted(d for d in os.listdir(os.path.join(T1, "gens")) if d.startswith("dose16k"))
arms += [a for a in ("nomark_s0", "nomarkB_s0", "dose8k_A_s0", "dose8k_B_s0", "local_base") if os.path.isdir(os.path.join(T1, "gens", a))]
res = {"model": "facebook/dinov2-base CLS", "copy_threshold_study": 0.90, "arms": {}}; cands = []; store = {}
for arm in arms:
    files = sorted(glob.glob(os.path.join(T1, "gens", arm, "*.png")))[:G]
    body = "B" if ("_B_" in arm or arm.startswith("nomarkB")) else "A"
    E = emb(files); S = E @ TE[body].T; k = S.argmax(1); m = S.max(1); store[arm] = E
    res["arms"][arm] = {"body": body, "n": len(files), "max_cos_mean": float(m.mean()), "max_cos_p95": float(np.percentile(m, 95)),
                        "frac_ge_0.90": float((m >= 0.90).mean()), "frac_ge_0.85": float((m >= 0.85).mean()), "max_cos_max": float(m.max())}
    if arm.startswith("dose16k"):
        cands += [(float(m[i]), files[i], train[body][k[i]], arm) for i in range(len(files))]
    print(arm, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in res["arms"][arm].items()}, flush=True)
cands.sort(key=lambda x: -x[0])
res["top16k"] = [{"cos": c[0], "generation": c[1], "training_crop": c[2], "arm": c[3]} for c in cands[:10]]
json.dump(res, open(os.path.join(T1, "dino_memorization.json"), "w"), indent=1)
np.savez_compressed(os.path.join(T1, "dino_memorization_emb.npz"), **{k: v for k, v in store.items()}, train_A=TE["A"], train_B=TE["B"])
for c in res["top16k"][:5]: print("top", round(c["cos"], 3), c["arm"], os.path.basename(c["generation"]), "vs", os.path.basename(c["training_crop"]))
