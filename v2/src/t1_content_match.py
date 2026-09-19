"""Entry 59, E3 — content-matched training sets for bodies A and B.

Pools: each body's images outside its E1, E2 and H splits (the fingerprint and held-out splits stay
untouched). DINOv2 (facebook/dinov2-base, CLS, CPU) embeddings of the centred 1024^2 crops; the one-to-one
matching that maximizes total cosine similarity (Hungarian); the 50 best-matched pairs become the training
sets cmA_a0 (body A) and cmB_a0 (body B), written as the crops the other arms use (no injection).
Also reports, for reference, the same similarity for the unmatched primary training splits (T of A vs T of B).
Writes out/t1/train_png/cmA_a0, cmB_a0 and out/t1/content_match.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np, torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, EINV.SRC)
from fingerprints import dv, splits
import t1_ladder as L

V2 = EINV.V2; T1 = os.path.join(V2, "out", "t1"); N_PAIRS = 50
torch.set_num_threads(4)

def pool(body):
    files = dv[body]; sp = splits(files); used = set(sp["E1"]) | set(sp["E2"]) | set(sp["H"])
    return [f for f in files if f not in used], sp["T"]

def main():
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base", local_files_only=True)
    model = AutoModel.from_pretrained("facebook/dinov2-base", local_files_only=True).eval()
    @torch.no_grad()
    def emb(files):
        out = []
        for i in range(0, len(files), 8):
            ims = [Image.fromarray(L.load_rgb_crop(f).astype(np.uint8)) for f in files[i:i + 8]]
            out.append(torch.nn.functional.normalize(model(**proc(images=ims, return_tensors="pt")).pooler_output, dim=-1).numpy())
        return np.concatenate(out)
    pA, tA = pool("Nikon_D200_1"); pB, tB = pool("Nikon_D200_0")
    print(f"[cm] pools: A {len(pA)}, B {len(pB)}", flush=True)
    EA, EB = emb(pA), emb(pB); S = EA @ EB.T
    r, c = linear_sum_assignment(-S); order = np.argsort(-S[r, c])[:N_PAIRS]; r, c = r[order], c[order]
    iA = {f: k for k, f in enumerate(pA)}; iB = {f: k for k, f in enumerate(pB)}
    unmatched = S[np.ix_([iA[f] for f in tA if f in iA], [iB[f] for f in tB if f in iB])]
    for body, files, key in (("A", [pA[i] for i in r], "cmA_a0"), ("B", [pB[j] for j in c], "cmB_a0")):
        d = os.path.join(T1, "train_png", key); os.makedirs(d, exist_ok=True)
        for k, f in enumerate(files):
            Image.fromarray(np.clip(np.rint(L.load_rgb_crop(f)), 0, 255).astype(np.uint8)).save(os.path.join(d, f"{k:04d}.png"), compress_level=1)
    sims = S[r, c]
    rep = {"pool_sizes": [len(pA), len(pB)], "n_pairs": N_PAIRS, "pair_cos_mean": float(sims.mean()), "pair_cos_min": float(sims.min()),
           "pairs": [[os.path.basename(pA[i]), os.path.basename(pB[j]), float(S[i, j])] for i, j in zip(r, c)],
           "reference_unmatched_T_split_best_match_mean": float(unmatched.max(1).mean()) if unmatched.size else None,
           "reference_unmatched_T_split_diagonal_free_mean": float(unmatched.mean()) if unmatched.size else None,
           "overlap_with_primary_T_split": [int(sum(pA[i] in set(tA) for i in r)), int(sum(pB[j] in set(tB) for j in c))]}
    json.dump(rep, open(os.path.join(T1, "content_match.json"), "w"), indent=1)
    print({k: v for k, v in rep.items() if k != "pairs"}, flush=True)

if __name__ == "__main__":
    main()
