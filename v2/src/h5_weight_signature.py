"""H5 (RESULTS.md Entry 94) - is the training body recoverable from the adapter weights themselves?

Every limit in this study is measured on generated images. This asks the same question of the weights: do
two adapters trained on the same body's photographs resemble each other more than two trained on different
bodies? Six adapters per body at each of two doses, on archived weights, no GPU.

The functional update of a LoRA layer is dW = B A. The factorisation is seed-dependent and not comparable
across adapters, but dW is, and the cosine between two adapters' concatenated updates is computed exactly
without ever forming dW: for rank r,

    <B_i A_i, B_j A_j>_F = trace(A_i^T B_i^T B_j A_j) = trace((B_i^T B_j)(A_j A_i^T)),

two r x r products per layer. The statistic is the mean same-body cosine minus the mean different-body
cosine, tested against the exact permutation null over balanced relabellings of body.

A positive result says the weights carry the identity of the training *set*, of which the camera is one
component; it cannot separate the camera from the photographs' content. Writes out/h5_weight_signature.json.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2
ADA = os.path.join(V2, "out", "t1", "adapters")
OUT = os.path.join(V2, "out", "h5_weight_signature.json")
DOSES = {
    "2000": {"A": [f"nomark_s{i}" for i in range(6)], "B": [f"nomarkB_s{i}" for i in range(6)]},
    "16000": {"A": [f"dose16k_A_s{i}" for i in range(6)], "B": [f"dose16k_B_s{i}" for i in range(6)]},
}


def load_pairs(tag):
    """{layer: (A, B)} for one adapter, as float64 numpy."""
    from safetensors.torch import load_file
    d = load_file(os.path.join(ADA, tag, "pytorch_lora_weights.safetensors"))
    out = {}
    for k, v in d.items():
        if k.endswith("lora_A.weight"):
            base = k[: -len("lora_A.weight")]
            b = d.get(base + "lora_B.weight")
            if b is not None:
                # weights are stored in bfloat16, which numpy cannot view; promote in torch first
                out[base] = (v.float().numpy().astype(np.float64),
                             b.float().numpy().astype(np.float64))
    return out


def gram(x, y):
    """<dW_x, dW_y>_F summed over the layers both share, without forming dW."""
    tot = 0.0
    for layer, (Ax, Bx) in x.items():
        if layer not in y: continue
        Ay, By = y[layer]
        tot += float(np.trace((Bx.T @ By) @ (Ay @ Ax.T)))
    return tot


def main():
    t0 = time.time()
    res = {"entry": "RESULTS.md Entry 94 (H5)", "doses": {}}
    for dose, arms in DOSES.items():
        tags = arms["A"] + arms["B"]
        missing = [t for t in tags if not os.path.exists(os.path.join(ADA, t, "pytorch_lora_weights.safetensors"))]
        if missing:
            print(f"[h5] dose {dose}: missing {missing}, skipped", flush=True); continue
        W = {t: load_pairs(t) for t in tags}
        n = len(tags)
        print(f"[h5] dose {dose}: {n} adapters, {len(next(iter(W.values())))} layers "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

        norm = {t: np.sqrt(gram(W[t], W[t])) for t in tags}
        C = np.eye(n)
        for i, j in itertools.combinations(range(n), 2):
            c = gram(W[tags[i]], W[tags[j]]) / (norm[tags[i]] * norm[tags[j]])
            C[i, j] = C[j, i] = c

        labels = np.array([0] * 6 + [1] * 6)          # index 0-5 body A, 6-11 body B
        iu = np.triu_indices(n, 1)
        cos = C[iu]

        def stat(lab):
            same = lab[iu[0]] == lab[iu[1]]
            return float(cos[same].mean() - cos[~same].mean())

        obs = stat(labels)
        # exact permutation over balanced relabellings; each split and its complement are one assignment
        perms = []
        for combo in itertools.combinations(range(n), 6):
            if 0 not in combo: continue               # fix adapter 0 to body A, removing the complement
            lab = np.ones(n, int); lab[list(combo)] = 0
            perms.append(stat(lab))
        perms = np.array(perms)
        p = float((perms >= obs).mean())

        # descriptive: does weight space cluster by seed group instead of by body? Seeds 0-2 and 3-5 of
        # each arm were trained in separate batches, so this asks whether batch explains the structure.
        grp = np.array([0 if int(t.split("_s")[-1]) < 3 else 1 for t in tags])
        gsame = grp[iu[0]] == grp[iu[1]]
        seed_group = {"mean_cos_same_batch": float(cos[gsame].mean()),
                      "mean_cos_diff_batch": float(cos[~gsame].mean()),
                      "D_batch": float(cos[gsame].mean() - cos[~gsame].mean())}
        print(f"[h5] dose {dose}: by batch - same {seed_group['mean_cos_same_batch']:+.4f} "
              f"diff {seed_group['mean_cos_diff_batch']:+.4f} D {seed_group['D_batch']:+.4f}", flush=True)

        same_mask = labels[iu[0]] == labels[iu[1]]
        rec = {"adapters": tags, "n_pairs": int(len(cos)), "n_permutations": int(len(perms)),
               "mean_cos_same_body": float(cos[same_mask].mean()),
               "mean_cos_diff_body": float(cos[~same_mask].mean()),
               "D": obs, "perm_p_one_sided": p, "perm_floor": 1.0 / len(perms),
               "perm_mean": float(perms.mean()), "perm_sd": float(perms.std(ddof=1)),
               "z_vs_perm": float((obs - perms.mean()) / perms.std(ddof=1)),
               "mean_cos_all": float(cos.mean()), "cosine_matrix": C.tolist(),
               "batch_diagnostic_descriptive": seed_group,
               "reading": ("the training body is recoverable from the adapter weights"
                           if p < 0.01 and obs > 0 else
                           "the adapter weights carry no detectable body signature at this dose")}
        res["doses"][dose] = rec
        print(f"[h5] dose {dose}: same {rec['mean_cos_same_body']:+.4f} diff {rec['mean_cos_diff_body']:+.4f} "
              f"D {obs:+.4f}  p {p:.4f} (floor {rec['perm_floor']:.4f}, z {rec['z_vs_perm']:+.2f})", flush=True)
        print(f"[h5] dose {dose}: {rec['reading']}", flush=True)

    res["caveat"] = ("the two bodies' adapters are trained on different photographs, so a positive result "
                     "attributes the training set, of which the camera is one component; it cannot "
                     "separate the camera from the photographs' content")
    res["runtime_min"] = (time.time() - t0) / 60
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"[h5] written {OUT}", flush=True)


if __name__ == "__main__":
    main()
