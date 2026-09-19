"""Entry 58, C8 — learned detector: texture or content.

The saved learned detector (out/t2_learned_net.pt) applied to the 24 primary adapters' generations (first
250 each; A_raw_s0..s11_r16, B_raw_s0..s11_r16) twice from the same residual:
  (i)  original patches, as t2_learned_ext.py (reproduction of its interaction, +0.381 logits);
       scored both from the float32 residual and from the float16-rounded residual (t2_learned.resid's
       cache format: Entry 17 read s0-s2 from the Entry 16 cache and computed s3-s11 fresh), so the
       exact Entry 17 path is fp16 for s0-s2 and fp32 for s3-s11;
  (ii) each 256x256 patch cut into 16x16 blocks, blocks permuted randomly (seeded per image and patch;
       destroys spatial arrangement, keeps local texture statistics).
Interaction theta_sym and the exact C(24,12) label permutation p for both. Also the held-out real H
photographs (40 per body) under both, as a descriptive check that the block-shuffled input still carries
the body difference in real photographs (not part of the registered reading).

Registered readings (Entry 58, C8): "texture" if the block-shuffled interaction stays >= 50 % of the
original with exact permutation p < 0.05; "spatial structure needed" if it falls below 25 %; otherwise
"unresolved". Content split: Entry 53 labels (out/t6_weapon_clip.json) — used only where they cover the
scored images.

CPU only, 3 worker processes, nothing cached.  Output: out/c8_learned_texture.json
  python c8_learned_texture.py [--n 250] [--out ...] [--workers 3] [--threads 2]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time, math, argparse, multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import t2_learned_arms as S

V2 = EINV.V2
DATA = EINV.DATA                                # the archived primary generations; set EINV_DATA
A_TAGS = [f"A_raw_s{s}_r16" for s in range(12)]; B_TAGS = [f"B_raw_s{s}_r16" for s in range(12)]
LABELS = os.path.join(V2, "out", "t6_weapon_clip.json")


def arm_dir(tag):
    s = int(tag.split("_s")[1].split("_")[0])
    return os.path.join(DATA, "gens" if s <= 2 else "gens_ext", tag)


def real_H():
    from fingerprints import splits, dv
    A, B = splits(dv["Nikon_D200_1"]), splits(dv["Nikon_D200_0"])
    return [(f, 0) for f in A["H"]] + [(f, 1) for f in B["H"]]


def auc_R(sc, lab):
    sc, lab = np.asarray(sc), np.asarray(lab); pos, neg = sc[lab == 0], sc[lab == 1]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    return auc, float(pos.mean() - neg.mean())


def reading(orig, shuf):
    ratio = shuf["theta_sym"] / orig["theta_sym"]; p = shuf["label_permutation_p_one_sided"]
    if ratio >= 0.5 and p < 0.05: r = "texture"
    elif ratio < 0.25: r = "spatial structure needed"
    else: r = "unresolved"
    return float(ratio), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250); ap.add_argument("--out", default=os.path.join(V2, "out", "c8_learned_texture.json"))
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--skip-real", action="store_true")
    a = ap.parse_args(); assert a.workers <= 3
    t0 = time.time(); log = lambda m: print(m, flush=True)
    ext = json.load(open(S.EXT_JSON))
    res = {"script": os.path.abspath(__file__), "entry": "58 C8", "model": S.NET_PATH, "device": "cpu", "n_per_arm": a.n,
           "block": S.BLK, "patch": S.P, "patches_per_image": S.NP, "shuffle_seed": "[58, crc32(arm tag), image index]; one permutation per patch"}

    # real held-out photographs (positive control, descriptive)
    if not a.skip_real:
        ho = real_H()
        with mp.get_context("spawn").Pool(a.workers, initializer=S.init_worker, initargs=(a.threads,)) as pool:
            out = pool.map(S.score_task, [(f, f"realH_{lab}", i, ("orig", "orig_fp16", "shuffle")) for i, (f, lab) in enumerate(ho)], chunksize=2)
        labs = [l for _, l in ho]; res["real_H"] = {"n": len(ho)}
        for v in ("orig", "orig_fp16", "shuffle"):
            auc, R = auc_R([o[v] for o in out], labs); res["real_H"][v] = {"auc": auc, "R": R}
        res["real_H"]["entry17_reference"] = {"auc": ext["real_H_auc"], "R": ext["real_paired_contrast_R"]}
        log(f"[c8] real H: {json.dumps(res['real_H'])}  ({(time.time()-t0)/60:.1f} min)")

    dirs = {t: arm_dir(t) for t in A_TAGS + B_TAGS}
    variants = ["orig", "orig_fp16", "shuffle"]
    scores, files_of = S.score_arms(dirs, a.n, variants, a.workers, a.threads, log)
    R = ext["real_paired_contrast_R"]
    # the exact Entry 17 path: fp16 (cache hit) for s0-s2, fp32 (cache miss) for s3-s11
    e17 = {t: scores[t]["orig_fp16"] if int(t.split("_s")[1].split("_")[0]) <= 2 else scores[t]["orig"] for t in dirs}
    arms = {}
    for t in dirs:
        arms[t] = {"dir": dirs[t], **{f"mean_{v}": float(scores[t][v].mean()) for v in variants},
                   "mean_entry17_path": float(e17[t].mean()), "se_orig": float(scores[t]["orig"].std(ddof=1) / math.sqrt(a.n)),
                   "se_shuffle": float(scores[t]["shuffle"].std(ddof=1) / math.sqrt(a.n)),
                   "entry17_mean": ext["arms"][t]["mean"] if a.n == ext["G_per_arm"] else None}
    res["arms"] = arms
    st = lambda f: S.interaction_stats([f(t) for t in A_TAGS], [f(t) for t in B_TAGS], R)
    res["stats_orig"] = st(lambda t: scores[t]["orig"].mean())
    res["stats_orig_fp16"] = st(lambda t: scores[t]["orig_fp16"].mean())
    res["stats_entry17_path"] = st(lambda t: e17[t].mean())
    res["stats_shuffle"] = st(lambda t: scores[t]["shuffle"].mean())
    d = np.array([arms[t]["mean_entry17_path"] - ext["arms"][t]["mean"] for t in dirs]) if a.n == ext["G_per_arm"] else None
    res["reproduction"] = {"entry17_theta_sym": ext["theta_sym"], "entry17_label_perm_p": ext["label_permutation_p_one_sided"],
                           "entry17_SE": ext["adapter_level_SE"],
                           "this_run_theta_sym_orig_fp32": res["stats_orig"]["theta_sym"],
                           "this_run_theta_sym_entry17_path": res["stats_entry17_path"]["theta_sym"],
                           "per_arm_mean_abs_diff_vs_entry17": None if d is None else float(np.abs(d).mean()),
                           "per_arm_max_abs_diff_vs_entry17": None if d is None else float(np.abs(d).max()),
                           "note": "Entry 17 ran on GPU (cuDNN) and read s0-s2 residuals from the float16 cache; this run is CPU."}
    ratio, rd = reading(res["stats_orig"], res["stats_shuffle"])
    ratio17, rd17 = reading(res["stats_entry17_path"], res["stats_shuffle"])
    res["C8_reading"] = {"rule": "texture if shuffled theta_sym >= 50% of original and exact permutation p < 0.05; "
                                 "spatial structure needed if < 25%; otherwise unresolved",
                         "original_theta_sym": res["stats_orig"]["theta_sym"], "shuffled_theta_sym": res["stats_shuffle"]["theta_sym"],
                         "ratio": ratio, "shuffled_perm_p": res["stats_shuffle"]["label_permutation_p_one_sided"], "reading": rd,
                         "ratio_vs_entry17_path_original": ratio17, "reading_vs_entry17_path_original": rd17}

    # content split: Entry 53 labels
    lab = json.load(open(LABELS))
    covered_dirs = {os.path.normcase(v["path"]) for v in lab["folders"].values()}
    cov = {t: os.path.normcase(dirs[t]) in covered_dirs for t in dirs}
    res["content_split"] = {"label_source": LABELS, "labelled_folders": {k: v["path"] for k, v in lab["folders"].items()},
                            "primary_arms_covered": sum(cov.values()), "possible": all(cov.values())}
    if all(cov.values()):
        res["content_split"].update(S.content_split({t: scores[t]["orig"] for t in dirs}, files_of, A_TAGS, B_TAGS, LABELS, log))
    else:
        res["content_split"]["note"] = ("Entry 53 labels cover only out/t1/gens (local_*, nomark*, nomarkB*) and out/t5/gens "
                                        "(five-caption bank). None of the 24 primary archived generation folders is labelled "
                                        "(t5/A_raw_s0_r16 and t5/B_raw_s0_r16 are different, five-caption generations), so the "
                                        "registered split of the primary interaction cannot be made from existing labels.")
    res["per_image"] = {t: {v: [round(float(x), 5) for x in scores[t][v]] for v in variants} for t in dirs}
    res["runtime_min"] = (time.time() - t0) / 60
    json.dump(res, open(a.out, "w"), indent=1)
    for k in ("stats_orig", "stats_entry17_path", "stats_shuffle"):
        s = res[k]
        log(f"[c8] {k:20s} theta_sym {s['theta_sym']:+.4f} SE {s['adapter_level_SE']:.4f} t {s['t']:.3f} p_t {s['p_t_one_sided']:.4g} "
            f"perm p {s['label_permutation_p_one_sided']:.4g} additive {s['additive_part']:+.3f}")
    log(f"[c8] reading: {json.dumps(res['C8_reading'])}")
    log(f"[c8] reproduction: {json.dumps(res['reproduction'])}")
    log(f"[c8] written {a.out} ({res['runtime_min']:.1f} min)")


if __name__ == "__main__":
    main()
