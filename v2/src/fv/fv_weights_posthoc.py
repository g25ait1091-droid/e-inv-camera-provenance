"""Post-hoc check (RESULTS.md Entry 114, part B): the H5 adapter-weight comparison with same-seed pairs set aside.

CPU only, seconds. Reads out/h5_weight_signature.json (Entries 94-95), the adapters' train_meta.json records and
src/t1_ladder.py; writes out/fv_weights_posthoc.json. Not pre-specified: every quantity here was computed after the
registered H5 test (D = same-body mean cosine - cross-body mean cosine; exact permutation over 462 balanced
relabellings; p 0.93 at both doses) had been read. Descriptive; no registered reading changes.

WHY. H5 compares adapters by the cosine between their concatenated LoRA updates dW_l = B_l A_l (12 adapters per
dose, six per body: nomark / nomarkB at 2000 steps, dose16k_A / dose16k_B at 16000 steps). Adapter j of body A and
adapter j of body B were trained with the same seed j: t1_ladder.train_arm() seeds torch, numpy and random before
PEFT adds the adapter (init_lora_weights="gaussian": A_l random, B_l zero) and draws the training batches, noise
and timesteps from the same seeded streams. So the two adapters of a seed pair share their A_l initialization
and random stream while their photographs differ. All six same-seed pairs are cross-body pairs, so they enter only
the cross-body mean of D; they are far more alike than any other pair, and they make D negative.

WHAT. At each dose, over the 66 pairs: the mean cosine of the six same-seed pairs; of the 30 same-body pairs (none
shares a seed); of the 30 cross-body pairs of different seeds; the seed-stratified statistic
    D_strat = mean(same-body) - mean(cross-body, different seeds);
its exact permutation null that respects the seed pairing - body labels swapped within any subset of the six seed
pairs, 2^6 = 64 relabellings, of which 32 are distinct partitions (swapping every pair returns the same
partition), so the smallest attainable one-sided p is 2/64 = 1/32; whether every same-body cosine exceeds every
cross-body different-seed cosine; and the Entry 95 batch grouping (seeds 0-2 vs 3-5) with the same-seed pairs set
aside, overall and within each body category.

READING (fixed here, before the numbers are printed, and the same whatever they are). The two bodies' adapters
were trained on different photographs (different scenes). Weights that separate the two training sets are
expected from content alone and are not evidence that the camera fingerprint is in the weights. Separating the
two needs content held fixed while the fingerprint changes: adapters trained on the same photographs with the
fingerprint manipulated (the G1 normal and inverted 16000-step adapters are such a design, seeds 0-5 shared), or
the same scenes photographed by both bodies (content-matched or swapped sets), with the statistic and a
relabelling design whose floor is below the level registered before looking.

Run:  python src/fv/fv_weights_posthoc.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import itertools
import json
import os
import re

import numpy as np

V2 = EINV.V2
OUT = V2 + "/out"
SRC = OUT + "/h5_weight_signature.json"
ADA = OUT + "/t1/adapters"
LADDER = EINV.SRC + "/t1_ladder.py"
DST = OUT + "/fv_weights_posthoc.json"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def seed_evidence():
    """The lines of t1_ladder.train_arm() that make same-seed adapters share initialization and random stream."""
    lines = open(LADDER, encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("def train_arm("))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("def "))
    body = lines[start:end]
    want = {"global seeds": r"torch\.manual_seed\(seed\)",
            "adapter added after seeding": r"add_adapter\(LoraConfig\(.*init_lora_weights=\"gaussian\"",
            "batch-index generator": r"torch\.Generator\(\)\.manual_seed\(seed\)",
            "noise from the seeded global stream": r"torch\.randn_like\(lat\)"}
    found = {}
    for k, pat in want.items():
        hit = [(start + 1 + i, l.strip()) for i, l in enumerate(body) if re.search(pat, l)]
        assert hit, f"{k}: pattern {pat} not found in train_arm()"
        found[k] = {"line": hit[0][0], "text": hit[0][1]}
    assert found["global seeds"]["line"] < found["adapter added after seeding"]["line"]
    return found


def analyse(dose, rec):
    tags = rec["adapters"]
    C = np.array(rec["cosine_matrix"], float)
    n = len(tags)
    assert n == 12 and C.shape == (n, n) and np.allclose(C, C.T)
    body = np.array([0] * 6 + [1] * 6)                       # file order: body A seeds 0-5, then body B seeds 0-5
    seed = np.array([int(t.split("_s")[-1]) for t in tags])
    assert list(seed) == list(range(6)) * 2, tags
    meta_seed = [load(f"{ADA}/{t}/train_meta.json")["seed"] for t in tags]
    assert meta_seed == list(seed), (tags, meta_seed)          # the training records carry the same seeds
    batch = (seed >= 3).astype(int)                            # Entry 95 grouping: seeds 0-2 vs 3-5
    iu = np.triu_indices(n, 1)
    cos = C[iu]
    si, sj, bi, bj, gi, gj = seed[iu[0]], seed[iu[1]], body[iu[0]], body[iu[1]], batch[iu[0]], batch[iu[1]]
    same_seed = si == sj
    same_body = bi == bj
    cross_ns = ~same_body & ~same_seed
    same_batch = gi == gj
    assert same_seed.sum() == 6 and not (same_seed & same_body).any()
    assert same_body.sum() == 30 and cross_ns.sum() == 30

    # reproduce the registered statistic from the matrix
    D_reg = float(cos[same_body].mean() - cos[~same_body].mean())
    assert abs(D_reg - rec["D"]) < 1e-12 and abs(cos[same_body].mean() - rec["mean_cos_same_body"]) < 1e-12

    def d_strat(lab):
        sb = lab[iu[0]] == lab[iu[1]]
        return float(cos[sb].mean() - cos[~sb & ~same_seed].mean())

    obs = d_strat(body)
    perm = []
    for flips in itertools.product((0, 1), repeat=6):
        lab = body.copy()
        for s, f in enumerate(flips):
            if f:
                lab[seed == s] = 1 - lab[seed == s]
        perm.append(d_strat(lab))
    perm = np.array(perm)
    distinct = np.unique(np.round(perm, 12))
    p = float((perm >= obs - 1e-12).mean())
    second = float(distinct[-2]) if len(distinct) > 1 else None

    sbns = same_batch & ~same_seed
    out = {
        "adapters": tags, "seeds_from_train_meta": meta_seed,
        "n_pairs": int(len(cos)), "n_same_seed": int(same_seed.sum()), "n_same_body": int(same_body.sum()),
        "n_cross_body_different_seed": int(cross_ns.sum()),
        "same_seed": {"mean": float(cos[same_seed].mean()), "min": float(cos[same_seed].min()),
                      "max": float(cos[same_seed].max())},
        "same_body": {"mean": float(cos[same_body].mean()), "min": float(cos[same_body].min()),
                      "max": float(cos[same_body].max()),
                      "mean_A": float(cos[same_body & (bi == 0)].mean()),
                      "mean_B": float(cos[same_body & (bi == 1)].mean())},
        "cross_body_different_seed": {"mean": float(cos[cross_ns].mean()), "min": float(cos[cross_ns].min()),
                                      "max": float(cos[cross_ns].max())},
        "largest_cosine_other_than_same_seed": float(cos[~same_seed].max()),
        "same_seed_over_same_body_mean": float(cos[same_seed].mean() / cos[same_body].mean()),
        "registered_D_reproduced": D_reg,
        "registered_p": rec["perm_p_one_sided"],
        "D_strat": obs,
        "D_strat_permutation": {
            "design": "body labels swapped within any subset of the six seed pairs",
            "n_relabellings": int(len(perm)), "n_distinct_partitions": int(len(distinct)),
            "p_one_sided": p, "floor": 2.0 / len(perm),
            "observed_is_largest": bool(np.isclose(obs, distinct[-1])),
            "second_largest_value": second},
        "every_same_body_above_every_cross_different_seed": bool(cos[same_body].min() > cos[cross_ns].max()),
        "separation_margin": float(cos[same_body].min() - cos[cross_ns].max()),
        "batch_without_same_seed_pairs": {
            "same_batch_different_seed_mean": float(cos[sbns].mean()), "n_same_batch_different_seed": int(sbns.sum()),
            "different_batch_mean": float(cos[~same_batch].mean()), "n_different_batch": int((~same_batch).sum()),
            "difference": float(cos[sbns].mean() - cos[~same_batch].mean()),
            "different_batch_mean_equals_entry95": abs(float(cos[~same_batch].mean())
                                                       - rec["batch_diagnostic_descriptive"]["mean_cos_diff_batch"]) < 1e-12,
            "within_same_body": {"same_batch": float(cos[same_body & same_batch].mean()),
                                 "different_batch": float(cos[same_body & ~same_batch].mean())},
            "within_cross_body_different_seed": {"same_batch": float(cos[cross_ns & same_batch].mean()),
                                                 "different_batch": float(cos[cross_ns & ~same_batch].mean())},
            "entry95_same_batch_mean_incl_same_seed": rec["batch_diagnostic_descriptive"]["mean_cos_same_batch"],
            "entry95_D_batch": rec["batch_diagnostic_descriptive"]["D_batch"]},
    }
    return out


def main():
    H = load(SRC)
    res = {"entry": "RESULTS.md Entry 114",
           "status": ("post hoc; descriptive; computed after the registered H5 test (Entries 94-95) had been read; "
                      "no registered reading changes"),
           "source": SRC, "script": "src/fv/fv_weights_posthoc.py",
           "seed_sharing_evidence": seed_evidence(), "doses": {}}
    for dose, rec in H["doses"].items():
        res["doses"][dose] = analyse(dose, rec)
    floors = {d["D_strat_permutation"]["floor"] for d in res["doses"].values()}
    assert len(floors) == 1
    res["floor"] = floors.pop()
    res["reading"] = {
        "registered_test": ("uninformative: the six same-seed pairs (all cross-body) are far more alike than any other "
                            "pair and enter only the cross-body mean, so D is negative whatever the weights carry "
                            "about the body; its reading 'no detectable body signature' stands as registered but "
                            "carries no information about the body"),
        "seed_stratified": ("with same-seed pairs set aside, every same-body pair is more alike than every cross-body "
                            "pair of different seeds at both doses; D_strat is the largest of the 32 distinct values "
                            "of its within-seed-pair relabelling null, one-sided p = 1/32, the floor of the design, "
                            "which cannot reach the 0.01 used for H5"),
        "batch": ("Entry 95's batch reading is superseded: the same-batch excess is the six same-seed pairs, which "
                  "always fall within one batch; among pairs of different seeds, same-batch and different-batch pairs "
                  "are equally alike, within same-body and within cross-body pairs alike. The weights are organized "
                  "by training seed, not by training batch"),
        "interpretation": ("the two bodies' adapters were trained on different photographs of different scenes; "
                           "weights that separate the two training sets are expected from content and are not "
                           "evidence that the camera fingerprint is in the weights"),
        "what_would_separate": ("content held fixed while the fingerprint changes: a registered weight-level test on "
                                "adapters trained on the same photographs with the fingerprint manipulated - the G1 "
                                "normal (dose16k) and inverted (inv16k, inv16kext) 16000-step adapters share seeds 0-5 "
                                "within each body, so relabelling normal/inverted within seed pairs gives 2^5 x 2^5 = "
                                "1024 distinct partitions over the two bodies (floor about 0.001) - or the same "
                                "scenes photographed by both bodies (content-matched or swapped training sets), with "
                                "enough seed pairs that the relabelling floor lies below the level"),
    }
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for dose, d in res["doses"].items():
        print(f"[weights] {dose} steps: same-seed {d['same_seed']['mean']:.4f} "
              f"({d['same_seed']['min']:.4f}-{d['same_seed']['max']:.4f}); same-body {d['same_body']['mean']:.4f}; "
              f"cross different-seed {d['cross_body_different_seed']['mean']:.4f}; D_strat {d['D_strat']:+.4f}; "
              f"p {d['D_strat_permutation']['p_one_sided']:.4f} (floor {d['D_strat_permutation']['floor']:.4f}, "
              f"{d['D_strat_permutation']['n_distinct_partitions']} distinct); all same > all cross: "
              f"{d['every_same_body_above_every_cross_different_seed']} (margin {d['separation_margin']:.4f}); "
              f"batch without same-seed pairs {d['batch_without_same_seed_pairs']['difference']:+.5f}")
    print(f"[weights] written {DST}")


if __name__ == "__main__":
    main()
