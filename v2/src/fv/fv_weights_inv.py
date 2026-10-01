"""Pre-specified test (RESULTS.md Entry 116, item 4): the normal-versus-inverted adapter-weight test at 16000 steps.

Do the LoRA weights distinguish training on a body's photographs with its fingerprint reversed from training on the
same photographs unchanged? CPU only, well under a minute. Writes out/fv_weights_inv.json and nothing else.

FILE NAMES. Entry 116 names this script src/fv/fv_weights_invert.py and its output out/fv_weights_invert.json. The run
that executed it was instructed to write src/fv/fv_weights_inv.py -> out/fv_weights_inv.json. Only the names differ;
the analysis is the one registered in Entry 116 item 4.

READS
  out/t1/adapters/<tag>/pytorch_lora_weights.safetensors and train_meta.json for the 28 adapters below;
  src/h5_weight_signature.py (load_pairs, gram: imported, not re-implemented);
  out/h5_weight_signature.json (reproduction check of the normal adapters' cosines);
  out/t1/invert.json and out/t1/train_png/{none,noneB,invA,invB}_a0 (design-fact check: same photographs, same index);
  logs/t1_{dose16k,dose16krep,dose16krep2,inv16k,inv16kext}_train.log (design-fact check: shared batch/noise stream).

ADAPTERS (Entry 116 item 4; all 16000 steps, rank 16, 14,587,296 bytes each)
  normal   A: dose16k_A_s0..s5 (field none)      normal   B: dose16k_B_s0..s5 (field noneB)
  inverted A: inv16k_A_s0..s2, inv16kext_A_s3..s7 (field invA)
  inverted B: inv16k_B_s0..s2, inv16kext_B_s3..s7 (field invB)
  dose16k_*_s6 and _s7 do not exist (asserted).

EXTRACTION. dW_l = B_l A_l per layer; the cosine between two adapters is the cosine between their concatenated dW,
computed exactly (never forming dW) with h5_weight_signature.gram: <dW_x, dW_y> = sum_l trace((B_x^T B_y)(A_y A_x^T)).

STATISTIC (primary, registered). For body X, over seeds 0-5 only:
  D_X = mean cosine over the 30 within-condition different-seed pairs (15 normal-normal, 15 inverted-inverted)
      - mean cosine over the 30 cross-condition different-seed pairs (N_j, I_k), j != k;
  the six same-seed pairs (N_j, I_j) and the inverted seeds 6-7 are excluded; D = (D_A + D_B) / 2.

NULL AND LEVEL (registered). Exact relabelling within seed pairs: for each body and each seed j the labels of N_j and
I_j are kept or exchanged. Exchanging all six in one body maps the partition to itself (D_X is unchanged), so each
body has 32 distinct relabellings and the two bodies 1,024. One-sided p = fraction of the 1,024 with D >= observed,
identity included; floor 1/1024. Level 0.01, one-sided, D > 0. A relabelled value counts as >= the observed one if
it is at least the observed value minus 1e-12 (ties counted against the hypothesis).

READINGS (registered, Entry 116):
  p < 0.01 with D > 0 -> "the adapter weights carry the fingerprint's sign" (they distinguish training on the
    photographs with the fingerprint reversed from training on the same photographs unchanged: a white-box trace of
    the stored fingerprint change). Ceiling fixed in advance: the two conditions also differ by the fixed dither
    field and by training date, so a positive result does not isolate the fingerprint pattern from the dither, and
    it says nothing about whether a natural-amplitude fingerprint is recoverable from standard 2000-step adapters.
  p >= 0.01 -> "the weights do not detectably carry the fingerprint's sign", at six seed pairs per body and 16000 steps.
  (p < 0.01 with D <= 0 is not among the registered readings; the script would say so rather than pick one.)

DESCRIPTIVE (registered as descriptive, not tested). Interpretation choices fixed here before any number was computed:
  - per-body D_X and exact p over its 32 distinct relabellings (floor 1/32);
  - the six twin cosines cos(N_j, I_j) per body;
  - matched variant: twin differences Delta_j = dW(I_j) - dW(N_j); the statistic is the mean cosine over the 15
    within-body pairs of Delta's in each body, averaged over the two bodies (= the mean over the 30 within-body
    pairs). Exchanging seed pair j flips the sign of Delta_j. Only this within-body form is unchanged when all six
    labels of one body are exchanged, which is what "the same 1,024 partitions" requires; cross-body pairs of
    Delta's are reported separately as diagnostic 4. Delta inner products are computed with gram() on the exact
    rank-32 factorisation Delta_j = [B_I, -B_N] [A_I; A_N] and checked against the Gram-matrix expansion;
  - D with the inverted seeds 6-7 kept at their fixed (inverted) label: within = the pooled mean over all same-label
    different-seed pairs (15 N-N + 28 I-I = 43), cross = the pooled mean over the 42 different-label different-seed
    pairs; 64 distinct relabellings per body, 4,096 over both;
  - adapter-weight norms by condition: the Frobenius norm of the concatenated dW (sqrt gram(x, x)) and the
    train_meta lora_B_norm (the Entry 89 / 110 covariate).

DIAGNOSTICS NOT IN THE PRE-SPECIFICATION (fixed here before any number was computed; no reading attaches):
  1. crop correspondence: inverted crop i against normal crop i reproduces out/t1/invert.json change_rms_gray, and
     against normal crop i+1 it is far larger (the index-aligned batch draws show the same photographs);
  2. shared stream: from the training logs, the correlation of the first differences of the 80 logged mean losses
     (every 200 steps) between twins, against different-seed pairs, and the twins' mean loss offset (I - N);
  3. training chain within condition: same-chain against different-chain pairs of different seeds (chains are the
     t1_ladder armsets: dose16k {0}, dose16krep {1,2}, dose16krep2 {3,4,5}; inv16k {0,1,2}, inv16kext {3..7});
  4. cross-body alignment of twin differences: the mean cosine between Delta^A_j and Delta^B_k over the 30 pairs
     j != k (and the 6 with j = k), with its within-seed-pair relabelling p (4,096 relabellings, 2,048 distinct);
     a common direction in both bodies' twin differences would be a weight change generic to the inverted
     condition (dither, any added high-frequency pattern) rather than specific to one body's fingerprint;
  5. numerical checks: the normal adapters' cosines reproduce out/h5_weight_signature.json; the Delta inner products
     agree between factorisation and expansion; p over the 1,024 distinct relabellings equals p over all 4,096.

POST-HOC CONTEXT (added after a scratch run of this script had shown the registered result; no reading attaches):
  a. how alike the two reversed patterns are in pixel space: NCC(K_A^E1, K_B^E1), beside gates.json kappa_model (E2);
  b. whether the training stack changed between the two training periods: install times of the packages that train
     (dist-info folders of the interpreter the chains call as `python`, which is this interpreter);
  c. the step-200 mean loss of seed-0 runs from the training logs: unchanged crops on different days and schedules
     (nomark, dose8k, dose16k), the inverted twins, and body-A arms that add another fine pattern to the same crops
     (mark_rand, kinj, kinjd; Entries 01, 59). At step 200 the adapter has barely moved, so the loss compares the
     crop sets under one random stream.

Run:  python src/fv/fv_weights_inv.py              (writes out/fv_weights_inv.json; refuses to overwrite)
      python src/fv/fv_weights_inv.py --selftest   (checks the relabelling code on synthetic matrices; writes nothing)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import argparse
import datetime
import hashlib
import itertools
import json
import os
import platform
import re
import sys
import time

import numpy as np

V2 = EINV.V2
os.environ["EINV_V2"] = V2            # h5_weight_signature resolves its adapter folder through einv_paths at import
SRC = EINV.SRC + ""
OUT = V2 + "/out"
ADA = OUT + "/t1/adapters"
DST = OUT + "/fv_weights_inv.json"
H5_JSON = OUT + "/h5_weight_signature.json"
INVERT_JSON = OUT + "/t1/invert.json"
TRAIN_PNG = OUT + "/t1/train_png"
LOGS = V2 + "/logs"
TRAIN_LOGS = ["t1_dose16k_train.log", "t1_dose16krep_train.log", "t1_dose16krep2_train.log",
              "t1_inv16k_train.log", "t1_inv16kext_train.log"]
H5_SCRIPT = SRC + "/h5_weight_signature.py"
THIS = os.path.abspath(__file__)

LEVEL = 0.01
TOL = 1e-12
FILE_BYTES = 14587296
PAIRED = list(range(6))                       # seeds with both a normal and an inverted adapter in each body
SEEDS = {"N": list(range(6)), "I": list(range(8))}
FIELD = {("A", "N"): "none", ("B", "N"): "noneB", ("A", "I"): "invA", ("B", "I"): "invB"}
DEVICE = {"A": "Nikon_D200_1", "B": "Nikon_D200_0"}
CHAIN = {"N": {0: "dose16k", 1: "dose16krep", 2: "dose16krep", 3: "dose16krep2", 4: "dose16krep2", 5: "dose16krep2"},
         "I": {0: "inv16k", 1: "inv16k", 2: "inv16k", 3: "inv16kext", 4: "inv16kext", 5: "inv16kext",
               6: "inv16kext", 7: "inv16kext"}}
CROPS = {"A": ("none_a0", "invA_a0"), "B": ("noneB_a0", "invB_a0")}

READ_POS = "the adapter weights carry the fingerprint's sign"
READ_NEG = "the weights do not detectably carry the fingerprint's sign"
CEILING = ("in the precise sense that they distinguish training on the photographs with the fingerprint reversed from "
           "training on the same photographs unchanged: a white-box trace of the stored fingerprint change. The two "
           "conditions also differ by the fixed dither field and by training date, so a positive result does not "
           "isolate the fingerprint pattern from the dither. It says nothing about whether a natural-amplitude "
           "fingerprint is recoverable from standard 2000-step adapters. The date is not expected to matter: among "
           "different-seed pairs, batches ten days apart differ by -0.0001 (Entry 114); cited, not tested.")


def tag(body, cond, seed):
    if cond == "N":
        return f"dose16k_{body}_s{seed}"
    return f"inv16k_{body}_s{seed}" if seed <= 2 else f"inv16kext_{body}_s{seed}"


ORDER = [(b, c, s) for b in "AB" for c in "NI" for s in SEEDS[c]]      # 28 adapters: A-N, A-I, B-N, B-I
IDX = {k: i for i, k in enumerate(ORDER)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_mtime(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------------------------------------------
# relabelling machinery (pure functions of a cosine matrix, so the self-test can exercise them)
# ---------------------------------------------------------------------------------------------------------------
def bit_vectors(n=6):
    """All 2^n swap vectors, identity first, in itertools.product order."""
    return [np.array(b, int) for b in itertools.product((0, 1), repeat=n)]


def split_stat(Cb, seeds, cond0, swap):
    """D for one body under one relabelling.

    Cb: cosine matrix of the body's adapters; seeds, cond0: seed and true condition (0 normal, 1 inverted) of each;
    swap: 0/1 per paired seed 0-5 (1 exchanges the labels of N_j and I_j; seeds 6-7 are never exchanged).
    Returns (D, mean_within, mean_cross, n_within, n_cross, mean_within_label0, mean_within_label1)."""
    lab = np.array([c ^ int(swap[s]) if s < len(swap) else c for c, s in zip(cond0, seeds)])
    iu = np.triu_indices(len(seeds), 1)
    diff_seed = seeds[iu[0]] != seeds[iu[1]]
    same_lab = lab[iu[0]] == lab[iu[1]]
    cos = Cb[iu]
    w = cos[diff_seed & same_lab]
    x = cos[diff_seed & ~same_lab]
    w0 = cos[diff_seed & same_lab & (lab[iu[0]] == 0)]
    w1 = cos[diff_seed & same_lab & (lab[iu[0]] == 1)]
    return float(w.mean() - x.mean()), float(w.mean()), float(x.mean()), int(w.size), int(x.size), \
        float(w0.mean()), float(w1.mean())


def body_null(Cb, seeds, cond0):
    """D_X under all 64 within-seed-pair relabellings (identity first)."""
    return np.array([split_stat(Cb, seeds, cond0, b)[0] for b in bit_vectors()])


def matched_stat(CD, sign):
    """Mean over j<k of sign_j sign_k cos(Delta_j, Delta_k) within one body."""
    iu = np.triu_indices(CD.shape[0], 1)
    return float((sign[iu[0]] * sign[iu[1]] * CD[iu]).mean())


def cross_matched_stat(X, sa, sb, same_seed):
    """Mean of sa_j sb_k cos(Delta^A_j, Delta^B_k) over j != k (same_seed False) or j == k (True)."""
    M = sa[:, None] * sb[None, :] * X
    mask = np.eye(X.shape[0], dtype=bool)
    return float(M[mask].mean() if same_seed else M[~mask].mean())


def exact_p(null, obs):
    return float(np.mean(null >= obs - TOL)), int(np.sum(null >= obs - TOL))


def summarize_null(null, obs):
    sd = float(null.std(ddof=1))
    return {"mean": float(null.mean()), "sd": sd, "min": float(null.min()), "max": float(null.max()),
            "z_of_observed": float((obs - null.mean()) / sd) if sd > 0 else None,
            "n_within_1e-9_of_observed_excluding_identity": int(np.sum(np.abs(null - obs) <= 1e-9) - 1)}


def primary_from_blocks(CbA, CbB, seeds, cond0):
    """The registered statistic and its 1,024-relabelling null, from the two bodies' 12 x 12 cosine blocks."""
    bits = bit_vectors()
    out = {}
    v64 = {}
    for b, Cb in (("A", CbA), ("B", CbB)):
        v = body_null(Cb, seeds, cond0)
        comp = np.array([int("".join(map(str, 1 - x)), 2) for x in bits])      # index of the complement relabelling
        sym = float(np.max(np.abs(v - v[comp])))
        assert sym < 1e-14, f"body {b}: D_X not invariant under exchanging all six pairs ({sym})"
        v64[b] = v
        out[b] = {"v32": v[[i for i, x in enumerate(bits) if x[0] == 0]], "symmetry_max_abs_diff": sym}
    assert all(len(out[b]["v32"]) == 32 for b in "AB")
    D1024 = ((out["A"]["v32"][:, None] + out["B"]["v32"][None, :]) / 2).ravel()
    D4096 = ((v64["A"][:, None] + v64["B"][None, :]) / 2).ravel()
    obs = float(D1024[0])
    p, n_ge = exact_p(D1024, obs)
    p4096, n_ge4096 = exact_p(D4096, obs)
    assert n_ge4096 == 4 * n_ge, (n_ge4096, n_ge)
    return obs, p, n_ge, D1024, p4096, n_ge4096, out, v64


# ---------------------------------------------------------------------------------------------------------------
# self-test on synthetic matrices (no file is read or written)
# ---------------------------------------------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(116004)
    seeds = np.array(PAIRED + PAIRED)
    cond0 = np.array([0] * 6 + [1] * 6)

    def synth(effect, twin=0.5):
        C = np.eye(12)
        for i, j in itertools.combinations(range(12), 2):
            c = 0.02 + 0.002 * rng.standard_normal()
            if seeds[i] == seeds[j]:
                c = twin
            elif cond0[i] == cond0[j]:
                c += effect
            C[i, j] = C[j, i] = c
        return C

    # counts and the identity
    s = split_stat(synth(0.0), seeds, cond0, np.zeros(6, int))
    assert (s[3], s[4]) == (30, 30), s
    # a strong within-condition excess puts the observed D at the top of both bodies' nulls: p = floor
    obs, p, n_ge, D1024, p4096, _, per, _ = primary_from_blocks(synth(0.05), synth(0.05), seeds, cond0)
    assert obs > 0 and n_ge == 1 and abs(p - 1 / 1024) < 1e-15, (obs, p, n_ge)
    # the reversed excess puts it at the bottom
    obs, p, n_ge, *_ = primary_from_blocks(synth(-0.05), synth(-0.05), seeds, cond0)
    assert obs < 0 and n_ge == 1024, (obs, n_ge)
    # under the null, p is roughly uniform over repeated synthetic draws
    ps = np.array([primary_from_blocks(synth(0.0), synth(0.0), seeds, cond0)[1] for _ in range(300)])
    frac = float(np.mean(ps <= 0.05))
    assert 0.01 <= frac <= 0.11, frac
    # the seeds 6-7 variant has 43 within and 42 cross pairs and no whole-body symmetry
    seeds14 = np.array(PAIRED + list(range(8)))
    cond14 = np.array([0] * 6 + [1] * 8)
    C14 = np.eye(14)
    for i, j in itertools.combinations(range(14), 2):
        C14[i, j] = C14[j, i] = 0.5 if seeds14[i] == seeds14[j] else 0.02 + 0.002 * rng.standard_normal()
    s14 = split_stat(C14, seeds14, cond14, np.zeros(6, int))
    assert (s14[3], s14[4]) == (43, 42), s14
    v = body_null(C14, seeds14, cond14)
    assert len(np.unique(np.round(v, 13))) == 64
    # matched statistic: whole-body flip invariance and sign behaviour
    CD = np.eye(6)
    for i, j in itertools.combinations(range(6), 2):
        CD[i, j] = CD[j, i] = 0.3
    sgn = 1 - 2 * np.array([1, 0, 0, 0, 0, 0])
    assert abs(matched_stat(CD, np.ones(6)) - 0.3) < 1e-15
    assert abs(matched_stat(CD, sgn) - (0.3 * (10 - 5) / 15)) < 1e-15
    assert abs(matched_stat(CD, -sgn) - matched_stat(CD, sgn)) < 1e-15
    print(f"[selftest] all checks passed; synthetic null P(p <= 0.05) = {frac:.3f} over 300 draws")


# ---------------------------------------------------------------------------------------------------------------
# data checks
# ---------------------------------------------------------------------------------------------------------------
def check_inputs():
    inputs, checks = {}, {}
    for key in ORDER:
        b, c, s = key
        t = tag(*key)
        wpath = f"{ADA}/{t}/pytorch_lora_weights.safetensors"
        mpath = f"{ADA}/{t}/train_meta.json"
        assert os.path.exists(wpath), wpath
        nbytes = os.path.getsize(wpath)
        assert nbytes == FILE_BYTES, (t, nbytes)
        with open(mpath, encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["seed"] == s and meta["steps"] == 16000 and meta["rank"] == 16, (t, meta)
        assert meta["field"] == FIELD[(b, c)] and float(meta["alpha"]) == 0.0, (t, meta)
        inputs[t] = {"body": b, "device": DEVICE[b], "condition": "normal" if c == "N" else "inverted", "seed": s,
                     "training_chain": CHAIN[c][s], "weights": wpath, "bytes": nbytes, "sha256": sha256(wpath),
                     "weights_mtime_utc": utc_mtime(wpath), "train_meta": meta}
    absent = [f"dose16k_{b}_s{s}" for b in "AB" for s in (6, 7)]
    for t in absent:
        assert not os.path.exists(f"{ADA}/{t}"), t
    checks["files_14587296_bytes_seed_steps_rank_field"] = True
    checks["absent_as_stated"] = absent
    return inputs, checks


def crop_correspondence():
    from PIL import Image
    with open(INVERT_JSON, encoding="utf-8") as f:
        inv = json.load(f)
    res = {"source": INVERT_JSON, "sha256": sha256(INVERT_JSON), "bodies": {}}
    for b, (ndir, idir) in CROPS.items():
        nf = sorted(os.listdir(f"{TRAIN_PNG}/{ndir}"))
        jf = sorted(os.listdir(f"{TRAIN_PNG}/{idir}"))
        assert nf == jf == [f"{i:04d}.png" for i in range(50)], (b, nf[:3], jf[:3])
        norm = [np.asarray(Image.open(f"{TRAIN_PNG}/{ndir}/{x}").convert("RGB"), np.float32) for x in nf]
        same, shifted = [], []
        for i, x in enumerate(jf):
            sup = np.asarray(Image.open(f"{TRAIN_PNG}/{idir}/{x}").convert("RGB"), np.uint8)
            same.append(float(np.sqrt(((sup.astype(np.float32) - norm[i]) ** 2).mean())))
            shifted.append(float(np.sqrt(((sup.astype(np.float32) - norm[(i + 1) % 50]) ** 2).mean())))
        rec = inv["bodies"][b]
        assert rec["source"] == ndir and rec["n"] == 50
        d = float(np.mean(same)) - rec["change_rms_gray"]
        assert abs(d) < 1e-6, (b, d)
        res["bodies"][b] = {"normal_dir": ndir, "inverted_dir": idir, "n": 50,
                            "rms_same_index_mean": float(np.mean(same)), "rms_same_index_max": float(np.max(same)),
                            "invert_json_change_rms_gray": rec["change_rms_gray"], "difference": d,
                            "rms_next_index_mean": float(np.mean(shifted)), "rms_next_index_min": float(np.min(shifted)),
                            "every_same_index_rms_below_every_next_index_rms": bool(max(same) < min(shifted))}
    return res


def parse_losses():
    pat = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]\s+(\S+)\s+(\d+)/(\d+)\s+loss\s+([0-9.]+)", re.M)
    rows = {}
    files = {}
    for name in TRAIN_LOGS:
        p = f"{LOGS}/{name}"
        txt = open(p, encoding="utf-8", errors="replace").read().replace("\r", "\n")
        files[name] = {"sha256": sha256(p), "mtime_utc": utc_mtime(p)}
        for m in pat.finditer(txt):
            if m.group(3) != "16000":
                continue
            rows.setdefault(m.group(1), []).append((int(m.group(2)), float(m.group(4)), name))
    traj, missing = {}, []
    want = list(range(200, 16001, 200))
    for key in ORDER:
        t = tag(*key)
        r = rows.get(t, [])
        if not r:
            missing.append(t)
            continue
        # a restart would repeat steps; keep the last complete 200..16000 sequence
        steps = [x[0] for x in r]
        last = max(i for i, st in enumerate(steps) if st == 200)
        seq = r[last:]
        assert [x[0] for x in seq] == want, (t, len(seq))
        traj[t] = np.array([x[1] for x in seq])
    return traj, missing, files


def loss_diagnostic(traj, missing):
    def r(a, b):
        return float(np.corrcoef(np.diff(a), np.diff(b))[0, 1])
    res = {"what": ("Pearson r of the first differences of the 80 logged 200-step mean losses; twins share seed j, "
                    "so the same batch order, noise and timesteps; offset = inverted minus normal"),
           "missing_traces": missing, "bodies": {}}
    for b in "AB":
        twins = {}
        for j in PAIRED:
            tn, ti = tag(b, "N", j), tag(b, "I", j)
            if tn in traj and ti in traj:
                twins[str(j)] = {"r_first_diff": r(traj[tn], traj[ti]),
                                 "mean_offset_I_minus_N": float(np.mean(traj[ti] - traj[tn])),
                                 "offset_at_step_200": float(traj[ti][0] - traj[tn][0]),
                                 "loss_step_200_N": float(traj[tn][0]), "loss_step_200_I": float(traj[ti][0])}
        nn = [r(traj[tag(b, "N", j)], traj[tag(b, "N", k)]) for j, k in itertools.combinations(PAIRED, 2)
              if tag(b, "N", j) in traj and tag(b, "N", k) in traj]
        ni = [r(traj[tag(b, "N", j)], traj[tag(b, "I", k)]) for j in PAIRED for k in PAIRED
              if j != k and tag(b, "N", j) in traj and tag(b, "I", k) in traj]
        tr = [v["r_first_diff"] for v in twins.values()]
        res["bodies"][b] = {"twins": twins, "twin_r_mean": float(np.mean(tr)), "twin_r_min": float(np.min(tr)),
                            "n_twins": len(tr),
                            "normal_different_seed_r_mean": float(np.mean(nn)), "n_normal_pairs": len(nn),
                            "cross_condition_different_seed_r_mean": float(np.mean(ni)),
                            "cross_condition_different_seed_r_max_abs": float(np.max(np.abs(ni))), "n_cross_pairs": len(ni),
                            "twin_mean_offset_I_minus_N": float(np.mean([v["mean_offset_I_minus_N"] for v in twins.values()]))}
    return res


def post_hoc_context(first_normal_mtime):
    """Context added after the scratch run (see header): pattern similarity, stack continuity, step-200 losses."""
    res = {"status": ("post hoc: added after a scratch run of this script had shown the registered result; context "
                      "for diagnostic 4 and the loss offsets; no reading attaches")}
    ka = np.load(OUT + "/fp/K_A_E1.npy").astype(np.float64)
    kb = np.load(OUT + "/fp/K_B_E1.npy").astype(np.float64)
    ka, kb = ka - ka.mean(), kb - kb.mean()
    with open(OUT + "/fp/gates.json", encoding="utf-8") as f:
        gates = json.load(f)
    res["a_pattern_similarity"] = {"ncc_K_A_E1_K_B_E1": float((ka * kb).sum() / np.sqrt((ka * ka).sum() * (kb * kb).sum())),
                                   "kappa_model_E2_from_gates_json": gates["kappa_model_E2"],
                                   "sources": [OUT + "/fp/K_A_E1.npy", OUT + "/fp/K_B_E1.npy", OUT + "/fp/gates.json"]}
    site = os.path.join(sys.prefix, "Lib", "site-packages")
    pk = {}
    for name in ("torch", "diffusers", "peft", "transformers", "accelerate", "safetensors", "numpy"):
        hits = sorted(d for d in os.listdir(site) if d.lower().startswith(name + "-") and d.endswith(".dist-info"))
        pk[name] = [{"dist_info": d, "mtime_utc": utc_mtime(os.path.join(site, d))} for d in hits]
    latest = max(os.path.getmtime(os.path.join(site, x["dist_info"])) for v in pk.values() for x in v)
    res["b_training_stack"] = {"interpreter": sys.executable, "site_packages": site, "packages": pk,
                               "latest_install_utc": datetime.datetime.fromtimestamp(latest, datetime.timezone.utc).isoformat(),
                               "all_installed_before_first_normal_adapter": bool(latest < first_normal_mtime),
                               "note": ("the chains (src/orchestrate*.sh) call `python`, which resolves to this interpreter "
                                        "in the chains' Git Bash; one dist-info folder per package means no second version")}
    pat = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]\s+(\S+)\s+200/(\d+)\s+loss\s+([0-9.]+)", re.M)
    want = {"t1_nomark_train.log": ["nomark_s0"], "t1_nomarkB_train.log": ["nomarkB_s0"],
            "t1_dose8k_train.log": ["dose8k_A_s0", "dose8k_B_s0"], "t1_dose16k_train.log": ["dose16k_A_s0", "dose16k_B_s0"],
            "t1_inv16k_train.log": ["inv16k_A_s0", "inv16k_B_s0"],
            "t1_train.log": ["mark_rand_a1_s0", "mark_rand_a3_s0", "mark_rand_a12_s0"],
            "t1_kinj_train.log": ["kinjd_a3_s0", "kinj_a12_s0", "kinj_a48_s0"]}
    rows = {}
    for name, tags in want.items():
        txt = open(f"{LOGS}/{name}", encoding="utf-8", errors="replace").read().replace("\r", "\n")
        found = {}
        for m in pat.finditer(txt):
            found.setdefault(m.group(1), []).append((int(m.group(2)), float(m.group(3))))
        for t in tags:
            assert len(found.get(t, [])) == 1, (name, t, found.get(t))
            meta = json.load(open(f"{ADA}/{t}/train_meta.json", encoding="utf-8"))
            assert meta["seed"] == 0 and meta["steps"] == found[t][0][0], (t, meta)
            rows[t] = {"log": name, "steps": meta["steps"], "field": meta["field"], "alpha": meta["alpha"],
                       "loss_step_200": found[t][0][1],
                       "weights_mtime_utc": utc_mtime(f"{ADA}/{t}/pytorch_lora_weights.safetensors")}
    res["c_step200_losses_seed0"] = {"rows": rows,
                                     "note": ("fields none/noneB are the unchanged crops; invA/invB the inverted crops; "
                                              "rand, kinj, kinjd add another fine pattern to body A's crops "
                                              "(Entries 01, 59); losses are printed to four decimals in the logs")}
    return res


# ---------------------------------------------------------------------------------------------------------------
def main(dst):
    if os.path.exists(dst):
        sys.exit(f"[wt-inv] {dst} exists; results are never overwritten")
    t0 = time.time()
    sys.path.insert(0, SRC)
    import h5_weight_signature as H5
    assert os.path.normcase(os.path.abspath(H5.ADA)) == os.path.normcase(os.path.abspath(ADA)), H5.ADA
    import torch
    import safetensors

    inputs, checks = check_inputs()
    print(f"[wt-inv] 28 adapters checked (sizes, seeds, steps, rank, fields)", flush=True)

    # ---- load and Gram ------------------------------------------------------------------------------------
    W = {key: H5.load_pairs(tag(*key)) for key in ORDER}
    layers = list(W[ORDER[0]].keys())
    shapes = {k: (a.shape, bb.shape) for k, (a, bb) in W[ORDER[0]].items()}
    for key in ORDER:
        assert list(W[key].keys()) == layers, tag(*key)
        assert all((a.shape, bb.shape) == shapes[k] for k, (a, bb) in W[key].items()), tag(*key)
    assert all(a.shape[0] == 16 and bb.shape[1] == 16 for a, bb in W[ORDER[0]].values())
    n_params = int(sum(a.size + bb.size for a, bb in W[ORDER[0]].values()))
    checks["identical_layer_sets_and_shapes"] = True
    n = len(ORDER)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            G[i, j] = G[j, i] = H5.gram(W[ORDER[i]], W[ORDER[j]])
    norm = np.array([np.sqrt(G[i, i]) for i in range(n)])
    C = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        C[i, j] = C[j, i] = G[i, j] / (norm[i] * norm[j])
    # symmetry of gram() on a few pairs
    sym = max(abs(H5.gram(W[ORDER[j]], W[ORDER[i]]) - G[i, j]) / abs(G[i, j])
              for i, j in [(0, 6), (0, 14), (6, 20), (13, 27), (3, 25)])
    assert sym < 1e-9, sym
    print(f"[wt-inv] Gram matrix of 28 adapters, {len(layers)} layers, {n_params} parameters each "
          f"({time.time() - t0:.1f} s)", flush=True)

    # ---- reproduction of H5 (normal adapters, 16000 steps) ----------------------------------------------------
    with open(H5_JSON, encoding="utf-8") as f:
        H = json.load(f)["doses"]["16000"]
    hidx = [IDX[("A", "N", s)] for s in PAIRED] + [IDX[("B", "N", s)] for s in PAIRED]
    assert H["adapters"] == [tag(*ORDER[i]) for i in hidx], H["adapters"]
    h5_diff = float(np.max(np.abs(C[np.ix_(hidx, hidx)] - np.array(H["cosine_matrix"]))))
    assert h5_diff < 1e-10, h5_diff
    checks["h5_cosines_reproduced_max_abs_diff"] = h5_diff

    # ---- primary ----------------------------------------------------------------------------------------------
    seeds12 = np.array(PAIRED + PAIRED)
    cond12 = np.array([0] * 6 + [1] * 6)

    def block(b, seeds_I):
        keys = [(b, "N", s) for s in PAIRED] + [(b, "I", s) for s in seeds_I]
        ii = [IDX[k] for k in keys]
        return keys, C[np.ix_(ii, ii)], G[np.ix_(ii, ii)]

    kA, CA, _ = block("A", PAIRED)
    kB, CB, _ = block("B", PAIRED)
    obs, p, n_ge, D1024, p4096, n_ge4096, per, v64 = primary_from_blocks(CA, CB, seeds12, cond12)
    ident = np.zeros(6, int)
    per_body = {}
    for b, Cb, keys in (("A", CA, kA), ("B", CB, kB)):
        D_X, w, x, nw, nx, wN, wI = split_stat(Cb, seeds12, cond12, ident)
        v32 = per[b]["v32"]
        pX, nX = exact_p(v32, D_X)
        assert abs(D_X - v32[0]) == 0.0
        per_body[b] = {"device": DEVICE[b], "adapters": [tag(*k) for k in keys],
                       "D_X": D_X, "mean_within": w, "mean_cross": x, "n_within": nw, "n_cross": nx,
                       "mean_normal_normal": wN, "mean_inverted_inverted": wI,
                       "p_one_sided_exact": pX, "n_ge_observed": nX, "n_distinct_relabellings": 32, "floor": 1 / 32,
                       "rank_of_observed_among_32": int(1 + np.sum(v32 > D_X + TOL)),
                       "null_32_values": v32.tolist(), "null_64_values_product_order": v64[b].tolist(),
                       "whole_body_exchange_max_abs_diff": per[b]["symmetry_max_abs_diff"],
                       "null_summary": summarize_null(v32, D_X)}
    assert abs(obs - (per_body["A"]["D_X"] + per_body["B"]["D_X"]) / 2) < 1e-15
    if p < LEVEL and obs > 0:
        reading = READ_POS
    elif p >= LEVEL:
        reading = READ_NEG
    else:
        reading = "not among the registered readings (p < 0.01 with D <= 0)"
    primary = {"statistic": ("D = (D_A + D_B)/2; D_X = mean cosine over the 30 within-condition different-seed pairs "
                             "(15 N-N, 15 I-I) - mean cosine over the 30 cross-condition different-seed pairs; seeds 0-5; "
                             "same-seed pairs and inverted seeds 6-7 excluded"),
               "D": obs, "p_one_sided_exact": p, "n_ge_observed": n_ge, "n_relabellings": 1024, "floor": 1 / 1024,
               "rank_of_observed_among_1024": int(1 + np.sum(D1024 > obs + TOL)),
               "level": LEVEL, "direction": "D > 0",
               "p_over_all_4096_relabellings": p4096, "n_ge_observed_4096": n_ge4096,
               "null_summary": summarize_null(D1024, obs),
               "null_1024_values": D1024.tolist(),
               "null_1024_order": "index = 32 * a + b; a, b index the 32 distinct relabellings of body A, B "
                                  "(product order of the six swap bits with seed 0 kept, identity first)",
               "per_body": per_body, "reading": reading,
               "reading_scope": {READ_POS: CEILING, READ_NEG: "at six seed pairs per body and 16000 steps"}.get(
                   reading, "no registered reading applies")}
    print(f"[wt-inv] D_A {per_body['A']['D_X']:+.6f} (p {per_body['A']['p_one_sided_exact']:.4f})  "
          f"D_B {per_body['B']['D_X']:+.6f} (p {per_body['B']['p_one_sided_exact']:.4f})  "
          f"D {obs:+.6f}  one-sided p {p:.5f} ({n_ge}/1024, floor {1 / 1024:.5f})", flush=True)
    print(f"[wt-inv] reading: {reading}", flush=True)

    # ---- descriptive: twin cosines ------------------------------------------------------------------------------
    twins = {}
    for b in "AB":
        tc = [float(C[IDX[(b, "N", j)], IDX[(b, "I", j)]]) for j in PAIRED]
        keys = [(b, "N", s) for s in PAIRED] + [(b, "I", s) for s in SEEDS["I"]]
        other = [C[IDX[p_], IDX[q_]] for p_, q_ in itertools.combinations(keys, 2) if p_[2] != q_[2]]
        twins[b] = {"cosines_by_seed": tc, "mean": float(np.mean(tc)), "min": float(np.min(tc)),
                    "max": float(np.max(tc)),
                    "largest_different_seed_cosine_in_body_incl_seeds_6_7": float(np.max(other))}
    same_seed_cross_body = {
        "normal": [float(C[IDX[("A", "N", j)], IDX[("B", "N", j)]]) for j in PAIRED],
        "inverted": [float(C[IDX[("A", "I", j)], IDX[("B", "I", j)]]) for j in SEEDS["I"]]}

    # ---- descriptive: matched variant (twin differences) ---------------------------------------------------------
    D_fac = {}
    for b in "AB":
        for j in PAIRED:
            wi, wn = W[(b, "I", j)], W[(b, "N", j)]
            D_fac[(b, j)] = {k: (np.vstack([wi[k][0], wn[k][0]]), np.hstack([wi[k][1], -wn[k][1]])) for k in layers}
    dkeys = [(b, j) for b in "AB" for j in PAIRED]
    GD = np.zeros((12, 12))
    GDx = np.zeros((12, 12))
    for p_, q_ in itertools.combinations_with_replacement(range(12), 2):
        (b1, j1), (b2, j2) = dkeys[p_], dkeys[q_]
        GD[p_, q_] = GD[q_, p_] = H5.gram(D_fac[dkeys[p_]], D_fac[dkeys[q_]])
        i1, n1, i2, n2 = IDX[(b1, "I", j1)], IDX[(b1, "N", j1)], IDX[(b2, "I", j2)], IDX[(b2, "N", j2)]
        GDx[p_, q_] = GDx[q_, p_] = G[i1, i2] - G[i1, n2] - G[n1, i2] + G[n1, n2]
    rel = float(np.max(np.abs(GD - GDx)) / np.max(np.abs(GD)))
    assert rel < 1e-9, rel
    checks["twin_difference_factorisation_vs_expansion_max_rel_diff"] = rel
    dn = np.sqrt(np.diag(GD))
    CD = GD / np.outer(dn, dn)
    bits = bit_vectors()
    mnull = {}
    for bi, b in enumerate("AB"):
        CDb = CD[6 * bi:6 * bi + 6, 6 * bi:6 * bi + 6]
        v = np.array([matched_stat(CDb, 1 - 2 * x) for x in bits])
        comp = np.array([int("".join(map(str, 1 - x)), 2) for x in bits])
        assert np.max(np.abs(v - v[comp])) < 1e-14
        mnull[b] = v[[i for i, x in enumerate(bits) if x[0] == 0]]
    M1024 = ((mnull["A"][:, None] + mnull["B"][None, :]) / 2).ravel()
    M_obs = float(M1024[0])
    pM, nM = exact_p(M1024, M_obs)
    matched = {"definition": ("Delta_j = dW(I_j) - dW(N_j); statistic = mean over the 15 within-body pairs j<k of "
                              "cos(Delta_j, Delta_k), averaged over the two bodies (= mean over 30 pairs); exchanging "
                              "seed pair j flips the sign of Delta_j; 1,024 distinct relabellings"),
               "M": M_obs, "p_one_sided_exact": pM, "n_ge_observed": nM, "floor": 1 / 1024,
               "per_body": {b: {"M_X": float(mnull[b][0]),
                                "p_one_sided_exact_32": exact_p(mnull[b], float(mnull[b][0]))[0],
                                "pairwise_cosines": CD[6 * bi:6 * bi + 6, 6 * bi:6 * bi + 6].tolist(),
                                "twin_difference_norms": dn[6 * bi:6 * bi + 6].tolist(),
                                "twin_difference_norm_over_normal_norm": [
                                    float(dn[6 * bi + j] / norm[IDX[(b, "N", j)]]) for j in PAIRED]}
                            for bi, b in enumerate("AB")},
               "null_summary": summarize_null(M1024, M_obs)}

    # ---- descriptive: D with inverted seeds 6-7 kept at their fixed label -----------------------------------------
    seeds14 = np.array(PAIRED + SEEDS["I"])
    cond14 = np.array([0] * 6 + [1] * 8)
    v14, d14 = {}, {}
    for b in "AB":
        _, Cb, _ = block(b, SEEDS["I"])
        v14[b] = body_null(Cb, seeds14, cond14)
        s_ = split_stat(Cb, seeds14, cond14, ident)
        assert (s_[3], s_[4]) == (43, 42)
        d14[b] = {"D_X": s_[0], "mean_within": s_[1], "mean_cross": s_[2], "n_within": s_[3], "n_cross": s_[4],
                  "mean_normal_normal": s_[5], "mean_inverted_inverted": s_[6],
                  "p_one_sided_exact_64": exact_p(v14[b], s_[0])[0],
                  "n_distinct_values_64": int(len(np.unique(np.round(v14[b], 13))))}
    V4096 = ((v14["A"][:, None] + v14["B"][None, :]) / 2).ravel()
    s67_obs = float(V4096[0])
    p67, n67 = exact_p(V4096, s67_obs)
    seeds67 = {"definition": ("inverted seeds 6-7 always labelled inverted; within = pooled mean over the 43 same-label "
                              "different-seed pairs (15 N-N + 28 I-I), cross = pooled mean over the 42 different-label "
                              "different-seed pairs; 64 relabellings per body, 4,096 over both"),
               "D": s67_obs, "p_one_sided_exact": p67, "n_ge_observed": n67, "n_relabellings": 4096,
               "floor": 1 / 4096, "n_distinct_values_4096": int(len(np.unique(np.round(V4096, 13)))),
               "per_body": d14, "null_summary": summarize_null(V4096, s67_obs)}

    # ---- descriptive: norms --------------------------------------------------------------------------------------
    norms = {}
    for b in "AB":
        rec = {}
        for c, seeds_c in (("N", SEEDS["N"]), ("I", SEEDS["I"])):
            fro = [float(norm[IDX[(b, c, s)]]) for s in seeds_c]
            lbn = [float(inputs[tag(b, c, s)]["train_meta"]["lora_B_norm"]) for s in seeds_c]
            rec["normal" if c == "N" else "inverted"] = {
                "seeds": seeds_c, "dW_frobenius": fro, "dW_frobenius_mean": float(np.mean(fro)),
                "dW_frobenius_sd": float(np.std(fro, ddof=1)), "lora_B_norm": lbn,
                "lora_B_norm_mean": float(np.mean(lbn)), "lora_B_norm_sd": float(np.std(lbn, ddof=1))}
        fro_tw = [float(norm[IDX[(b, "I", j)]] - norm[IDX[(b, "N", j)]]) for j in PAIRED]
        lbn_tw = [float(inputs[tag(b, "I", j)]["train_meta"]["lora_B_norm"] - inputs[tag(b, "N", j)]["train_meta"]["lora_B_norm"])
                  for j in PAIRED]
        rec["inverted_seeds_0_5_dW_frobenius_mean"] = float(np.mean([norm[IDX[(b, "I", j)]] for j in PAIRED]))
        rec["twin_differences_I_minus_N"] = {"dW_frobenius": fro_tw, "dW_frobenius_mean": float(np.mean(fro_tw)),
                                             "lora_B_norm": lbn_tw, "lora_B_norm_mean": float(np.mean(lbn_tw))}
        norms[b] = rec

    # ---- diagnostics not in the pre-specification -----------------------------------------------------------------
    diag = {"status": "not pre-specified; fixed in the script header before any number was computed; no reading attaches"}
    diag["1_crop_correspondence"] = crop_correspondence()
    traj, missing, logfiles = parse_losses()
    diag["2_shared_stream_from_training_logs"] = loss_diagnostic(traj, missing)
    diag["2_shared_stream_from_training_logs"]["logs"] = logfiles
    chains = {}
    for b in "AB":
        rec = {}
        for c, seeds_c in (("N", SEEDS["N"]), ("I", SEEDS["I"])):
            same, diff = [], []
            for s1, s2 in itertools.combinations(seeds_c, 2):
                v = C[IDX[(b, c, s1)], IDX[(b, c, s2)]]
                (same if CHAIN[c][s1] == CHAIN[c][s2] else diff).append(v)
            rec["normal" if c == "N" else "inverted"] = {
                "same_chain_mean": float(np.mean(same)), "n_same_chain": len(same),
                "different_chain_mean": float(np.mean(diff)), "n_different_chain": len(diff),
                "difference": float(np.mean(same) - np.mean(diff))}
        chains[b] = rec
    diag["3_training_chain_within_condition"] = {"chains": {c: {str(s): v for s, v in CHAIN[c].items()} for c in CHAIN},
                                                 "bodies": chains}
    X = CD[:6, 6:]                                  # cos(Delta^A_j, Delta^B_k)
    ones = np.ones(6)
    xobs = cross_matched_stat(X, ones, ones, False)
    xs_obs = cross_matched_stat(X, ones, ones, True)
    xnull = np.array([cross_matched_stat(X, 1 - 2 * a, 1 - 2 * c, False) for a in bits for c in bits])
    px, nx = exact_p(xnull, xobs)
    diag["4_cross_body_alignment_of_twin_differences"] = {
        "different_seed_mean_cos": xobs, "same_seed_mean_cos": xs_obs,
        "matrix_cos_DeltaA_j_DeltaB_k": X.tolist(),
        "p_one_sided_relabelling": px, "n_ge_observed": nx, "n_relabellings": 4096,
        "n_distinct_values": int(len(np.unique(np.round(xnull, 13)))),
        "null_summary": summarize_null(xnull, xobs)}
    diag["5_numerical_checks"] = {"h5_cosines_reproduced_max_abs_diff": h5_diff,
                                  "gram_symmetry_max_rel_diff": float(sym),
                                  "twin_difference_factorisation_vs_expansion_max_rel_diff": rel,
                                  "p_1024_equals_p_4096": p4096 == p}
    first_normal = min(os.path.getmtime(inputs[tag("A" if i < 6 else "B", "N", i % 6)]["weights"]) for i in range(12))
    post_hoc = post_hoc_context(first_normal)

    res = {
        "entry": "RESULTS.md Entry 116, item 4 (normal-versus-inverted weight test)",
        "status": ("registered test at level 0.01, one-sided (D > 0); exact relabelling within seed pairs, 1,024 "
                   "distinct partitions, floor 1/1024. Descriptive items as registered; diagnostics not pre-specified "
                   "are labelled as such"),
        "script": "src/fv/fv_weights_inv.py", "script_sha256": sha256(THIS),
        "file_names_note": ("Entry 116 names src/fv/fv_weights_invert.py -> out/fv_weights_invert.json; this run was "
                            "instructed to write src/fv/fv_weights_inv.py -> out/fv_weights_inv.json. Names only."),
        "run": {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "python": platform.python_version(),
                "numpy": np.__version__, "torch": torch.__version__, "safetensors": safetensors.__version__,
                "platform": platform.platform()},
        "settings": {"level": LEVEL, "sidedness": "one-sided, D > 0", "tie_tolerance": TOL,
                     "paired_seeds": PAIRED, "inverted_seeds_all": SEEDS["I"],
                     "relabelling": "labels of N_j and I_j kept or exchanged independently for each body and seed j",
                     "n_distinct_relabellings": 1024, "floor": 1 / 1024,
                     "extraction": "dW = B A per layer; cosine of concatenated dW via h5_weight_signature.gram (exact)",
                     "n_layers": len(layers), "n_lora_parameters_per_adapter": n_params,
                     "lora_targets": "to_q, to_k, to_v, to_out.0 (t1_ladder.LORA_TARGETS)"},
        "inputs": {"adapters": inputs, "import": {"module": H5_SCRIPT, "sha256": sha256(H5_SCRIPT),
                                                  "functions": ["load_pairs", "gram"]},
                   "h5_json": H5_JSON, "h5_json_sha256": sha256(H5_JSON)},
        "checks": checks,
        "gram": {"order": [tag(*k) for k in ORDER], "inner_products": G.tolist(), "cosines": C.tolist(),
                 "dW_frobenius_norms": norm.tolist()},
        "primary": primary,
        "descriptive": {"per_body": {b: {k: per_body[b][k] for k in ("D_X", "p_one_sided_exact", "floor")}
                                     for b in "AB"},
                        "twin_cosines": twins, "same_seed_cross_body_cosines_context": same_seed_cross_body,
                        "matched_variant": matched, "seeds_6_7_kept": seeds67, "norms": norms},
        "diagnostics_not_prespecified": diag,
        "post_hoc_context_added_after_scratch_run": post_hoc,
        "runtime_s": time.time() - t0,
    }
    with open(dst, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"[wt-inv] twin cosines A {np.round(twins['A']['cosines_by_seed'], 4).tolist()} "
          f"B {np.round(twins['B']['cosines_by_seed'], 4).tolist()}")
    print(f"[wt-inv] matched M {M_obs:+.6f} p {pM:.5f}; seeds 6-7 kept D {s67_obs:+.6f} p {p67:.5f}; "
          f"cross-body Delta alignment {xobs:+.6f} p {px:.5f} (not pre-specified)")
    print(f"[wt-inv] written {dst} ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=DST)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        main(a.out)
