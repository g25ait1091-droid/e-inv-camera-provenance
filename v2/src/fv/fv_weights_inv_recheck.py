"""Independent recheck of the registered normal-versus-inverted weight test (RESULTS.md Entry 116, item 4), and the
checks a skeptic should run on it. CPU only, a few minutes. Writes out/fv_weights_inv_recheck.json and nothing else.

WHY THIS FILE, AND NOT A SECOND out/fv_weights_inv.json. The registered test was run on 30 Sep 2026 by
src/fv/fv_weights_inv.py -> out/fv_weights_inv.json (the names the workflow gave; Entry 116 names
fv_weights_invert.*). A second workflow run on 1 Oct 2026 was asked to write the same two files. Result files are
never overwritten, and replacing the script would orphan the result it produced, so both are left untouched. This
script recomputes every registered quantity with code written independently of that script (only
h5_weight_signature.load_pairs and gram are shared, because Entry 116 requires them), compares each with the file of
record, and adds checks that bear on how far the registered reading can be taken. The adapter weights are fixed
inputs, so a re-run is a reproduction, not a second test.

REGISTERED (Entry 116 item 4, unchanged)
  adapters   normal dose16k_{A,B}_s0..s5; inverted inv16k_{A,B}_s0..s2 and inv16kext_{A,B}_s3..s7 (16000 steps, rank 16)
  extraction dW_l = B_l A_l; cosine of the concatenated dW via h5_weight_signature.gram (exact, never forms dW)
  statistic  D_X = mean cos over the 30 within-condition different-seed pairs (15 N-N, 15 I-I) - mean cos over the 30
             cross-condition different-seed pairs (N_j, I_k), j != k; seeds 0-5 only; twins (N_j, I_j) and inverted
             seeds 6-7 excluded; D = (D_A + D_B) / 2
  null       labels of N_j and I_j kept or exchanged, independently per body and seed; exchanging all six in a body
             returns the same partition, so 32 distinct per body and 1,024 in all; one-sided p = fraction of the
             1,024 with D >= observed, identity included; floor 1/1024; level 0.01, D > 0
  readings   p < 0.01 with D > 0 -> "the adapter weights carry the fingerprint's sign";
             p >= 0.01 -> "the weights do not detectably carry the fingerprint's sign"
  descriptive per-body D_X and exact p (floor 1/32); the six twin cosines per body; matched variant on the twin
             differences Delta_j = dW(I_j) - dW(N_j) over the same 1,024 partitions; D with inverted seeds 6-7 kept
             at their label (4,096 relabellings); adapter-weight norms by condition

TWO ROUTES TO D (they must agree to rounding)
  (i)  enumeration: labels assigned, every different-seed pair classified, the two means differenced;
  (ii) closed form: with T_jk = c(N_j,N_k) + c(I_j,I_k) - c(N_j,I_k) - c(I_j,N_k) (j < k),
       D_X(s) = (1/30) sum_{j<k} (-1)^(s_j + s_k) T_jk, because every relabelling keeps 30 within and 30 cross pairs
       and exchanging one seed pair swaps the within and cross pairs of each block it belongs to.
  The Gram matrix is likewise computed twice: by h5_weight_signature.gram (registered) and by a second algebraic route
  (per layer, the Hadamard product of the stacked B^T B and A A^T Gram matrices), plus explicit dW on three layers.

SKEPTIC CHECKS (not pre-specified; written after the registered result was known from the file of record; no
reading attaches to any of them; each states its own floor where it has one)
  F1 per-pair contributions T_jk / 30 and leave-one-seed-out D (exact p over 256 relabellings, floor 1/256);
  F2 the twin-difference alignment split into a part common to both bodies (which cannot be either body's own
     fingerprint pattern: the two E1 estimates are nearly orthogonal) and a body-specific remainder; the common part
     is tested by the registered relabelling (2,048 distinct), the remainder by exchanging the body labels of Delta^A_j
     and Delta^B_j within seed pairs (32 distinct, floor 1/32);
  F3 training-code identity across the training processes, from the editor's file-history snapshots of
     src/t1_ladder.py, each process matched to the snapshot in force when it started (start = weights mtime minus the
     recorded training minutes, checked against the log's start line);
  F4 environment identity: installed packages, the SD-3.5 snapshot, the prompt embeddings, the normal and inverted
     crops, the GPU driver library, all dated against the first normal adapter;
  F5 the inverted crops rebuilt bit-exactly by g1_invert.py's rule (rng 20260919), and the stored change split into
     the reversed-fingerprint term -6 K Y (one spatial pattern shared by all 50 crops of a body) and the
     dither-and-rounding residual (drawn afresh for every crop);
  F6 within-condition different-seed pairs grouped by training process (does the process or week make adapters
     alike?);
  F7 the twins' training-loss traces (shared random stream) and their loss offset;
  F8 for scale, at the 2000-step dose: the same twin-difference quantities for arms that add other fixed patterns to
     body A's unchanged crops, and for v1's Colab-decoded crops (a decoder change with no added pattern), each against
     the nomark adapter of the same seed;
  F9 alignment across patterns: the mean cosine between twin differences of different seeds for every pair of
     conditions (the F8 arms and the 16000-step inverted arms of each body), i.e. whether the inverted condition's
     direction in weight space is shared with unrelated fixed patterns of either sign.
  F8 and F9 use one Gram matrix over all adapters by the second route, checked against h5_weight_signature.gram.

Run:  python -B src/fv/fv_weights_inv_recheck.py              (writes out/fv_weights_inv_recheck.json; refuses to overwrite)
      python -B src/fv/fv_weights_inv_recheck.py --out PATH    (write elsewhere, e.g. a scratch copy)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import argparse
import datetime
import glob
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
os.environ["EINV_V2"] = V2            # einv_paths (imported by h5_weight_signature) resolves the adapter folder from this
SRC = EINV.SRC + ""
OUT = V2 + "/out"
ADA = OUT + "/t1/adapters"
TRAIN_PNG = OUT + "/t1/train_png"
FP = OUT + "/fp"
LOGS = V2 + "/logs"
DST = OUT + "/fv_weights_inv_recheck.json"
RECORD_JSON = OUT + "/fv_weights_inv.json"
RECORD_SCRIPT = SRC + "/fv/fv_weights_inv.py"
H5_SCRIPT = SRC + "/h5_weight_signature.py"
H5_JSON = OUT + "/h5_weight_signature.json"
G1_SCRIPT = SRC + "/g1_invert.py"
INVERT_JSON = OUT + "/t1/invert.json"
LADDER = SRC + "/t1_ladder.py"
PROMPT_PT = V2 + "/data/adapters/_prompt_embeds_a1759453b6c6cd42.pt"
HF_MODEL_DIR = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub",
                            "models--stabilityai--stable-diffusion-3.5-medium")
# F3 read the editor's local file-history snapshots of src/t1_ladder.py on the machine that trained the
# adapters; they are not part of this repository, so F3 reports "not found" unless this is set
FILE_HISTORY = os.environ.get("EINV_EDITOR_HISTORY", "")
LADDER_KEY = "d054c1454045ecd5"
NVCUDA = os.path.join(os.environ.get("SYSTEMROOT", ""), "System32", "nvcuda.dll")
THIS = os.path.abspath(__file__)

LEVEL = 0.01
TIE = 1e-12
BYTES = 14587296
PAIRED = [0, 1, 2, 3, 4, 5]
INV_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
FIELD = {("A", "N"): "none", ("B", "N"): "noneB", ("A", "I"): "invA", ("B", "I"): "invB"}
DEVICE = {"A": "Nikon_D200_1", "B": "Nikon_D200_0"}
CROPS = {("A", "N"): "none_a0", ("B", "N"): "noneB_a0", ("A", "I"): "invA_a0", ("B", "I"): "invB_a0"}
READ_POS = "the adapter weights carry the fingerprint's sign"
READ_NEG = "the weights do not detectably carry the fingerprint's sign"
SCOPE_POS = ("in the precise sense registered in Entry 116: the weights distinguish training on the photographs with "
             "the fingerprint reversed from training on the same photographs unchanged, a white-box trace of the "
             "stored change. The two conditions also differ by the fixed dither field and by training date, so the "
             "result does not isolate the fingerprint pattern from the dither, and it says nothing about whether a "
             "natural-amplitude fingerprint is recoverable from standard 2000-step adapters.")
SCOPE_NEG = "at six seed pairs per body and 16000 steps"

# Training processes, from the logs (one `python t1_ladder.py train` per armset; inv16kext was restarted after A_s3).
PROCESSES = [
    ("dose16k", "t1_dose16k_train.log", [("A", "N", 0), ("B", "N", 0)]),
    ("dose16krep", "t1_dose16krep_train.log", [("A", "N", 1), ("B", "N", 1), ("A", "N", 2), ("B", "N", 2)]),
    ("dose16krep2", "t1_dose16krep2_train.log", [("A", "N", 3), ("B", "N", 3), ("A", "N", 4), ("B", "N", 4),
                                                 ("A", "N", 5), ("B", "N", 5)]),
    ("inv16k", "t1_inv16k_train.log", [("A", "I", 0), ("B", "I", 0), ("A", "I", 1), ("B", "I", 1), ("A", "I", 2),
                                       ("B", "I", 2)]),
    ("inv16kext_first", None, [("A", "I", 3)]),
    ("inv16kext_second", "t1_inv16kext_train.log", [("B", "I", 3), ("A", "I", 4), ("B", "I", 4), ("A", "I", 5),
                                                    ("B", "I", 5), ("A", "I", 6), ("B", "I", 6), ("A", "I", 7),
                                                    ("B", "I", 7)]),
]
ARMSET_OF = {"dose16k": "dose16k", "dose16krep": "dose16krep", "dose16krep2": "dose16krep2", "inv16k": "inv16k",
             "inv16kext_first": "inv16kext", "inv16kext_second": "inv16kext"}
# what each armset block of t1_ladder.py must define: (tag, field, alpha, seed) at STEPS = 16000
ARMSET_EXPECT = {
    "dose16k": [(f"dose16k_{b}_s0", "none" if b == "A" else "noneB", 0.0, 0) for b in "AB"],
    "dose16krep": [(f"dose16k_{b}_s{s}", "none" if b == "A" else "noneB", 0.0, s) for s in (1, 2) for b in "AB"],
    "dose16krep2": [(f"dose16k_{b}_s{s}", "none" if b == "A" else "noneB", 0.0, s) for s in (3, 4, 5) for b in "AB"],
    "inv16k": [(f"inv16k_{b}_s{s}", f"inv{b}", 0.0, s) for s in (0, 1, 2) for b in "AB"],
    "inv16kext": [(f"inv16kext_{b}_s{s}", f"inv{b}", 0.0, s) for s in (3, 4, 5, 6, 7) for b in "AB"],
}

# F8: 2000-step arms on body A's unchanged crops (none_a0) plus an added pattern, or v1's Colab-decoded crops.
F8_FAMILIES = [
    ("colab", "colab", 0.0, "colab_a0", "v1's Colab-decoded crops of the same photographs: a decoder change, no added pattern", [0, 1, 2]),
    ("mark_rand_a3", "rand", 3.0, "rand_a3", "random +-1 field at 3x the fingerprint RMS, multiplicative, rounded (Entry 01)", [0, 1, 2]),
    ("gkadd_a1", "gkadd", 1.0, "gkadd_a1", "Gaussian field with K's spectrum, additive, 1 gray (Entry 59, E2)", [0, 1]),
    ("gkadd_a4", "gkadd", 4.0, "gkadd_a4", "Gaussian field with K's spectrum, additive, 4 gray (Entry 59, E2)", [0, 1]),
    ("gkmul_a4", "gkmul", 4.0, "gkmul_a4", "Gaussian field with K's spectrum, multiplicative, equal RMS to the 4-gray additive arm (Entry 59, E2)", [0, 1]),
    ("kinj_a12", "kinj", 12.0, "kinj_a12", "body B's E2 fingerprint estimate injected at 12x (Entry 59, E1)", [0, 1, 2]),
    ("band0", "band0", 4.0, "band0_a4", "band-pass Gaussian field, 0.25-0.5 cycles/px (finest octave), additive, 4 gray (Entries 37, 51)", [0, 1, 2]),
    ("band1", "band1", 4.0, "band1_a4", "band-pass Gaussian field, 0.125-0.25 cycles/px, additive, 4 gray (Entries 37, 51)", [0, 1, 2]),
    ("per24", "per24", 4.0, "per24_a4", "periodic tile, 24-px period, additive, 4 gray (Entries 51, 59)", [0, 1, 2]),
    ("per28", "per28", 4.0, "per28_a4", "periodic tile, 28-px period, additive, 4 gray (Entries 51, 59)", [0, 1, 2]),
    ("per32", "per32", 4.0, "per32_a4", "periodic tile, 32-px period, additive, 4 gray (Entries 40, 51)", [0, 1, 2]),
    ("per36", "per36", 4.0, "per36_a4", "periodic tile, 36-px period, additive, 4 gray (Entries 40, 51)", [0, 1, 2]),
    ("per40", "per40", 4.0, "per40_a4", "periodic tile, 40-px period, additive, 4 gray (Entries 51, 59)", [0, 1, 2]),
    ("per48", "per48", 4.0, "per48_a4", "periodic tile, 48-px period, additive, 4 gray (Entries 51, 59)", [0, 1, 2]),
    ("wm_ds", "dswm", 1.0, "dswm_a1", "DiffusionShield released watermark (Entry 26)", [0, 1, 2]),
]


def tag(b, c, s):
    if c == "N":
        return f"dose16k_{b}_s{s}"
    return f"inv16k_{b}_s{s}" if s <= 2 else f"inv16kext_{b}_s{s}"


KEYS = [(b, c, s) for b in "AB" for c in "NI" for s in (PAIRED if c == "N" else INV_SEEDS)]   # 28
IDX = {k: i for i, k in enumerate(KEYS)}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_folder(folder, pattern="*.png"):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    h = hashlib.sha256()
    for p in files:
        h.update(os.path.basename(p).encode())
        h.update(sha256_file(p).encode())
    return {"folder": folder, "n_files": len(files), "sha256_of_name_and_file_sha256_list": h.hexdigest()}


def utc(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat()


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------------
# Gram matrices
# ---------------------------------------------------------------------------------------------------------------
def gram_h5(H5, W, keys):
    n = len(keys)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            G[i, j] = G[j, i] = H5.gram(W[keys[i]], W[keys[j]])
    return G


def gram_hadamard(W, keys, layers):
    """Second route: per layer, sum_{a,b} (B_x^T B_y)[a,b] (A_x A_y^T)[a,b] from two stacked Gram matrices."""
    n = len(keys)
    G = np.zeros((n, n))
    for layer in layers:
        As = np.concatenate([W[k][layer][0] for k in keys], axis=0)           # (n r, in)
        Bs = np.concatenate([W[k][layer][1].T for k in keys], axis=0)         # (n r, out)
        r = W[keys[0]][layer][0].shape[0]
        P = Bs @ Bs.T                                                          # (x,a),(y,b) -> (B_x^T B_y)[a,b]
        Q = As @ As.T                                                          # (x,a),(y,b) -> (A_x A_y^T)[a,b]
        G += (P * Q).reshape(n, r, n, r).sum(axis=(1, 3))
    return G


def explicit_dw_check(W, pairs, layers):
    """<B_x A_x, B_y A_y>_F with dW formed explicitly, against the per-layer trace formula, for a few layers."""
    out = []
    for (kx, ky) in pairs:
        for layer in layers:
            Ax, Bx = W[kx][layer]
            Ay, By = W[ky][layer]
            explicit = float(np.sum((Bx @ Ax) * (By @ Ay)))
            trace = float(np.trace((Bx.T @ By) @ (Ay @ Ax.T)))
            out.append({"x": tag(*kx), "y": tag(*ky), "layer": layer, "explicit": explicit, "trace_formula": trace,
                        "rel_diff": abs(explicit - trace) / max(abs(explicit), 1e-300)})
    return out


# ---------------------------------------------------------------------------------------------------------------
# relabelling statistics
# ---------------------------------------------------------------------------------------------------------------
BITS6 = [tuple(b) for b in itertools.product((0, 1), repeat=6)]       # identity first
DISTINCT6 = [b for b in BITS6 if b[0] == 0]                             # one of each complementary pair (32)


def d_enumerate(Cb, members, swap):
    """members: list of (seed, true_cond 0=N 1=I, matrix index). swap: dict seed -> 0/1 (absent seeds never swap).
    Returns (D, n_within, n_cross, mean_within, mean_cross)."""
    lab = [cond ^ swap.get(seed, 0) for seed, cond, _ in members]
    within, cross = [], []
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            if members[a][0] == members[b][0]:
                continue
            v = Cb[members[a][2], members[b][2]]
            (within if lab[a] == lab[b] else cross).append(v)
    mw, mc = float(np.mean(within)), float(np.mean(cross))
    return mw - mc, len(within), len(cross), mw, mc


def t_matrix(C, b, seeds):
    T = {}
    for j, k in itertools.combinations(seeds, 2):
        nj, nk, ij, ik = IDX[(b, "N", j)], IDX[(b, "N", k)], IDX[(b, "I", j)], IDX[(b, "I", k)]
        T[(j, k)] = C[nj, nk] + C[ij, ik] - C[nj, ik] - C[ij, nk]
    return T


def d_closed(T, swap, n_pairs_each):
    return float(sum(((-1) ** (swap[j] + swap[k])) * v for (j, k), v in T.items()) / n_pairs_each)


def exact_p(null, obs):
    n_ge = int(np.sum(null >= obs - TIE))
    return n_ge / len(null), n_ge, int(np.sum(null >= obs))


def null_summary(null, obs):
    sd = float(np.std(null, ddof=1))
    return {"n": int(len(null)), "mean": float(np.mean(null)), "sd": sd, "min": float(np.min(null)),
            "max": float(np.max(null)), "z_of_observed": (float((obs - np.mean(null)) / sd) if sd > 0 else None),
            "n_distinct_at_1e-13": int(len(np.unique(np.round(null, 13)))),
            "second_largest": float(np.sort(null)[-2]) if len(null) > 1 else None}


# ---------------------------------------------------------------------------------------------------------------
# F3 / F4 / F5 / F7 helpers
# ---------------------------------------------------------------------------------------------------------------
def ladder_parts(text):
    """The parts of t1_ladder.py that determine how an adapter is trained, normalised for line endings."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [l.rstrip() for l in lines]
    consts = [l for l in lines if re.match(r"^(V2|DEV|HF_MODEL|CAPTION|LORA_RANK, LORA_TARGETS|"
                                           r"LR, STEPS, BATCH, GRAD_ACC|PROMPT_PT) *=", l)]

    def block(start_pat):
        i0 = next((i for i, l in enumerate(lines) if re.match(start_pat, l)), None)
        if i0 is None:
            return None
        i1 = next((i for i in range(i0 + 1, len(lines)) if lines[i] and not lines[i].startswith((" ", "\t"))),
                  len(lines))
        return "\n".join(lines[i0:i1]).rstrip()

    funcs = {name: block(rf"^def {name}\(") for name in ("load_prompt", "train_arm", "train_dir", "adapter_dir",
                                                          "stage_train")}
    main = block(r"^if __name__ == \"__main__\":")
    main_tf32 = None
    if main:
        main_tf32 = "\n".join(l for l in main.split("\n") if "allow_tf32" in l)
    armsets = {}
    for name in ("dose16k", "dose16krep", "dose16krep2", "inv16k", "inv16kext"):
        armsets[name] = block(rf"^if os\.environ\.get\(\"T1_ARMSET\"\) == \"{name}\":")
    recipe = "\n#--\n".join(consts + [funcs[k] or "" for k in sorted(funcs)] + [main_tf32 or ""])
    return {"constants": consts, "functions_present": {k: v is not None for k, v in funcs.items()},
            "recipe_sha256": hashlib.sha256(recipe.encode()).hexdigest(),
            "train_arm_sha256": hashlib.sha256((funcs["train_arm"] or "").encode()).hexdigest(),
            "main_tf32": main_tf32, "armsets": armsets}


def armset_ok(block, expect):
    """Does the armset block define exactly the expected (tag, field, alpha, seed) arms at 16000 steps?"""
    if block is None:
        return False, "absent"
    code = block.split("\n", 1)[1]
    code = "\n".join(l[4:] if l.startswith("    ") else l for l in code.split("\n"))
    ns = {"ARMS": None, "STEPS": 2000, "G_PER_ADAPTER": 500}
    # the block only assigns literals and comprehensions; evaluate it with no builtins except range
    exec(compile(code, "<armset>", "exec"), {"__builtins__": {"range": range}}, ns)
    arms = sorted(tuple(a) for a in ns["ARMS"])
    ok = arms == sorted(expect) and ns["STEPS"] == 16000
    return ok, {"arms": [list(a) for a in arms], "steps": ns["STEPS"]}


def parse_loss_logs(names):
    pat = re.compile(r"^\[(\d\d:\d\d:\d\d)\]\s+(\S+)\s+(\d+)/(\d+)\s+loss\s+([0-9.]+)", re.M)
    rows = {}
    for name in names:
        txt = open(os.path.join(LOGS, name), encoding="utf-8", errors="replace").read().replace("\r", "\n")
        for m in pat.finditer(txt):
            rows.setdefault(m.group(2), []).append((int(m.group(3)), int(m.group(4)), float(m.group(5))))
    traces = {}
    for t, r in rows.items():
        steps = [x[0] for x in r]
        starts = [i for i, s in enumerate(steps) if s == 200]
        seq = r[starts[-1]:]
        total = seq[0][1]
        if [x[0] for x in seq] == list(range(200, total + 1, 200)):
            traces[t] = np.array([x[2] for x in seq])
    return traces


def log_start_lines(name):
    pat = re.compile(r"^\[(\d\d:\d\d:\d\d)\]\s+(\S+): 50 imgs @", re.M)
    txt = open(os.path.join(LOGS, name), encoding="utf-8", errors="replace").read().replace("\r", "\n")
    skips = re.findall(r"^\[(\d\d:\d\d:\d\d)\] skip existing (\S+)", txt, re.M)
    return {m.group(2): m.group(1) for m in pat.finditer(txt)}, skips


_CROP_CACHE = {}


def _crops(folder):
    from PIL import Image
    if folder not in _CROP_CACHE:
        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        _CROP_CACHE[folder] = [np.asarray(Image.open(p).convert("RGB"), np.float32) for p in files]
    return _CROP_CACHE[folder]


def rms_change(dir_a, dir_b, n=50, cache_b=False):
    A = _crops(dir_a)
    B = _crops(dir_b)
    assert len(A) == len(B) == n, (dir_a, len(A), dir_b, len(B))
    if not cache_b:
        _CROP_CACHE.pop(dir_b, None)
    same = [float(np.sqrt(np.mean((B[i] - A[i]) ** 2))) for i in range(n)]
    nxt = [float(np.sqrt(np.mean((B[i] - A[(i + 1) % n]) ** 2))) for i in range(n)]
    return {"rms_gray_same_index_mean": float(np.mean(same)), "rms_gray_same_index_max": float(np.max(same)),
            "rms_gray_next_index_mean": float(np.mean(nxt)), "rms_gray_next_index_min": float(np.min(nxt)),
            "index_correspondence": bool(max(same) < min(nxt))}


def f5_inverted_crops():
    """Rebuild invA_a0 / invB_a0 exactly as g1_invert.py wrote them and split the stored change."""
    from PIL import Image
    rng = np.random.default_rng(20260919)                 # g1_invert.py: one generator, body A's 50 crops then B's
    alpha = 6.0
    res = {"rule": "round(clip(Y (1 - 6 K_E1) + u)), u ~ U[-0.5, 0.5) per pixel and channel, rng 20260919, "
                   "A crops 0-49 then B crops 0-49 (g1_invert.py)", "bodies": {}}
    inv = load_json(INVERT_JSON)
    for role, ndir, idir in (("A", "none_a0", "invA_a0"), ("B", "noneB_a0", "invB_a0")):
        K = np.load(os.path.join(FP, f"K_{role}_E1.npy")).astype(np.float32)
        src = sorted(glob.glob(os.path.join(TRAIN_PNG, ndir, "*.png")))
        assert len(src) == 50
        exact, rms_d, ms_d, ms_f, ms_r, cross, clip, r_lag = 0, [], [], [], [], [], [], []
        prev_r = None
        for i, f in enumerate(src):
            rgb = np.asarray(Image.open(f).convert("RGB"), np.float32)
            pre = rgb * (1.0 - alpha * K[..., None])        # float32, as in g1_invert.py
            ys = pre + rng.uniform(-0.5, 0.5, rgb.shape)
            rebuilt = np.clip(np.rint(ys), 0, 255).astype(np.uint8)
            stored = np.asarray(Image.open(os.path.join(TRAIN_PNG, idir, f"{i:04d}.png")).convert("RGB"), np.uint8)
            exact += int(np.array_equal(rebuilt, stored))
            d = stored.astype(np.float64) - rgb.astype(np.float64)
            fterm = pre.astype(np.float64) - rgb.astype(np.float64)       # the reversed fingerprint term, -6 K Y
            r = d - fterm                                                  # dither, rounding and clipping
            rms_d.append(float(np.sqrt(np.mean(d ** 2))))
            ms_d.append(float(np.mean(d ** 2)))
            ms_f.append(float(np.mean(fterm ** 2)))
            ms_r.append(float(np.mean(r ** 2)))
            cross.append(float(2 * np.mean(fterm * r)))
            clip.append(float(((ys < 0) | (ys > 255)).mean()))
            if prev_r is not None:
                a, b_ = prev_r.ravel() - prev_r.mean(), r.ravel() - r.mean()
                r_lag.append(float(a @ b_ / np.sqrt((a @ a) * (b_ @ b_))))
            prev_r = r
        rec = inv["bodies"][role]
        res["bodies"][role] = {
            "normal_crops": ndir, "inverted_crops": idir, "n": 50, "bit_exact_rebuilds": exact,
            "change_rms_gray_mean": float(np.mean(rms_d)),
            "invert_json_change_rms_gray": rec["change_rms_gray"],
            "abs_diff_vs_invert_json": abs(float(np.mean(rms_d)) - rec["change_rms_gray"]),
            "clipped_fraction_mean": float(np.mean(clip)),
            "invert_json_clipped_fraction": rec["clipped_pixel_fraction"],
            "mean_square_gray2": {"total_change": float(np.mean(ms_d)), "fingerprint_term": float(np.mean(ms_f)),
                                  "dither_rounding_clipping": float(np.mean(ms_r)),
                                  "cross_term": float(np.mean(cross))},
            "rms_gray": {"fingerprint_term": float(np.sqrt(np.mean(ms_f))),
                         "dither_rounding_clipping": float(np.sqrt(np.mean(ms_r)))},
            "share_of_mean_square": {"fingerprint_term": float(np.mean(ms_f) / np.mean(ms_d)),
                                     "dither_rounding_clipping": float(np.mean(ms_r) / np.mean(ms_d)),
                                     "cross_term": float(np.mean(cross) / np.mean(ms_d))},
            "residual_correlation_between_consecutive_crops_mean": float(np.mean(r_lag)),
            "residual_correlation_between_consecutive_crops_max_abs": float(np.max(np.abs(r_lag))),
            "note": ("the fingerprint term is -6 K_E1 Y with one K for all 50 crops (a fixed pattern modulated by "
                     "content); the residual is drawn afresh per crop and per pixel (independent across crops)")}
    return res


# ---------------------------------------------------------------------------------------------------------------
def main(dst):
    if os.path.exists(dst):
        sys.exit(f"[wt-recheck] {dst} exists; results are never overwritten")
    t0 = time.time()
    sys.path.insert(0, SRC)
    import h5_weight_signature as H5
    assert os.path.normcase(os.path.abspath(H5.ADA)) == os.path.normcase(os.path.abspath(ADA)), H5.ADA
    import torch
    import safetensors
    record = load_json(RECORD_JSON)

    # ---- inputs ------------------------------------------------------------------------------------------------
    inputs = {}
    for k in KEYS:
        b, c, s = k
        t = tag(*k)
        wp = f"{ADA}/{t}/pytorch_lora_weights.safetensors"
        meta = load_json(f"{ADA}/{t}/train_meta.json")
        assert os.path.getsize(wp) == BYTES, (t, os.path.getsize(wp))
        assert (meta["seed"], meta["steps"], meta["rank"], meta["field"], float(meta["alpha"])) == \
               (s, 16000, 16, FIELD[(b, c)], 0.0), (t, meta)
        sh = sha256_file(wp)
        rec_sh = record["inputs"]["adapters"][t]["sha256"]
        inputs[t] = {"body": b, "device": DEVICE[b], "condition": "normal" if c == "N" else "inverted", "seed": s,
                     "bytes": os.path.getsize(wp), "sha256": sh, "sha256_equals_file_of_record": sh == rec_sh,
                     "weights_mtime_utc": utc(os.path.getmtime(wp)), "train_meta": meta,
                     "training_start_utc_estimate": utc(os.path.getmtime(wp) - 60 * meta["minutes"])}
    assert all(v["sha256_equals_file_of_record"] for v in inputs.values())
    for b in "AB":
        for s in (6, 7):
            assert not os.path.exists(f"{ADA}/dose16k_{b}_s{s}"), (b, s)
    print(f"[wt-recheck] 28 adapters: sizes, train_meta and SHA-256 agree with the file of record "
          f"({time.time() - t0:.1f} s)", flush=True)

    # ---- weights and Gram matrices -----------------------------------------------------------------------------
    W = {k: H5.load_pairs(tag(*k)) for k in KEYS}
    layers = sorted(W[KEYS[0]].keys())
    for k in KEYS:
        assert sorted(W[k].keys()) == layers, tag(*k)
        for layer in layers:
            assert W[k][layer][0].shape == W[KEYS[0]][layer][0].shape and \
                   W[k][layer][1].shape == W[KEYS[0]][layer][1].shape, (tag(*k), layer)
    n_params = int(sum(a.size + bb.size for a, bb in W[KEYS[0]].values()))
    G = gram_h5(H5, W, KEYS)
    G2 = gram_hadamard(W, KEYS, layers)
    gram_rel = float(np.max(np.abs(G - G2)) / np.max(np.abs(G)))
    assert gram_rel < 1e-10, gram_rel
    nrm = np.sqrt(np.diag(G))
    C = G / np.outer(nrm, nrm)
    np.fill_diagonal(C, 1.0)
    dw_pairs = [(("A", "N", 0), ("A", "I", 0)), (("A", "N", 1), ("A", "I", 2)), (("B", "N", 3), ("B", "I", 7)),
                (("A", "I", 4), ("B", "I", 5))]
    dw_layers = [layers[0], layers[len(layers) // 2], layers[-1]]
    dw = explicit_dw_check(W, dw_pairs, dw_layers)
    assert max(x["rel_diff"] for x in dw) < 1e-10
    rec_order = record["gram"]["order"]
    rec_G = np.array(record["gram"]["inner_products"])
    perm = [rec_order.index(tag(*k)) for k in KEYS]
    gram_vs_record = float(np.max(np.abs(G - rec_G[np.ix_(perm, perm)])) / np.max(np.abs(G)))
    H = load_json(H5_JSON)["doses"]["16000"]
    hidx = [IDX[("A", "N", s)] for s in PAIRED] + [IDX[("B", "N", s)] for s in PAIRED]
    assert H["adapters"] == [tag(*KEYS[i]) for i in hidx]
    h5_diff = float(np.max(np.abs(C[np.ix_(hidx, hidx)] - np.array(H["cosine_matrix"]))))
    print(f"[wt-recheck] Gram: two routes agree to {gram_rel:.1e}; file of record {gram_vs_record:.1e}; "
          f"H5 cosines {h5_diff:.1e} ({time.time() - t0:.1f} s)", flush=True)

    # ---- registered primary --------------------------------------------------------------------------------------
    per_body, v64, v32, Tm = {}, {}, {}, {}
    for b in "AB":
        members = [(s, 0, IDX[(b, "N", s)]) for s in PAIRED] + [(s, 1, IDX[(b, "I", s)]) for s in PAIRED]
        T = t_matrix(C, b, PAIRED)
        Tm[b] = T
        enum_vals, closed_vals = [], []
        for bits in BITS6:
            sw = dict(zip(PAIRED, bits))
            d_e, nw, nx, mw, mc = d_enumerate(C, members, sw)
            assert (nw, nx) == (30, 30)
            enum_vals.append(d_e)
            closed_vals.append(d_closed(T, sw, 30))
        enum_vals, closed_vals = np.array(enum_vals), np.array(closed_vals)
        route_diff = float(np.max(np.abs(enum_vals - closed_vals)))
        assert route_diff < 1e-15, route_diff
        comp_diff = float(max(abs(enum_vals[BITS6.index(bb)] - enum_vals[BITS6.index(tuple(1 - x for x in bb))])
                              for bb in BITS6))
        v64[b] = enum_vals
        v32[b] = np.array([enum_vals[BITS6.index(bb)] for bb in DISTINCT6])
        d_obs, nw, nx, mw, mc = d_enumerate(C, members, {})
        assert d_obs == v32[b][0]
        nn = [C[IDX[(b, "N", j)], IDX[(b, "N", k)]] for j, k in itertools.combinations(PAIRED, 2)]
        ii = [C[IDX[(b, "I", j)], IDX[(b, "I", k)]] for j, k in itertools.combinations(PAIRED, 2)]
        p_b, nge_b, nge_b_exact = exact_p(v32[b], d_obs)
        per_body[b] = {"device": DEVICE[b], "D_X": d_obs, "mean_within": mw, "mean_cross": mc, "n_within": nw,
                       "n_cross": nx, "mean_normal_normal": float(np.mean(nn)),
                       "mean_inverted_inverted": float(np.mean(ii)),
                       "p_one_sided_exact": p_b, "n_ge_observed": nge_b, "n_ge_observed_without_tolerance": nge_b_exact,
                       "n_distinct_relabellings": 32, "floor": 1 / 32,
                       "rank_of_observed_among_32": int(1 + np.sum(v32[b] > d_obs + TIE)),
                       "null_32": v32[b].tolist(), "null_64_in_bit_order": enum_vals.tolist(),
                       "enumeration_vs_closed_form_max_abs_diff": route_diff,
                       "whole_body_exchange_max_abs_diff": comp_diff,
                       "null_summary": null_summary(v32[b], d_obs)}
    D1024 = ((v32["A"][:, None] + v32["B"][None, :]) / 2).ravel()
    D4096 = ((v64["A"][:, None] + v64["B"][None, :]) / 2).ravel()
    D_obs = float(D1024[0])
    assert abs(D_obs - (per_body["A"]["D_X"] + per_body["B"]["D_X"]) / 2) < 1e-18
    p, n_ge, n_ge_exact = exact_p(D1024, D_obs)
    p4096, n_ge4096, _ = exact_p(D4096, D_obs)
    if p < LEVEL and D_obs > 0:
        reading, scope = READ_POS, SCOPE_POS
    elif p >= LEVEL:
        reading, scope = READ_NEG, SCOPE_NEG
    else:
        reading, scope = "not among the registered readings (p < 0.01 with D <= 0)", "none"
    primary = {"statistic": "D = (D_A + D_B)/2 as registered (see header)", "D": D_obs,
               "p_one_sided_exact": p, "n_ge_observed": n_ge, "n_ge_observed_without_tolerance": n_ge_exact,
               "n_relabellings": 1024, "floor": 1 / 1024, "level": LEVEL, "direction": "D > 0",
               "rank_of_observed_among_1024": int(1 + np.sum(D1024 > D_obs + TIE)),
               "p_over_all_4096_relabellings": p4096, "n_ge_observed_4096": n_ge4096,
               "null_summary": null_summary(D1024, D_obs), "null_1024_sorted": np.sort(D1024).tolist(),
               "per_body": per_body, "reading": reading, "reading_scope": scope}
    print(f"[wt-recheck] D_A {per_body['A']['D_X']:+.6e} (p {per_body['A']['p_one_sided_exact']:.5f})  "
          f"D_B {per_body['B']['D_X']:+.6e} (p {per_body['B']['p_one_sided_exact']:.5f})  D {D_obs:+.6e}  "
          f"p {p:.6f} ({n_ge}/1024, floor {1 / 1024:.6f})", flush=True)
    print(f"[wt-recheck] reading: {reading}", flush=True)

    # ---- registered descriptives -----------------------------------------------------------------------------------
    twins = {}
    for b in "AB":
        tc = [float(C[IDX[(b, "N", j)], IDX[(b, "I", j)]]) for j in PAIRED]
        others = [C[IDX[x], IDX[y]] for x, y in itertools.combinations([k for k in KEYS if k[0] == b], 2)
                  if x[2] != y[2]]
        twins[b] = {"by_seed": tc, "mean": float(np.mean(tc)), "min": float(np.min(tc)), "max": float(np.max(tc)),
                    "largest_different_seed_cosine_in_body": float(np.max(others))}

    def dprod(x1, x0, y1, y0):          # <dW(x1) - dW(x0), dW(y1) - dW(y0)> from G
        return G[x1, y1] - G[x1, y0] - G[x0, y1] + G[x0, y0]

    dn = {}
    for b in "AB":
        for j in PAIRED:
            i1, i0 = IDX[(b, "I", j)], IDX[(b, "N", j)]
            dn[(b, j)] = float(np.sqrt(dprod(i1, i0, i1, i0)))

    def dcos(b1, j, b2, k):
        return float(dprod(IDX[(b1, "I", j)], IDX[(b1, "N", j)], IDX[(b2, "I", k)], IDX[(b2, "N", k)])
                     / (dn[(b1, j)] * dn[(b2, k)]))

    # second route for the twin differences: exact rank-32 factors, Hadamard Gram route
    DW = {}
    for b in "AB":
        for j in PAIRED:
            wi, wn = W[(b, "I", j)], W[(b, "N", j)]
            DW[(b, j)] = {l: (np.vstack([wi[l][0], wn[l][0]]), np.hstack([wi[l][1], -wn[l][1]])) for l in layers}
    dkeys = [(b, j) for b in "AB" for j in PAIRED]
    GD = gram_hadamard(DW, dkeys, layers)
    GDx = np.array([[dprod(IDX[(b1, "I", j)], IDX[(b1, "N", j)], IDX[(b2, "I", k)], IDX[(b2, "N", k)])
                     for (b2, k) in dkeys] for (b1, j) in dkeys])
    delta_route = float(np.max(np.abs(GD - GDx)) / np.max(np.abs(GD)))
    assert delta_route < 1e-9, delta_route
    del DW
    m32 = {}
    for b in "AB":
        vals = []
        for bits in DISTINCT6:
            sw = dict(zip(PAIRED, bits))
            vals.append(float(np.mean([((-1) ** (sw[j] + sw[k])) * dcos(b, j, b, k)
                                       for j, k in itertools.combinations(PAIRED, 2)])))
        m32[b] = np.array(vals)
    M1024 = ((m32["A"][:, None] + m32["B"][None, :]) / 2).ravel()
    M_obs = float(M1024[0])
    pM, nM, _ = exact_p(M1024, M_obs)
    matched = {"definition": ("Delta_j = dW(I_j) - dW(N_j); M_X = mean over the 15 within-body pairs j < k of "
                              "cos(Delta_j, Delta_k); M = (M_A + M_B)/2; exchanging seed pair j flips Delta_j's sign; "
                              "1,024 distinct relabellings (the cross-body pairs are in F2)"),
               "M": M_obs, "p_one_sided_exact": pM, "n_ge_observed": nM, "floor": 1 / 1024,
               "per_body": {b: {"M_X": float(m32[b][0]), "p_one_sided_exact_32": exact_p(m32[b], float(m32[b][0]))[0],
                                "cosines": {f"{j}-{k}": dcos(b, j, b, k) for j, k in itertools.combinations(PAIRED, 2)},
                                "delta_norm": [dn[(b, j)] for j in PAIRED],
                                "delta_norm_over_normal_norm": [dn[(b, j)] / nrm[IDX[(b, "N", j)]] for j in PAIRED]}
                            for b in "AB"},
               "factor_route_vs_expansion_max_rel_diff": delta_route,
               "null_summary": null_summary(M1024, M_obs)}
    # first-order link between D_X and M_X: D_X ~ (1/2) (|Delta| / |dW|)^2 M_X
    for b in "AB":
        q = float(np.mean([(dn[(b, j)] / nrm[IDX[(b, "N", j)]]) ** 2 for j in PAIRED]))
        matched["per_body"][b]["half_relative_delta_energy_times_M_X"] = 0.5 * q * float(m32[b][0])
        matched["per_body"][b]["D_X_for_comparison"] = per_body[b]["D_X"]

    s67_vals, s67 = {}, {}
    for b in "AB":
        members = [(s, 0, IDX[(b, "N", s)]) for s in PAIRED] + [(s, 1, IDX[(b, "I", s)]) for s in INV_SEEDS]
        vals = []
        for bits in BITS6:
            d_e, nw, nx, mw, mc = d_enumerate(C, members, dict(zip(PAIRED, bits)))
            assert (nw, nx) == (43, 42)
            vals.append(d_e)
        s67_vals[b] = np.array(vals)
        d_e, nw, nx, mw, mc = d_enumerate(C, members, {})
        s67[b] = {"D_X": d_e, "mean_within": mw, "mean_cross": mc, "n_within": nw, "n_cross": nx,
                  "p_one_sided_exact_64": exact_p(s67_vals[b], d_e)[0], "floor": 1 / 64}
    V4096 = ((s67_vals["A"][:, None] + s67_vals["B"][None, :]) / 2).ravel()
    s67_obs = float(V4096[0])
    p67, n67, _ = exact_p(V4096, s67_obs)
    seeds67 = {"definition": ("inverted seeds 6-7 always labelled inverted; within = pooled mean over the 43 same-label "
                              "different-seed pairs (15 + 28), cross = pooled mean over the 42 different-label "
                              "different-seed pairs; 64 relabellings per body, 4,096 in all"),
               "D": s67_obs, "p_one_sided_exact": p67, "n_ge_observed": n67, "n_relabellings": 4096,
               "floor": 1 / 4096, "per_body": s67, "null_summary": null_summary(V4096, s67_obs)}

    norms = {}
    for b in "AB":
        rec = {}
        for c, seeds in (("N", PAIRED), ("I", INV_SEEDS)):
            fro = [float(nrm[IDX[(b, c, s)]]) for s in seeds]
            lbn = [float(inputs[tag(b, c, s)]["train_meta"]["lora_B_norm"]) for s in seeds]
            rec["normal" if c == "N" else "inverted"] = {
                "seeds": seeds, "dW_frobenius": fro, "dW_frobenius_mean": float(np.mean(fro)),
                "dW_frobenius_sd": float(np.std(fro, ddof=1)), "lora_B_norm": lbn,
                "lora_B_norm_mean": float(np.mean(lbn)), "lora_B_norm_sd": float(np.std(lbn, ddof=1))}
        rec["twin_difference_I_minus_N_dW_frobenius"] = [float(nrm[IDX[(b, "I", j)]] - nrm[IDX[(b, "N", j)]])
                                                         for j in PAIRED]
        rec["twin_difference_I_minus_N_dW_frobenius_mean"] = float(np.mean(rec["twin_difference_I_minus_N_dW_frobenius"]))
        norms[b] = rec

    # ---- comparison with the file of record ------------------------------------------------------------------------
    rp = record["primary"]
    rd = record["descriptive"]
    cmp_ = {
        "D_abs_diff": abs(D_obs - rp["D"]),
        "D_A_abs_diff": abs(per_body["A"]["D_X"] - rp["per_body"]["A"]["D_X"]),
        "D_B_abs_diff": abs(per_body["B"]["D_X"] - rp["per_body"]["B"]["D_X"]),
        "p_equal": p == rp["p_one_sided_exact"],
        "per_body_p_equal": all(per_body[b]["p_one_sided_exact"] == rp["per_body"][b]["p_one_sided_exact"] for b in "AB"),
        "null_1024_sorted_max_abs_diff": float(np.max(np.abs(np.sort(D1024) - np.sort(np.array(rp["null_1024_values"]))))),
        "reading_equal": reading == rp["reading"],
        "twin_cosines_max_abs_diff": float(max(abs(a - b_) for b in "AB" for a, b_ in
                                               zip(twins[b]["by_seed"], rd["twin_cosines"][b]["cosines_by_seed"]))),
        "matched_M_abs_diff": abs(M_obs - rd["matched_variant"]["M"]),
        "matched_p_equal": pM == rd["matched_variant"]["p_one_sided_exact"],
        "seeds67_D_abs_diff": abs(s67_obs - rd["seeds_6_7_kept"]["D"]),
        "seeds67_p_equal": p67 == rd["seeds_6_7_kept"]["p_one_sided_exact"],
        "norms_max_abs_diff": float(max(abs(a - b_) for b in "AB" for c in ("normal", "inverted") for a, b_ in
                                        zip(norms[b][c]["dW_frobenius"], rd["norms"][b][c]["dW_frobenius"]))),
        "gram_inner_products_max_rel_diff": gram_vs_record,
    }
    cmp_["all_agree"] = bool(cmp_["D_abs_diff"] < 1e-15 and cmp_["D_A_abs_diff"] < 1e-15 and cmp_["D_B_abs_diff"] < 1e-15
                             and cmp_["p_equal"] and cmp_["per_body_p_equal"] and cmp_["reading_equal"]
                             and cmp_["null_1024_sorted_max_abs_diff"] < 1e-15 and cmp_["twin_cosines_max_abs_diff"] < 1e-14
                             and cmp_["matched_M_abs_diff"] < 1e-13 and cmp_["matched_p_equal"]
                             and cmp_["seeds67_D_abs_diff"] < 1e-15 and cmp_["seeds67_p_equal"]
                             and cmp_["norms_max_abs_diff"] < 1e-12 and cmp_["gram_inner_products_max_rel_diff"] < 1e-12)
    print(f"[wt-recheck] file of record reproduced: {cmp_['all_agree']} ({time.time() - t0:.1f} s)", flush=True)

    # ================================================================================================================
    # skeptic checks (not pre-specified)
    # ================================================================================================================
    sk = {"status": ("not pre-specified; written after the registered result was known from the file of record; "
                     "descriptive context, no reading attaches")}

    # F1 per-pair contributions and leave-one-seed-out ------------------------------------------------------------
    f1 = {"per_pair_T_over_30": {}, "leave_one_seed_out": {}}
    for b in "AB":
        vals = {f"{j}-{k}": float(v / 30) for (j, k), v in Tm[b].items()}
        f1["per_pair_T_over_30"][b] = {"values": vals, "n_positive": int(sum(v > 0 for v in vals.values())),
                                       "n": 15, "min": float(min(vals.values())), "max": float(max(vals.values())),
                                       "sum_equals_D_X": abs(sum(vals.values()) - per_body[b]["D_X"]) < 1e-15}
    bits5 = [tuple(x) for x in itertools.product((0, 1), repeat=5) if x[0] == 0]      # 16 distinct per body
    for m in PAIRED:
        keep = [s for s in PAIRED if s != m]
        per = {}
        nulls = {}
        for b in "AB":
            T = {jk: v for jk, v in Tm[b].items() if m not in jk}
            nulls[b] = np.array([d_closed(T, dict(zip(keep, bits)), 20) for bits in bits5])
            per[b] = float(nulls[b][0])
        N256 = ((nulls["A"][:, None] + nulls["B"][None, :]) / 2).ravel()
        f1["leave_one_seed_out"][str(m)] = {"D_A": per["A"], "D_B": per["B"], "D": float(N256[0]),
                                            "p_one_sided_exact_256": exact_p(N256, float(N256[0]))[0],
                                            "floor": 1 / 256}
    f1["leave_one_seed_out_D_range"] = [min(v["D"] for v in f1["leave_one_seed_out"].values()),
                                        max(v["D"] for v in f1["leave_one_seed_out"].values())]
    sk["F1_per_pair_and_leave_one_seed_out"] = f1

    # F2 common versus body-specific ----------------------------------------------------------------------------------
    w_pairs = [dcos(b, j, b, k) for b in "AB" for j, k in itertools.combinations(PAIRED, 2)]
    X = np.array([[dcos("A", j, "B", k) for k in PAIRED] for j in PAIRED])
    off = ~np.eye(6, dtype=bool)
    w_mean, x_mean = float(np.mean(w_pairs)), float(X[off].mean())
    # registered-style relabelling of the cross-body part (condition labels exchanged within seed pairs, both bodies)
    xnull = np.array([float(np.mean([((-1) ** (sa[j] + sb[k])) * X[j, k] for j in PAIRED for k in PAIRED if j != k]))
                      for sa in BITS6 for sb in BITS6])
    px, nx_, _ = exact_p(xnull, x_mean)
    # body-specific remainder: body labels of Delta^A_j and Delta^B_j exchanged within seed pairs
    V = {}
    for j, k in itertools.combinations(PAIRED, 2):
        V[(j, k)] = dcos("A", j, "A", k) + dcos("B", j, "B", k) - X[j, k] - X[k, j]
    snull = np.array([d_closed(V, dict(zip(PAIRED, bits)), 30) for bits in DISTINCT6])
    s_obs = float(snull[0])
    assert abs(s_obs - (w_mean - x_mean)) < 1e-15
    ka = np.load(os.path.join(FP, "K_A_E1.npy")).astype(np.float64)
    kb = np.load(os.path.join(FP, "K_B_E1.npy")).astype(np.float64)
    ka -= ka.mean()
    kb -= kb.mean()
    sk["F2_common_versus_body_specific"] = {
        "within_body_delta_alignment_mean_30": w_mean,
        "cross_body_delta_alignment_different_seed_mean_30": x_mean,
        "cross_body_delta_alignment_same_seed_mean_6": float(np.mean(np.diag(X))),
        "cross_body_matrix_cos_DeltaA_j_DeltaB_k": X.tolist(),
        "common_share_of_within_alignment": x_mean / w_mean,
        "cross_body_relabelling": {"p_one_sided_exact": px, "n_ge_observed": nx_, "n_relabellings": 4096,
                                   "n_distinct": 2048, "floor": 1 / 2048, "null_summary": null_summary(xnull, x_mean)},
        "body_specific_remainder": {"value_within_minus_cross": s_obs,
                                    "p_one_sided_exact_32": exact_p(snull, s_obs)[0],
                                    "n_ge_observed": exact_p(snull, s_obs)[1], "floor": 1 / 32,
                                    "null_32": snull.tolist()},
        "ncc_K_A_E1_K_B_E1": float((ka * kb).sum() / np.sqrt((ka * ka).sum() * (kb * kb).sum())),
        "note": ("A direction shared by both bodies' twin differences cannot be either body's own fingerprint pattern, "
                 "because the two reversed patterns are nearly orthogonal; it is what the two inverted conditions have "
                 "in common (added fine-scale structure and the dither field's statistics) or anything else that "
                 "differs between the training weeks. The registered statistic contains both parts, to first order "
                 "in the same proportion (D_X ~ (1/2)(|Delta|/|dW|)^2 M_X).")}
    del ka, kb

    # F3 training-code identity ---------------------------------------------------------------------------------------
    f3 = {"source": FILE_HISTORY, "key": LADDER_KEY,
          "what": ("the editor's file-history snapshots of src/t1_ladder.py (@v1..@vN; each holds the file's content and "
                   "keeps its modification time); the recipe hash covers DEV, HF_MODEL, CAPTION, LORA_RANK/TARGETS, "
                   "LR/STEPS/BATCH/GRAD_ACC, PROMPT_PT, load_prompt, train_arm, train_dir, adapter_dir, stage_train and "
                   "the TF32 settings of __main__; armset blocks are checked separately")}
    snaps = sorted(glob.glob(os.path.join(FILE_HISTORY, LADDER_KEY + "@v*")),
                   key=lambda p: int(p.rsplit("@v", 1)[1]))
    if snaps:
        cur = open(LADDER, "rb").read()
        versions = []
        for pth in snaps:
            raw = open(pth, "rb").read()
            parts = ladder_parts(raw.decode("utf-8"))
            versions.append({"version": int(pth.rsplit("@v", 1)[1]), "path": pth, "sha256": hashlib.sha256(raw).hexdigest(),
                             "mtime_utc": utc(os.path.getmtime(pth)), "mtime": os.path.getmtime(pth),
                             "recipe_sha256": parts["recipe_sha256"], "train_arm_sha256": parts["train_arm_sha256"],
                             "armsets_present": [k for k, v in parts["armsets"].items() if v is not None],
                             "_parts": parts})
        f3["current_file_equals_last_snapshot"] = hashlib.sha256(cur).hexdigest() == versions[-1]["sha256"]
        f3["current_file_mtime_utc"] = utc(os.path.getmtime(LADDER))
        f3["distinct_recipe_hashes_over_all_snapshots"] = sorted({v["recipe_sha256"] for v in versions})
        procs = []
        for name, logname, members in PROCESSES:
            starts = [os.path.getmtime(f"{ADA}/{tag(*k)}/pytorch_lora_weights.safetensors")
                      - 60 * inputs[tag(*k)]["train_meta"]["minutes"] for k in members]
            start = min(starts)
            in_force = [v for v in versions if v["mtime"] <= start]
            v = in_force[-1] if in_force else None
            gap = (start - v["mtime"]) if v else None
            armset = ARMSET_OF[name]
            assert all((tag(*k), FIELD[(k[0], k[1])], 0.0, k[2]) in ARMSET_EXPECT[armset] for k in members), name
            ok, detail = (armset_ok(v["_parts"]["armsets"][armset], ARMSET_EXPECT[armset]) if v
                          else (False, "no snapshot before start"))
            log_check = None
            if logname:
                starts_log, skips = log_start_lines(logname)
                first = tag(*members[0])
                est = datetime.datetime.fromtimestamp(min(starts), datetime.timezone.utc)
                hh = starts_log.get(first)
                if hh:
                    lt = datetime.datetime.strptime(hh, "%H:%M:%S").time()
                    est_t = est.time()
                    diff_s = abs((lt.hour * 3600 + lt.minute * 60 + lt.second) -
                                 (est_t.hour * 3600 + est_t.minute * 60 + est_t.second))
                    diff_s = min(diff_s, 86400 - diff_s)
                else:
                    diff_s = None
                log_check = {"log": logname, "first_arm": first, "log_start_time_of_day": hh,
                             "estimated_start_utc": est.isoformat(), "abs_diff_seconds_time_of_day": diff_s,
                             "skip_existing_lines": skips,
                             "arms_started_in_log": sorted(starts_log.keys())}
            procs.append({"process": name, "adapters": [tag(*k) for k in members],
                          "start_utc_estimate": utc(start), "snapshot_in_force": v["version"] if v else None,
                          "snapshot_mtime_utc": v["mtime_utc"] if v else None,
                          "seconds_between_snapshot_and_start": gap,
                          "recipe_sha256": v["recipe_sha256"] if v else None,
                          "armset_defines_these_arms_at_16000_steps": ok, "armset_detail": detail,
                          "log_check": log_check})
        f3["processes"] = procs
        f3["recipe_identical_across_processes"] = len({p_["recipe_sha256"] for p_ in procs}) == 1
        f3["snapshots"] = [{k: v for k, v in x.items() if not k.startswith("_") and k != "mtime"} for x in versions]
        f3["recipe_constants_last_snapshot"] = versions[-1]["_parts"]["constants"]
        f3["tf32_last_snapshot"] = versions[-1]["_parts"]["main_tf32"]
        f3["limitation"] = ("snapshots exist only for edits made through the editor; an edit made outside it and "
                            "reverted between two snapshots would not be seen")
    else:
        f3["available"] = False
    sk["F3_training_code_identity"] = f3

    # F4 environment identity -----------------------------------------------------------------------------------------
    first_normal_start = min(os.path.getmtime(f"{ADA}/{tag(*k)}/pytorch_lora_weights.safetensors")
                             - 60 * inputs[tag(*k)]["train_meta"]["minutes"] for k in KEYS if k[1] == "N")
    first_inv_start = min(os.path.getmtime(f"{ADA}/{tag(*k)}/pytorch_lora_weights.safetensors")
                          - 60 * inputs[tag(*k)]["train_meta"]["minutes"] for k in KEYS if k[1] == "I")
    site = os.path.join(sys.prefix, "Lib", "site-packages")
    pk = {}
    for name in ("torch", "diffusers", "peft", "transformers", "accelerate", "safetensors", "numpy", "huggingface_hub",
                 "pillow"):
        hits = sorted(d for d in os.listdir(site) if d.lower().startswith(name + "-") and d.endswith(".dist-info"))
        pk[name] = [{"dist_info": d, "mtime_utc": utc(os.path.getmtime(os.path.join(site, d))),
                     "before_first_normal_adapter": os.path.getmtime(os.path.join(site, d)) < first_normal_start}
                    for d in hits]
    model_files = [os.path.join(dp, f) for dp, _, fs in os.walk(HF_MODEL_DIR) for f in fs]
    newest_model = max(os.path.getmtime(p_) for p_ in model_files)
    crops_dates = {}
    for key, d in CROPS.items():
        fs = glob.glob(os.path.join(TRAIN_PNG, d, "*.png"))
        crops_dates[d] = {"n": len(fs), "oldest_utc": utc(min(os.path.getmtime(x) for x in fs)),
                          "newest_utc": utc(max(os.path.getmtime(x) for x in fs))}
    crops_dates["normal_crops_written_before_first_normal_adapter"] = max(
        os.path.getmtime(x) for d in ("none_a0", "noneB_a0") for x in glob.glob(os.path.join(TRAIN_PNG, d, "*.png"))) \
        < first_normal_start
    crops_dates["inverted_crops_written_before_first_inverted_adapter"] = max(
        os.path.getmtime(x) for d in ("invA_a0", "invB_a0") for x in glob.glob(os.path.join(TRAIN_PNG, d, "*.png"))) \
        < first_inv_start
    sk["F4_environment_identity"] = {
        "first_normal_adapter_start_utc": utc(first_normal_start),
        "first_inverted_adapter_start_utc": utc(first_inv_start),
        "interpreter": sys.executable, "packages": pk,
        "all_packages_single_version_installed_before_first_normal_adapter": all(
            len(v) == 1 and v[0]["before_first_normal_adapter"] for v in pk.values()),
        "sd35_model_dir": HF_MODEL_DIR,
        "sd35_refs_main": open(os.path.join(HF_MODEL_DIR, "refs", "main")).read().strip(),
        "sd35_snapshots": sorted(os.listdir(os.path.join(HF_MODEL_DIR, "snapshots"))),
        "sd35_newest_file_mtime_utc": utc(newest_model),
        "sd35_unchanged_since_before_first_normal_adapter": newest_model < first_normal_start,
        "prompt_embeddings": {"path": PROMPT_PT, "sha256": sha256_file(PROMPT_PT),
                              "mtime_utc": utc(os.path.getmtime(PROMPT_PT)),
                              "before_first_normal_adapter": os.path.getmtime(PROMPT_PT) < first_normal_start},
        "crops": crops_dates,
        "gpu_driver_library": ({"path": NVCUDA, "created_utc": utc(os.path.getctime(NVCUDA)),
                                "modified_utc": utc(os.path.getmtime(NVCUDA)),
                                "created_before_first_normal_adapter": os.path.getctime(NVCUDA) < first_normal_start}
                               if os.path.exists(NVCUDA) else "not found")}

    # F5 inverted crops -------------------------------------------------------------------------------------------------
    print(f"[wt-recheck] F5: rebuilding the inverted crops ({time.time() - t0:.1f} s)", flush=True)
    sk["F5_inverted_crops_rebuilt_and_decomposed"] = f5_inverted_crops()

    # F6 training process within condition -------------------------------------------------------------------------------
    proc_of = {k: name for name, _, members in PROCESSES for k in members}
    f6 = {}
    for b in "AB":
        for c, seeds in (("N", PAIRED), ("I", INV_SEEDS)):
            same, diff = [], []
            for j, k in itertools.combinations(seeds, 2):
                v = C[IDX[(b, c, j)], IDX[(b, c, k)]]
                (same if proc_of[(b, c, j)] == proc_of[(b, c, k)] else diff).append(v)
            f6[f"{b}_{'normal' if c == 'N' else 'inverted'}"] = {
                "same_process_mean": float(np.mean(same)) if same else None, "n_same": len(same),
                "different_process_mean": float(np.mean(diff)) if diff else None, "n_different": len(diff),
                "difference": (float(np.mean(same) - np.mean(diff)) if same and diff else None)}
    sk["F6_training_process_within_condition"] = {"processes": {n: [tag(*k) for k in m] for n, _, m in PROCESSES},
                                                  "by_body_and_condition": f6}

    # F7 loss traces ---------------------------------------------------------------------------------------------------------
    traces = parse_loss_logs([p_[1] for p_ in PROCESSES if p_[1]])
    f7 = {"missing": [tag(*k) for k in KEYS if tag(*k) not in traces], "bodies": {}}
    for b in "AB":
        tw = {}
        for j in PAIRED:
            tn, ti = tag(b, "N", j), tag(b, "I", j)
            if tn in traces and ti in traces:
                tw[str(j)] = {"r_first_differences": float(np.corrcoef(np.diff(traces[tn]), np.diff(traces[ti]))[0, 1]),
                              "mean_offset_I_minus_N": float(np.mean(traces[ti] - traces[tn])),
                              "offset_step_200": float(traces[ti][0] - traces[tn][0]),
                              "offset_step_16000": float(traces[ti][-1] - traces[tn][-1])}
        nn = [float(np.corrcoef(np.diff(traces[tag(b, "N", j)]), np.diff(traces[tag(b, "N", k)]))[0, 1])
              for j, k in itertools.combinations(PAIRED, 2) if tag(b, "N", j) in traces and tag(b, "N", k) in traces]
        f7["bodies"][b] = {"twins": tw, "twin_r_mean": float(np.mean([v["r_first_differences"] for v in tw.values()])),
                           "twin_mean_offset": float(np.mean([v["mean_offset_I_minus_N"] for v in tw.values()])),
                           "normal_different_seed_r_mean": float(np.mean(nn)), "n_normal_pairs": len(nn)}
    sk["F7_loss_traces"] = f7

    # F8 / F9 other fixed changes at 2000 steps, and alignment across patterns -----------------------------------------
    print(f"[wt-recheck] F8/F9: 2000-step arms ({time.time() - t0:.1f} s)", flush=True)
    extra, extra_sha = {}, {}
    base_tags = {s: f"nomark_s{s}" for s in (0, 1, 2)}
    for s, t in base_tags.items():
        m = load_json(f"{ADA}/{t}/train_meta.json")
        assert (m["seed"], m["steps"], m["field"], float(m["alpha"])) == (s, 2000, "none", 0.0), (t, m)
    fam_tags = {fam: {s: f"{fam}_s{s}" for s in seeds} for fam, _, _, _, _, seeds in F8_FAMILIES}
    for fam, field, alpha, cdir, desc, seeds in F8_FAMILIES:
        for s, t in fam_tags[fam].items():
            m = load_json(f"{ADA}/{t}/train_meta.json")
            assert (m["seed"], m["steps"], m["field"], float(m["alpha"])) == (s, 2000, field, alpha), (t, m)
    for t in list(base_tags.values()) + [t for fam in fam_tags for t in fam_tags[fam].values()]:
        wp = f"{ADA}/{t}/pytorch_lora_weights.safetensors"
        assert os.path.getsize(wp) == BYTES, t
        extra[t] = H5.load_pairs(t)
        extra_sha[t] = sha256_file(wp)
        assert sorted(extra[t].keys()) == layers, t
    all_tags = [tag(*k) for k in KEYS] + list(extra)
    Wall = {tag(*k): W[k] for k in KEYS}
    Wall.update(extra)
    GA = gram_hadamard(Wall, all_tags, layers)
    IA = {t: i for i, t in enumerate(all_tags)}
    ga_vs_g = float(np.max(np.abs(GA[:len(KEYS), :len(KEYS)] - G)) / np.max(np.abs(G)))
    spot = [("nomark_s0", "colab_s1"), ("mark_rand_a3_s2", "kinj_a12_s0"), ("per32_s1", tag("A", "I", 4)),
            ("band0_s0", tag("B", "N", 2))]
    spot_rel = max(abs(GA[IA[a], IA[b_]] - H5.gram(Wall[a], Wall[b_])) / abs(GA[IA[a], IA[b_]]) for a, b_ in spot)
    assert ga_vs_g < 1e-10 and spot_rel < 1e-10, (ga_vs_g, spot_rel)

    def cosA(a, b_):
        return float(GA[IA[a], IA[b_]] / np.sqrt(GA[IA[a], IA[a]] * GA[IA[b_], IA[b_]]))

    deltas = {}
    for fam in fam_tags:
        for s, t in fam_tags[fam].items():
            deltas[(fam, s)] = (t, base_tags[s])
    for b in "AB":
        for s in PAIRED:
            deltas[(f"inverted_16000_{b}", s)] = (tag(b, "I", s), tag(b, "N", s))

    def gip(k1, k2):
        (a1, a0), (b1, b0) = deltas[k1], deltas[k2]
        return GA[IA[a1], IA[b1]] - GA[IA[a1], IA[b0]] - GA[IA[a0], IA[b1]] + GA[IA[a0], IA[b0]]

    dnorm = {k: float(np.sqrt(gip(k, k))) for k in deltas}

    def dc(k1, k2):
        return float(gip(k1, k2) / (dnorm[k1] * dnorm[k2]))

    f8 = {"twins": base_tags, "adapter_sha256": extra_sha,
          "gram_route": "Hadamard route over all adapters at once; checked against h5_weight_signature.gram",
          "gram_check_primary_block_max_rel_diff": ga_vs_g, "gram_check_spot_pairs_max_rel_diff": float(spot_rel),
          "nomark_different_seed_cosines": {f"{j}-{k}": cosA(base_tags[j], base_tags[k])
                                            for j, k in itertools.combinations((0, 1, 2), 2)},
          "families": {}}
    for fam, field, alpha, cdir, desc, seeds in F8_FAMILIES:
        tf = fam_tags[fam]
        twin = {s: cosA(tf[s], base_tags[s]) for s in seeds}
        align = {f"{j}-{k}": dc((fam, j), (fam, k)) for j, k in itertools.combinations(seeds, 2)}
        within = [cosA(base_tags[j], base_tags[k]) for j, k in itertools.combinations(seeds, 2)] + \
                 [cosA(tf[j], tf[k]) for j, k in itertools.combinations(seeds, 2)]
        cross = [cosA(base_tags[j], tf[k]) for j in seeds for k in seeds if j != k]
        crop = rms_change(os.path.join(TRAIN_PNG, "none_a0"), os.path.join(TRAIN_PNG, cdir))
        f8["families"][fam] = {"description": desc, "field": field, "alpha": alpha, "adapters": list(tf.values()),
                               "crops": cdir, "crop_change_vs_none_a0": crop,
                               "twin_cosine_by_seed": twin, "twin_cosine_mean": float(np.mean(list(twin.values()))),
                               "delta_norm_over_nomark_norm_mean": float(np.mean(
                                   [dnorm[(fam, s)] / np.sqrt(GA[IA[base_tags[s]], IA[base_tags[s]]]) for s in seeds])),
                               "delta_alignment_by_pair": align,
                               "delta_alignment_mean": float(np.mean(list(align.values()))),
                               "registered_style_D": float(np.mean(within) - np.mean(cross)),
                               "n_within": len(within), "n_cross": len(cross)}
        print(f"[wt-recheck]   {fam:13s} crop change {crop['rms_gray_same_index_mean']:.3f} gray, twin "
              f"{f8['families'][fam]['twin_cosine_mean']:.4f}, alignment {f8['families'][fam]['delta_alignment_mean']:+.4f}, "
              f"D-style {f8['families'][fam]['registered_style_D']:+.6f}", flush=True)
    inv_crop = {b: rms_change(os.path.join(TRAIN_PNG, CROPS[(b, "N")]), os.path.join(TRAIN_PNG, CROPS[(b, "I")]))
                for b in "AB"}
    f8["sixteen_thousand_step_inverted_for_comparison"] = {
        b: {"crop_change_vs_normal": inv_crop[b], "twin_cosine_mean": twins[b]["mean"],
            "delta_norm_over_normal_norm_mean": float(np.mean(matched["per_body"][b]["delta_norm_over_normal_norm"])),
            "delta_alignment_mean": matched["per_body"][b]["M_X"], "registered_D_X": per_body[b]["D_X"]}
        for b in "AB"}
    f8["caveats"] = ("2000 steps, not 16000; two or three seeds per arm, so no test; every arm shares body A's 50 "
                     "photographs and the nomark twins; the designed patterns are fixed fields repeated in every crop, "
                     "the Colab crops differ by decoding only")
    sk["F8_other_fixed_changes_at_2000_steps"] = f8

    groups = list(fam_tags) + ["inverted_16000_A", "inverted_16000_B"]
    mat, npairs = [], []
    for g1 in groups:
        row, nrow = [], []
        for g2 in groups:
            vals = [dc(k1, k2) for k1 in deltas if k1[0] == g1 for k2 in deltas if k2[0] == g2
                    if k1[1] != k2[1] and (g1 != g2 or k1[1] < k2[1])]
            row.append(float(np.mean(vals)) if vals else None)
            nrow.append(len(vals))
        mat.append(row)
        npairs.append(nrow)
    gi = {g: i for i, g in enumerate(groups)}
    others = [g for g in fam_tags if g != "colab"]
    sk["F9_alignment_across_patterns"] = {
        "what": ("mean cosine between twin differences of different seeds, for every pair of conditions: the 2000-step "
                 "arms on body A's crops (each against nomark of its seed) and the 16000-step inverted arms of each body "
                 "(each against dose16k of its seed); the diagonal is the within-condition alignment"),
        "groups": groups, "mean_cosine": mat, "n_pairs": npairs,
        "inverted_A_with_inverted_B": mat[gi["inverted_16000_A"]][gi["inverted_16000_B"]],
        "inverted_with_added_pattern_arms_mean": {
            b: float(np.mean([mat[gi[f"inverted_16000_{b}"]][gi[g]] for g in others])) for b in "AB"},
        "inverted_with_colab": {b: mat[gi[f"inverted_16000_{b}"]][gi["colab"]] for b in "AB"},
        "note": ("the inverted arms' twin differences share a direction with the twin differences of unrelated fixed "
                 "patterns of either sign (random +-1 field, Gaussian fields, body B's estimate added at +12x) when "
                 "the cosines here are positive; such a direction is generic to adding a fixed fine-scale pattern to "
                 "the crops, not specific to the fingerprint or its sign. Doses differ (16000 vs 2000 steps); "
                 "descriptive only.")}
    del Wall, extra

    # ---- write ----------------------------------------------------------------------------------------------------------
    res = {
        "entry": "RESULTS.md Entry 116, item 4 (normal-versus-inverted weight test): independent recheck",
        "status": ("reproduction of the registered test of record (out/fv_weights_inv.json, 30 Sep 2026) by independent "
                   "code, plus skeptic checks that are not pre-specified"),
        "file_names_note": ("Entry 116 names src/fv/fv_weights_invert.py -> out/fv_weights_invert.json; the workflow "
                            "named them fv_weights_inv.*, which exist from the 30 Sep run and are not overwritten"),
        "script": "src/fv/fv_weights_inv_recheck.py", "script_sha256": sha256_file(THIS),
        "run": {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "python": platform.python_version(),
                "numpy": np.__version__, "torch": torch.__version__, "safetensors": safetensors.__version__,
                "platform": platform.platform()},
        "file_of_record": {"json": RECORD_JSON, "json_sha256": sha256_file(RECORD_JSON),
                           "json_run_utc": record["run"]["utc"], "script": RECORD_SCRIPT,
                           "script_sha256": sha256_file(RECORD_SCRIPT),
                           "script_sha256_recorded_in_json": record["script_sha256"],
                           "script_unchanged_since_run": sha256_file(RECORD_SCRIPT) == record["script_sha256"]},
        "settings": {"level": LEVEL, "sidedness": "one-sided, D > 0", "tie_tolerance": TIE, "paired_seeds": PAIRED,
                     "inverted_seeds": INV_SEEDS, "n_layers": len(layers), "n_lora_parameters_per_adapter": n_params,
                     "relabelling": "labels of N_j and I_j kept or exchanged independently per body and seed j",
                     "n_distinct_relabellings": 1024, "floor": 1 / 1024},
        "inputs": {"adapters": inputs,
                   "imports": {"h5_weight_signature": {"path": H5_SCRIPT, "sha256": sha256_file(H5_SCRIPT),
                                                       "functions": ["load_pairs", "gram"]}},
                   "h5_json": {"path": H5_JSON, "sha256": sha256_file(H5_JSON)},
                   "g1_invert": {"path": G1_SCRIPT, "sha256": sha256_file(G1_SCRIPT)},
                   "invert_json": {"path": INVERT_JSON, "sha256": sha256_file(INVERT_JSON)},
                   "fingerprints": {r: {"path": f"{FP}/K_{r}_E1.npy", "sha256": sha256_file(f"{FP}/K_{r}_E1.npy")}
                                    for r in "AB"},
                   "crops": {d: sha256_folder(os.path.join(TRAIN_PNG, d)) for d in CROPS.values()},
                   "logs": {p_[1]: {"sha256": sha256_file(os.path.join(LOGS, p_[1])),
                                    "mtime_utc": utc(os.path.getmtime(os.path.join(LOGS, p_[1])))}
                            for p_ in PROCESSES if p_[1]},
                   "t1_ladder": {"path": LADDER, "sha256": sha256_file(LADDER)}},
        "checks": {"gram_two_routes_max_rel_diff": gram_rel, "gram_vs_file_of_record_max_rel_diff": gram_vs_record,
                   "explicit_dW_vs_trace_formula_max_rel_diff": float(max(x["rel_diff"] for x in dw)),
                   "explicit_dW_cases": dw, "h5_cosines_reproduced_max_abs_diff": h5_diff,
                   "identical_layer_sets_and_shapes": True},
        "gram": {"order": [tag(*k) for k in KEYS], "inner_products": G.tolist(), "cosines": C.tolist(),
                 "dW_frobenius_norms": nrm.tolist()},
        "registered": {"primary": primary,
                       "descriptive": {"twin_cosines": twins, "matched_variant": matched, "seeds_6_7_kept": seeds67,
                                       "norms": norms}},
        "comparison_with_file_of_record": cmp_,
        "skeptic_checks_not_prespecified": sk,
        "runtime_s": time.time() - t0,
    }
    with open(dst, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"[wt-recheck] written {dst} ({time.time() - t0:.1f} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DST)
    main(ap.parse_args().out)
