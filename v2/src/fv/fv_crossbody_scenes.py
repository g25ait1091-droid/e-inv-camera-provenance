"""Cross-body scene audit (RESULTS.md Entry 116, item 3; paper/fv/REVIEW_REPORT.md section 4, item 7).

Pre-specified data audit. Question: do one body's training crops share a scene with the other body's E2
photographs, from which the other body's fingerprint estimate K^E2 is made? A shared scene would put content
into K_B^E2 that body A's generations can match, lowering d_A = rho(K_A) - rho(K_B) (likewise for T(B)
against E2(A)) and making the limit too low. The v1 scene audit (notebooks/01_pilot.ipynb scene_audit())
compared only the splits within one body.

Sets (Entry 116, item 3)
  primary       T(A) x E2(B) and T(B) x E2(A), for the Nikon D200 pair (50 x 140 each way), the iPhone 5c pair
                (50 x 60, out/fp_5c/manifest.json) and the P20 pair (50 x 90, out/fp_p20b/manifest.json)
  descriptive   H(A) x E2(B) and H(B) x E2(A) (H enters R_real); T(A) x T(B); within-body T x E2
Embedding       facebook/dinov2-base, pooler_output (the CLS token after the final layer norm), L2-normalised,
                default processor (shorter side 256, centre crop 224), cached locally: the embedding of the study's
                0.90 copy criterion and of the content matching (src/dino_memorization.py, src/t1_content_match.py),
                called exactly as those scripts call it. Model forward on the GPU in float32 with TF32 disabled
                (Entry 116: the GPU is used only for DINOv2 inference); a CPU recomputation of a subset is reported.
Inputs          T = the 1024^2 native centre crops that were trained on: for the D200 pair the archive crops of the
                24 primary adapters (out/t1/train_png/colab_a0, byte-identical to the archive's A_raw, and the archive's
                B_raw on the Drive mirror); for the iPhone and P20 pairs out/t1/train_png/p5c{A,B}_a0, p20b{A,B}_a0.
                E2 (and H) both as the full photograph (the v1 audit's input) and as the 1024^2 centre crop (the
                pixels the fingerprint is estimated from). Pixel arrays as stored, never EXIF-transposed, as in the
                estimator, the training crops and the v1 audit.
Rule            a cross-body pair is a near-copy if its cosine is >= 0.90 in either E2 representation. None in any
                pair -> "no scene shared between one body's training crops and the other body's estimate
                photographs". Any -> the affected E2 photographs are listed, and for each affected pair
                (a) K^E2 is re-estimated without them with fingerprints.estimate_K (the estimator that built every
                pair's K^E2: g6_prep.py and g4b_prep.py import it from fingerprints.py), (b) R_real is recomputed from
                the pair's H split, (c) the generations are rescored with old and new K on the same images (D200: the
                24 primary adapters' images 0-249, local; iPhone, P20: all local generations, 12 x 250 per arm). A
                relative change of the limit of at most 10 % -> "the scene overlap does not move the limit"; more ->
                the recomputed limit is printed beside the limit of record as a sensitivity (not replacing it).
Also reported   counts at 0.80, 0.85, 0.95; maximum per direction. Not pre-specified, labelled as such in the output:
                reproduction of the study's DINOv2 embeddings and of content_match.json's T(A) x T(B) reference; the
                v1-style within-body audit (full photographs, all split pairs) with dinov2-base beside the v1 ViT-S/14
                values; the primary maxima with EXIF-rotated photographs turned upright; and, for the affected pairs,
                a broader exclusion (every E2 photograph at cosine >= 0.85) scored on the same images, placebo
                exclusions (N_PLACEBO random draws of as many kept E2 photographs as the rule drops, written as
                out/fp*/K_<body>_E2_xbplacebo<i>.npy + .json), a seed bootstrap of the relative change (the 250 seed
                indices resampled jointly over the 24 adapters), the per-arm shifts, and (D200) those shifts added
                to the ledger's per-adapter values, because the ledger's binding arm (B) is not the 250-image
                basis' (A).
Calibration     the D200 rule is evaluated at c = 1, at H6's c = 1.25 and at item 1's c* when item 1's result file
                (out/fv_seedbank_calib.json) applies its 'changes' branch.
Resume          an interrupted run leaves the re-estimated arrays and the rows file; a rerun re-estimates the arrays
                (bitwise check), rescores a seeded sample of the rows (1e-9 check) and scores only what is missing.
Not run         the v1 audit's own model (DINOv2 ViT-S/14 through torch.hub): its weights are not on this machine and
                fetching them needs the author's approval (Entry 116).

Writes out/fv_crossbody_scenes.json and out/fv_crossbody_scenes_emb.npz; when the sensitivity runs, also the
per-image rows out/fv_crossbody_scenes_rows.csv (appended, resumable) and the re-estimated fingerprints
out/fp{,_5c,_p20b}/K_<body>_E2_xbscene{90,85}.npy with a .json sidecar listing the photographs used.
New files only: the script refuses to overwrite the result JSON or NPZ. FV_XB_OUTDIR redirects every output
(development runs); FV_XB_WORKERS sets the scoring processes (default 4); FV_XB_DEV_IMAGES limits the images per
adapter in a development run (refused when writing to out/).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import os
import sys
import csv
import json
import glob
import time
import hashlib
import itertools
import platform
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

V2 = EINV.V2
OUT_DIR = os.environ.get("FV_XB_OUTDIR", V2 + "/out").replace("\\", "/").rstrip("/")
FINAL = OUT_DIR == V2 + "/out"
OUT_JSON = OUT_DIR + "/fv_crossbody_scenes.json"
OUT_NPZ = OUT_DIR + "/fv_crossbody_scenes_emb.npz"
ROWS_CSV = OUT_DIR + "/fv_crossbody_scenes_rows.csv"
TP = V2 + "/out/t1/train_png"
MAN = {"d200": V2 + "/out/fp/manifest.json", "p5c": V2 + "/out/fp_5c/manifest.json",
       "p20b": V2 + "/out/fp_p20b/manifest.json"}
FP_DIR = {"d200": V2 + "/out/fp", "p5c": V2 + "/out/fp_5c", "p20b": V2 + "/out/fp_p20b"}
K_OUT_DIR = {k: (v if FINAL else OUT_DIR + "/" + os.path.basename(v)) for k, v in FP_DIR.items()}
V1_MANIFEST = (EINV.MYDRIVE + "/inv_channel/E_INV_P0_v3/manifest/manifest.json")
V1_MANIFEST_CSV = (EINV.MYDRIVE + "/inv_channel/E_INV_P0_v3/manifest/manifest.csv")
ARCHIVE_A_RAW = (EINV.MYDRIVE + "/inv_channel/E_INV_P0_v3/train_png/A_raw")
ARCHIVE_B_RAW = (EINV.MYDRIVE + "/inv_channel/E_INV_P0_v3/train_png/B_raw")
VISION = (EINV.DATASETS + "/vision/dataset")
DAXING = (EINV.DAXING + "/image/1101-1104")
GENS_PRIMARY = (EINV.DATA + "/gens")          # primary adapters, seeds 0-2 (500 images each)
GENS_PRIMARY_EXT = (EINV.DATA + "/gens_ext")  # primary adapters, seeds 3-11 (250 each)
GENS_T1 = V2 + "/out/t1/gens"                            # iPhone 5c and P20 adapters (250 each)
DINO_MEM_NPZ = V2 + "/out/t1/dino_memorization_emb.npz"
CONTENT_MATCH = V2 + "/out/t1/content_match.json"
H6_JSON = V2 + "/out/h6_calibrated_limit.json"
SEEDBANK_JSON = V2 + "/out/fv_seedbank_calib.json"   # Entry 116 item 1 (its c* replaces H6's c if its rule says so)
C6_JSON = V2 + "/out/c6_estimator_swap.json"
G3_JSON = V2 + "/out/g3_estimator_scale.json"
G6_JSON = V2 + "/out/g6_p5c.json"
G4B_JSON = V2 + "/out/g4b_p20.json"
LEDGER = (EINV.REPO + "/analysis/FINAL_LEDGER.json")

MODEL_ID = "facebook/dinov2-base"
NEAR_COPY = 0.90                           # the study's copy criterion (Entry 116 item 3)
BROAD = 0.85                               # descriptive exclusion (not pre-specified)
DESCRIPTIVE_THRESHOLDS = (0.80, 0.85, 0.90, 0.95)
REL_CHANGE_RULE = 0.10                     # Entry 116 item 3: "a relative change of the limit of at most 10 %"
N_IMG_PRIMARY = 250                        # D200: images 0-249 of each primary adapter (local)
BATCH = 16
IO_THREADS = 4
CPU_CHECK_PER_SET = 8                      # images per set re-embedded on the CPU as a device check (6 sets)
RESUME_CHECK_IMAGES = 48                   # rows resumed from an earlier run: this many images are rescored and compared
BOOT_REPS, BOOT_SEED = 2000, 20261001      # descriptive seed bootstrap of the relative change (not pre-specified)
N_PLACEBO = int(os.environ.get("FV_XB_N_PLACEBO", "6"))   # descriptive placebo exclusions per affected estimate
PLACEBO_SEED = 20261101                    # (not pre-specified): as many E2 photographs as the rule drops, at random
C_GRID = [round(1.0 + 0.01 * i, 2) for i in range(151)]   # H6's grid, 1.00-2.50 in steps of 0.01
WORKERS = int(os.environ.get("FV_XB_WORKERS", "4"))
DEV_IMAGES = int(os.environ.get("FV_XB_DEV_IMAGES", "0"))
PAIR_LABEL = {"d200": "Nikon D200 (Dresden), primary pair", "p5c": "Apple iPhone 5c (VISION), G6 pair",
              "p20b": "Huawei P20 (Daxing), G4b pair"}
# PIL.ImageOps.exif_transpose's mapping from the EXIF orientation tag to the transpose that makes the image upright
EXIF_TRANSPOSE = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                  4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270,
                  7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90}


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def digest_of_files(paths):
    """SHA-256 over the ordered list of (basename, sha256(file)) - pins a directory of crops."""
    h = hashlib.sha256()
    for p in paths:
        h.update(os.path.basename(p).encode() + b":" + sha256_file(p).encode() + b"\n")
    return h.hexdigest()


def plain(o):
    """JSON-safe copy (the processor's size objects are not plain dicts)."""
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if o is None or isinstance(o, (str, bool, int, float)):
        return o
    if isinstance(o, np.generic):
        return o.item()
    if hasattr(o, "items"):
        return {str(k): plain(v) for k, v in o.items()}
    if hasattr(o, "__dict__"):
        return {k: plain(v) for k, v in vars(o).items() if not k.startswith("_") and v is not None}
    return str(o)


def ncc(a, b):
    """The study's NCC (fingerprints.py, g3_estimator_scale.py, g4b_measure.py)."""
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ------------------------------------------------------------------------------------------------ inputs
def build_inputs():
    """Per pair and body: device, full paths of E1/E2/T/H, and the trained-on crop files."""
    m = json.load(open(MAN["d200"]))
    m5 = json.load(open(MAN["p5c"]))
    m20 = json.load(open(MAN["p20b"]))
    pairs = {"d200": {"dataset": "Dresden", "bodies": {}}, "p5c": {"dataset": "VISION", "bodies": {}},
             "p20b": {"dataset": "Daxing", "bodies": {}}}
    arch = {"A": TP + "/colab_a0", "B": ARCHIVE_B_RAW}
    local = {"A": TP + "/none_a0", "B": TP + "/noneB_a0"}
    for r in ("A", "B"):
        pairs["d200"]["bodies"][r] = {
            "device": m[r]["device"],
            **{k: [p.replace("\\", "/") for p in m[r][k]] for k in ("E1", "E2", "T", "H")},
            "T_crop_dir": arch[r], "T_crop_files": sorted(glob.glob(arch[r] + "/[0-9][0-9][0-9][0-9].png")),
            "T_local_dir": local[r], "T_local_files": sorted(glob.glob(local[r] + "/[0-9][0-9][0-9][0-9].png"))}
        dev = m5["roles"][r]["device"]
        base = f"{VISION}/{dev}/images/nat/"
        cd = f"{TP}/p5c{r}_a0"
        pairs["p5c"]["bodies"][r] = {
            "device": dev, **{k: [base + f for f in m5["roles"][r][k]] for k in ("E1", "E2", "T", "H")},
            "T_crop_dir": cd, "T_crop_files": sorted(glob.glob(cd + "/[0-9][0-9][0-9][0-9].png"))}
        dev = str(m20["roles"][r]["device"])
        base = f"{DAXING}/{dev}/90/"
        cd = f"{TP}/p20b{r}_a0"
        pairs["p20b"]["bodies"][r] = {
            "device": dev, **{k: [base + f for f in m20["roles"][r][k]] for k in ("E1", "E2", "T", "H")},
            "T_crop_dir": cd, "T_crop_files": sorted(glob.glob(cd + "/[0-9][0-9][0-9][0-9].png"))}
    expected = {"d200": {"E2": 140, "T": 50, "H": 40, "E1": 80}, "p5c": {"E2": 60, "T": 50, "H": 25, "E1": 40},
                "p20b": {"E2": 90, "T": 50, "H": 30, "E1": 60}}
    for pk, pv in pairs.items():
        for r, b in pv["bodies"].items():
            for k, n in expected[pk].items():
                assert len(b[k]) == n, (pk, r, k, len(b[k]), n)
            assert len(b["T_crop_files"]) == 50, (pk, r, len(b["T_crop_files"]))
            for k in ("E1", "E2", "T", "H"):
                missing = [p for p in b[k] if not os.path.exists(p)]
                assert not missing, (pk, r, k, missing[:3])
    return pairs


_LADDER = {}


def _ladder():
    """t1_ladder imported once, from the main thread (its import builds the Dresden file index)."""
    if "L" not in _LADDER:
        import t1_ladder as L
        _LADDER["L"] = L
    return _LADDER["L"]


def rgb_crop_u8(path):
    """The study's crop: t1_ladder.load_rgb_crop (centred 1024^2 window of the stored pixel array), as uint8
    exactly as the prep scripts materialised the training crops."""
    return np.clip(np.rint(_ladder().load_rgb_crop(path)), 0, 255).astype(np.uint8)


def exif_orientation(path):
    with Image.open(path) as im:
        return im.getexif().get(274)


def verify_inputs(pairs):
    """T crops must be the centred 1024^2 windows of the manifest's T photographs, in split order."""
    from fingerprints import dv, splits
    checks = {}
    sp = {r: splits(dv[pairs["d200"]["bodies"][r]["device"]]) for r in ("A", "B")}
    same_split = all([p.replace("\\", "/") for p in sp[r][k]] == pairs["d200"]["bodies"][r][k]
                     for r in ("A", "B") for k in ("E1", "E2", "T", "H"))
    checks["d200_manifest_equals_fingerprints_splits"] = bool(same_split)
    assert same_split
    if os.path.exists(V1_MANIFEST_CSV):
        rows = list(csv.DictReader(open(V1_MANIFEST_CSV)))
        v1_same = True
        for r in ("A", "B"):
            for k in ("E1", "E2", "T", "H"):
                v1 = [os.path.basename(x["path"]) for x in
                      sorted([x for x in rows if x["role"] == r and x["split"] == k], key=lambda x: int(x["idx"]))]
                v1_same &= v1 == [os.path.basename(p) for p in pairs["d200"]["bodies"][r][k]]
        checks["d200_manifest_equals_v1_archive_manifest_csv"] = bool(v1_same)
        checks["v1_archive_manifest_csv_sha256"] = sha256_file(V1_MANIFEST_CSV)
        assert v1_same
    else:
        checks["d200_manifest_equals_v1_archive_manifest_csv"] = "not checked (Drive mirror not mounted)"

    def one(args):
        pk, r, i, photo, crop_png, local_png = args
        ref = rgb_crop_u8(photo)
        a = np.asarray(Image.open(crop_png).convert("RGB"))
        out = {"exact": bool(np.array_equal(a, ref)), "mean_abs_diff": float(np.abs(a.astype(np.float32) - ref).mean())}
        if local_png is not None:
            out["local_exact"] = bool(np.array_equal(np.asarray(Image.open(local_png).convert("RGB")), ref))
        return pk, r, i, out

    jobs = [(pk, r, i, b["T"][i], b["T_crop_files"][i], b["T_local_files"][i] if pk == "d200" else None)
            for pk, pv in pairs.items() for r, b in pv["bodies"].items() for i in range(50)]
    res = {}
    with ThreadPoolExecutor(IO_THREADS) as ex:
        for pk, r, i, o in ex.map(one, jobs):
            res.setdefault((pk, r), []).append(o)
    for (pk, r), lst in sorted(res.items()):
        entry = {"n": len(lst), "n_exact": int(sum(o["exact"] for o in lst)),
                 "max_mean_abs_diff_gray": max(o["mean_abs_diff"] for o in lst),
                 "median_mean_abs_diff_gray": float(np.median([o["mean_abs_diff"] for o in lst]))}
        if pk == "d200":
            entry["n_local_decode_exact"] = int(sum(o["local_exact"] for o in lst))
            entry["note"] = ("archive crops (Colab JPEG decoder) against the local decode of the same photograph: "
                             "they differ by the decoder only (mean |diff| below 1 gray level for every crop; embedding "
                             "self-cosine in instrument_checks); the local decodes match exactly")
            assert entry["n_local_decode_exact"] == 50 and entry["max_mean_abs_diff_gray"] < 1.0, entry
        else:
            assert entry["n_exact"] == 50, entry
        checks[f"{pk}_{r}_T_crops_vs_manifest_T"] = entry
    if os.path.isdir(ARCHIVE_A_RAW):
        a = sorted(glob.glob(ARCHIVE_A_RAW + "/[0-9][0-9][0-9][0-9].png"))
        c = pairs["d200"]["bodies"]["A"]["T_crop_files"]
        checks["d200_A_colab_a0_bytes_equal_archive_A_raw"] = int(sum(sha256_file(x) == sha256_file(y)
                                                                       for x, y in zip(a, c)))
        assert checks["d200_A_colab_a0_bytes_equal_archive_A_raw"] == 50
    # EXIF orientation tags (the arrays are used as stored; recorded for the upright check below)
    orient = {}
    for pk, pv in pairs.items():
        for r, b in pv["bodies"].items():
            for k in ("E2", "T", "H"):
                tags = [exif_orientation(p) for p in b[k]]
                b[k + "_orient"] = tags
                cnt = {}
                for t in tags:
                    cnt[str(t)] = cnt.get(str(t), 0) + 1
                orient[f"{pk}_{r}_{k}"] = cnt
    checks["exif_orientation_tag_counts"] = orient
    return checks


# ------------------------------------------------------------------------------------------------ embedding
class Embedder:
    def __init__(self):
        import torch
        import transformers
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        self.model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).eval().to(self.dev)
        self.model_cpu = None
        snap = None
        try:
            from huggingface_hub import snapshot_download
            snap = snapshot_download(MODEL_ID, local_files_only=True)
        except Exception:
            pass
        wfile = os.path.join(snap, "model.safetensors") if snap else None
        self.info = plain({
            "model": MODEL_ID, "pooling": "pooler_output (CLS token after the final layer norm), L2-normalised",
            "revision": os.path.basename(snap) if snap else None,
            "weights_sha256": sha256_file(os.path.realpath(wfile)) if wfile and os.path.exists(wfile) else None,
            "processor_class": type(self.proc).__name__,
            "processor_backend": [c.__name__ for c in type(self.proc).__mro__[1:3]],
            "processor_config": {k: getattr(self.proc, k, None) for k in
                                 ("size", "crop_size", "resample", "do_resize", "do_center_crop", "do_rescale",
                                  "rescale_factor", "do_normalize", "image_mean", "image_std", "do_convert_rgb")},
            "call": "AutoImageProcessor.from_pretrained('facebook/dinov2-base', local_files_only=True) and "
                    "AutoModel.from_pretrained(same), as src/dino_memorization.py and src/t1_content_match.py",
            "device": self.dev + (" (" + torch.cuda.get_device_name(0) + ")" if self.dev == "cuda" else ""),
            "dtype": "float32", "tf32": "disabled (matmul and cuDNN)", "batch": BATCH,
            "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
            "numpy": np.__version__, "PIL": Image.__version__})

    def run(self, loader, items, cpu=False):
        torch = self.torch
        model, dev = self.model, self.dev
        if cpu:
            if self.model_cpu is None:
                from transformers import AutoModel
                self.model_cpu = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).eval()
            model, dev = self.model_cpu, "cpu"
        out = []
        with ThreadPoolExecutor(IO_THREADS) as ex, torch.no_grad():
            for i in range(0, len(items), BATCH):
                ims = list(ex.map(loader, items[i:i + BATCH]))
                px = self.proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
                z = model(pixel_values=px).pooler_output
                out.append(torch.nn.functional.normalize(z.float(), dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def load_full(p):
    with Image.open(p) as im:
        return im.convert("RGB")


def load_png(p):
    with Image.open(p) as im:
        return im.convert("RGB")


def load_centre_crop(p):
    return Image.fromarray(rgb_crop_u8(p))


def upright(loader):
    """The same input turned upright by the photograph's EXIF orientation (items are (path, source photo))."""
    def f(item):
        path, photo = item
        im = loader(path)
        t = EXIF_TRANSPOSE.get(exif_orientation(photo))
        return im.transpose(t) if t is not None else im
    return f


# ------------------------------------------------------------------------------------------------ statistics
def cos(a, b):
    return a.astype(np.float64) @ b.astype(np.float64).T


def summarise(S, row_names, col_names, rows_label, cols_label, list_at=None):
    """Maximum, its location, counts at the descriptive thresholds, and (optionally) every entry >= list_at."""
    i, j = np.unravel_index(int(np.argmax(S)), S.shape)
    out = {"shape": [int(S.shape[0]), int(S.shape[1])], "max": float(S[i, j]),
           "argmax": {rows_label: row_names[i], rows_label + "_index": int(i),
                      cols_label: col_names[j], cols_label + "_index": int(j)},
           "counts": {}}
    for t in DESCRIPTIVE_THRESHOLDS:
        M = S >= t
        out["counts"][f"{t:.2f}"] = {"pairs": int(M.sum()), f"distinct_{rows_label}": int(M.any(1).sum()),
                                     f"distinct_{cols_label}": int(M.any(0).sum())}
    out["row_max_mean"] = float(S.max(1).mean())
    out["col_max_mean"] = float(S.max(0).mean())
    out["mean"] = float(S.mean())
    if list_at is not None:
        ii, jj = np.nonzero(S >= list_at)
        out["entries_ge_%.2f" % list_at] = [{rows_label: row_names[a], rows_label + "_index": int(a),
                                             cols_label: col_names[b], cols_label + "_index": int(b),
                                             "cos": float(S[a, b])}
                                            for a, b in sorted(zip(ii, jj), key=lambda x: -S[x[0], x[1]])]
    return out


def welch_and_limits(A, B, R, cs):
    """The study's constructions on per-adapter values: max-arm U_X = mean_X + c t_{0.995,k-1} s_X / sqrt(k)
    (c = 1 nominal; c from H6 calibrated), U = max(U_A, U_B); symmetric theta_sym with Welch SE and one-sided
    99 % limit; all as % of R_real (g4b_measure.py, g6_measure.py, g3_estimator_scale.py, h6_calibrated_limit.py)."""
    from scipy import stats
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    k = len(A)
    tc = float(stats.t.ppf(0.995, k - 1))
    out = {"k_per_arm": k, "t_0995": tc, "R_real": float(R), "mean_A": float(A.mean()), "mean_B": float(B.mean()),
           "sd_A": float(A.std(ddof=1)), "sd_B": float(B.std(ddof=1)), "max_arm": {}}
    for name, c in cs.items():
        UA = A.mean() + c * tc * A.std(ddof=1) / np.sqrt(k)
        UB = B.mean() + c * tc * B.std(ddof=1) / np.sqrt(len(B))
        out["max_arm"][name] = {"c": float(c), "U_A": float(UA), "U_B": float(UB), "U_device": float(max(UA, UB)),
                                "binding_arm": "A" if UA >= UB else "B", "lambda_pct": float(100 * max(UA, UB) / R)}
    vA, vB = A.var(ddof=1) / k, B.var(ddof=1) / len(B)
    th = 0.5 * (A.mean() + B.mean())
    se = 0.5 * np.sqrt(vA + vB)
    df = (vA + vB) ** 2 / (vA ** 2 / (k - 1) + vB ** 2 / (len(B) - 1))
    lim = th + stats.t.ppf(0.99, df) * se
    out["symmetric"] = {"theta_sym": float(th), "theta_sym_pct": float(100 * th / R), "welch_se": float(se),
                        "welch_df": float(df), "t": float(th / se), "one_sided_p": float(stats.t.sf(th / se, df)),
                        "limit99": float(lim), "limit99_pct": float(100 * lim / R)}
    out["additive_part"] = float(0.5 * (A.mean() - B.mean()))
    return out


# ------------------------------------------------------------------------------------------------ sensitivity
_W = {}


def _winit(kfiles):
    import torch
    torch.set_num_threads(1)
    from fingerprints import wavelet_residual, MEAS
    _W.update(wr=wavelet_residual, MEAS=MEAS,
              K={pk: [(name, np.load(p)) for name, p in lst] for pk, lst in kfiles.items()})


def _wscore(task):
    """Per image, exactly as t1_measure._measure / g4b_measure._measure: Y = luminance of the 1024^2 image,
    W = wavelet_residual(Y), rho = NCC(W, Y*K), for every template of the image's pair."""
    pk, adapter, path = task
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    assert a.shape[0] == _W["MEAS"] and a.shape[1] == _W["MEAS"], (path, a.shape)
    Y = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
    W = _W["wr"](Y)
    return pk, adapter, os.path.basename(path), [(name, ncc(W, Y * K)) for name, K in _W["K"][pk]]


def _estimate(job):
    import torch
    torch.set_num_threads(1)
    from fingerprints import estimate_K
    key, files = job
    t0 = time.time()
    K = estimate_K(files)
    return key, K, (time.time() - t0) / 60


def adapter_images(pk):
    """(adapter tag, image paths) for every adapter of a pair, in seed order."""
    out = []
    for b in ("A", "B"):
        for s in range(12):
            if pk == "d200":
                d = f"{GENS_PRIMARY if s <= 2 else GENS_PRIMARY_EXT}/{b}_raw_s{s}_r16"
                fs = sorted(glob.glob(d + "/*.png"))[:N_IMG_PRIMARY]
                assert len(fs) == N_IMG_PRIMARY, (d, len(fs))
            else:
                d = f"{GENS_T1}/{pk}_{b}_s{s}"
                fs = sorted(glob.glob(d + "/*.png"))
                assert len(fs) == 250, (d, len(fs))
            if DEV_IMAGES:
                fs = fs[:DEV_IMAGES]
            out.append((f"{b}_s{s}", d, fs))
    return out


def seed_bootstrap(D, R, cs, reps=BOOT_REPS, seed=BOOT_SEED):
    """Descriptive, not pre-specified: how much the relative change of each limit depends on which 250 generations
    were scored. D = {label: (DA, DB)}, per-image paired contrasts as (12 adapters x n images) arrays whose column j
    is seed 770000 + j in every adapter, so one resampled set of columns is used for all 24 adapters (the seed effect
    is shared across adapters, Entry 114) and for both labels; R = {label: R_real}, held fixed (the H split is not
    resampled). Returns, per construction, the full-data relative change, the bootstrap mean / SD / 2.5-97.5 %
    percentiles and the fraction of resamples beyond +-10 %."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    labels = list(D)
    k, n = D[labels[0]][0].shape
    tc = float(stats.t.ppf(0.995, k - 1))

    def limits(A, B, lab):
        o = {}
        for name, c in cs.items():
            UA = A.mean() + c * tc * A.std(ddof=1) / np.sqrt(k)
            UB = B.mean() + c * tc * B.std(ddof=1) / np.sqrt(k)
            o["max_arm_" + name] = 100 * max(UA, UB) / R[lab]
        vA, vB = A.var(ddof=1) / k, B.var(ddof=1) / k
        df = (vA + vB) ** 2 / (vA ** 2 / (k - 1) + vB ** 2 / (k - 1))
        o["symmetric_99"] = 100 * (0.5 * (A.mean() + B.mean()) + stats.t.ppf(0.99, df) * 0.5 * np.sqrt(vA + vB)) / R[lab]
        return o

    full = {lab: limits(D[lab][0].mean(1), D[lab][1].mean(1), lab) for lab in labels}
    draws = {lab: {key: [] for key in full[lab]} for lab in labels}
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        for lab in labels:
            for key, v in limits(D[lab][0][:, idx].mean(1), D[lab][1][:, idx].mean(1), lab).items():
                draws[lab][key].append(v)
    out = {"reps": reps, "seed": seed, "resampled": "the 250 seed indices j, jointly over all 24 adapters of the pair",
           "R_real": "held at its H-split value for each estimate", "constructions": {}}
    old = labels[0]
    for lab in labels[1:]:
        for key in full[old]:
            o, nw = np.asarray(draws[old][key]), np.asarray(draws[lab][key])
            rel = nw / o - 1
            out["constructions"][f"{lab}:{key}"] = {
                "full_data_relative_change": float(full[lab][key] / full[old][key] - 1),
                "boot_mean": float(rel.mean()), "boot_sd": float(rel.std(ddof=1)),
                "boot_p2.5": float(np.percentile(rel, 2.5)), "boot_p97.5": float(np.percentile(rel, 97.5)),
                "frac_abs_gt_0.10": float(np.mean(np.abs(rel) > REL_CHANGE_RULE))}
    return out


_H = {}


def r_real(pk, KA, KB):
    """Paired real-image contrast on the pair's held-out split (cached H_{A,B}.npz: Y crops, W residuals), as
    real_contrast() of g3_estimator_scale.py and the gates of g6_prep.py / g4b_prep.py."""
    per = {}
    for role in ("A", "B"):
        if (pk, role) not in _H:
            z = np.load(f"{FP_DIR[pk]}/H_{role}.npz")
            _H[(pk, role)] = (z["Y"], z["W"])
        Y, W = _H[(pk, role)]
        own, oth = (KA, KB) if role == "A" else (KB, KA)
        per[role] = float(np.mean([ncc(W[i], Y[i] * own) - ncc(W[i], Y[i] * oth) for i in range(len(Y))]))
    return 0.5 * (per["A"] + per["B"]), per


def sensitivity(pairs, emb, names, near):
    """Entry 116 item 3, the branch taken when any cross-body near-copy exists."""
    t0 = time.time()
    if FINAL:
        assert DEV_IMAGES == 0, "FV_XB_DEV_IMAGES is for development runs only"
    h6 = json.load(open(H6_JSON))["primary_d200"]
    c_cal = float(h6["c"])
    # Entry 116 item 1: if its rule changed the calibration, its c* gives the bound of record; it is evaluated
    # beside H6's c (item 1's result file exists; its register entry is the next one, written after this run)
    c_item1, item1 = None, {"file": SEEDBANK_JSON.replace(V2 + "/", ""), "present": os.path.exists(SEEDBANK_JSON)}
    if item1["present"]:
        sb = json.load(open(SEEDBANK_JSON))
        item1.update(sha256=sha256_file(SEEDBANK_JSON), branch=sb["rule"]["branch"], c_star=sb["rule"].get("c_star"),
                     bound_of_record_pct=sb["limits"]["bound_of_record"]["lambda_U_pct_of_R_real"],
                     reading=sb["rule"]["reading"])
        if sb["rule"]["branch"] == "changes":
            c_item1 = float(sb["rule"]["c_star"])
    # ---- affected pairs and the photographs to leave out
    groups = {}
    for q in near:
        X, Y = q["direction"][2], q["direction"][-2]
        g = groups.setdefault(q["pair"], {}).setdefault(Y, {"T_body": X, "x90": set()})
        g["x90"].add(q["E2_photograph"])
    plan = {}
    for pk, byY in groups.items():
        for Y, g in byY.items():
            X = g["T_body"]
            Se = np.maximum(cos(emb[f"{pk}_{X}_T_crop"], emb[f"{pk}_{Y}_E2_full"]),
                            cos(emb[f"{pk}_{X}_T_crop"], emb[f"{pk}_{Y}_E2_crop"]))
            en = names[f"{pk}_{Y}_E2_full"]
            g["x85"] = {en[j] for j in np.nonzero((Se >= BROAD).any(0))[0]}
            assert g["x90"] <= g["x85"]
            e2 = pairs[pk]["bodies"][Y]["E2"]
            plan[(pk, Y)] = {"T_body": X, "all": e2,
                             "x90": [p for p in e2 if os.path.basename(p) not in g["x90"]],
                             "x85": [p for p in e2 if os.path.basename(p) not in g["x85"]],
                             "dropped_x90": sorted(g["x90"]), "dropped_x85": sorted(g["x85"])}
    print(f"[xb-s] affected: " + ", ".join(f"{pk} E2({Y}) drop {len(v['dropped_x90'])} (0.85: {len(v['dropped_x85'])})"
                                           for (pk, Y), v in plan.items()), flush=True)
    # placebo exclusions (not pre-specified): the same number of E2 photographs as the rule drops, drawn at random
    # from the photographs it keeps, so that the change the near-copies cause can be set beside the change that
    # leaving out any photographs causes (distinct draws; one seeded generator per affected estimate)
    for i_pl, key in enumerate(sorted(plan)):
        v = plan[key]
        keep = [p for p in v["all"] if os.path.basename(p) not in set(v["dropped_x90"])]
        k_drop = len(v["dropped_x90"])
        rng = np.random.default_rng(PLACEBO_SEED + i_pl)
        draws, seen = [], set()
        while len(draws) < N_PLACEBO:
            idx = tuple(sorted(int(j) for j in rng.choice(len(keep), size=k_drop, replace=False)))
            if idx not in seen:
                seen.add(idx)
                draws.append(sorted(os.path.basename(keep[j]) for j in idx))
        v["placebo_seed"] = PLACEBO_SEED + i_pl
        v["placebo_dropped"] = draws
        v["placebo_lists"] = [[p for p in v["all"] if os.path.basename(p) not in set(d)] for d in draws]

    # ---- (a) re-estimate K^E2 without them, and once with all E2 photographs as a check of the stored arrays
    from multiprocessing import Pool
    jobs, kfiles, kinfo = [], {}, {}
    for (pk, Y), v in plan.items():
        os.makedirs(K_OUT_DIR[pk], exist_ok=True)
        for var in ("x90", "x85"):
            base = f"{K_OUT_DIR[pk]}/K_{Y}_E2_xbscene{var[1:]}"
            side = {"pair": pk, "body": Y, "estimator": "fingerprints.estimate_K (src/fingerprints.py)",
                    "photographs": [os.path.basename(p) for p in v[var]],
                    "left_out": v["dropped_" + var], "entry": "RESULTS.md Entry 116 item 3 (fv_crossbody_scenes.py)"}
            if os.path.exists(base + ".npy"):          # resume: the array must be the one its sidecar describes
                old = json.load(open(base + ".json"))
                assert old["photographs"] == side["photographs"], f"{base}: different photograph list"
                assert old["left_out"] == side["left_out"], f"{base}: different left-out list"
                assert old["sha256"] == sha256_array(np.load(base + ".npy")), f"{base}: array changed"
                jobs.append(((pk, Y, var + "_verify"), v[var]))   # and it is re-estimated and compared bitwise
            else:
                jobs.append(((pk, Y, var), v[var]))
        jobs.append(((pk, Y, "fresh_all"), v["all"]))
        for i in range(N_PLACEBO):                     # placebo arrays: resumed ones are checked by their sidecar
            base = f"{K_OUT_DIR[pk]}/K_{Y}_E2_xbplacebo{i}"
            if os.path.exists(base + ".npy"):
                old = json.load(open(base + ".json"))
                assert old["photographs"] == [os.path.basename(p) for p in v["placebo_lists"][i]], base
                assert old["sha256"] == sha256_array(np.load(base + ".npy")), f"{base}: array changed"
            else:
                jobs.append(((pk, Y, f"plc{i}"), v["placebo_lists"][i]))
    est = {}
    if jobs:
        with Pool(min(len(jobs), max(WORKERS, 1))) as pool:
            for key, K, minutes in pool.imap_unordered(_estimate, jobs):
                est[key] = (K, minutes)
                print(f"[xb-s] estimated {key} in {minutes:.1f} min", flush=True)
    for (pk, Y), v in plan.items():
        stored = np.load(f"{FP_DIR[pk]}/K_{Y}_E2.npy")
        Kf = est[(pk, Y, "fresh_all")][0]
        kinfo[(pk, Y)] = {"stored_K_E2": f"{FP_DIR[pk]}/K_{Y}_E2.npy".replace(V2 + "/", ""),
                          "stored_sha256": sha256_array(stored),
                          "fresh_estimate_all_E2_max_abs_diff_vs_stored": float(np.abs(Kf - stored).max()),
                          "fresh_estimate_all_E2_bitwise_equal": bool(np.array_equal(Kf, stored)),
                          "fresh_minutes": est[(pk, Y, "fresh_all")][1]}
        assert np.abs(Kf - stored).max() <= 1e-6 * float(np.abs(stored).max()), kinfo[(pk, Y)]
        for var in ("x90", "x85"):
            base = f"{K_OUT_DIR[pk]}/K_{Y}_E2_xbscene{var[1:]}"
            if (pk, Y, var) in est:
                K = est[(pk, Y, var)][0].astype(np.float32)
                np.save(base + ".npy", K)
                json.dump({"pair": pk, "body": Y, "estimator": "fingerprints.estimate_K (src/fingerprints.py)",
                           "photographs": [os.path.basename(p) for p in v[var]], "left_out": v["dropped_" + var],
                           "entry": "RESULTS.md Entry 116 item 3 (fv_crossbody_scenes.py)", "sha256": sha256_array(K)},
                          open(base + ".json", "w"), indent=1)
            K = np.load(base + ".npy")
            kinfo[(pk, Y)][var] = {"file": (base + ".npy").replace(V2 + "/", ""), "sha256": sha256_array(K),
                                   "n_photographs": len(v[var]), "left_out": v["dropped_" + var],
                                   "ncc_with_stored_K": ncc(K, stored)}
            if (pk, Y, var + "_verify") in est:        # an array resumed from an earlier run, re-estimated here
                Kv = est[(pk, Y, var + "_verify")][0].astype(np.float32)
                kinfo[(pk, Y)][var]["resumed_from_earlier_run"] = True
                kinfo[(pk, Y)][var]["reestimated_bitwise_equal"] = bool(np.array_equal(Kv, K))
                kinfo[(pk, Y)][var]["reestimated_max_abs_diff"] = float(np.abs(Kv - K).max())
                assert np.array_equal(Kv, K), f"{base}: the stored array does not reproduce"
            else:
                kinfo[(pk, Y)][var]["resumed_from_earlier_run"] = False
        kinfo[(pk, Y)]["placebos_not_prespecified"] = {"seed": v["placebo_seed"], "n": N_PLACEBO,
                                                       "drawn_from": "the E2 photographs the 0.90 rule keeps",
                                                       "n_left_out_each": len(v["dropped_x90"]), "arrays": []}
        for i in range(N_PLACEBO):
            base = f"{K_OUT_DIR[pk]}/K_{Y}_E2_xbplacebo{i}"
            if (pk, Y, f"plc{i}") in est:
                K = est[(pk, Y, f"plc{i}")][0].astype(np.float32)
                np.save(base + ".npy", K)
                json.dump({"pair": pk, "body": Y, "estimator": "fingerprints.estimate_K (src/fingerprints.py)",
                           "kind": "placebo exclusion (descriptive, not pre-specified)", "seed": v["placebo_seed"],
                           "draw": i, "photographs": [os.path.basename(p) for p in v["placebo_lists"][i]],
                           "left_out": v["placebo_dropped"][i],
                           "entry": "RESULTS.md Entry 116 item 3 (fv_crossbody_scenes.py)", "sha256": sha256_array(K)},
                          open(base + ".json", "w"), indent=1)
            K = np.load(base + ".npy")
            kinfo[(pk, Y)]["placebos_not_prespecified"]["arrays"].append(
                {"file": (base + ".npy").replace(V2 + "/", ""), "sha256": sha256_array(K),
                 "left_out": v["placebo_dropped"][i], "ncc_with_stored_K": ncc(K, stored)})
    # templates per pair: the stored pair of estimates, then the re-estimates of each affected body
    for pk in groups:
        lst = [("KA_E2", f"{FP_DIR[pk]}/K_A_E2.npy"), ("KB_E2", f"{FP_DIR[pk]}/K_B_E2.npy")]
        for Y in sorted(groups[pk]):
            for var in ("x90", "x85"):
                lst.append((f"K{Y}_E2_{var}", f"{K_OUT_DIR[pk]}/K_{Y}_E2_xbscene{var[1:]}.npy"))
            for i in range(N_PLACEBO):
                lst.append((f"K{Y}_E2_plc{i}", f"{K_OUT_DIR[pk]}/K_{Y}_E2_xbplacebo{i}.npy"))
        kfiles[pk] = lst

    # ---- (c) rescore the generations (appended, resumable)
    ims = {pk: adapter_images(pk) for pk in groups}
    done = {}
    if os.path.exists(ROWS_CSV):
        with open(ROWS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                try:                                   # a line cut by an interruption is skipped and rescored
                    done.setdefault((row["pair"], row["adapter"], row["image"]), {})[row["template"]] = float(row["rho"])
                except (TypeError, ValueError, KeyError):
                    continue
    need = {pk: [n for n, _ in kfiles[pk]] for pk in kfiles}
    tasks = [(pk, tag, p) for pk in groups for tag, _, fs in ims[pk] for p in fs
             if not all(t in done.get((pk, tag, os.path.basename(p)), {}) for t in need[pk])]
    n_total = sum(len(fs) for pk in groups for _, _, fs in ims[pk])
    # rows resumed from an earlier run of this script: a seeded sample is rescored and must agree
    path_of = {(pk, tag, os.path.basename(p)): p for pk in groups for tag, _, fs in ims[pk] for p in fs}
    resumed = sorted(k for k in path_of if all(t in done.get(k, {}) for t in need[k[0]]))
    pick = []
    if resumed:
        rng = np.random.default_rng(BOOT_SEED + 1)
        pick = [resumed[i] for i in sorted(rng.choice(len(resumed), size=min(RESUME_CHECK_IMAGES, len(resumed)),
                                                      replace=False))]
    # images an earlier run scored with fewer templates are rescored in full; their earlier values are compared
    resumed_set = set(resumed)
    partial = {k: dict(done[k]) for k in path_of if k in done and k not in resumed_set}
    resume_check = {"n_resumed_images": len(resumed), "n_rescored_for_check": len(pick),
                    "n_partially_scored_images_rescored": len(partial)}
    # the stored estimates' rows of earlier-scored D200 images must equal the recorded rows of c6_estimator_swap.json
    # (same scorer, same arrays, same images) before anything new is scored
    early = [k for k in path_of if k[0] == "d200" and "KA_E2" in done.get(k, {}) and "KB_E2" in done.get(k, {})]
    if early:
        c6rows = json.load(open(C6_JSON))["rows"]
        dd = []
        for pk, tag, nm in early:
            b, s = tag.split("_s")
            ref = c6rows[f"{b}_raw_s{s}_r16"][nm]
            dd += [abs(done[(pk, tag, nm)]["KA_E2"] - ref[0]), abs(done[(pk, tag, nm)]["KB_E2"] - ref[1])]
        resume_check["earlier_d200_rows_vs_c6_max_abs_diff"] = float(max(dd))
        resume_check["earlier_d200_rows_vs_c6_n_values"] = len(dd)
        print(f"[xb-s] earlier D200 rows against c6_estimator_swap.json: max |diff| {max(dd):.3e} "
              f"over {len(dd)} values", flush=True)
        assert max(dd) <= 1e-9, resume_check
    print(f"[xb-s] scoring {len(tasks)} of {n_total} images with {WORKERS} workers "
          f"({n_total - len(tasks)} resumed from {ROWS_CSV}; {len(pick)} of them rescored as a check)", flush=True)
    if tasks or pick:
        new_file = not os.path.exists(ROWS_CSV)
        if not new_file:                               # never append onto a line cut by an interruption
            with open(ROWS_CSV, "rb") as fb:
                fb.seek(0, os.SEEK_END)
                if fb.tell() > 0:
                    fb.seek(-1, os.SEEK_END)
                    cut = fb.read(1) != b"\n"
                else:
                    cut = False
            if cut:
                with open(ROWS_CSV, "a", newline="") as fa:
                    fa.write("\r\n")
        with open(ROWS_CSV, "a", newline="") as f, Pool(WORKERS, initializer=_winit, initargs=(kfiles,)) as pool:
            w = csv.writer(f)
            if new_file:
                w.writerow(["pair", "adapter", "image", "template", "rho"])
            if pick:
                diffs = []
                for pk, tag, name, vals in pool.imap(_wscore, [(k[0], k[1], path_of[k]) for k in pick], chunksize=2):
                    for t, v in vals:
                        diffs.append(abs(v - done[(pk, tag, name)][t]))
                resume_check.update(max_abs_diff=float(max(diffs)), n_values=len(diffs),
                                    n_exactly_equal=int(sum(d == 0.0 for d in diffs)),
                                    images=[f"{k[0]}/{k[1]}/{k[2]}" for k in pick])
                print(f"[xb-s] resume check: {len(pick)} images rescored, max |diff| {max(diffs):.3e}", flush=True)
                assert max(diffs) <= 1e-9, resume_check
            t1 = time.time()
            pdiff = []
            for n, (pk, tag, name, vals) in enumerate(pool.imap_unordered(_wscore, tasks, chunksize=4), 1):
                for t, v in vals:
                    w.writerow([pk, tag, name, t, repr(v)])
                    if t in partial.get((pk, tag, name), {}):
                        pdiff.append(abs(v - partial[(pk, tag, name)][t]))
                    done.setdefault((pk, tag, name), {})[t] = v
                if n % 25 == 0:
                    f.flush()
                if n % 500 == 0 or n == len(tasks):
                    el = (time.time() - t1) / 60
                    print(f"[xb-s] {n}/{len(tasks)} scored, {el:.1f} min, eta {el / n * (len(tasks) - n):.0f} min",
                          flush=True)
            if partial:
                resume_check["partially_scored_rescored_max_abs_diff"] = float(max(pdiff)) if pdiff else None
                resume_check["partially_scored_rescored_n_values"] = len(pdiff)
                resume_check["partially_scored_rescored_agree_1e-9"] = bool(pdiff and max(pdiff) <= 1e-9)
                print(f"[xb-s] earlier partial rows rescored: max |diff| {max(pdiff) if pdiff else float('nan'):.3e} "
                      f"over {len(pdiff)} values", flush=True)

    # ---- (b) R_real and the limits, old against new on the same images
    out = {"c_calibrated_H6": c_cal, "item1_seedbank": item1, "c_calibrated_item1": c_item1,
           "rule": "relative change of the limit <= 10 % -> 'the scene overlap does not "
                   "move the limit'; > 10 % -> the recomputed limit is printed beside the "
                   "limit of record as a sensitivity and the overlap is named in "
                   "Limitations; the limit of record is not replaced",
           "rule_applied_to": ("the max-arm limits (Entry 116: 'the change in the nominal and calibrated max-arm limits'); "
                               "D200 at c = 1 (nominal), H6's c and, when item 1's rule changed the calibration, item 1's "
                               "c*; the iPhone and P20 pairs at c = 1 (their limits are nominal, Entry 106). The "
                               "symmetric limit is reported beside, descriptive"),
           "resume_check": resume_check, "pairs": {}}
    for pk in groups:
        KA0, KB0 = np.load(f"{FP_DIR[pk]}/K_A_E2.npy"), np.load(f"{FP_DIR[pk]}/K_B_E2.npy")
        variants = {"old": ("KA_E2", "KB_E2")}
        for var in ["x90", "x85"] + [f"plc{i}" for i in range(N_PLACEBO)]:
            variants[var] = (f"KA_E2_{var}" if "A" in groups[pk] else "KA_E2",
                             f"KB_E2_{var}" if "B" in groups[pk] else "KB_E2")
        Kmap = {n: np.load(p) for n, p in kfiles[pk]}
        cs = {"nominal": 1.0}
        if pk == "d200":                               # H6's c is measured for the D200 pair only (Entry 106)
            cs["calibrated_c_H6"] = c_cal
            if c_item1 is not None:
                cs["calibrated_c_item1"] = c_item1
        pr = {"label": PAIR_LABEL[pk], "affected_E2_bodies": sorted(groups[pk]),
              "estimates": {Y: kinfo[(pk, Y)] for Y in sorted(groups[pk])},
              "images": {tag: {"dir": d, "n": len(fs)} for tag, d, fs in ims[pk]},
              "basis": ("24 primary adapters, images 0-249 each (local archive copies); the ledger's 500-image basis "
                        "would need images 250-499 of seeds 3-11, on the Drive archive only, not fetched (Entry 116)"
                        if pk == "d200" else "all local generations, 12 adapters x 250 per arm"),
              "variants": {}}
        per_adapter = {}
        for label, (ta, tb) in variants.items():
            R, Rper = r_real(pk, Kmap[ta], Kmap[tb])
            A, B = [], []
            for tag, _, fs in ims[pk]:
                rows = [done[(pk, tag, os.path.basename(p))] for p in fs]
                d = [r[ta] - r[tb] if tag.startswith("A") else r[tb] - r[ta] for r in rows]
                (A if tag.startswith("A") else B).append(float(np.mean(d)))
            st = welch_and_limits(A, B, R, cs)
            st.update(templates={"A": ta, "B": tb}, R_real_per_body=Rper, per_adapter_A=A, per_adapter_B=B)
            pr["variants"][label] = st
            per_adapter[label] = (A, B)
        # relative changes, the rule (x90), and the descriptive broader exclusion (x85)
        for var in ("x90", "x85"):
            ch = {}
            for name in cs:
                o = pr["variants"]["old"]["max_arm"][name]["lambda_pct"]
                n = pr["variants"][var]["max_arm"][name]["lambda_pct"]
                ch[f"max_arm_{name}"] = {"old_pct": o, "new_pct": n, "relative_change": (n - o) / o}
            o = pr["variants"]["old"]["symmetric"]["limit99_pct"]
            n = pr["variants"][var]["symmetric"]["limit99_pct"]
            ch["symmetric_99_descriptive"] = {"old_pct": o, "new_pct": n, "relative_change": (n - o) / o}
            ch["R_real"] = {"old": pr["variants"]["old"]["R_real"], "new": pr["variants"][var]["R_real"],
                            "relative_change": (pr["variants"][var]["R_real"] - pr["variants"]["old"]["R_real"])
                            / pr["variants"]["old"]["R_real"]}
            ch["theta_sym"] = {"old": pr["variants"]["old"]["symmetric"]["theta_sym"],
                               "new": pr["variants"][var]["symmetric"]["theta_sym"],
                               "old_p": pr["variants"]["old"]["symmetric"]["one_sided_p"],
                               "new_p": pr["variants"][var]["symmetric"]["one_sided_p"]}
            ch["per_adapter_max_abs_change"] = float(max(
                np.abs(np.subtract(per_adapter[var][0], per_adapter["old"][0])).max(),
                np.abs(np.subtract(per_adapter[var][1], per_adapter["old"][1])).max()))
            # per arm: the shift of each arm's adapter means and of each arm's upper limit (U_device = max of the two)
            dA = np.subtract(per_adapter[var][0], per_adapter["old"][0])
            dB = np.subtract(per_adapter[var][1], per_adapter["old"][1])
            ch["per_arm"] = {
                "mean_shift_A": float(dA.mean()), "sd_shift_A": float(dA.std(ddof=1)),
                "mean_shift_B": float(dB.mean()), "sd_shift_B": float(dB.std(ddof=1)),
                "by_c": {name: {"U_A_old": pr["variants"]["old"]["max_arm"][name]["U_A"],
                                "U_A_new": pr["variants"][var]["max_arm"][name]["U_A"],
                                "U_B_old": pr["variants"]["old"]["max_arm"][name]["U_B"],
                                "U_B_new": pr["variants"][var]["max_arm"][name]["U_B"],
                                "binding_arm_old": pr["variants"]["old"]["max_arm"][name]["binding_arm"],
                                "binding_arm_new": pr["variants"][var]["max_arm"][name]["binding_arm"]}
                         for name in cs}}
            # calibrated multiplier sensitivity on H6's grid (a later c can be read off without rescoring)
            if pk == "d200":
                ch["relative_change_by_c"] = {}
                for c in C_GRID:
                    lo = welch_and_limits(*per_adapter["old"], pr["variants"]["old"]["R_real"], {"c": c})["max_arm"]["c"]
                    ln = welch_and_limits(*per_adapter[var], pr["variants"][var]["R_real"], {"c": c})["max_arm"]["c"]
                    ch["relative_change_by_c"][f"{c:.2f}"] = (ln["lambda_pct"] - lo["lambda_pct"]) / lo["lambda_pct"]
            key = "rule_exclusion_0.90" if var == "x90" else "broader_exclusion_0.85_not_prespecified"
            pr[key] = ch
        rule_rel = {k: v["relative_change"] for k, v in pr["rule_exclusion_0.90"].items() if k.startswith("max_arm_")}
        moves = any(abs(v) > REL_CHANGE_RULE for v in rule_rel.values())
        pr["reading"] = ("the recomputed limit is printed beside the limit of record as a sensitivity, and the overlap "
                         "is named in Limitations (the limit of record is not replaced)" if moves
                         else "the scene overlap does not move the limit")
        pr["rule_relative_changes"] = rule_rel
        pr["rule_threshold"] = REL_CHANGE_RULE
        pr["rule_limits_evaluated"] = {f"max_arm_{name}": float(c) for name, c in cs.items()}
        # ---- descriptive additions (not pre-specified)
        # (o) placebo exclusions: the same changes when as many photographs, drawn at random, are left out instead
        def changes(lab):
            o, nv = pr["variants"]["old"], pr["variants"][lab]
            e = {f"max_arm_{name}": (nv["max_arm"][name]["lambda_pct"] - o["max_arm"][name]["lambda_pct"])
                 / o["max_arm"][name]["lambda_pct"] for name in cs}
            e["symmetric_99"] = (nv["symmetric"]["limit99_pct"] - o["symmetric"]["limit99_pct"]) / o["symmetric"]["limit99_pct"]
            e["R_real"] = (nv["R_real"] - o["R_real"]) / o["R_real"]
            e["mean_shift_A"] = float(np.mean(np.subtract(per_adapter[lab][0], per_adapter["old"][0])))
            e["mean_shift_B"] = float(np.mean(np.subtract(per_adapter[lab][1], per_adapter["old"][1])))
            return e

        if N_PLACEBO:
            near_e = changes("x90")
            pl = {"n": N_PLACEBO, "note": ("each draw leaves out as many E2 photographs as the 0.90 rule does, drawn at "
                                           "random from those it keeps, re-estimated and scored on the same images; "
                                           "relative changes of the limits and R_real, and arm-mean shifts of the "
                                           "paired contrast, against the stored estimates"),
                  "near_copy_exclusion": near_e, "draws": [], "comparison": {}}
            for i in range(N_PLACEBO):
                e = changes(f"plc{i}")
                e["left_out"] = {Y: plan[(pk, Y)]["placebo_dropped"][i] for Y in sorted(groups[pk])}
                pl["draws"].append(e)
            for key in near_e:
                vals = np.array([d[key] for d in pl["draws"]])
                pl["comparison"][key] = {"near_copy_exclusion": near_e[key], "placebo_min": float(vals.min()),
                                         "placebo_max": float(vals.max()), "placebo_mean": float(vals.mean()),
                                         "placebo_sd": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                                         "n_placebo_abs_ge_near_copy": int(np.sum(np.abs(vals) >= abs(near_e[key])))}
            pr["placebo_exclusions_not_prespecified"] = pl
        # (i) seed bootstrap of the relative change: how far the 250-image basis could move it
        names_ref = None
        Dm = {}
        for label, (ta, tb) in variants.items():
            DA, DB = [], []
            for tag, _, fs in ims[pk]:
                nm = [os.path.basename(p) for p in fs]
                names_ref = names_ref or nm
                assert nm == names_ref, (pk, tag, "image names differ between adapters")
                rows = [done[(pk, tag, x)] for x in nm]
                (DA if tag.startswith("A") else DB).append(
                    [r[ta] - r[tb] if tag.startswith("A") else r[tb] - r[ta] for r in rows])
            Dm[label] = (np.asarray(DA, np.float64), np.asarray(DB, np.float64))
        pr["seed_bootstrap_not_prespecified"] = {
            var: seed_bootstrap({"old": Dm["old"], var: Dm[var]},
                                {"old": pr["variants"]["old"]["R_real"], var: pr["variants"][var]["R_real"]}, cs)
            for var in ("x90", "x85")}
        # (ii) D200 only: the per-adapter shifts measured here added to the ledger's per-adapter values (archive rows,
        # 500 images), with the ledger R_real scaled by the measured relative change of R_real. An approximation: the
        # archive instrument's own estimates are not re-estimated, and its binding arm (B) differs from this basis' (A).
        if pk == "d200":
            ledj = json.load(open(LEDGER))
            LA, LB = np.asarray(ledj["primary"]["per_adapter_A"]), np.asarray(ledj["primary"]["per_adapter_B"])
            Rl = float(ledj["denominators"]["R_real"])
            tr = {}
            for var in ("x90", "x85"):
                sA = np.subtract(per_adapter[var][0], per_adapter["old"][0])
                sB = np.subtract(per_adapter[var][1], per_adapter["old"][1])
                Rs = pr["variants"][var]["R_real"] / pr["variants"]["old"]["R_real"]
                lo = welch_and_limits(LA, LB, Rl, cs)["max_arm"]
                ln = welch_and_limits(LA + sA, LB + sB, Rl * Rs, cs)["max_arm"]
                lm = welch_and_limits(LA + sA.mean(), LB + sB.mean(), Rl * Rs, cs)["max_arm"]
                tr[var] = {name: {"ledger_pct": lo[name]["lambda_pct"],
                                  "per_adapter_shifts_pct": ln[name]["lambda_pct"],
                                  "per_adapter_shifts_relative_change": (ln[name]["lambda_pct"] - lo[name]["lambda_pct"])
                                  / lo[name]["lambda_pct"],
                                  "arm_mean_shift_pct": lm[name]["lambda_pct"],
                                  "arm_mean_shift_relative_change": (lm[name]["lambda_pct"] - lo[name]["lambda_pct"])
                                  / lo[name]["lambda_pct"],
                                  "binding_arm_ledger": lo[name]["binding_arm"],
                                  "binding_arm_per_adapter_shifts": ln[name]["binding_arm"],
                                  "binding_arm_arm_mean_shift": lm[name]["binding_arm"]} for name in cs}
                tr[var]["R_real_scale"] = Rs
                tr[var]["arm_mean_shift"] = {"A": float(sA.mean()), "B": float(sB.mean())}
            tr["ledger_reproduced"] = {name: lo[name]["lambda_pct"] for name in cs}
            tr["placebos_arm_mean_shift_relative_change"] = {}
            for i in range(N_PLACEBO):
                lab = f"plc{i}"
                sA = np.subtract(per_adapter[lab][0], per_adapter["old"][0])
                sB = np.subtract(per_adapter[lab][1], per_adapter["old"][1])
                Rs = pr["variants"][lab]["R_real"] / pr["variants"]["old"]["R_real"]
                lm = welch_and_limits(LA + sA.mean(), LB + sB.mean(), Rl * Rs, cs)["max_arm"]
                tr["placebos_arm_mean_shift_relative_change"][lab] = {
                    name: (lm[name]["lambda_pct"] - lo[name]["lambda_pct"]) / lo[name]["lambda_pct"] for name in cs}
            tr["note"] = ("the shifts measured here (new minus old estimate, 250 local images per adapter) added to the "
                          "ledger's per-adapter values (archive rows, 500 images): either each adapter's own shift, or "
                          "each arm's mean shift (which moves U_X by exactly that mean and leaves s_X unchanged; the "
                          "per-adapter version also adds the 250-image sampling noise of the shifts to s_X). The ledger "
                          "R_real is scaled by the local relative change of R_real. The ledger's binding arm (B) is not "
                          "the 250-image basis' (A), so this shows the change where the limit of record is decided. "
                          "Approximate: the archive's own fingerprint estimates are not re-estimated here.")
            pr["ledger_transport_not_prespecified"] = tr
        # reproduction checks of the old variant against the files that produced the recorded results
        chk = {}
        if pk == "d200":
            c6 = json.load(open(C6_JSON))
            diffs = []
            for tag, _, fs in ims[pk]:
                b, s = tag.split("_s")
                ref = c6["rows"][f"{b}_raw_s{s}_r16"]
                for p in fs:
                    nm = os.path.basename(p)
                    r = done[(pk, tag, nm)]
                    diffs += [abs(r["KA_E2"] - ref[nm][0]), abs(r["KB_E2"] - ref[nm][1])]
            chk["old_rows_vs_c6_estimator_swap_rows_max_abs_diff"] = float(max(diffs))
            chk["old_rows_vs_c6_n_values"] = len(diffs)
            g3 = json.load(open(G3_JSON))["estimates"]["E2"]
            chk["R_real_old_vs_g3_E2_local"] = {"recomputed": pr["variants"]["old"]["R_real"],
                                                "g3_estimator_scale.json": g3["R_real_own_scale"]}
            chk["R_real_ledger_archive_instrument"] = json.load(open(LEDGER))["denominators"]["R_real"]
            if not DEV_IMAGES:
                c6s = c6["summary"]["E2"]
                chk["old_vs_c6_summary_E2_250"] = {
                    "maxarm_U": [pr["variants"]["old"]["max_arm"]["nominal"]["U_device"], c6s["maxarm_U"]],
                    "theta_sym": [pr["variants"]["old"]["symmetric"]["theta_sym"], c6s["theta_sym"]]}
            led = json.load(open(LEDGER))["primary"]
            chk["limit_of_record"] = {"nominal_pct": h6["lambda_U_nominal_pct"],
                                      "calibrated_pct": h6["lambda_U_calibrated_pct"], "c": c_cal,
                                      "item1_bound_of_record_pct": item1.get("bound_of_record_pct"),
                                      "item1_c_star": item1.get("c_star"), "item1_branch": item1.get("branch"),
                                      "basis": "archive rows, 500 images per adapter, ledger R_real",
                                      "ledger_U_device": led["U_device"]}
        else:
            ref = json.load(open(G4B_JSON if pk == "p20b" else G6_JSON))
            if pk == "p5c":
                ref = dict(ref["estimators"]["E2"], R_real=ref["estimators"]["E2"]["R_real"])
            man = json.load(open(MAN[pk]))
            chk["R_real_old_vs_manifest_gate"] = {"recomputed": pr["variants"]["old"]["R_real"],
                                                  "manifest": man["gates"]["R_real"]}
            if not DEV_IMAGES:
                chk["old_per_adapter_vs_recorded_max_abs_diff"] = float(max(
                    np.abs(np.subtract(pr["variants"]["old"]["per_adapter_A"], ref["per_adapter_A"])).max(),
                    np.abs(np.subtract(pr["variants"]["old"]["per_adapter_B"], ref["per_adapter_B"])).max()))
                chk["old_max_arm_vs_recorded"] = [pr["variants"]["old"]["max_arm"]["nominal"]["lambda_pct"],
                                                  ref["max_arm"]["lambda_U_pct"]]
                chk["old_symmetric_vs_recorded"] = [pr["variants"]["old"]["symmetric"]["limit99_pct"],
                                                    ref["symmetric"]["lambda_sym_pct"]]
            chk["limit_of_record"] = {"max_arm_nominal_pct": ref["max_arm"]["lambda_U_pct"],
                                      "symmetric_99_pct": ref["symmetric"]["lambda_sym_pct"],
                                      "file": (G4B_JSON if pk == "p20b" else G6_JSON).replace(V2 + "/", "")}
        pr["reproduction_checks"] = chk
        out["pairs"][pk] = pr
        print(f"[xb-s] {pk}: rule relative changes {json.dumps(rule_rel)} -> {pr['reading']}", flush=True)
    out["rows_csv"] = ROWS_CSV.replace(V2 + "/", "")
    out["dev_images_per_adapter"] = DEV_IMAGES or None
    out["minutes"] = (time.time() - t0) / 60
    return out


def make_summary(res, prim):
    """Compact summary of the pre-specified quantities (every value also appears in its block of the output)."""
    summ = {"reading": res["reading"], "reading_per_pair": res.get("reading_per_pair"), "pairs": {}}
    for pk, pr in prim.items():
        s = {"max_cross_body_cosine_either": pr["max_cross_body_cosine"],
             "max_cross_body_cosine_E2_full": pr["max_cross_body_cosine_E2_full"],
             "max_cross_body_cosine_E2_crop": pr["max_cross_body_cosine_E2_crop"],
             "max_by_direction": {k: {"either": v["either"]["max"], "E2_full": v["E2_full"]["max"],
                                      "E2_crop": v["E2_crop"]["max"],
                                      "largest_cos_below_threshold": v.get("largest_cos_below_threshold")}
                                  for k, v in pr["directions"].items()},
             "n_near_copies": pr["n_near_copies"], "n_affected_E2_photographs": pr["n_affected_E2_photographs"],
             "near_copies": [{"direction": k, "T_photograph": q["T_photograph"], "E2_photograph": q["E2_photograph"],
                              "cos_E2_full": q["cos_E2_full"], "cos_E2_crop": q["cos_E2_crop"]}
                             for k, v in pr["directions"].items() for q in v["near_copies"]]}
        if res.get("sensitivity") and pk in res["sensitivity"]["pairs"]:
            sp = res["sensitivity"]["pairs"][pk]
            s["rule_relative_changes"] = sp["rule_relative_changes"]
            s["rule_reading"] = sp["reading"]
            s["limits_250_basis_pct"] = {
                lab: {**{f"max_arm_{n}": v["max_arm"][n]["lambda_pct"] for n in v["max_arm"]},
                      "symmetric_99": v["symmetric"]["limit99_pct"], "R_real": v["R_real"]}
                for lab, v in sp["variants"].items() if lab in ("old", "x90", "x85")}
            if "placebo_exclusions_not_prespecified" in sp:
                s["placebo_comparison_not_prespecified"] = sp["placebo_exclusions_not_prespecified"]["comparison"]
            if "ledger_transport_not_prespecified" in sp:
                t = sp["ledger_transport_not_prespecified"]["x90"]
                s["ledger_transport_x90_not_prespecified"] = {
                    n: {"arm_mean_shift_relative_change": t[n]["arm_mean_shift_relative_change"],
                        "per_adapter_shifts_relative_change": t[n]["per_adapter_shifts_relative_change"]}
                    for n in t if isinstance(t[n], dict) and "ledger_pct" in t[n]}
        summ["pairs"][pk] = s
    return summ


# ------------------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    if os.path.exists(OUT_JSON) or os.path.exists(OUT_NPZ):
        raise SystemExit(f"refusing to overwrite an existing result file: {OUT_JSON} / {OUT_NPZ}")
    if FINAL and DEV_IMAGES:
        raise SystemExit("FV_XB_DEV_IMAGES is for development runs only")
    os.makedirs(OUT_DIR, exist_ok=True)
    _ladder()
    pairs = build_inputs()
    checks = verify_inputs(pairs)
    print(f"[xb] input checks passed ({time.time()-t0:.0f}s)", flush=True)

    E = Embedder()
    print(f"[xb] embedder: {E.info['device']}, {E.info['processor_class']} {E.info['processor_backend']}", flush=True)
    emb, names = {}, {}
    for pk, pv in pairs.items():
        for r, b in pv["bodies"].items():
            jobs = [("T_crop", load_png, b["T_crop_files"], b["T"]),
                    ("E2_full", load_full, b["E2"], b["E2"]),
                    ("E2_crop", load_centre_crop, b["E2"], b["E2"]),
                    ("H_full", load_full, b["H"], b["H"]),
                    ("H_crop", load_centre_crop, b["H"], b["H"])]
            if pk == "d200":   # the local decode (check) and the v1-style full-photograph audit (not pre-specified)
                jobs += [("T_local", load_png, b["T_local_files"], b["T"]),
                         ("T_full", load_full, b["T"], b["T"]),
                         ("E1_full", load_full, b["E1"], b["E1"])]
            for tag, loader, items, photos in jobs:
                key = f"{pk}_{r}_{tag}"
                emb[key] = E.run(loader, items)
                names[key] = [os.path.basename(p) for p in photos]
            print(f"[xb] embedded {pk} {r} ({time.time()-t0:.0f}s)", flush=True)
    # upright versions of every EXIF-rotated input of the primary sets (not pre-specified)
    up = {}
    for pk, pv in pairs.items():
        for r, b in pv["bodies"].items():
            for tag, loader, items, photos, ok in (("T_crop", load_png, b["T_crop_files"], b["T"], b["T_orient"]),
                                                   ("E2_full", load_full, b["E2"], b["E2"], b["E2_orient"]),
                                                   ("E2_crop", load_centre_crop, b["E2"], b["E2"], b["E2_orient"])):
                idx = [i for i, t in enumerate(ok) if t in EXIF_TRANSPOSE]
                if idx:
                    Z = emb[f"{pk}_{r}_{tag}"].copy()
                    Z[idx] = E.run(upright(loader), [(items[i], photos[i]) for i in idx])
                    up[f"{pk}_{r}_{tag}"] = (Z, idx)

    res = {"entry": "RESULTS.md Entry 116, item 3 - cross-body scene audit (paper/fv/REVIEW_REPORT.md section 4 item 7)",
           "status": ("pre-specified data-audit rule with a fixed threshold and a fixed sensitivity (Entry 116); "
                      "blocks whose names end in _not_prespecified are descriptive additions"),
           "script": "src/fv/fv_crossbody_scenes.py",
           "script_sha256": sha256_file(os.path.abspath(__file__)),
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "deviations_and_choices": [
               "File names: Entry 116 names the script and output fv_crossbody_scene.py / fv_crossbody_scene.json "
               "(+ _emb.npz); the run's instructions named them fv_crossbody_scenes.*, which are used. Nothing else "
               "depends on the name.",
               "T for the D200 pair is the archive crops the 24 primary adapters trained on (colab_a0 = archive A_raw, "
               "archive B_raw); the local decodes of the same photographs (none_a0, noneB_a0) are embedded as a check.",
               "The v1 audit's ViT-S/14 replication was not run: its weights are not on this machine and fetching them "
               "needs the author's approval (as Entry 116 foresees).",
               "Old and new estimates are compared on the local instrument (out/fp*/K_*_E2.npy, H_{A,B}.npz, local "
               "generations), which is the basis Entry 116 fixes; the D200 limit of record comes from the archive "
               "rows and the ledger's R_real and is quoted beside, not recomputed.",
               "The rule is evaluated on the D200 max-arm limit at c = 1, at H6's c = 1.25 and, because item 1's result "
               "file (out/fv_seedbank_calib.json) applies its rule's 'changes' branch, at item 1's c* as well; the "
               "relative change at every c of H6's grid (1.00-2.50, step 0.01) is tabulated (relative_change_by_c).",
               "This run resumes an interrupted run of the same script (30 Sep 2026, killed during scoring): the "
               "re-estimated fingerprint arrays it saved are re-estimated here and must agree bitwise, a seeded sample "
               "of its scored rows is rescored and must agree to 1e-9, and every other image is scored here. The "
               "embeddings, the near-copy search and everything downstream are recomputed in this run.",
               "Added, not pre-specified (labelled _not_prespecified): placebo exclusions (as many E2 photographs as "
               "the rule drops, at random, re-estimated and scored on the same images), a seed bootstrap of the "
               "relative change, the per-arm shifts transported to the ledger's per-adapter values (D200), the "
               "broader 0.85 exclusion, what the near-copy crops look like (luminance SD), the upright (EXIF) "
               "check, the v1-style within-body audit with dinov2-base, and instrument checks."],
           "settings": {"embedding": E.info, "near_copy_threshold": NEAR_COPY,
                        "near_copy_rule": "cross-body pair (T crop, E2 photograph) with cosine >= 0.90 in either E2 "
                                          "representation (full photograph or 1024^2 centre crop)",
                        "descriptive_thresholds": list(DESCRIPTIVE_THRESHOLDS),
                        "representations": {
                            "T_crop": "the 1024^2 native centre crop that was trained on (PNG as stored)",
                            "E2_full / H_full": "the full photograph (pixel array as stored, RGB), as in the v1 audit; the "
                                                "processor resizes the shorter side to 256 and centre-crops 224",
                            "E2_crop / H_crop": "the centred 1024^2 window of the stored pixel array "
                                                "(t1_ladder.load_rgb_crop), the pixels the fingerprint (E2) or R_real (H) "
                                                "is computed from"},
                        "cosine_arithmetic": "float64 dot products of the float32 L2-normalised embeddings",
                        "exif_orientation": "not applied (as the estimator, the training crops and the v1 audit); an "
                                            "upright check is reported separately",
                        "scoring_workers": WORKERS},
           "inputs": {}, "input_checks": checks}
    for pk, pv in pairs.items():
        res["inputs"][pk] = {"label": PAIR_LABEL[pk], "dataset": pv["dataset"], "manifest": MAN[pk].replace(V2 + "/", ""),
                             "manifest_sha256": sha256_file(MAN[pk]), "bodies": {}}
        for r, b in pv["bodies"].items():
            ent = {"device": b["device"], "T_crop_dir": b["T_crop_dir"],
                   "T_crop_dir_digest_sha256": digest_of_files(b["T_crop_files"]),
                   **{f"{k}_files": [os.path.basename(p) for p in b[k]] for k in ("E2", "T", "H")},
                   "n": {k: len(b[k]) for k in ("E1", "E2", "T", "H")}}
            if pk == "d200":
                ent["T_local_dir"] = b["T_local_dir"]
                ent["T_local_dir_digest_sha256"] = digest_of_files(b["T_local_files"])
                ent["E1_files"] = [os.path.basename(p) for p in b["E1"]]
            res["inputs"][pk]["bodies"][r] = ent

    # ---------------- instrument checks (not pre-specified)
    inst = {}
    if os.path.exists(DINO_MEM_NPZ):
        z = np.load(DINO_MEM_NPZ)
        ref = {r: z["train_" + r].astype(np.float64) for r in ("A", "B")}
        mine = {r: emb[f"d200_{r}_T_local"].astype(np.float64) for r in ("A", "B")}
        for r in ("A", "B"):
            inst[f"dino_memorization_train_{r}_vs_this_run_T_local"] = {
                "max_abs_embedding_diff": float(np.abs(ref[r] - mine[r]).max()),
                "min_self_cosine": float((ref[r] * mine[r]).sum(1).min())}
        inst["dino_memorization_T_A_x_T_B_max_abs_cosine_diff"] = float(np.abs(ref["A"] @ ref["B"].T
                                                                              - mine["A"] @ mine["B"].T).max())
        inst["dino_memorization_source"] = ("out/t1/dino_memorization_emb.npz train_A/train_B: the local decodes "
                                            "none_a0/noneB_a0 embedded on the CPU on 16 Sep 2026")
    if os.path.exists(CONTENT_MATCH):
        cm = json.load(open(CONTENT_MATCH))
        S = cos(emb["d200_A_T_local"], emb["d200_B_T_local"])
        inst["content_match_reference_T_split"] = {
            "file": "out/t1/content_match.json",
            "recorded_best_match_mean": cm.get("reference_unmatched_T_split_best_match_mean"),
            "recomputed_best_match_mean": float(S.max(1).mean()),
            "recorded_mean": cm.get("reference_unmatched_T_split_diagonal_free_mean"),
            "recomputed_mean": float(S.mean()),
            "content_matched_pairs_cos_mean_min": [cm.get("pair_cos_mean"), cm.get("pair_cos_min")],
            "note": "the content-matched D200 training sets (Entry 71) pair the two bodies' photographs at cosine "
                    "0.936-0.988 with this embedding: same-scene cross-body photographs do reach the 0.90 criterion"}
    for r in ("A", "B"):
        a, b = emb[f"d200_{r}_T_crop"].astype(np.float64), emb[f"d200_{r}_T_local"].astype(np.float64)
        inst[f"d200_{r}_archive_vs_local_decode_self_cosine_min"] = float((a * b).sum(1).min())
    if E.dev == "cuda":
        diffs, used = [], []
        PB = pairs
        for pk, r, tag, loader, items, a in (("d200", "A", "T_crop", load_png, PB["d200"]["bodies"]["A"]["T_crop_files"], 0),
                                             ("d200", "B", "E2_full", load_full, PB["d200"]["bodies"]["B"]["E2"], 132),
                                             ("d200", "B", "E2_crop", load_centre_crop, PB["d200"]["bodies"]["B"]["E2"], 132),
                                             ("p5c", "A", "E2_full", load_full, PB["p5c"]["bodies"]["A"]["E2"], 0),
                                             ("p20b", "A", "E2_crop", load_centre_crop, PB["p20b"]["bodies"]["A"]["E2"], 76),
                                             ("p20b", "B", "T_crop", load_png, PB["p20b"]["bodies"]["B"]["T_crop_files"], 8)):
            sl = slice(a, a + CPU_CHECK_PER_SET)
            c = E.run(loader, items[sl], cpu=True)
            g = emb[f"{pk}_{r}_{tag}"][sl]
            diffs.append(float(np.abs(c.astype(np.float64) - g.astype(np.float64)).max()))
            used.append(f"{pk}_{r}_{tag}[{a}:{a + CPU_CHECK_PER_SET}]")
        inst["gpu_vs_cpu_max_abs_embedding_diff"] = max(diffs)
        inst["gpu_vs_cpu_sets"] = used
        inst["gpu_vs_cpu_note"] = ("GPU float32 forward (TF32 off) against the CPU forward of the same processed "
                                   "inputs; the sets cover the near-copy candidates")
    res["instrument_checks_not_prespecified"] = inst
    print("[xb] instrument checks: " + json.dumps(plain(inst))[:600], flush=True)

    # ---------------- primary: T(X) x E2(Y), cross-body
    prim, near = {}, []
    for pk, pv in pairs.items():
        pr = {"label": PAIR_LABEL[pk], "directions": {}}
        for X, Y in (("A", "B"), ("B", "A")):
            tn, en = names[f"{pk}_{X}_T_crop"], names[f"{pk}_{Y}_E2_full"]
            Sf = cos(emb[f"{pk}_{X}_T_crop"], emb[f"{pk}_{Y}_E2_full"])
            Sc = cos(emb[f"{pk}_{X}_T_crop"], emb[f"{pk}_{Y}_E2_crop"])
            Se = np.maximum(Sf, Sc)
            d = {"T_body": X, "T_device": pv["bodies"][X]["device"], "E2_body": Y, "E2_device": pv["bodies"][Y]["device"],
                 "E2_full": summarise(Sf, tn, en, "T", "E2"), "E2_crop": summarise(Sc, tn, en, "T", "E2"),
                 "either": summarise(Se, tn, en, "T", "E2", list_at=BROAD)}
            ii, jj = np.nonzero(Se >= NEAR_COPY)
            d["near_copies"] = [{"T_crop_index": int(a), "T_photograph": tn[a], "E2_index": int(b), "E2_photograph": en[b],
                                 "cos_E2_full": float(Sf[a, b]), "cos_E2_crop": float(Sc[a, b]),
                                 "cos_either": float(Se[a, b])}
                                for a, b in sorted(zip(ii, jj), key=lambda x: -Se[x[0], x[1]])]
            d["max_either_minus_threshold"] = float(Se.max() - NEAR_COPY)
            if d["near_copies"]:
                d["smallest_near_copy_margin"] = float(min(q["cos_either"] for q in d["near_copies"]) - NEAR_COPY)
            below = Se[Se < NEAR_COPY]                 # how close the nearest non-copy comes (robustness of the split)
            d["largest_cos_below_threshold"] = float(below.max()) if below.size else None
            by_e2 = {}
            for q in d["near_copies"]:
                by_e2.setdefault(q["E2_photograph"], []).append(
                    {"T_photograph": q["T_photograph"], "T_crop_index": q["T_crop_index"],
                     "cos_E2_full": q["cos_E2_full"], "cos_E2_crop": q["cos_E2_crop"]})
            d["affected_E2_photographs"] = [{"E2_photograph": k, "E2_index": en.index(k), "matching_T_crops": v}
                                            for k, v in sorted(by_e2.items())]
            for q in d["near_copies"]:
                near.append({"pair": pk, "direction": f"T({X}) x E2({Y})", **q})
            pr["directions"][f"T({X}) x E2({Y})"] = d
        pr["max_cross_body_cosine"] = max(v["either"]["max"] for v in pr["directions"].values())
        pr["max_cross_body_cosine_E2_full"] = max(v["E2_full"]["max"] for v in pr["directions"].values())
        pr["max_cross_body_cosine_E2_crop"] = max(v["E2_crop"]["max"] for v in pr["directions"].values())
        pr["n_near_copies"] = sum(len(v["near_copies"]) for v in pr["directions"].values())
        pr["n_affected_E2_photographs"] = sum(len(v["affected_E2_photographs"]) for v in pr["directions"].values())
        prim[pk] = pr
        print(f"[xb] {pk}: max cross-body T x E2 cosine {pr['max_cross_body_cosine']:.4f} "
              f"(full {pr['max_cross_body_cosine_E2_full']:.4f}, crop {pr['max_cross_body_cosine_E2_crop']:.4f}); "
              f"near-copies {pr['n_near_copies']} on {pr['n_affected_E2_photographs']} E2 photographs", flush=True)
    res["primary"] = prim

    # ---------------- descriptive (pre-specified)
    desc = {}
    for pk in pairs:
        dd = {}
        for X, Y in (("A", "B"), ("B", "A")):
            hn, en = names[f"{pk}_{X}_H_crop"], names[f"{pk}_{Y}_E2_full"]
            combos = {f"{h} x {e}": cos(emb[f"{pk}_{X}_{h}"], emb[f"{pk}_{Y}_{e}"])
                      for h in ("H_crop", "H_full") for e in ("E2_full", "E2_crop")}
            dd[f"H({X}) x E2({Y})"] = {**{k: summarise(v, hn, en, "H", "E2") for k, v in combos.items()},
                                       "any_representation": summarise(np.maximum.reduce(list(combos.values())), hn, en,
                                                                       "H", "E2", list_at=NEAR_COPY)}
        tA, tB = names[f"{pk}_A_T_crop"], names[f"{pk}_B_T_crop"]
        dd["T(A) x T(B)"] = summarise(cos(emb[f"{pk}_A_T_crop"], emb[f"{pk}_B_T_crop"]), tA, tB, "T_A", "T_B",
                                      list_at=NEAR_COPY)
        for X in ("A", "B"):
            tn, en = names[f"{pk}_{X}_T_crop"], names[f"{pk}_{X}_E2_full"]
            Sf = cos(emb[f"{pk}_{X}_T_crop"], emb[f"{pk}_{X}_E2_full"])
            Sc = cos(emb[f"{pk}_{X}_T_crop"], emb[f"{pk}_{X}_E2_crop"])
            dd[f"within-body T({X}) x E2({X})"] = {"E2_full": summarise(Sf, tn, en, "T", "E2"),
                                                   "E2_crop": summarise(Sc, tn, en, "T", "E2"),
                                                   "either": summarise(np.maximum(Sf, Sc), tn, en, "T", "E2",
                                                                       list_at=NEAR_COPY)}
        desc[pk] = dd
    res["descriptive"] = desc

    # ---------------- upright check (not pre-specified)
    upc = {}
    for pk in pairs:
        for X, Y in (("A", "B"), ("B", "A")):
            def g(key):
                return up[key][0] if key in up else emb[key]
            Se = np.maximum(cos(g(f"{pk}_{X}_T_crop"), g(f"{pk}_{Y}_E2_full")),
                            cos(g(f"{pk}_{X}_T_crop"), g(f"{pk}_{Y}_E2_crop")))
            ii, jj = np.nonzero(Se >= NEAR_COPY)
            tn, en = names[f"{pk}_{X}_T_crop"], names[f"{pk}_{Y}_E2_full"]
            upc[f"{pk} T({X}) x E2({Y})"] = {
                "n_rotated_T": len(up.get(f"{pk}_{X}_T_crop", (None, []))[1]),
                "n_rotated_E2": len(up.get(f"{pk}_{Y}_E2_full", (None, []))[1]),
                "max_either": float(Se.max()),
                "near_copies": [{"T_photograph": tn[a], "E2_photograph": en[b], "cos_either": float(Se[a, b])}
                                for a, b in zip(ii, jj)]}
    res["exif_upright_check_not_prespecified"] = {
        "note": ("photographs with an EXIF orientation tag other than 1 re-embedded upright (full photograph and centre "
                 "crop turned by the tag; T crops by their source photograph's tag); every other input unchanged. "
                 "Content rotated relative to the other body's generations cannot align pixel-for-pixel with them, "
                 "so the stored orientation is the one the bias mechanism acts in; this check only asks whether a "
                 "shared scene was hidden by rotation."),
        "directions": upc}

    # ---------------- v1-style within-body audit with dinov2-base (not pre-specified; D200 only)
    v1 = json.load(open(V1_MANIFEST)) if os.path.exists(V1_MANIFEST) else None
    v1s = {}
    for r in ("A", "B"):
        sets = {k: emb[f"d200_{r}_{k}_full"] for k in ("E1", "E2", "T", "H")}
        per = {f"{a}-{b}": float(cos(sets[a], sets[b]).max()) for a, b in itertools.combinations(("E1", "E2", "T", "H"), 2)}
        worst = max(per, key=per.get)
        v1s[r] = {"device": pairs["d200"]["bodies"][r]["device"], "per_split_pair_max": per, "worst": per[worst],
                  "worst_pair": worst, "v1_recorded_vits14": (v1["scene_audit"][r] if v1 else None)}
    v1s["note"] = ("v1's audit (notebooks/01_pilot.ipynb scene_audit()) used DINOv2 ViT-S/14 via torch.hub on the full "
                   "photographs of all four splits, halt at 0.95; recomputed here with dinov2-base on the same inputs. "
                   "The two models' cosine scales differ, so the comparison is of order, not identity.")
    res["v1_style_within_body_audit_dinov2_base_not_prespecified"] = v1s
    res["v1_vits14_replication"] = ("not run: DINOv2 ViT-S/14 (torch.hub facebookresearch/dinov2) weights are not on "
                                    "this machine and fetching them needs the author's approval (Entry 116 item 3)")

    # ---------------- what the near-copies look like (not pre-specified): luminance SD of the 1024^2 crops, in grey
    # levels, against the median over the pair's T crops and E2 crops (a featureless wall has a small SD)
    if near:
        from fingerprints import load_lum_crop

        def png_lum_sd(p):
            a = np.asarray(Image.open(p).convert("RGB"), np.float32)
            return float((0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).std())

        look = {"note": ("luminance standard deviation of the 1024^2 crop (grey levels): the T crop as trained on, the E2 "
                         "crop as the estimator reads it (fingerprints.load_lum_crop); medians over all T crops of the "
                         "T body and all E2 photographs of the E2 body of that pair for reference"), "pairs": {}}
        for pk in sorted({q["pair"] for q in near}):
            ent = {"near_copies": [], "median": {}}
            for q in [q for q in near if q["pair"] == pk]:
                X, Y = q["direction"][2], q["direction"][-2]
                b_t, b_e = pairs[pk]["bodies"][X], pairs[pk]["bodies"][Y]
                ent["near_copies"].append({
                    "direction": q["direction"], "T_photograph": q["T_photograph"], "E2_photograph": q["E2_photograph"],
                    "T_crop_lum_sd": png_lum_sd(b_t["T_crop_files"][q["T_crop_index"]]),
                    "E2_crop_lum_sd": float(load_lum_crop(b_e["E2"][q["E2_index"]]).std())})
                key = q["direction"]
                if key not in ent["median"]:
                    ent["median"][key] = {
                        "T_crops": float(np.median([png_lum_sd(p) for p in b_t["T_crop_files"]])),
                        "E2_crops": float(np.median([float(load_lum_crop(p).std()) for p in b_e["E2"]]))}
            look["pairs"][pk] = ent
        res["near_copy_content_not_prespecified"] = look

    # ---------------- rule
    res["near_copies"] = near
    res["n_near_copies_total"] = len(near)
    res["pairs_with_near_copies"] = sorted({q["pair"] for q in near})
    if not near:
        res["reading"] = "no scene shared between one body's training crops and the other body's estimate photographs"
        res["sensitivity"] = None
    else:
        res["reading"] = ("near-copies found in " + ", ".join(res["pairs_with_near_copies"])
                          + ": the affected E2 photographs are listed and the pre-specified sensitivity was run")
        res["sensitivity"] = sensitivity(pairs, emb, names, near)
        res["reading_per_pair"] = {pk: v["reading"] for pk, v in res["sensitivity"]["pairs"].items()}
        for pk in pairs:
            if pk not in res["reading_per_pair"]:
                res["reading_per_pair"][pk] = ("no near-copy: no scene shared between one body's training crops and "
                                               "the other body's estimate photographs")
    res["summary"] = make_summary(res, prim)
    res["runtime_min"] = (time.time() - t0) / 60
    # written last, so that an interrupted run leaves no result file and resumes from the rows file; the JSON text
    # is built before any file is opened, and the NPZ is written under a temporary name and moved into place
    payload = json.dumps(plain(res), indent=1)
    np.savez_compressed(OUT_NPZ + ".partial.npz", **emb, **{"upright__" + k: v[0] for k, v in up.items()},
                        **{"names__" + k: np.array(v) for k, v in names.items()})
    os.replace(OUT_NPZ + ".partial.npz", OUT_NPZ)
    with open(OUT_JSON, "w") as f:
        f.write(payload)
    print(f"[xb] reading: {res['reading']}", flush=True)
    if near:
        print("[xb] per pair: " + json.dumps(res["reading_per_pair"]), flush=True)
    print(f"[xb] wrote {OUT_JSON} and {OUT_NPZ} in {res['runtime_min']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
