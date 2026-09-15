"""t6_weapon_clip.py -- zero-shot CLIP estimate of the fraction of generated images that depict a firearm.

Why: the personalization caption "a photograph, sks style" contains the token "sks", which also names
the SKS carbine, and SD 3.5 Medium often draws a rifle for it.  This script measures how often.

Method
  * open_clip ViT-L-14-quickgelu, pretrained="openai" (OpenAI CLIP ViT-L/14, from the local HF cache),
    CPU only (CUDA_VISIBLE_DEVICES="" is set before torch is imported).
  * Each image: standard CLIP preprocessing (resize 224 + centre crop), L2-normalised embedding.
  * Labels: FIREARM_LABELS vs NON_WEAPON_LABELS (fixed below).  Softmax over all labels of
    100 * cosine similarity (CLIP's learned logit scale); an image is flagged as a firearm if the
    summed probability of the firearm labels exceeds THRESHOLD (0.5).
  * Per-folder n, count, fraction, Wilson 95% interval; pooled groups.
  * Manual validation labels (made by visually inspecting a seeded stratified sample) are read from
    --validation (JSON {"folder/file": 0|1}) and compared with the classifier.

Usage
  python t6_weapon_clip.py --stage embed    --cache <dir>          # image embeddings -> <dir>/*.npz
  python t6_weapon_clip.py --stage classify --cache <dir> [--validation v.json] [--out ...json]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse, glob, json, math, time
import numpy as np

ROOT = os.path.join(EINV.V2, 'out')
FOLDERS = {
    # single training caption "a photograph, sks style"
    "t1/local_base":     r"t1\gens\local_base",
    "t1/local_A_raw_s0": r"t1\gens\local_A_raw_s0",
    "t1/local_B_raw_s0": r"t1\gens\local_B_raw_s0",
    "t1/nomark_s0":      r"t1\gens\nomark_s0",
    "t1/nomark_s1":      r"t1\gens\nomark_s1",
    "t1/nomark_s2":      r"t1\gens\nomark_s2",
    "t1/nomarkB_s0":     r"t1\gens\nomarkB_s0",
    "t1/nomarkB_s1":     r"t1\gens\nomarkB_s1",
    "t1/nomarkB_s2":     r"t1\gens\nomarkB_s2",
    # five-caption bank (caption j mod 5)
    "t5/base":           r"t5\gens\base",
    "t5/A_raw_s0_r16":   r"t5\gens\A_raw_s0_r16",
    "t5/B_raw_s0_r16":   r"t5\gens\B_raw_s0_r16",
}
POOLS = {
    "single_caption_base":     ["t1/local_base"],
    "single_caption_adapters": ["t1/local_A_raw_s0", "t1/local_B_raw_s0", "t1/nomark_s0", "t1/nomark_s1",
                                "t1/nomark_s2", "t1/nomarkB_s0", "t1/nomarkB_s1", "t1/nomarkB_s2"],
    "five_caption_all":        ["t5/base", "t5/A_raw_s0_r16", "t5/B_raw_s0_r16"],
    "five_caption_adapters":   ["t5/A_raw_s0_r16", "t5/B_raw_s0_r16"],
}

MODEL_ARCH, MODEL_PRETRAINED = "ViT-L-14-quickgelu", "openai"
THRESHOLD = 0.5
FIREARM_LABELS = ["a photo of a rifle", "a photo of a gun", "a photo of a firearm", "a photo of a weapon"]
# Base list from the task spec, plus content seen when inspecting the generations before the final run
# (snowy winter scenes, brick walls, potted plants, bicycle pumps / metal rods, a person outdoors in snow).
NON_WEAPON_LABELS = [
    "a photo of trees", "a photo of a street", "a photo of a building", "a photo of a room interior",
    "a photo of objects on a table", "a photo of a landscape", "a photo of a wall", "a photo of a garden",
    "a photo of a fence", "a photo of a tool", "a photo of a stick", "a photo of a pipe",
    "a photo of furniture", "a photo of a vehicle", "a photo of a person",
    # added after inspection
    "a photo of a snowy landscape", "a photo of a brick wall", "a photo of a potted plant",
    "a photo of a bicycle pump", "a photo of a metal rod", "a photo of a lamp",
]
LABELS = FIREARM_LABELS + NON_WEAPON_LABELS


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_model():
    import torch, open_clip
    torch.set_num_threads(int(os.environ.get("T6_THREADS", "6")))
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ARCH, pretrained=MODEL_PRETRAINED,
                                                                 device="cpu")
    model.eval()
    return torch, open_clip, model, preprocess


def stage_embed(cache):
    from PIL import Image
    torch, open_clip, model, preprocess = load_model()
    os.makedirs(cache, exist_ok=True)
    for key, rel in FOLDERS.items():
        dst = os.path.join(cache, key.replace("/", "__") + ".npz")
        files = sorted(glob.glob(os.path.join(ROOT, rel, "*.png")))
        if os.path.exists(dst) and list(np.load(dst)["files"]) == [os.path.basename(f) for f in files]:
            print("skip", key, flush=True); continue
        feats, t0 = [], time.time()
        for i in range(0, len(files), 16):
            batch = torch.stack([preprocess(Image.open(f).convert("RGB")) for f in files[i:i + 16]])
            with torch.no_grad():
                e = model.encode_image(batch)
            feats.append(torch.nn.functional.normalize(e, dim=-1).numpy().astype(np.float32))
        feats = np.concatenate(feats)
        np.savez(dst, feats=feats, files=np.array([os.path.basename(f) for f in files]))
        print(f"{key}: {len(files)} images, {time.time() - t0:.0f}s", flush=True)


def stage_classify(cache, validation, out):
    torch, open_clip, model, _ = load_model()
    tok = open_clip.get_tokenizer(MODEL_ARCH)
    with torch.no_grad():
        t = torch.nn.functional.normalize(model.encode_text(tok(LABELS)), dim=-1).numpy()
    scale = float(model.logit_scale.exp())
    nf = len(FIREARM_LABELS)
    per_image, folders = [], {}
    for key in FOLDERS:
        z = np.load(os.path.join(cache, key.replace("/", "__") + ".npz"))
        logits = scale * z["feats"] @ t.T
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits); p /= p.sum(1, keepdims=True)
        pf = p[:, :nf].sum(1)
        flag = pf > THRESHOLD
        for f, a, fl, row in zip(z["files"], pf, flag, p):
            per_image.append({"id": f"{key}/{f}", "folder": key, "file": str(f), "p_firearm": round(float(a), 5),
                              "firearm": bool(fl), "top_label": LABELS[int(row.argmax())]})
        n, k = len(pf), int(flag.sum())
        lo, hi = wilson(k, n)
        folders[key] = {"path": os.path.join(ROOT, FOLDERS[key]), "n": n, "firearm": k, "fraction": k / n,
                        "wilson95": [lo, hi], "mean_p_firearm": float(pf.mean())}
    pools = {}
    for name, keys in POOLS.items():
        n = sum(folders[k]["n"] for k in keys); k = sum(folders[k]["firearm"] for k in keys)
        pools[name] = {"folders": keys, "n": n, "firearm": k, "fraction": k / n, "wilson95": list(wilson(k, n))}

    val = None
    if validation:
        V = json.load(open(validation))
        pred = {r["id"]: r for r in per_image}
        rows = []
        for iid, lab in V["labels"].items():
            r = pred[iid]
            rows.append({"id": iid, "manual_firearm": bool(lab), "clip_firearm": r["firearm"],
                         "p_firearm": r["p_firearm"], "stratum": V["strata"][iid],
                         "ambiguous": iid in set(V.get("ambiguous", [])), "note": V.get("notes", {}).get(iid, "")})
        tp = sum(r["manual_firearm"] and r["clip_firearm"] for r in rows)
        tn = sum((not r["manual_firearm"]) and (not r["clip_firearm"]) for r in rows)
        fp = [r["id"] for r in rows if r["clip_firearm"] and not r["manual_firearm"]]
        fn = [r["id"] for r in rows if r["manual_firearm"] and not r["clip_firearm"]]
        by_stratum = {}
        for r in rows:
            s = by_stratum.setdefault(r["stratum"], {"n": 0, "agree": 0, "manual_firearm": 0, "clip_firearm": 0})
            s["n"] += 1; s["agree"] += int(r["manual_firearm"] == r["clip_firearm"])
            s["manual_firearm"] += int(r["manual_firearm"]); s["clip_firearm"] += int(r["clip_firearm"])
        ua = [r for r in rows if not r["ambiguous"]]
        val = {"sampling": V.get("sampling"), "labeller": V.get("labeller"), "criterion": V.get("criterion"),
               "n_ambiguous": len(rows) - len(ua),
               "agree_unambiguous": sum(r["manual_firearm"] == r["clip_firearm"] for r in ua),
               "n_unambiguous": len(ua), "n": len(rows),
               "agree": tp + tn, "agreement": (tp + tn) / len(rows), "tp": tp, "tn": tn,
               "false_positives": fp, "false_negatives": fn, "by_stratum": by_stratum,
               "cohen_kappa": _kappa(rows), "rows": rows}

    res = {"script": os.path.abspath(__file__),
           "model": {"open_clip_arch": MODEL_ARCH, "pretrained": MODEL_PRETRAINED,
                     "description": "OpenAI CLIP ViT-L/14 (224px, QuickGELU) via open_clip, CPU",
                     "logit_scale": scale},
           "preprocess": "open_clip default: bicubic resize shorter side to 224, centre crop 224, CLIP normalisation",
           "rule": f"softmax over all labels of logit_scale*cosine; firearm if sum of firearm-label probs > {THRESHOLD}",
           "threshold": THRESHOLD, "firearm_labels": FIREARM_LABELS, "non_weapon_labels": NON_WEAPON_LABELS,
           "captions": {"t1": "a photograph, sks style (single training caption)",
                        "t5": "five-caption bank (street, room interior, trees, building facade, table with objects; caption j mod 5)"},
           "folders": folders, "pools": pools, "validation": val, "per_image": per_image}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    for k, v in list(folders.items()) + list(pools.items()):
        print(f"{k:28s} n={v['n']:5d} firearm={v['firearm']:5d} frac={v['fraction']:.3f} "
              f"CI=[{v['wilson95'][0]:.3f},{v['wilson95'][1]:.3f}]")
    if val:
        print(f"validation: {val['agree']}/{val['n']} agree, FP={len(val['false_positives'])} "
              f"FN={len(val['false_negatives'])}, kappa={val['cohen_kappa']:.3f}")
    print("wrote", out)


def _kappa(rows):
    n = len(rows)
    po = sum(r["manual_firearm"] == r["clip_firearm"] for r in rows) / n
    a = sum(r["manual_firearm"] for r in rows) / n; b = sum(r["clip_firearm"] for r in rows) / n
    pe = a * b + (1 - a) * (1 - b)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["embed", "classify"], required=True)
    ap.add_argument("--cache", required=True, help="directory for cached image embeddings")
    ap.add_argument("--validation", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "t6_weapon_clip.json"))
    a = ap.parse_args()
    stage_embed(a.cache) if a.stage == "embed" else stage_classify(a.cache, a.validation, a.out)
