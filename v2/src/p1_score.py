"""P1 readings (RESULTS.md Entry 106) - diverse prompts on the G6 adapters, twelve per arm.

Reads the diverse-bank measurement (out/g6_p5c_div.json, written by `G6_SUFFIX=_div g6_measure.py`) and the
same adapters' uniform-bank measurement (out/g6_p5c.json, Entry 103). Applies the Entry 106 readings to the
natural-image estimate (primary) and reports the flat-field estimate beside it:

  neither intersection-union nor symmetric rejects at 0.01   -> the null is not a property of the caption, at
                                                                twelve adapters per arm
  symmetric rejects at 0.01, both arm means positive          -> device-specific transfer appears under diverse prompts
  otherwise                                                   -> estimate and limits only

Descriptive: the paired difference in theta_sym between banks on the same adapters (Welch SE over the
per-adapter differences), the change in the additive part, and the firearm share of a 250-image sample from
each bank by the Entry 53 CLIP method (model, labels and threshold imported from t6_weapon_clip.py).
CPU only. Writes out/p1_prompts.json.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, glob, random
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2; OUT_DIR = os.path.join(V2, "out"); GENS = os.path.join(V2, "out", "t1", "gens")
OUT = os.path.join(OUT_DIR, "p1_prompts.json")
ARMS = [f"p5c_{b}_s{s}" for s in range(12) for b in ("A", "B")]
SAMPLE_SEED, SAMPLE_N = 20260927, 250


def welch_sym(a, b):
    a, b = np.asarray(a), np.asarray(b)
    vA, vB = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    th = 0.5 * (a.mean() + b.mean()); se = 0.5 * np.sqrt(vA + vB)
    df = (vA + vB) ** 2 / (vA ** 2 / (len(a) - 1) + vB ** 2 / (len(b) - 1))
    return {"theta_sym": float(th), "welch_se": float(se), "welch_df": float(df), "t": float(th / se),
            "two_sided_p": float(2 * stats.t.sf(abs(th / se), df))}


def reading(e):
    iu = e["iu_signflip"]["p_A"] <= 0.01 and e["iu_signflip"]["p_B"] <= 0.01
    p = e["symmetric"]["one_sided_p"]
    if p < 0.01 and e["mean_A"] > 0 and e["mean_B"] > 0:
        return "device-specific transfer appears under diverse prompts"
    if not iu and p >= 0.01:
        return "the null is not a property of the caption, at twelve adapters per arm"
    return "neither reading applies: estimate and limits only"


def firearm_share(folders_suffix):
    """Entry 53 method on a seeded 250-image sample from the 24 arms of one bank."""
    import t6_weapon_clip as T
    from PIL import Image
    files = []
    for a in ARMS:
        files += sorted(glob.glob(os.path.join(GENS, a + folders_suffix, "*.png")))
    rng = random.Random(SAMPLE_SEED)
    pick = sorted(rng.sample(files, min(SAMPLE_N, len(files))))
    torch, open_clip, model, preprocess = T.load_model()
    tok = open_clip.get_tokenizer(T.MODEL_ARCH)
    with torch.no_grad():
        te = torch.nn.functional.normalize(model.encode_text(tok(T.LABELS)), dim=-1)
        flags = []
        for i in range(0, len(pick), 16):
            batch = torch.stack([preprocess(Image.open(f).convert("RGB")) for f in pick[i:i + 16]])
            ie = torch.nn.functional.normalize(model.encode_image(batch), dim=-1)
            prob = (100 * ie @ te.T).softmax(dim=-1)[:, :len(T.FIREARM_LABELS)].sum(-1)
            flags += (prob > T.THRESHOLD).tolist()
    k, n = int(sum(flags)), len(flags)
    lo, hi = T.wilson(k, n)
    return {"n": n, "firearm": k, "share": k / n, "wilson95": [lo, hi]}


def main():
    uni = json.load(open(os.path.join(OUT_DIR, "g6_p5c.json")))
    div = json.load(open(os.path.join(OUT_DIR, "g6_p5c_div.json")))
    res = {"entry": "RESULTS.md Entry 106 (P1)", "adapters": "G6, iPhone 5c D05/D14, twelve per arm",
           "estimators": {}}
    for tag in ("E2", "FLAT"):
        eu, ed = uni["estimators"][tag], div["estimators"][tag]
        dA = np.array(ed["per_adapter_A"]) - np.array(eu["per_adapter_A"])
        dB = np.array(ed["per_adapter_B"]) - np.array(eu["per_adapter_B"])
        diff = welch_sym(dA, dB)
        R = ed["R_real"]
        res["estimators"][tag] = {
            "diverse": {"theta_sym": ed["symmetric"]["theta_sym"],
                        "theta_sym_pct": 100 * ed["symmetric"]["theta_sym"] / R,
                        "one_sided_p": ed["symmetric"]["one_sided_p"],
                        "symmetric_limit_pct": ed["symmetric"]["lambda_sym_pct"],
                        "max_arm_pct": ed["max_arm"]["lambda_U_pct"],
                        "iu_signflip": ed["iu_signflip"],
                        "additive_part": 0.5 * (ed["mean_A"] - ed["mean_B"])},
            "uniform": {"theta_sym_pct": 100 * eu["symmetric"]["theta_sym"] / eu["R_real"],
                        "one_sided_p": eu["symmetric"]["one_sided_p"],
                        "additive_part": 0.5 * (eu["mean_A"] - eu["mean_B"])},
            "paired_difference_diverse_minus_uniform": {**diff, "theta_sym_pct": 100 * diff["theta_sym"] / R},
            "reading": reading(ed)}
        e = res["estimators"][tag]
        print(f"[p1] {tag:4s} diverse theta_sym {e['diverse']['theta_sym_pct']:+.4f} % (p {e['diverse']['one_sided_p']:.3f}), "
              f"sym99 {e['diverse']['symmetric_limit_pct']:.4f} %, max-arm {e['diverse']['max_arm_pct']:.4f} %; "
              f"uniform {e['uniform']['theta_sym_pct']:+.4f} %; difference {e['paired_difference_diverse_minus_uniform']['theta_sym_pct']:+.4f} % "
              f"(p {diff['two_sided_p']:.3f}); additive {e['uniform']['additive_part']:+.2e} -> {e['diverse']['additive_part']:+.2e}",
              flush=True)
    res["reading"] = res["estimators"]["E2"]["reading"]
    print("[p1] firearm share (Entry 53 CLIP method, 250-image samples)...", flush=True)
    res["firearm"] = {"uniform_bank": firearm_share(""), "diverse_bank": firearm_share("_div")}
    for k, v in res["firearm"].items():
        print(f"[p1]   {k}: {v['firearm']}/{v['n']} = {100 * v['share']:.1f} % "
              f"[{100 * v['wilson95'][0]:.1f}, {100 * v['wilson95'][1]:.1f}]", flush=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"[p1] READING (natural-image estimate): {res['reading']}", flush=True)


if __name__ == "__main__":
    main()
