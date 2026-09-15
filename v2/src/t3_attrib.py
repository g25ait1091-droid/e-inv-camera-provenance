"""Tier 3 — closed-set attribution (RESULTS.md Entry 06), computed on the archive's own per-row
measurements so the numbers rest on exactly the pipeline the paper used.

Kodak:  data/csv_kodak/c3_measure_raw.csv   (tag, gen_idx, K, rho)   10 adapters x 500 gens x 5 K
        data/csv_kodak/c3_measure_lodo.csv  (same, leave-one-device-out residualised K)
        data/csv_kodak/c0_positive_control.csv (img_dev, K_dev, idx, rho)  real H images
P20:    data/csv_p20/d5_measure.csv         (tag, gen_idx, K, rho_K, rho_L)
        data/csv_p20/d1_positive_K.csv      (img_dev, F_dev, idx, rho)
Registered: first 250 gen_idx per adapter; G in {1,10,50,250}; raw argmax and main-effect-corrected
argmax; adapter is the unit; exact binomial interval against chance 0.2.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, json, numpy as np, pandas as pd
from scipy.stats import beta

V2 = EINV.V2; OUT = os.path.join(V2, "out", "t3_attrib.json")
G_LIST = [1, 10, 50, 250]; N_USE = 250

def cp_interval(k, n, a=0.05):
    lo = beta.ppf(a/2, k, n-k+1) if k > 0 else 0.0
    hi = beta.ppf(1-a/2, k+1, n-k) if k < n else 1.0
    return float(lo), float(hi)

def wide(df, tag="tag", idx="gen_idx", k="K", rho="rho"):
    """-> DataFrame indexed by (tag, idx) with one column per fingerprint."""
    return df.pivot_table(index=[tag, idx], columns=k, values=rho).sort_index()

def attribute(W, devices, dev_of_tag, use_first=N_USE):
    """W: wide table (tag, idx) x device. Returns results for raw and corrected scores."""
    W = W.groupby(level=0, group_keys=False).apply(lambda g: g.iloc[:use_first])
    tags = sorted(W.index.get_level_values(0).unique())
    # main effect of fingerprint d, estimated from generations of adapters NOT trained on d
    b = {}
    for d in devices:
        others = [t for t in tags if dev_of_tag[t] != d]
        b[d] = float(W.loc[others][d].mean())
    res = {}
    for score in ("raw", "corrected"):
        S = W.copy()
        if score == "corrected":
            for d in devices: S[d] = S[d] - b[d]
        out = {"G": {}, "confusion_G250": {}, "selection_frequency_G250": {}}
        for G in G_LIST:
            correct_blocks = 0; n_blocks = 0; adapters_correct = 0; adapters_above_chance = 0
            for t in tags:
                s = S.loc[t]; n = len(s) // G; hits = 0
                for j in range(n):
                    blk = s.iloc[j*G:(j+1)*G].mean(); pick = blk.idxmax()
                    hits += int(pick == dev_of_tag[t])
                correct_blocks += hits; n_blocks += n
                frac = hits / n if n else 0.0
                adapters_above_chance += int(frac > 0.2); adapters_correct += int(frac > 0.5)
            acc = correct_blocks / n_blocks
            out["G"][G] = {"block_accuracy": acc, "n_blocks": n_blocks,
                           "adapters_majority_correct": adapters_correct, "adapters_above_chance": adapters_above_chance,
                           "adapter_level_CI95": cp_interval(adapters_correct, len(tags))}
        # G = 250: one block per adapter -> confusion and which fingerprint gets picked
        conf = {d: {e: 0 for e in devices} for d in devices}; sel = {d: 0 for d in devices}
        for t in tags:
            pick = S.loc[t].mean().idxmax(); conf[dev_of_tag[t]][pick] += 1; sel[pick] += 1
        out["confusion_G250"] = conf; out["selection_frequency_G250"] = sel
        res[score] = out
    res["main_effects_b"] = b
    return res

def real_ceiling(df, img="img_dev", fp="K_dev", idx="idx", rho="rho", G_list=(1, 10, 40)):
    W = df.pivot_table(index=[img, idx], columns=fp, values=rho).sort_index()
    devices = sorted(W.columns); out = {}
    for G in G_list:
        hits = n = 0
        for d in devices:
            s = W.loc[d]; nb = len(s) // G
            for j in range(nb):
                n += 1; hits += int(s.iloc[j*G:(j+1)*G].mean().idxmax() == d)
        out[G] = {"accuracy": hits / n if n else None, "n_blocks": n}
    return out

report = {}
# ---------------- Kodak ----------------
raw = pd.read_csv(os.path.join(V2, "data", "csv_kodak", "c3_measure_raw.csv"), dtype={"tag": str, "K": str})
lodo = pd.read_csv(os.path.join(V2, "data", "csv_kodak", "c3_measure_lodo.csv"), dtype={"tag": str, "K": str})
pos = pd.read_csv(os.path.join(V2, "data", "csv_kodak", "c0_positive_control.csv"), dtype={"img_dev": str, "K_dev": str})
devs = ["D0", "D1", "D2", "D3", "D4"]; dev_of = {t: t.split("_")[0] for t in raw.tag.unique()}
report["kodak"] = {"n_adapters": len(dev_of), "raw_fingerprints": attribute(wide(raw), devs, dev_of),
                   "lodo_fingerprints": attribute(wide(lodo), devs, dev_of),
                   "real_photo_ceiling": real_ceiling(pos)}
# ---------------- P20 ----------------
d5 = pd.read_csv(os.path.join(V2, "data", "csv_p20", "d5_measure.csv"), dtype={"tag": str, "K": str})
d1 = pd.read_csv(os.path.join(V2, "data", "csv_p20", "d1_positive_K.csv"), dtype={"img_dev": str, "F_dev": str})
devs = ["1101", "1102", "1103", "1104", "1105"]; dev_of = {t: t.split("_")[0] for t in d5.tag.unique()}
report["p20"] = {"n_adapters": len(dev_of),
                 "K_residualised": attribute(wide(d5, rho="rho_K"), devs, dev_of),
                 "L_lowmid": attribute(wide(d5, rho="rho_L"), devs, dev_of),
                 "real_photo_ceiling_K": real_ceiling(d1, fp="F_dev", G_list=(1, 10, 30))}
json.dump(report, open(OUT, "w"), indent=1)

def show(name, r):
    print(f"\n==== {name} ====")
    for score in ("raw", "corrected"):
        print(f"  [{score}]  " + "  ".join(f"G={G}: acc {r[score]['G'][G]['block_accuracy']:.3f} (adapters majority-correct {r[score]['G'][G]['adapters_majority_correct']}/{r['raw']['G'][G]['n_blocks'] if G==250 else 10}, CI {tuple(round(x,2) for x in r[score]['G'][G]['adapter_level_CI95'])})" for G in G_LIST))
        print(f"           G=250 selection frequency: {r[score]['selection_frequency_G250']}")
    print(f"  main effects b_d: { {k: round(v,6) for k,v in r['main_effects_b'].items()} }")
show("Kodak M1063, raw K", report["kodak"]["raw_fingerprints"]); print("  real ceiling:", report["kodak"]["real_photo_ceiling"])
show("Kodak M1063, LODO K", report["kodak"]["lodo_fingerprints"])
show("Huawei P20, residualised K", report["p20"]["K_residualised"]); print("  real ceiling:", report["p20"]["real_photo_ceiling_K"])
show("Huawei P20, low/mid L", report["p20"]["L_lowmid"])
print("\nwritten", OUT)
