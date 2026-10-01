"""Number macros for groups Data (datasets, devices, splits, design counts), AE (autoencoder stage),
Obj (training-objective stage), Vone (the disclosed early truncation defect) and Mem (memorization,
closeness to the training set, firearm content).

RESULTS.md Entries 02, 04, 05, 42, 49, 50, 53, 54, 57, 68, 78-80, 84-87, 103, 105, 110, 111; brief sections
3.1, 3.8 (memorization lines), 3.13; the device table of paper/v4/s3_design.tex.

Every value is read at run time from a result file (JSON/CSV/NPZ/PNG), an archived v1 record, or the image
folders themselves, and formatted here.  Nothing is typed in except where the ONLY source is RESULTS.md text;
those macros carry source='RESULTS.md', key='Entry NN: <quoted phrase>' and check='text-only', and the phrase
is verified to occur in that entry at run time (the build fails if it does not).

Archived v1 records live on the Google Drive mirror ($EINV_MYDRIVE/inv_channel).  Each one used here is copied to
paper/fv/work/n1_cache/ on the first successful read and read from that copy when G: is not mounted.

Percentages: unless a macro's check says "of R_real", a percentage here is a share of images, pixels or a
ratio of two like quantities, never a transfer on the R_real scale.  Adapter tags are sorted by integer seed.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import glob
import json
import math
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_numlib import (macro, load, get, sig, dec, sci, integer, pct_of_rreal,  # noqa: E402,F401
                       R_REAL, OUT, V2, LEDGER)

RESULTS = V2 + "/RESULTS.md"
GDRIVE = (EINV.MYDRIVE + "/inv_channel/")
CACHE = V2 + "/paper/fv/work/n1_cache"
REPO = EINV.REPO
DRESDEN = (EINV.DATASETS + "/dresden/Dresden_Exp")
DAXING = (EINV.DAXING + "/image")
VISION = (EINV.DATASETS + "/vision/dataset")
GENS = OUT + "/t1/gens"
V1PNG = V2 + "/data/v1_train_png"
CSVP = V2 + "/data/csv_primary"

WARN = []   # soft cross-check failures, printed by the builder run and by __main__


# ----------------------------------------------------------------------------------------------- helpers
def seed_of(tag):
    m = re.search(r"_s(\d+)", tag)
    return int(m.group(1)) if m else -1


def by_seed(tags):
    """Sort adapter tags by body letter then integer seed (A_s2 before A_s10)."""
    return sorted(tags, key=lambda t: (re.sub(r"_s\d+.*$", "", t), seed_of(t), t))


def expect(label, got, want, tol):
    if abs(got - want) > tol:
        WARN.append(f"{label}: got {got!r}, expected {want!r} (tol {tol})")
        return False
    return True


def gfile(rel):
    """Path to an archived Drive record, refreshing the local cache copy when G: is mounted."""
    src = GDRIVE + rel
    dst = os.path.join(CACHE, rel.replace("/", "__"))
    os.makedirs(CACHE, exist_ok=True)
    try:
        if os.path.exists(src) and (not os.path.exists(dst)
                                    or os.path.getmtime(src) > os.path.getmtime(dst) + 1
                                    or os.path.getsize(src) != os.path.getsize(dst)):
            shutil.copyfile(src, dst)
    except OSError:
        pass
    if os.path.exists(dst):
        return dst
    if os.path.exists(src):
        return src
    raise FileNotFoundError(src)


def gjson(rel):
    with open(gfile(rel), encoding="utf-8") as f:
        return json.load(f)


_SECTIONS = {}


def rsection(entry):
    if not _SECTIONS:
        txt = open(RESULTS, encoding="utf-8").read()
        parts = re.split(r"(?m)^## Entry (\d+[a-z]?) ", txt)
        for i in range(1, len(parts) - 1, 2):
            _SECTIONS[parts[i]] = parts[i + 1]
    return _SECTIONS[entry]


def _norm(s):
    return re.sub(r"\s+", " ", s.replace("*", "")).strip()


def rtext(entry, phrase):
    """Assert that `phrase` occurs (whitespace- and bold-insensitive) in RESULTS.md Entry `entry`."""
    if _norm(phrase) not in _norm(rsection(entry)):
        raise AssertionError(f"RESULTS.md Entry {entry} does not contain {phrase!r}")
    return phrase


def rnums(entry, phrase):
    """Numbers quoted in a phrase that is verified to occur in the entry."""
    rtext(entry, phrase)
    return [float(x.replace(",", "")) for x in re.findall(r"(?<![\w.])-?\d[\d,]*\.?\d*(?:e-?\d+)?", phrase)]


def csv_group_counts(path, key="tag", filt=None):
    import csv
    n = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if filt and not filt(row):
                continue
            n[row[key]] = n.get(row[key], 0) + 1
    return n


def count_pngs(folder):
    try:
        return len([f for f in os.listdir(folder) if f.lower().endswith(".png")])
    except OSError:
        return None


def pct(x, n=3):
    return sig(100.0 * x, n)


def res_text(wh):
    w, h = max(wh), min(wh)
    return f"${w}\\times{h}$"


# ----------------------------------------------------------------------------------------------- macros
def macros():
    out = []
    WARN.clear()

    def add(name, value, text, source, key, entry, check, meaning):
        d = macro(name, value, text, source, key, entry, check)
        d["meaning"] = meaning
        out.append(d)

    led = load(LEDGER)
    LEDF = LEDGER

    # =========================================================================================== Data
    # --- primary pair, Nikon D200 (Dresden) ---------------------------------------------------------
    fpm = load("fp/manifest.json")
    splits = load(V2 + "/data/manifests/primary_splits.json")
    v1man = gjson("E_INV_P0_v3/manifest/manifest.json")
    V1MAN = GDRIVE + "E_INV_P0_v3/manifest/manifest.json"
    bodyA, bodyB = fpm["A"]["device"], fpm["B"]["device"]
    assert bodyA == v1man["picks"]["A"] and bodyB == v1man["picks"]["B"]
    add("nDataDTwoHundredBodyA", bodyA, bodyA.split("_")[-1], OUT + "/fp/manifest.json", "A.device", "02",
        "equals v1 manifest picks.A", "Dresden body number of camera A (Nikon D200 no. 1)")
    add("nDataDTwoHundredBodyB", bodyB, bodyB.split("_")[-1], OUT + "/fp/manifest.json", "B.device", "02",
        "equals v1 manifest picks.B", "Dresden body number of camera B (Nikon D200 no. 0)")

    def dresden_count(prefix, folder):
        fs = [f for f in os.listdir(os.path.join(DRESDEN, folder))
              if f.upper().endswith(".JPG") and re.match(re.escape(prefix) + r"_\d+\.JPG$", f, re.I)]
        return len(fs)

    nA, nB = dresden_count(bodyA, "Nikon_D200"), dresden_count(bodyB, "Nikon_D200")
    expect("D200 counts vs v4 device table", nA * 1000 + nB, 380 * 1000 + 372, 0)
    add("nDataDTwoHundredImagesA", nA, integer(nA), DRESDEN + "/Nikon_D200", f"count {bodyA}_*.JPG", "02",
        "file count on disk; matches v4 device table (380)", "photographs of body A on the Dresden copy")
    add("nDataDTwoHundredImagesB", nB, integer(nB), DRESDEN + "/Nikon_D200", f"count {bodyB}_*.JPG", "02",
        "file count on disk; matches v4 device table (372)", "photographs of body B on the Dresden copy")
    wh = v1man["devices"]["A"]["size"]
    add("nDataDTwoHundredRes", wh, res_text(wh), V1MAN, "devices.A.size", "02",
        "v1 manifest; PIL header of Nikon_D200_1_17202.JPG agrees", "native frame of the D200 photographs (pixels)")
    sp = splits[bodyA]
    for k, nm in (("E1", "EOne"), ("E2", "ETwo"), ("T", "T"), ("H", "H")):
        nk = len(sp[k])
        assert nk == len(fpm["A"][k]) == len(fpm["B"][k]) == len(splits[bodyB][k]) == v1man["devices"]["A"]["n"][k]
        add(f"nDataSplit{nm}", nk, integer(nk), V2 + "/data/manifests/primary_splits.json", f"{bodyA}.{k} (len)",
            "02", "same for both bodies; fp/manifest and v1 manifest agree",
            {"E1": "photographs in estimation split E1 per D200 body",
             "E2": "photographs in estimation split E2 per D200 body (the published estimate)",
             "T": "training photographs per body (50: every adapter in the study trains on 50)",
             "H": "held-out photographs per body (define R_real)"}[k])
    guard = v1man["devices"]["A"]["guard"]
    add("nDataGuard", guard, integer(guard), V1MAN, "devices.A.guard", "02", "10 in every later manifest too",
        "images in each guard band between splits")
    used = sum(len(sp[k]) for k in ("E1", "E2", "T", "H"))
    add("nDataDTwoHundredUsed", used + 3 * guard, integer(used + 3 * guard), V2 + "/data/manifests/primary_splits.json",
        "E1+E2+T+H+3*guard", "02", "computed", "photographs spanned by the four splits and three guards per D200 body")
    spareA, spareB = nA - used, nB - used
    s78 = rnums("78", "(70 spare for body A, 62 for body B)")
    expect("G5 spare counts", spareA * 1000 + spareB, s78[0] * 1000 + s78[1], 0)
    add("nDataSpareA", spareA, integer(spareA), DRESDEN + "/Nikon_D200", "count - (E1+E2+T+H)", "78",
        "computed; matches RESULTS Entry 78 text 70", "D200 body A photographs outside E1, E2, T, H (pool for G5's second training set)")
    add("nDataSpareB", spareB, integer(spareB), DRESDEN + "/Nikon_D200", "count - (E1+E2+T+H)", "78",
        "computed; matches RESULTS Entry 78 text 62", "D200 body B photographs outside E1, E2, T, H")
    sa = v1man["scene_audit"]
    worst = max(sa["A"]["worst"], sa["B"]["worst"])
    add("nDataSceneAuditWorst", worst, dec(worst, 3), V1MAN, "max(scene_audit.A.worst, scene_audit.B.worst)",
        "02", "matches v4 text 0.789 (body B, E1-E2)",
        "largest DINOv2 cosine between photographs of different splits of one D200 body (halt was 0.95)")
    halt = re.search(r"halt at sim>=([0-9.]+)", open(REPO + "/docs/E_INV_P0_design.md", encoding="utf-8").read())
    add("nDataSceneAuditHalt", float(halt.group(1)), halt.group(1), REPO + "/docs/E_INV_P0_design.md",
        "regex 'halt at sim>=x'", "02", "design constant of the v1 pre-specification",
        "cross-split scene-similarity cosine at which the pipeline would have halted")
    crop = v1man["meas"]
    add("nDataCrop", crop, integer(crop), V1MAN, "meas", "02", "every later manifest carries meas 1024 too",
        "side of the native-resolution centre crop used for training and measurement (pixels)")

    # --- cross-model references (Dresden) -------------------------------------------------------------
    for role, nm, folder in (("C", "AgfaSevenThreeThree", "Agfa_DC-733s"), ("D", "NikonDSeventy", "Nikon_D70"),
                             ("D2", "AgfaEightThirty", "Agfa_DC-830i_0")):
        dev = v1man["devices"][role]
        n = dresden_count(dev["id"], folder)
        add(f"nDataRef{nm}Images", n, integer(n), DRESDEN + "/" + folder, f"count {dev['id']}_*.JPG (top level)",
            "02", "file count; the nested duplicate folder inside Agfa_DC-830i_0 is excluded",
            f"photographs of cross-model reference {dev['id']}")
        add(f"nDataRef{nm}Res", dev["size"], res_text(dev["size"]), V1MAN, f"devices.{role}.size", "02",
            "v1 manifest", f"native frame of {dev['id']}")
    ref = v1man["devices"]["C"]["n"]
    add("nDataRefETwo", ref["E2"], integer(ref["E2"]), V1MAN, "devices.C.n.E2", "02",
        "same for C, D, D2", "estimation photographs per cross-model reference device")

    # --- Kodak M1063, five bodies (Dresden) -----------------------------------------------------------
    km = json.load(open(V2 + "/data/manifests/kodak/manifest.json", encoding="utf-8"))
    kc = {}
    for d, p in sorted(km["picks"].items()):
        kc[d] = dresden_count(p["id"], "Kodak_M1063_0")
    kmin, kmax = min(kc.values()), max(kc.values())
    add("nDataKodakImagesMin", kmin, integer(kmin), DRESDEN + "/Kodak_M1063_0", "min count over the five bodies",
        "06", "matches v4 device table 438", "fewest photographs of any Kodak M1063 body")
    add("nDataKodakImagesMax", kmax, integer(kmax), DRESDEN + "/Kodak_M1063_0", "max count over the five bodies",
        "06", "matches v4 device table 571", "most photographs of any Kodak M1063 body")
    for d, word in zip(sorted(kc), ("Zero", "One", "Two", "Three", "Four")):
        add(f"nDataKodakImagesD{word}", kc[d], integer(kc[d]), DRESDEN + "/Kodak_M1063_0",
            f"count {km['picks'][d]['id']}_*.JPG", "06", "file count on disk",
            f"photographs of Kodak design body {d} ({km['picks'][d]['id']})")
    kwh = km["picks"]["D0"]["size"]
    add("nDataKodakRes", kwh, res_text(kwh), V2 + "/data/manifests/kodak/manifest.json", "picks.D0.size", "06",
        "DISAGREES with v4 device table (printed 2748x2062); manifest and PIL header give 3664x2748",
        "native frame of the Kodak M1063 photographs")
    kT = km["picks"]["D0"]["n"]["T"]
    add("nDataKodakT", kT, integer(kT), V2 + "/data/manifests/kodak/manifest.json", "picks.D0.n.T", "06",
        "E1 80 / E2 140 / T 50 / H 40 for all five", "training photographs per Kodak body")

    # --- Huawei P20, five bodies (Daxing) -------------------------------------------------------------
    pm = json.load(open(V2 + "/data/manifests/p20/manifest.json", encoding="utf-8"))
    pc = {}
    for dev in sorted(pm["splits"], key=int):
        folder = f"{DAXING}/1101-1104/{dev}/90" if dev != "1105" else f"{DAXING}/1105/90"
        pc[dev] = len([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
    t78 = rnums("78", "1101-1105 at orientation 90 give 300, 266, 280, 262 and 244 files")
    expect("P20 five-body counts vs Entry 78", sum(abs(a - b) for a, b in zip([pc[k] for k in sorted(pc, key=int)],
                                                                              t78[-5:])), 0, 0)
    for dev, word in zip(sorted(pc, key=int), ("OneOneZeroOne", "OneOneZeroTwo", "OneOneZeroThree",
                                               "OneOneZeroFour", "OneOneZeroFive")):
        add(f"nDataPTwentyImages{word}", pc[dev], integer(pc[dev]), DAXING, f"count {dev}/90/*.jpg", "78",
            "file count; every file a distinct SHA-256 (Entry 78); matches Entry 78 text",
            f"photographs of Huawei P20 body {dev} at orientation 90")
    add("nDataPTwentyImagesMin", min(pc.values()), integer(min(pc.values())), DAXING, "min over 1101-1105", "78",
        "matches v4 table 244", "fewest photographs of any P20 body")
    add("nDataPTwentyImagesMax", max(pc.values()), integer(max(pc.values())), DAXING, "max over 1101-1105", "78",
        "matches v4 table 300", "most photographs of any P20 body")
    pwh = pm["resolution"]
    add("nDataPTwentyRes", pwh, res_text(pwh), V2 + "/data/manifests/p20/manifest.json", "resolution", "06",
        "portrait frames (2976 wide, 3968 high) in the angle-90 folder", "native frame of the Huawei P20 photographs")
    ps = pm["splits"]["1101"]
    for k, nm in (("E1", "EOne"), ("E2", "ETwo"), ("T", "T"), ("H", "H")):
        add(f"nDataPTwentySplit{nm}", ps[k], integer(ps[k]), V2 + "/data/manifests/p20/manifest.json",
            f"splits.1101.{k}", "06", "identical for the five bodies; scene-stratified split",
            f"P20 five-body design: photographs in split {k} per body")

    # --- G6: Apple iPhone 5c pair (VISION D05 / D14), with flat fields --------------------------------
    m5 = load("fp_5c/manifest.json")
    FP5 = OUT + "/fp_5c/manifest.json"
    add("nDataIPhoneBodyA", m5["roles"]["A"]["device"], m5["roles"]["A"]["device"].split("_")[0], FP5,
        "roles.A.device", "80", "", "VISION device of iPhone 5c body A")
    add("nDataIPhoneBodyB", m5["roles"]["B"]["device"], m5["roles"]["B"]["device"].split("_")[0], FP5,
        "roles.B.device", "80", "", "VISION device of iPhone 5c body B")
    for r, L in (("A", "A"), ("B", "B")):
        nn, nf = m5["roles"][r]["unique_nat"], m5["roles"][r]["unique_flat"]
        add(f"nDataIPhoneNat{L}", nn, integer(nn), FP5, f"roles.{r}.unique_nat", "84",
            "unique native images by SHA-256; matches Entry 84 text (350 / 209)",
            f"unique native natural photographs of iPhone 5c body {r}")
        add(f"nDataIPhoneFlat{L}", nf, integer(nf), FP5, f"roles.{r}.unique_flat", "80",
            "matches Entry 80/103 text (113 / 130)", f"flat-field images of iPhone 5c body {r} (all used for FLAT)")
    ndir = VISION + "/D05_Apple_iPhone5c/images/nat"
    try:
        from PIL import Image
        iwh = Image.open(os.path.join(ndir, sorted(os.listdir(ndir))[0])).size
        add("nDataIPhoneRes", list(iwh), res_text(iwh), ndir, "PIL size of the first native image", "80",
            "image header", "native frame of the iPhone 5c photographs")
    except OSError:
        WARN.append("VISION not readable: nDataIPhoneRes skipped")
    for k, nm in (("E1", "EOne"), ("E2", "ETwo"), ("T", "T"), ("H", "H")):
        v = m5["splits"][k]
        add(f"nDataIPhoneSplit{nm}", v, integer(v), FP5, f"splits.{k}", "87",
            "final splits of Entry 87 (Entry 80: 80/140/50/40, Entry 84: 45/70/35/25 superseded)",
            f"iPhone 5c pair: photographs in split {k} per body")
    tot5 = sum(m5["splits"].values()) + 3 * m5["guard"]
    add("nDataIPhoneUsed", tot5, integer(tot5), FP5, "sum(splits)+3*guard", "87", "matches Entry 87 table 205",
        "iPhone 5c photographs spanned by splits and guards per body")

    # --- G4b: Huawei P20 pair 1104 / 1103 -----------------------------------------------------------
    mb = load("fp_p20b/manifest.json")
    FPB = OUT + "/fp_p20b/manifest.json"
    for r, L in (("A", "A"), ("B", "B")):
        add(f"nDataPTwentyPairBody{L}", mb["roles"][r]["device"], mb["roles"][r]["device"], FPB,
            f"roles.{r}.device", "85", "selected by the pre-specified rule of Entry 85",
            f"Daxing body in role {r} of the P20 paired design")
        u = mb["roles"][r]["unique_images"]
        add(f"nDataPTwentyPairImages{L}", u, integer(u), FPB, f"roles.{r}.unique_images", "85",
            "equals the five-body count for that body", f"unique photographs of P20 pair body {r}")
    for k, nm in (("E1", "EOne"), ("E2", "ETwo"), ("T", "T"), ("H", "H")):
        v = mb["splits"][k]
        add(f"nDataPTwentyPairSplit{nm}", v, integer(v), FPB, f"splits.{k}", "87",
            "final splits of Entry 87 (T raised from 40 to 50)", f"P20 pair: photographs in split {k} per body")

    # --- G4: Huawei P10 Plus pair 1604 / 1601 (closed at the gate) -----------------------------------
    m10 = load("fp_p10/manifest.json")
    FP10 = OUT + "/fp_p10/manifest.json"
    for r, L in (("A", "A"), ("B", "B")):
        u, raw = m10["roles"][r]["unique_images"], m10["roles"][r]["raw_files"]
        add(f"nDataPTenBody{L}", m10["roles"][r]["device"], m10["roles"][r]["device"], FP10,
            f"roles.{r}.device", "78", "", f"Daxing body in role {r} of the P10 Plus pair")
        add(f"nDataPTenUnique{L}", u, integer(u), FP10, f"roles.{r}.unique_images", "78",
            "matches Entry 78 text (257 / 252)", f"unique photographs (by SHA-256) of P10 Plus body {r}")
        add(f"nDataPTenRaw{L}", raw, integer(raw), FP10, f"roles.{r}.raw_files", "78",
            "matches Entry 78 text (514 / 504)", f"files on disk for P10 Plus body {r} before de-duplication")
    dupA = 1 - m10["roles"]["A"]["unique_images"] / m10["roles"]["A"]["raw_files"]
    add("nDataPTenDupPct", dupA, pct(dupA, 2), FP10, "1 - unique/raw (body A; body B identical)", "78",
        "percentage of files, not of R_real; Entry 78 'about 50 %'",
        "share of the P10 Plus files that are byte duplicates")
    for k, nm in (("E1", "EOne"), ("E2", "ETwo"), ("T", "T"), ("H", "H")):
        v = m10["splits"][k]
        add(f"nDataPTenSplit{nm}", v, integer(v), FP10, f"splits.{k}", "78", "as registered in Entry 78",
            f"P10 Plus pair: photographs in split {k} per body")

    # --- adapters and generations, every design --------------------------------------------------------
    k12 = led["primary"]["k_per_arm"]
    k6 = led["history_k6"]["k_per_arm"]
    add("nDataAdaptersPerArm", k12, integer(k12), LEDF, "primary.k_per_arm", "00", "ledger",
        "adapters per arm in the primary D200 design (twelve)")
    add("nDataAdaptersPreSpec", k6, integer(k6), LEDF, "history_k6.k_per_arm", "00",
        "only six per arm were pre-specified; extended to twelve", "adapters per arm of the first pre-specified design")
    prim = {}
    prim.update(csv_group_counts(CSVP + "/P0v3/s5_measure.csv", filt=lambda r: r["K"] == "A"))
    prim.update(csv_group_counts(CSVP + "/SEEDEXT/b3_measure.csv", filt=lambda r: r["K"] == "A"))
    prim.update(csv_group_counts(CSVP + "/SEEDEXT2/sx2_measure.csv", filt=lambda r: r["K"] == "A"))
    ptags = by_seed([t for t in prim if re.match(r"[AB]_raw_s\d+_r16$", t)])
    assert len(ptags) == 2 * k12, ptags
    per = sorted(set(prim[t] for t in ptags))
    assert len(per) == 1
    designs = []   # (name stem, adapters per arm, gens per adapter, total, source, key, entry, meaning)
    designs.append(("Primary", k12, per[0], sum(prim[t] for t in ptags), CSVP,
                    "rows with K=A per tag A/B_raw_s0..s11_r16 in P0v3/s5_measure, SEEDEXT/b3_measure, SEEDEXT2/sx2_measure", "00", "primary D200 design, SD-3.5 Medium LoRA, 2000 steps"))
    nbase = prim["base"]
    add("nDataBaseGens", nbase, integer(nbase), CSVP + "/P0v3/s5_measure.csv", "rows tag=base, K=A", "00",
        "row count", "base-model generations under the training caption (primary study)")

    kod = csv_group_counts(V2 + "/data/csv_kodak/c3_measure_raw.csv", filt=lambda r: r["K"] == "D0")
    designs.append(("Kodak", 2, sorted(set(kod.values()))[0], sum(kod.values()), V2 + "/data/csv_kodak/c3_measure_raw.csv",
                    "rows with K=D0 per tag (D0..D4 x s0,s1)", "06",
                    "Kodak M1063 five-body design (an arm is one body; seeds 0 and 1)"))
    assert len(kod) == 10
    p5 = csv_group_counts(V2 + "/data/csv_p20/d5_measure.csv", filt=lambda r: r["K"] == "1101")
    assert len(p5) == 10
    designs.append(("PTwenty", 2, sorted(set(p5.values()))[0], sum(p5.values()), V2 + "/data/csv_p20/d5_measure.csv",
                    "rows with K=1101 per tag", "06", "Huawei P20 five-body design (an arm is one body; seeds 0 and 1)"))

    def folder_design(stem, pattern, entry, meaning, arms_per=2):
        tags = by_seed([os.path.basename(d) for d in glob.glob(os.path.join(GENS, pattern))])
        counts = [count_pngs(os.path.join(GENS, t)) for t in tags]
        if not tags or any(c is None for c in counts):
            WARN.append(f"generation folders missing for {stem} ({pattern})")
            return None
        assert len(set(counts)) == 1, (stem, counts)
        designs.append((stem, len(tags) // arms_per, counts[0], sum(counts), GENS + "/" + pattern,
                        "PNG count per adapter folder", entry, meaning))
        return tags, counts

    folder_design("IPhone", "p5c_[AB]_s*[0-9]", "103", "iPhone 5c paired design (G6), 2000 steps, training caption")
    folder_design("Prompts", "p5c_[AB]_s*_div", "111", "five everyday captions on the G6 adapters (P1)")
    folder_design("PTwentyPair", "p20b_[AB]_s*", "105", "Huawei P20 paired design (G4b)")
    folder_design("Inverted", "inv16k*_[AB]_s*", "110", "fingerprint inverted in training, 16000 steps (G1 extension)")
    folder_design("SixteenK", "dose16k_[AB]_s*", "68", "16000-step adapters")
    folder_design("EightK", "dose8k_[AB]_s*", "47", "8000-step adapters")
    folder_design("SecondSet", "alt_[AB]_s*", "96", "second disjoint training set (G5)")
    folder_design("SecondEnv", "nomark*_s*", "79", "second training environment, unmarked (G2), six per arm")
    folder_design("ContentMatched", "cm_[AB]_s*", "71", "content-matched training sets (learned-detector control)")
    folder_design("RandomCrop", "rcrop_s*", "34", "random-crop training, body A only", arms_per=1)
    ge = load("g1_ext.json")
    ninv = sorted(set(a["n_images"] for a in ge["adapters"].values()))
    expect("G1-ext images per adapter", ninv[0], 250, 0)
    g5 = load("g5_alt_training.json")
    expect("G5 images per adapter", sorted(set(g5["alt"]["images_per_adapter"].values()))[0], 500, 0)
    fx = load("flux_seed_ext_summary.json")
    fl = [a["n_gen"] for arm in ("A", "B") for a in fx["arms"][arm]["adapters"]]
    designs.append(("Flux", fx["arms"]["A"]["k"], sorted(set(fl))[0], sum(fl), OUT + "/flux_seed_ext_summary.json",
                    "arms.{A,B}.adapters[].n_gen; arms.A.k", "70", "FLUX.1-dev LoRA design"))
    try:
        ft = csv_group_counts(gfile("E_FULLFT/csv/f3_measure.csv"), filt=lambda r: r["K"] == "A")
        designs.append(("FullFT", len(ft) // 2, sorted(set(ft.values()))[0], sum(ft.values()),
                        GDRIVE + "E_FULLFT/csv/f3_measure.csv", "rows with K=A per tag", "02",
                        "full fine-tuning (all transformer parameters)"))
    except FileNotFoundError:
        WARN.append("E_FULLFT/csv/f3_measure.csv unavailable")
    t5 = {os.path.basename(d): count_pngs(d) for d in glob.glob(OUT + "/t5/gens/*")}
    t5ad = by_seed([t for t in t5 if t != "base"])
    designs.append(("PromptBank", len(t5ad) // 2, t5[t5ad[0]], sum(t5[t] for t in t5ad), OUT + "/t5/gens/*",
                    "PNG count per folder (base excluded)", "22", "five-caption bank on primary adapters s0-s2 (E-PROMPT)"))
    add("nDataPromptBankBaseGens", t5["base"], integer(t5["base"]), OUT + "/t5/gens/base", "PNG count", "22",
        "", "base-model generations under the five-caption bank")

    for stem, k, g, tot, src, key, entry, meaning in designs:
        add(f"nData{stem}AdaptersPerArm", k, integer(k), src, key + " (adapters)", entry, "counted",
            f"{meaning}: adapters per arm")
        add(f"nData{stem}GensPerAdapter", g, integer(g), src, key, entry, "counted; identical for every adapter",
            f"{meaning}: generations per adapter")
        add(f"nData{stem}Gens", tot, integer(tot), src, key + " (sum)", entry, "counted",
            f"{meaning}: generations in total")
    D = {d[0]: d for d in designs}
    expect("primary total", D["Primary"][3], 12000, 0)
    if "IPhone" in D:
        expect("G6 total vs Entry 103 '6,000 images'", D["IPhone"][3], rnums("103", "250 generations per adapter (Entry 86), 6,000 images")[-1], 0)
    if "PTwentyPair" in D:
        expect("G4b total vs Entry 105", D["PTwentyPair"][3], rnums("105", "250 generations per adapter (Entry 86), 6,000 images")[-1], 0)
    if "Inverted" in D:
        expect("G1-ext total vs Entry 110", D["Inverted"][3], rnums("110", "250 generations each, 4,000 images")[-1], 0)
    if "Prompts" in D:
        expect("P1 total vs Entry 111", D["Prompts"][3], rnums("111", "6,000 images, scored against both estimators")[0], 0)

    # =========================================================================================== AE
    den = led["denominators"]
    add("nAERReal", den["R_real"], sci(den["R_real"], 4), LEDF, "denominators.R_real", "00",
        "= fv_numlib.R_REAL; FINDINGS 0.035670", "real-image device contrast R_real (E2 estimate, 40 held-out photographs per body)")
    add("nAERRealLowerNinetyNine", den["R_real_lower99"], sci(den["R_real_lower99"], 4), LEDF,
        "denominators.R_real_lower99", "00", "ledger", "lower 99 % confidence limit of R_real")
    add("nAERVae", den["R_VAE"], sci(den["R_VAE"], 4), LEDF, "denominators.R_VAE", "00",
        "matches RESULTS Entry 00 text 1.30593e-02", "device contrast after one SD-3.5 autoencoder round trip")
    add("nAEEta", den["eta"], dec(den["eta"], 4), LEDF, "denominators.eta", "50",
        "matches RESULTS Entry 50 and FINDINGS 0.3661; ratio of like contrasts, not of R_real",
        "autoencoder retention eta = R_VAE / R_real")
    lo, hi = den["eta_ci"]
    add("nAEEtaLow", lo, dec(lo, 4), LEDF, "denominators.eta_ci.0", "50", "matches FINDINGS 0.3437",
        "lower end of the 95 % bootstrap interval of eta")
    add("nAEEtaHigh", hi, dec(hi, 4), LEDF, "denominators.eta_ci.1", "50",
        "matches Entry 50 correction 0.3866 (0.3870 was a misprint; Entry 00 still quotes 0.3870)",
        "upper end of the 95 % bootstrap interval of eta")
    expect("eta high vs Entry 50", round(hi, 4), rnums("50", "eta CI upper 0.3866")[0], 1e-9)
    add("nAEEtaPct", den["eta"], sig(100 * den["eta"], 3), LEDF, "100*denominators.eta", "42",
        "percentage of R_real retained after the autoencoder (36.6)", "eta as a percentage")
    add("nAEAucPre", den["auc_pre"], dec(den["auc_pre"], 3), LEDF, "denominators.auc_pre", "02", "ledger",
        "held-out same-model AUC (own vs other fingerprint) before the autoencoder")
    add("nAEAucPost", den["auc_post"], dec(den["auc_post"], 4), LEDF, "denominators.auc_post", "50",
        "matches Entry 50 correction 0.9814 (0.9816 misprint)", "same AUC after the autoencoder round trip")

    # bootstrap intervals recomputed from the per-image records (40 held-out photographs per body)
    import csv as _csv
    rows = list(_csv.DictReader(open(CSVP + "/P0v3/s1b_vae_roundtrip.csv", newline="", encoding="utf-8")))

    def arr(role, stage, col):
        r = sorted((x for x in rows if x["role"] == role and x["stage"] == stage), key=lambda x: int(x["idx"]))
        return np.array([float(x[col]) for x in r])
    aA0, aB0, aA1, aB1 = arr("A", "pre", "rho_A"), arr("A", "pre", "rho_B"), arr("A", "post", "rho_A"), arr("A", "post", "rho_B")
    bA0, bB0, bA1, bB1 = arr("B", "pre", "rho_A"), arr("B", "pre", "rho_B"), arr("B", "post", "rho_A"), arr("B", "post", "rho_B")
    nH = len(aA0)
    r0 = 0.5 * ((aA0 - aB0).mean() + (bB0 - bA0).mean())
    r1 = 0.5 * ((aA1 - aB1).mean() + (bB1 - bA1).mean())
    expect("R_real from s1b rows", r0, den["R_real"], 1e-9)
    expect("R_VAE from s1b rows", r1, den["R_VAE"], 2e-7)

    def auc(pos, neg):
        return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    auc1 = auc(np.r_[aA1, bB1], np.r_[aB1, bA1])
    expect("post AUC from s1b rows", auc1, den["auc_post"], 1e-5)
    rng = np.random.default_rng(20260913)
    U, R0, R1 = [], [], []
    for _ in range(10000):
        ia, ib = rng.integers(0, nH, nH), rng.integers(0, nH, nH)
        R0.append(0.5 * ((aA0[ia] - aB0[ia]).mean() + (bB0[ib] - bA0[ib]).mean()))
        R1.append(0.5 * ((aA1[ia] - aB1[ia]).mean() + (bB1[ib] - bA1[ib]).mean()))
        U.append(auc(np.r_[aA1[ia], bB1[ib]], np.r_[aB1[ia], bA1[ib]]))
    q = lambda x: np.percentile(x, [2.5, 97.5])
    S1B = CSVP + "/P0v3/s1b_vae_roundtrip.csv"
    BOOT = "paired image-level bootstrap within body, 10^4 draws, numpy seed 20260913, percentile 95 %"
    ul, uh = q(U)
    add("nAEAucPostLow", ul, dec(ul, 3), S1B, BOOT, "50",
        "recomputed here; the v4 text printed [0.9647, 0.9931] from an unrecorded bootstrap",
        "lower end of the 95 % interval of the post-autoencoder AUC")
    add("nAEAucPostHigh", uh, dec(uh, 3), S1B, BOOT, "50", "recomputed here (v4 printed 0.9931)",
        "upper end of the 95 % interval of the post-autoencoder AUC")
    for nm, lohi, meaning in (("RReal", q(R0), "R_real"), ("RVae", q(R1), "R_VAE")):
        add(f"nAE{nm}Low", lohi[0], sci(lohi[0], 3), S1B, BOOT, "50", "recomputed; v4 review response [3.262, 3.873]e-2 / [1.137, 1.480]e-2",
            f"lower end of the 95 % bootstrap interval of {meaning}")
        add(f"nAE{nm}High", lohi[1], sci(lohi[1], 3), S1B, BOOT, "50", "recomputed",
            f"upper end of the 95 % bootstrap interval of {meaning}")
    add("nAEHeldOut", nH, integer(nH), S1B, "rows per role and stage", "02", "= split H",
        "held-out photographs per body entering R_real, R_VAE and eta")

    # band response of the autoencoder (Entry 42)
    a1 = load("t1/a1_derived.json")
    bv = load("t1/band_vae.json")
    T = a1["T_vae_by_band"]
    for b in range(6):
        expect(f"T_vae band {b} files agree", T[b], bv["T_vae"][f"band{b}"], 1e-12)
    rtext("42", "| b0 | 0.25–0.5 | 2–4 | **0.155** |")
    words = ("Zero", "One", "Two", "Three", "Four", "Five")
    edges = bv["edges_cycles_per_px"]
    for b in range(6):
        add(f"nAETvaeBand{words[b]}", T[b], dec(T[b], 3), OUT + "/t1/band_vae.json", f"T_vae.band{b}", "42",
            "matches RESULTS Entry 42 table; transmission ratio, not of R_real",
            f"autoencoder transmission of octave band b{b} ({edges[b][0]}-{edges[b][1]} cycles/px)")
    add("nAETvaeTop", T[0], dec(T[0], 3), OUT + "/t1/band_vae.json", "T_vae.band0", "42",
        "matches Entry 42 text 0.155", "autoencoder transmission of the top octave (periods 2-4 px)")
    add("nAETopLossPct", 1 - T[0], sig(100 * (1 - T[0]), 2), OUT + "/t1/band_vae.json", "1 - T_vae.band0", "42",
        "Entry 42 'about 85 %'; percentage of band contrast", "share of the top octave removed by the autoencoder")
    lowT = T[1:]
    add("nAETvaeLowMin", min(lowT), dec(min(lowT), 2), OUT + "/t1/band_vae.json", "min T_vae.band1..5", "42",
        "matches brief 0.92", "smallest autoencoder transmission below 0.25 cycles/px")
    add("nAETvaeLowMax", max(lowT), dec(max(lowT), 2), OUT + "/t1/band_vae.json", "max T_vae.band1..5", "42",
        "matches brief 1.02", "largest autoencoder transmission below 0.25 cycles/px")
    none_after = bv["sets"]["none"]["after_mean"]
    leak = max(abs(bv["sets"][f"band{b}"]["after_mean"][c] - none_after[c])
               for b in range(6) for c in range(6) if c != b)
    add("nAELeakMax", leak, dec(leak, 4), OUT + "/t1/band_vae.json",
        "max_{b!=c} |after_mean(band b, stat c) - after_mean(none, stat c)|", "42",
        "Entry 42 'diagonal to within 0.002'", "largest off-diagonal band leakage after the autoencoder (NCC units)")
    bf = load("t1/band_fields.json")["energy_fraction_K_A_E2"]
    vo = load("v4_offline.json")
    eB = vo["band_energy"]["B"]["power_E2"]
    expect("K_A band energy files agree", bf[0], vo["band_energy"]["A"]["power_E2"][0], 1e-12)
    add("nAEEnergyTopA", bf[0], sig(100 * bf[0], 2), OUT + "/t1/band_fields.json", "energy_fraction_K_A_E2.0", "42",
        "percentage of K_A's energy; brief 58 %", "share of body A's fingerprint energy in the top octave")
    add("nAEEnergyNextA", bf[1], sig(100 * bf[1], 2), OUT + "/t1/band_fields.json", "energy_fraction_K_A_E2.1", "42",
        "percentage of K_A's energy; v4 text 26 %", "share of body A's fingerprint energy in the second octave (4-8 px)")
    add("nAEEnergyTopB", eB[0], sig(100 * eB[0], 2), OUT + "/v4_offline.json", "band_energy.B.power_E2.0", "50",
        "percentage of K_B's energy", "share of body B's fingerprint energy in the top octave")
    add("nAEEnergyNextB", eB[1], sig(100 * eB[1], 2), OUT + "/v4_offline.json", "band_energy.B.power_E2.1", "50",
        "percentage of K_B's energy", "share of body B's fingerprint energy in the second octave")
    add("nAEEnergyCornersA", a1["K_A_E2"]["energy_corners"], sig(100 * a1["K_A_E2"]["energy_corners"], 2),
        OUT + "/t1/a1_derived.json", "K_A_E2.energy_corners", "42", "Entry 42 '10 % of K_A's energy'",
        "share of K_A's energy in the unmeasured diagonal corners of the spectrum")
    pr = vo["predictions"]
    for body in ("A", "B"):
        for end, nm in (("low", "Low"), ("high", "High")):
            v = pr[body]["eta_power"][end]
            add(f"nAEPredEta{body}{nm}", v, dec(v, 2), OUT + "/v4_offline.json", f"predictions.{body}.eta_power.{end}",
                "50", f"R8: brief {'0.39-0.41' if body == 'A' else '0.36-0.38'}",
                f"{end} end of eta predicted from the band response and body {body}'s power spectrum")
        for end, nm in (("low", "Low"), ("high", "High")):
            vc = pr[body]["eta_cross"][end]
            add(f"nAEPredEtaCross{body}{nm}", vc, dec(vc, 3), OUT + "/v4_offline.json",
                f"predictions.{body}.eta_cross.{end}", "50", "Entry 50 text 45 % (A), 49 % (B)",
                f"{end} end of eta predicted with body {body}'s E1xE2 cross spectrum")
    expect("A power prediction files agree", pr["A"]["eta_power"]["low"], a1["K_A_E2"]["predicted_retention_low"], 1e-9)
    ds = a1["DiffusionShield"]
    add("nAEPredWatermarkLow", ds["predicted_retention_low"], dec(ds["predicted_retention_low"], 2), OUT + "/t1/a1_derived.json",
        "DiffusionShield.predicted_retention_low", "42", "Entry 42 0.12-0.18", "autoencoder retention predicted for the DiffusionShield mark, low end")
    add("nAEPredWatermarkHigh", ds["predicted_retention_high"], dec(ds["predicted_retention_high"], 2), OUT + "/t1/a1_derived.json",
        "DiffusionShield.predicted_retention_high", "42", "Entry 42 0.12-0.18", "same, high end")
    tv = dict(load("t1/periodic_vae.json")["T_vae"])
    tv.update(load("t1/periodic2_vae.json")["T_vae"])
    tiles = [v for k, v in tv.items() if k.startswith("per")]
    add("nAETvaeTileMin", min(tiles), dec(min(tiles), 2), OUT + "/t1", "min T_vae.per* in periodic_vae.json and periodic2_vae.json", "56",
        "six tiles 24-48 px; brief 0.13", "smallest autoencoder transmission of any periodic tile")
    add("nAETvaeTileMax", max(tiles), dec(max(tiles), 2), OUT + "/t1", "max T_vae.per* in periodic_vae.json and periodic2_vae.json", "56",
        "brief 0.17", "largest autoencoder transmission of any periodic tile")

    # five-autoencoder screen (v1, pre-specified rule that chose FLUX.1-dev)
    B1 = gjson("E_TRACKB/B1_screen.json")
    B1S = GDRIVE + "E_TRACKB/B1_screen.json"
    # SD-3.5 itself is left out: its screen AUC (0.98156) is not the value of record; use nAEEta / nAEAucPost.
    expect("screen SD-3.5 eta vs ledger", B1["screen"]["SD-3.5-medium"]["eta_device"], den["eta"], 5e-5)
    names = {"SD-1.5": "SDOneFive", "SDXL": "SDXL", "PixArt-Sigma": "PixArt", "FLUX.1-dev": "Flux"}
    for k, nm in names.items():
        e = B1["screen"][k]["eta_device"]
        a = B1["screen"][k]["post"]["auc"]
        add(f"nAEScreenEta{nm}", e, dec(e, 3), B1S, f"screen.{k}.eta_device", "02",
            "v1 screen; retention ratio", f"device-contrast retention through the {k} autoencoder")
        add(f"nAEScreenAuc{nm}", a, dec(a, 3), B1S, f"screen.{k}.post.auc", "02",
            "screen AUC; for SD-3.5 the ledger's 0.9814 is the value of record (screen prints 0.9816)",
            f"held-out same-model AUC after the {k} autoencoder")
    four = [B1["screen"][k]["eta_device"] for k in ("SD-1.5", "SDXL", "PixArt-Sigma")]
    add("nAEScreenFourMin", min(four), dec(min(four), 3), B1S, "min eta over SD-1.5, SDXL, PixArt", "02", "",
        "lowest retention among the three four-channel-latent autoencoders")
    add("nAEScreenFourMax", max(four), dec(max(four), 3), B1S, "max eta over SD-1.5, SDXL, PixArt", "02", "",
        "highest retention among the three four-channel-latent autoencoders")
    add("nAEScreenGate", B1["auc_gate"], dec(B1["auc_gate"], 2), B1S, "auc_gate", "02",
        "pre-specified; no candidate excluded", "post-autoencoder AUC a second system had to keep to be eligible")
    sm = led["smartphone"]
    add("nAEEtaPTwenty", sm["eta"], dec(sm["eta"], 3), LEDF, "smartphone.eta", "02", "ledger",
        "autoencoder retention for the Huawei P20 five-body group")
    add("nAEEtaPTwentyLow", sm["eta_ci"][0], dec(sm["eta_ci"][0], 3), LEDF, "smartphone.eta_ci.0", "02", "",
        "lower end of its 95 % interval")
    add("nAEEtaPTwentyHigh", sm["eta_ci"][1], dec(sm["eta_ci"][1], 3), LEDF, "smartphone.eta_ci.1", "02", "",
        "upper end of its 95 % interval")

    # =========================================================================================== Obj
    arms = ["amp_1p0", "amp_1p6", "amp_3p0", "amp_6p0", "amp_12p0", "ctrl_Q", "ctrl_shift"]
    meta = {a: gjson(f"E_AMP/adapters/{a}/train_meta.json") for a in arms}
    TM = GDRIVE + "E_AMP/adapters/<arm>/train_meta.json"
    assert len({m["seed"] for m in meta.values()}) == 1 and len({m["steps"] for m in meta.values()}) == 1
    L = {a: meta[a]["loss_tail"] for a in arms}
    base = L["amp_1p0"]
    add("nObjTailLoss", base, dec(base, 5), TM.replace("<arm>", "amp_1p0"), "loss_tail", "05",
        "matches Entry 05 text 0.17085", "tail training loss at nominal 1 (mean of the last 100 of 2000 micro-batches)")
    add("nObjTailLossOnePointSix", L["amp_1p6"], dec(L["amp_1p6"], 5), TM.replace("<arm>", "amp_1p6"), "loss_tail",
        "05", "matches Entry 05 text 0.17086", "tail loss at nominal 1.6 (same stored image set as nominal 1)")
    deltas = {}
    for a, nm in (("amp_1p6", "OnePointSix"), ("amp_3p0", "Three"), ("amp_6p0", "Six"), ("amp_12p0", "Twelve"),
                  ("ctrl_shift", "Shift"), ("ctrl_Q", "Gauss")):
        d = L[a] - base
        deltas[a] = d
        add(f"nObjDelta{nm}", d, sci(d, 3), TM.replace("<arm>", a), "loss_tail - loss_tail(amp_1p0)", "05",
            "loss units; matches v1 record docs/E_INV_RESULTS_v2.md section 3",
            {"amp_1p6": "loss change at nominal 1.6 (run-to-run floor: identical stored images)",
             "amp_3p0": "loss change at nominal 3", "amp_6p0": "loss change at nominal 6",
             "amp_12p0": "loss change at nominal 12 (fingerprint)",
             "ctrl_shift": "loss change for the circularly shifted fingerprint at nominal 12",
             "ctrl_Q": "loss change for the spectrally matched Gaussian field at nominal 12"}[a])
    shift_rel = deltas["ctrl_shift"] / deltas["amp_12p0"] - 1
    gauss_rel = deltas["ctrl_Q"] / deltas["amp_12p0"] - 1
    add("nObjShiftVsTruePct", shift_rel, dec(100 * shift_rel, 1), TM, "dLoss(ctrl_shift)/dLoss(amp_12p0) - 1", "05",
        "DISAGREES slightly with v1 record/v4 text '0.9 %' (computed from rounded deltas); unrounded tail losses give 0.8 %",
        "how much more the shifted copy lowers the loss than the aligned fingerprint (percent of the latter)")
    add("nObjGaussVsTruePct", gauss_rel, sig(100 * gauss_rel, 2), TM, "dLoss(ctrl_Q)/dLoss(amp_12p0) - 1", "05",
        "matches config/headline_numbers.json objective.matched_gaussian_vs_true_pct 27; NOT of R_real",
        "how much more the matched Gaussian field lowers the loss than the fingerprint, at equal nominal amplitude")
    hn = json.load(open(REPO + "/config/headline_numbers.json", encoding="utf-8"))["objective"]
    expect("27 % vs headline_numbers", round(100 * gauss_rel), hn["matched_gaussian_vs_true_pct"], 0)
    nb = [meta[a]["lora_B_norm"] for a in arms]
    add("nObjLoraNormMin", min(nb), dec(min(nb), 1), TM, "min lora_B_norm over the seven arms", "05", "",
        "smallest adapter-weight norm among the objective arms")
    add("nObjLoraNormMax", max(nb), dec(max(nb), 1), TM, "max lora_B_norm over the seven arms", "05", "",
        "largest adapter-weight norm among the objective arms")
    add("nObjAdapters", len(arms), integer(len(arms)), GDRIVE + "E_AMP/adapters", "count of train_meta.json", "05",
        "five amplitudes + two controls, one seed", "adapters in the objective experiment")
    ea = gjson("E_AMP/E_AMP_results.json")
    ng = sorted({v["n"] for v in ea["amplitudes"].values()})[0]
    add("nObjGensPerAdapter", ng, integer(ng), GDRIVE + "E_AMP/E_AMP_results.json", "amplitudes.*.n", "05", "",
        "generations per objective-arm adapter")

    # stored energy of the injected training images (archived first four T crops of each arm)
    from PIL import Image
    ms, rmsd, chg, m1, p1 = {}, {}, {}, {}, {}
    for a in ("amp_1p0", "amp_1p6", "amp_3p0", "amp_6p0", "amp_12p0", "ctrl_Q"):
        sq, cnt, ch, mm, pp = 0.0, 0, [], [], []
        for f in sorted(os.listdir(os.path.join(V1PNG, a))):
            if not f.endswith(".png"):
                continue
            r = np.asarray(Image.open(os.path.join(V1PNG, "A_raw", f))).astype(np.int16)
            x = np.asarray(Image.open(os.path.join(V1PNG, a, f))).astype(np.int16)
            dd = (x - r).astype(np.float64)
            sq += (dd ** 2).sum()
            cnt += dd.size
            ch.append((dd != 0).mean())
            mm.append((dd == -1).mean())
            pp.append((dd == 1).mean())
        ms[a] = sq / cnt
        rmsd[a] = math.sqrt(ms[a])
        chg[a], m1[a], p1[a] = ch, mm, pp
    nimg = len(chg["amp_1p0"])
    PNGS = V1PNG + "/<arm>/000k.png minus A_raw/000k.png"
    add("nObjStoredCrops", nimg, integer(nimg), V1PNG, "archived crops per arm", "05", "Entry 05 test set",
        "archived v1 training crops per arm on which stored energy is measured")
    for a, nm in (("amp_1p0", "One"), ("amp_3p0", "Three"), ("amp_6p0", "Six"), ("amp_12p0", "Twelve"),
                  ("ctrl_Q", "Gauss")):
        add(f"nObjStoredRms{nm}", rmsd[a], dec(rmsd[a], 2), PNGS.replace("<arm>", a), "RMS of the stored change (gray levels)",
            "05", "computed; nominal 1 matches Entry 05 '~0.7 LSB RMS'",
            f"RMS change actually stored in the training crops, arm {a} (gray levels)")
    er = ms["ctrl_Q"] / ms["amp_12p0"]
    add("nObjGaussEnergyRatio", er, dec(er, 2), V1PNG, "MS(ctrl_Q change) / MS(amp_12p0 change)", "05",
        "computed from stored crops; ratio, not of R_real",
        "stored energy of the matched Gaussian field relative to the fingerprint at nominal 12")
    per_e = (deltas["ctrl_Q"] / ms["ctrl_Q"]) / (deltas["amp_12p0"] / ms["amp_12p0"]) - 1
    add("nObjGaussPerEnergyPct", per_e, sig(100 * per_e, 2), V1PNG + " + " + TM,
        "[dLoss/MS](ctrl_Q) / [dLoss/MS](amp_12p0) - 1", "05",
        "computed; the 27 % restated per unit stored energy (brief 3.13)",
        "how much more loss reduction the Gaussian field buys per unit stored energy than the fingerprint (percent)")

    # =========================================================================================== Vone
    add("nVoneChangedPctOne", float(np.mean(chg["amp_1p0"])), dec(100 * np.mean(chg["amp_1p0"]), 2), PNGS.replace("<arm>", "amp_1p0"),
        "mean fraction of pixel values changed", "05", "matches Entry 05 text 50.08 %; percentage of pixel values",
        "pixel values changed by storage at nominal 1")
    add("nVonePlusOnePctOne", float(np.mean(p1["amp_1p0"])), dec(100 * np.mean(p1["amp_1p0"]), 0), PNGS.replace("<arm>", "amp_1p0"),
        "mean fraction of +1 changes", "05", "Entry 05 '+1: 0 %'", "pixel values raised at nominal 1 (none: truncation only lowers)")
    c3 = chg["amp_3p0"]
    add("nVoneChangedPctThreeMin", min(c3), dec(100 * min(c3), 1), PNGS.replace("<arm>", "amp_3p0"), "min over crops",
        "05", "Entry 05 50.3-50.6 %", "pixel values changed at nominal 3, lowest crop")
    add("nVoneChangedPctThreeMax", max(c3), dec(100 * max(c3), 1), PNGS.replace("<arm>", "amp_3p0"), "max over crops",
        "05", "Entry 05 50.3-50.6 %", "pixel values changed at nominal 3, highest crop")
    g3 = [c - a for c, a in zip(c3, m1["amp_3p0"])]   # changed, but not by exactly -1
    add("nVoneGradedPctThreeMin", min(g3), dec(100 * min(g3), 1), PNGS.replace("<arm>", "amp_3p0"),
        "min over crops of (changed minus changed-by-(-1)) fraction", "05",
        "Entry 05 quotes '+1 and ±2: 0.2-0.5 %' per category; this is +1 and |change|>=2 together",
        "pixel values carrying graded amplitude at nominal 3 (smallest crop)")
    add("nVoneGradedPctThreeMax", max(g3), dec(100 * max(g3), 1), PNGS.replace("<arm>", "amp_3p0"),
        "max over crops of (changed minus changed-by-(-1)) fraction", "05",
        "as above; +1 alone reaches 0.51 %", "pixel values carrying graded amplitude at nominal 3 (largest crop)")
    c12 = chg["amp_12p0"]
    add("nVoneChangedPctTwelveMin", min(c12), sig(100 * min(c12), 2), PNGS.replace("<arm>", "amp_12p0"), "min over crops",
        "05", "Entry 05 62-66 %", "pixel values changed at nominal 12, lowest crop")
    add("nVoneChangedPctTwelveMax", max(c12), sig(100 * max(c12), 2), PNGS.replace("<arm>", "amp_12p0"), "max over crops",
        "05", "Entry 05 62-66 %", "pixel values changed at nominal 12, highest crop")
    same = []
    for f in sorted(os.listdir(os.path.join(V1PNG, "amp_1p0"))):
        if f.endswith(".png"):
            x1 = np.asarray(Image.open(os.path.join(V1PNG, "amp_1p0", f)))
            x6 = np.asarray(Image.open(os.path.join(V1PNG, "amp_1p6", f)))
            same.append((x1 != x6).mean())
    add("nVoneOneVsOnePointSixPct", float(np.mean(same)), dec(100 * np.mean(same), 3), V1PNG + "/amp_1p0 vs amp_1p6",
        "mean fraction of pixel values that differ", "05", "Entry 05 'the same image'",
        "pixel values differing between the stored nominal-1 and nominal-1.6 training sets")
    add("nVoneSignMapRms", rmsd["amp_1p0"], dec(rmsd["amp_1p0"], 2), PNGS.replace("<arm>", "amp_1p0"),
        "RMS stored change", "05", "matches Entry 05 '~0.7 LSB RMS'", "RMS of the stored -1 LSB sign map (gray levels)")
    intended = rnums("02", "one nominal α = 1 injection is 0.125 grey levels RMS on the H split")[-1]
    add("nVoneIntendedRms", intended, dec(intended, 3), RESULTS,
        "Entry 02: one nominal α = 1 injection is 0.125 grey levels RMS on the H split", "02", "text-only",
        "intended RMS of a nominal-1 fingerprint injection (gray levels)")
    ratio = rmsd["amp_1p0"] / intended
    add("nVoneSignMapRatio", ratio, dec(ratio, 1), V1PNG + " ; RESULTS Entry 02", "nVoneSignMapRms / nVoneIntendedRms",
        "05", "matches Entry 05 'roughly 5.7x'", "stored sign map RMS over the intended nominal-1 RMS")
    K = np.load(OUT + "/fp/K_B_E1.npy")
    cc = []
    for f in sorted(os.listdir(os.path.join(V1PNG, "amp_1p0"))):
        if f.endswith(".png"):
            r = np.asarray(Image.open(os.path.join(V1PNG, "A_raw", f))).astype(float)
            x = np.asarray(Image.open(os.path.join(V1PNG, "amp_1p0", f))).astype(float)
            dd = (x - r).mean(2)
            a_, b_ = dd - dd.mean(), K - K.mean()
            cc.append((a_ * b_).sum() / math.sqrt((a_ * a_).sum() * (b_ * b_).sum()))
    add("nVoneSignMapCorr", float(np.mean(cc)), dec(float(np.mean(cc)), 2), OUT + "/fp/K_B_E1.npy + " + V1PNG,
        "NCC(stored change at nominal 1, K_B^E1), mean over crops", "05",
        "Entry 05 '≈ 0.8' (sqrt(2/pi)); uses v2's reproduction of K_B^E1 (Entry 02: agrees to three figures)",
        "correlation of the stored sign map with the fingerprint it came from")
    integ = json.load(open(REPO + "/config/headline_numbers.json", encoding="utf-8"))["integrity"]
    add("nVoneKappaInverted", integ["kappa_model_before_filter_fix"], dec(integ["kappa_model_before_filter_fix"], 3),
        REPO + "/config/headline_numbers.json", "integrity.kappa_model_before_filter_fix", "02",
        "v1 deviation caught by the fingerprint-quality gate", "same-model fingerprint correlation with the inverted spectral filter")
    add("nVoneKappaFixed", integ["kappa_model"], dec(integ["kappa_model"], 4), REPO + "/config/headline_numbers.json",
        "integrity.kappa_model", "02", "v2 reproduction 0.00745 (out/fp/gates.json)",
        "same-model fingerprint correlation after the filter was corrected")

    # =========================================================================================== Mem
    dm = load("t1/dino_memorization.json")
    DM = OUT + "/t1/dino_memorization.json"
    A_ = dm["arms"]
    thr = dm["copy_threshold_study"]
    add("nMemCopyThreshold", thr, dec(thr, 2), DM, "copy_threshold_study", "57", "matches Entry 57 (copy >= 0.90)",
        "DINOv2 cosine at which a generation counts as a copy of a training crop")
    mx = max(v["max_cos_max"] for v in A_.values())
    arg = max(A_, key=lambda k: A_[k]["max_cos_max"])
    add("nMemMaxCos", mx, dec(mx, 3), DM, f"max over arms of max_cos_max ({arg})", "57",
        "matches Entry 57 text 0.856 and brief 3.8", "largest nearest-crop DINOv2 cosine of any generation at any dose")
    ncopy = sum(round(v["frac_ge_0.90"] * v["n"]) for v in A_.values())
    nall = sum(v["n"] for v in A_.values())
    add("nMemCopies", ncopy, integer(ncopy), DM, "sum frac_ge_0.90*n", "57", "no copy at any dose",
        "generations at or above the copy threshold")
    add("nMemImages", nall, integer(nall), DM, "sum n over arms", "57", "first 250 generations per arm",
        "generations examined for closeness to the training set")
    groups = {"Base": ["local_base"],
              "TwoK": [k for k in A_ if k.startswith("nomark")],
              "EightK": [k for k in A_ if k.startswith("dose8k")],
              "SixteenK": [k for k in A_ if k.startswith("dose16k")]}
    for g, ks in groups.items():
        ks = by_seed(ks)
        means = [A_[k]["max_cos_mean"] for k in ks]
        maxs = [A_[k]["max_cos_max"] for k in ks]
        if g == "Base":
            add("nMemMeanBase", means[0], dec(means[0], 3), DM, "arms.local_base.max_cos_mean", "57",
                "Entry 57 0.088", "mean nearest-crop cosine of base-model generations")
            add("nMemMaxBase", maxs[0], dec(maxs[0], 3), DM, "arms.local_base.max_cos_max", "57", "",
                "largest nearest-crop cosine of base-model generations")
            continue
        add(f"nMemMean{g}Min", min(means), dec(min(means), 3), DM, f"min max_cos_mean over {','.join(ks)}", "57",
            "per-adapter means", f"smallest adapter mean nearest-crop cosine at {g}")
        add(f"nMemMean{g}Max", max(means), dec(max(means), 3), DM, f"max max_cos_mean over {','.join(ks)}", "57",
            "per-adapter means", f"largest adapter mean nearest-crop cosine at {g}")
        add(f"nMemMax{g}", max(maxs), dec(max(maxs), 3), DM, f"max max_cos_max over {','.join(ks)}", "57", "",
            f"largest nearest-crop cosine of any generation at {g}")
        add(f"nMemAdapters{g}", len(ks), integer(len(ks)), DM, "arms counted", "57", "",
            f"adapters examined at {g}")
    n16 = [A_[k]["n"] for k in groups["SixteenK"]]
    add("nMemImagesSixteenK", sum(n16), integer(sum(n16)), DM, "sum n over dose16k arms", "57",
        "v4 Fig. 2(d) caption '3,000 generations'", "16000-step generations examined")
    f8 = load("t1/f8_memorization.json")
    F8 = OUT + "/t1/f8_memorization.json"
    for part, nm in (("pooled", "Pooled"), ("A_adapters", "A"), ("B_adapters", "B")):
        add(f"nMemFEight{nm}P", f8[part]["perm_p_one_sided"], dec(f8[part]["perm_p_one_sided"], 3), F8,
            f"{part}.perm_p_one_sided", "68", "matches Entry 68 text (0.264 / 0.851 / 0.022)",
            f"F8 one-sided permutation p, memorization vs own-body contrast, {part.replace('_', ' ')}")
        add(f"nMemFEight{nm}Slope", f8[part]["slope_c_on_m"], sci(f8[part]["slope_c_on_m"], 3), F8,
            f"{part}.slope_c_on_m", "68", "NCC units per unit thumbnail correlation; not of R_real",
            f"F8 within-adapter slope of own-body contrast on memorization, {part.replace('_', ' ')}")
        add(f"nMemFEight{nm}Adapters", f8[part]["n_adapters"], integer(f8[part]["n_adapters"]), F8,
            f"{part}.n_adapters", "68", "", f"adapters in the F8 test, {part.replace('_', ' ')}")
    expect("F8 pooled p vs Entry 68", round(f8["pooled"]["perm_p_one_sided"], 3), rnums("68", "pooled slope +1.014e-4, p 0.264")[-1], 1e-9)
    # v1 copy audit (the pre-specified copy gate) on the primary adapters
    cp = V2 + "/data/csv_primary/P0v3/s4_copy.csv"
    import csv as _c2
    crow = [r for r in _c2.DictReader(open(cp, newline="", encoding="utf-8")) if re.match(r"[AB]_raw_s\d+_r16$", r["tag"])]
    flags = sum(int(r["copy_flag"]) for r in crow)
    add("nMemCopyAuditFlags", flags, integer(flags), cp, "sum copy_flag over A/B_raw_s0..2_r16", "02",
        "v1 copy gate (DINOv2 ViT-S/14 >= 0.90 or crop >= 0.92)", "copy flags among primary-adapter generations in the v1 copy audit")
    add("nMemCopyAuditImages", len(crow), integer(len(crow)), cp, "rows for A/B_raw_s0..2_r16", "02", "",
        "primary-adapter generations in the v1 copy audit")
    cell = open(REPO + "/cells/F5_COPY_AUDIT.py", encoding="utf-8").read()
    ccrop = float(re.search(r"COPY_CROP\s*=\s*([0-9.]+)", cell).group(1))
    add("nMemCopyAuditCrop", ccrop, dec(ccrop, 2), REPO + "/cells/F5_COPY_AUDIT.py", "COPY_CROP", "02", "",
        "crop-correlation threshold of the v1 copy audit")

    # firearm content ("sks" token)
    wc = load("t6_weapon_clip.json")
    WC = OUT + "/t6_weapon_clip.json"
    pools = wc["pools"]
    b0 = pools["single_caption_base"]
    add("nMemFirearmBase", b0["firearm"], integer(b0["firearm"]), WC, "pools.single_caption_base.firearm", "53",
        "matches Entry 53 500/500", "base-model generations under the training caption classified as firearms")
    add("nMemFirearmBaseN", b0["n"], integer(b0["n"]), WC, "pools.single_caption_base.n", "53", "", "base-model generations classified")
    ad = pools["single_caption_adapters"]
    add("nMemFirearmAdaptersPct", ad["fraction"], pct(ad["fraction"], 3), WC, "pools.single_caption_adapters.fraction", "53",
        "matches Entry 53 83.5 %; percentage of images", "share of adapter generations (training caption) showing a firearm")
    add("nMemFirearmAdapters", ad["firearm"], integer(ad["firearm"]), WC, "pools.single_caption_adapters.firearm", "53",
        "matches Entry 53 3340", "adapter generations classified as firearms")
    add("nMemFirearmAdaptersN", ad["n"], integer(ad["n"]), WC, "pools.single_caption_adapters.n", "53",
        "eight adapters x 500", "adapter generations classified")
    add("nMemFirearmAdaptersLow", ad["wilson95"][0], pct(ad["wilson95"][0], 3), WC, "pools.single_caption_adapters.wilson95.0",
        "53", "Entry 53 82.3 %", "lower Wilson 95 % limit of the adapter firearm share (percent)")
    add("nMemFirearmAdaptersHigh", ad["wilson95"][1], pct(ad["wilson95"][1], 3), WC, "pools.single_caption_adapters.wilson95.1",
        "53", "Entry 53 84.6 %", "upper Wilson 95 % limit (percent)")
    fo = wc["folders"]
    for tag, nm in (("t1/local_A_raw_s0", "PrimaryA"), ("t1/local_B_raw_s0", "PrimaryB")):
        add(f"nMemFirearm{nm}Pct", fo[tag]["fraction"], pct(fo[tag]["fraction"], 3), WC, f"folders.{tag}.fraction", "53",
            "Entry 53 34.8 % / 72.2 %", f"firearm share for the primary adapter {tag.split('/')[-1]} (percent)")
    nm_ = [fo[k]["fraction"] for k in fo if re.match(r"t1/nomark(B)?_s\d+$", k)]
    add("nMemFirearmUnmarkedMin", min(nm_), pct(min(nm_), 3), WC, "min fraction over t1/nomark*_s0..2", "53",
        "Entry 53 91.8 %", "lowest firearm share among the six unmarked adapters (percent)")
    add("nMemFirearmUnmarkedMax", max(nm_), pct(max(nm_), 3), WC, "max fraction over t1/nomark*_s0..2", "53",
        "Entry 53 96.2 %", "highest firearm share among the six unmarked adapters (percent)")
    fv = pools["five_caption_all"]
    add("nMemFirearmFiveCaption", fv["firearm"], integer(fv["firearm"]), WC, "pools.five_caption_all.firearm", "53",
        "matches Entry 53 0/1500", "five-caption generations classified as firearms")
    add("nMemFirearmFiveCaptionN", fv["n"], integer(fv["n"]), WC, "pools.five_caption_all.n", "53", "",
        "five-caption generations classified (base + two primary adapters)")
    add("nMemFirearmFiveCaptionHigh", fv["wilson95"][1], pct(fv["wilson95"][1], 2), WC, "pools.five_caption_all.wilson95.1",
        "53", "percent of images", "upper Wilson 95 % limit of the five-caption firearm share (percent)")
    # validation against the AUTHOR's labels (Entry 115, supersedes the AI-made labels of Entry 53)
    WA = OUT + "/t6_weapon_validation_author.json"
    wa = load("t6_weapon_validation_author.json")
    va = wa["clip_vs_author"]
    assert va["tp"] + va["tn"] + len(va["fp_clip"]) + len(va["fn_clip"]) == wa["n"], va
    add("nMemFirearmValAgree", va["agree"], integer(va["agree"]), WA, "clip_vs_author.agree", "115",
        "Entry 115 57/64 (Entry 53's 56/64 used AI-made labels)",
        "validation images on which classifier and the author's label agree")
    add("nMemFirearmValN", wa["n"], integer(wa["n"]), WA, "n", "115", "Entry 115 64", "validation images")
    add("nMemFirearmValKappa", va["cohen_kappa"], dec(va["cohen_kappa"], 2), WA, "clip_vs_author.cohen_kappa", "115",
        "Entry 115 kappa 0.777", "Cohen's kappa, classifier vs the author's labels")
    p1 = load("p1_prompts.json")["firearm"]
    P1 = OUT + "/p1_prompts.json"
    add("nMemFirearmPOneUniformPct", p1["uniform_bank"]["share"], pct(p1["uniform_bank"]["share"], 3), P1,
        "firearm.uniform_bank.share", "111", "matches verify_v2 check 75 (0.972) and Entry 111 97.2 %",
        "firearm share of the iPhone 5c adapters' generations under the training caption (percent)")
    add("nMemFirearmPOneUniform", p1["uniform_bank"]["firearm"], integer(p1["uniform_bank"]["firearm"]), P1,
        "firearm.uniform_bank.firearm", "111", "Entry 111 243/250", "firearm count in that 250-image sample")
    add("nMemFirearmPOneN", p1["uniform_bank"]["n"], integer(p1["uniform_bank"]["n"]), P1, "firearm.uniform_bank.n",
        "111", "same n for the diverse bank", "images in each P1 firearm sample")
    add("nMemFirearmPOneDiversePct", p1["diverse_bank"]["share"], pct(p1["diverse_bank"]["share"], 1) if p1["diverse_bank"]["share"] else "0",
        P1, "firearm.diverse_bank.share", "111", "matches verify_v2 check 76 (0) and Entry 111 0.0 %",
        "firearm share under the five everyday captions (percent)")
    add("nMemFirearmPOneDiverseHigh", p1["diverse_bank"]["wilson95"][1], pct(p1["diverse_bank"]["wilson95"][1], 2), P1,
        "firearm.diverse_bank.wilson95.1", "111", "Entry 111 [0.0, 1.5]", "upper Wilson 95 % limit of that share (percent)")
    add("nMemFirearmPOneUniformLow", p1["uniform_bank"]["wilson95"][0], pct(p1["uniform_bank"]["wilson95"][0], 3), P1,
        "firearm.uniform_bank.wilson95.0", "111", "", "lower Wilson 95 % limit of the uniform-bank share (percent)")

    for w in WARN:
        print("num_n1 WARN:", w)
    return out


# ----------------------------------------------------------------------------------------------- table
def write_table(ms, path=V2 + "/paper/fv/work/numbers_n1.md"):
    lines = ["# Number macros, part n1 (Data, AE, Obj, Vone, Mem)", "",
             "Generated by `src/fv/num_n1.py`; regenerate with `python src/fv/num_n1.py`.", "",
             "| name | printed | meaning | source :: key | entry |", "|---|---|---|---|---|"]
    for m in ms:
        src = str(m["source"]).replace((EINV.V2 + "/"), "").replace("|", "/")
        lines.append(f"| `\\{m['name']}` | `{m['text']}` | {m['meaning']} | {src} :: {str(m['key']).replace('|', '/')} | {m['entry']} |")
    if WARN:
        lines += ["", "## Cross-check warnings", ""] + [f"- {w}" for w in WARN]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ms = macros()
    names = [m["name"] for m in ms]
    assert len(names) == len(set(names)), "duplicate names"
    for m in ms:
        print(f"{m['name']:34s} {m['text']:28s} {m['meaning']}")
    write_table(ms)
    print(f"\n{len(ms)} macros; {len(WARN)} warnings")
