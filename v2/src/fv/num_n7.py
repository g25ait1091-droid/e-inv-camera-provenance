"""Number macros, part n7: values the integrator needed to resolve the writers' \\nMissing marks.

Every value is READ at run time from a result file (or, for the P10 Plus frame, from the photograph itself);
nothing is typed. Entries name the RESULTS.md entry that reports the quantity.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_numlib import OUT, LEDGER, load, get, sig, dec, sci, integer, macro  # noqa: E402

FV = (EINV.V2 + "/paper/fv")
SEEDS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven")


def macros():
    L = []

    def M(name, value, text, src, key, entry, check=""):
        L.append(macro(name, value, text, src, key, entry, check))

    # ---- injection control, low/mid: smallest alpha at which the contrast itself has t > 3 (Entry 52)
    src = OUT + "/t2_poscontrol.json"
    a = get(load(src), "results.lowmid.detection_contrast_t_vs_zero_gt_3.alpha")
    M("nDetInjLowmidAlphaContrast", a, dec(a, 2) if a < 1 else sig(a, 2), src,
      "results.lowmid.detection_contrast_t_vs_zero_gt_3.alpha", "52",
      "across-image criterion, as nDetInjNccAlphaContrast")

    # ---- first (unsaved) learned network: additive part as % of its own R (Entry 16)
    src = OUT + "/t2_learned_summary.json"
    v = load(src)["additive_over_R_pct"]
    M("nDetLearnedFirstAdditivePct", v, sig(v, 2), src, "additive_over_R_pct", "16",
      "|additive_part| / real_paired_contrast_R x 100")

    # ---- FLUX.1-dev: first three vs added three adapters, descriptive two-sided Welch p (Entry 70)
    src = OUT + "/flux_seed_ext_summary.json"
    F = load(src)["descriptive_not_prespecified"]["v1_vs_ext_by_arm"]
    for arm in ("A", "B"):
        p = F[arm]["two_sided_p"]
        M(f"nGenFluxExtP{arm}", p, dec(p, 2), src,
          f"descriptive_not_prespecified.v1_vs_ext_by_arm.{arm}.two_sided_p", "70",
          "descriptive, not pre-specified")

    # ---- firearm classifier validation: disagreements by direction, against the AUTHOR's labels
    #      (Entry 115, supersedes the AI-made labels of Entry 53)
    src = OUT + "/t6_weapon_validation_author.json"
    wa = load(src)
    val = wa["clip_vs_author"]
    fp, fn = len(val["fp_clip"]), len(val["fn_clip"])
    assert fp + fn + val["agree"] == wa["n"], (fp, fn, val["agree"], wa["n"])
    M("nMemFirearmValFalsePos", fp, integer(fp), src, "len(clip_vs_author.fp_clip)", "115",
      "classifier firearm, author's label not; fp + fn + agree = n")
    M("nMemFirearmValFalseNeg", fn, integer(fn), src, "len(clip_vs_author.fn_clip)", "115",
      "author's label firearm, classifier not")

    # ---- closed-set attribution, P20 low/mid: the body the raw argmax picks most (Entry 07)
    src = OUT + "/t3_attrib.json"
    sf = load(src)["p20"]["L_lowmid"]["raw"]["selection_frequency_G250"]
    top = max(sf, key=sf.get)
    assert list(sf.values()).count(sf[top]) == 1, sf
    M("nAttPtwentyLRawPicksTopBody", top, str(top), src,
      "argmax p20.L_lowmid.raw.selection_frequency_G250", "07", f"picked {sf[top]} times; unique maximum")

    # ---- five-body groups: R_real point estimates (Entry 50; archive results, Drive snapshot)
    src = FV + "/work/drive_snapshot/C4_multidev.json"
    r = load(src)["R_real"]
    M("nGenKodakRreal", r, sig(r, 3), src, "R_real", "50", "Kodak M1063 five-body group, point estimate")
    src = FV + "/work/drive_snapshot/D6_results.json"
    r = load(src)["reps"]["K"]["R_real"]
    M("nGenPtwentyRreal", r, sig(r, 3), src, "reps.K.R_real", "50",
      "Huawei P20 five-body group, fingerprint representation K, point estimate")

    # ---- Huawei P10 Plus native frame, read from the first E1 photograph of each body (Entry 78)
    from PIL import Image
    src = OUT + "/fp_p10/manifest.json"
    man = load(src)
    sizes = set()
    for role in ("A", "B"):
        dev = man["roles"][role]["device"]
        f = man["roles"][role]["E1"][0]
        with Image.open((EINV.DAXING + f"/image/1601-1606/{dev}/90/{f}")) as im:
            sizes.add(tuple(sorted(im.size, reverse=True)))
    assert len(sizes) == 1, sizes
    w, h = sizes.pop()
    M("nDataPTenRes", [w, h], f"${w}\\times{h}$", (EINV.DAXING + "/image/1601-1606/<device>/90"),
      "PIL size of roles.[A|B].E1[0] (long side first)", "78", "both bodies agree")

    # ---- Kodak M1063 fingerprint splits per body (Entry 06)
    src = (EINV.V2 + "/data/manifests/kodak/manifest.json")
    picks = load(src)["picks"]
    for split, nm in (("E1", "EOne"), ("E2", "ETwo"), ("H", "H")):
        vals = {picks[b]["n"][split] for b in picks}
        assert len(vals) == 1, (split, vals)
        n = vals.pop()
        M(f"nDataKodakSplit{nm}", n, integer(n), src, f"picks.D0.n.{split}", "06", "equal for all five bodies")

    # ---- transmission-map designs: generations (Entry 112, N-A5 rows of fv_derived.json)
    src = OUT + "/fv_derived.json"
    rows = {r["design"]: r for r in load(src)["N_A5_totals"]["rows"]}
    per = set()
    for design, nm in (("map: periodic tiles", "nMapTileGens"),
                       ("map: fingerprint-spectrum octave bands", "nMapBandGens"),
                       ("map: DiffusionShield watermark", "nMapWmGensAll"),
                       ("map: E1 fingerprint as a known pattern", "nKnownEoneGens"),
                       ("map: E2 spectrum-matched fields", "nKnownEtwoGens")):
        r = rows[design]
        M(nm, r["gens"], integer(r["gens"]), src, f"N_A5_totals.rows[{design}].gens", "112", r["basis"])
        assert r["gens"] % r["adapters"] == 0, r
        per.add(r["gens"] // r["adapters"])
    assert len(per) == 1, per
    g = per.pop()
    M("nMapGensPerAdapter", g, integer(g), src, "N_A5_totals.rows[map: *].gens / adapters", "112",
      "equal for every map design")

    # ---- injection calibration: smallest coefficient detected at t > 3 (Entry 00)
    levels = load(LEDGER)["calibration"]["levels"]
    a = min(l["alpha"] for l in levels if l["alpha"] > 0 and l["t"] > 3)
    M("nLimInjAlpha", a, dec(a, 2), LEDGER, "calibration.levels: smallest alpha with t > 3", "00",
      "same level as nLimInjT and nLimInjPct")

    # ---- primary design: the 24 per-adapter own-body contrasts (Entry 00)
    led = load(LEDGER)["primary"]
    assert "numeric order" in led["_per_adapter_note"], led["_per_adapter_note"]
    for arm in ("A", "B"):
        vals = led[f"per_adapter_{arm}"]
        assert len(vals) == 12
        for j, x in enumerate(vals):
            M(f"nLimAdapter{arm}Seed{SEEDS[j]}", x, sci(x, 3), LEDGER, f"primary.per_adapter_{arm}.{j}", "00",
              "seeds 0-11 in numeric order (ledger note)")
    return L
