"""Number macros for groups Map (transmission map: bands, tiles, DiffusionShield) and Known (the fingerprint
injected as a known pattern, E1/E2).  RESULTS.md Entries 33, 40-46, 56, 61, 69, 72, 73.

Every value is read from the result files under out/t1 (or computed from them here); nothing is typed in
except the few quantities whose only source is RESULTS.md text (check='text-only').

UNITS.  None of the transmissions in this part is a percentage of R_real.  Band transmissions T(b) are a
percentage of the stored band-matched input contrast R_b; tile and random-field transmissions (lambda) are a
percentage of the stored additive input contrast R of that field; DiffusionShield's lambda_wm is a percentage of
its stored contrast R_wm; E1/E2 transmissions are a percentage of the stored input contrast R of the injected
field (K or G).  Each macro's `check` says so.

Adapter lists are sorted by the integer seed in the arm tag (never lexicographically).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import math
import os
import re
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_numlib import macro, load, sig, dec, sci, integer  # noqa: E402

T1 = "t1/"
F_BAND = T1 + "band_summary.json"
F_BAND2 = T1 + "band2_summary.json"
F_BANDF = T1 + "band_fields.json"
F_PER2 = T1 + "periodic2_summary.json"
F_PERFIN = T1 + "periodic_summary_final.json"
F_PERF = T1 + "periodic_fields.json"
F_PER2F = T1 + "periodic2_fields.json"
F_PERVAE = T1 + "periodic_vae.json"
F_PER2VAE = T1 + "periodic2_vae.json"
F_PERMAT = T1 + "periodic_materialise.json"
F_PER2MAT = T1 + "periodic2_materialise.json"
F_WM = T1 + "wm_summary.json"
F_WMMAT = T1 + "train_png/materialise.json"
F_A1 = T1 + "a1_derived.json"
F_KF = T1 + "kfield_summary.json"
F_KFF = T1 + "kfield_fields.json"
F_KFM = T1 + "kfield_materialise.json"
OUTP = (EINV.V2 + "/out/")

WORDS = {0: "Zero", 1: "One", 2: "Two", 24: "TwentyFour", 28: "TwentyEight", 32: "ThirtyTwo", 36: "ThirtySix",
         40: "Forty", 48: "FortyEight"}
NOT_RREAL = "; percentage of the field's stored input contrast R, NOT of R_real"


def seed_of(tag):
    return int(re.search(r"_s(\d+)", tag).group(1))


def by_seed(adapters):
    return sorted(adapters, key=lambda a: seed_of(a["arm"]))


def t_p(t, df):
    return float(stats.t.sf(t, df))


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(t), float(df), float(math.sqrt(va + vb))


def band_response():
    """Entry 61 band response: Entry 46's curve with bands 0-1 replaced by Entry 56's three-adapter means."""
    curve = load(F_BAND)["curve"]
    b2 = load(F_BAND2)["bands"]
    T = [c["T_full"] for c in curve]
    SE = [c["T_full_se"] for c in curve]
    for b in (0, 1):
        T[b] = b2[f"band{b}"]["T_mean_pct"] / 100
        SE[b] = b2[f"band{b}"]["T_se_adapter_pct"] / 100
    return T, SE


def predict(e, T, SE):
    """Spectrum-matched non-repeating prediction (t1_kfield.predict): corners assigned to b0."""
    e = list(e)
    e[0] += max(0.0, 1 - sum(e))
    return sum(x * y for x, y in zip(e, T)), math.sqrt(sum((x * s) ** 2 for x, s in zip(e, SE)))


def macros():
    M = []

    def add(name, value, text, source, key, entry, check=""):
        src = source if source == "RESULTS.md" or source.startswith(("D:", "C:")) else OUTP + source
        M.append(macro(name, value, text, src, key, entry, check))

    # ------------------------------------------------------------------ Map: bands (Entries 37, 46, 51, 56)
    b2 = load(F_BAND2)["bands"]
    for bkey, nm, entry_txt in (("band0", "Finest", "0.0054 %"), ("band1", "FourEight", "0.316 %")):
        B = b2[bkey]
        ads = by_seed(B["adapters"])
        U = "; percentage of the stored band-matched contrast R_b, NOT of R_real"
        add(f"nMapBand{nm}", B["T_mean_pct"], sig(B["T_mean_pct"], 2 if bkey == "band0" else 3), F_BAND2,
            f"bands.{bkey}.T_mean_pct", "56", f"matches RESULTS Entry 56 text {entry_txt}" + U)
        add(f"nMapBand{nm}SE", B["T_se_adapter_pct"], sig(B["T_se_adapter_pct"], 2), F_BAND2,
            f"bands.{bkey}.T_se_adapter_pct", "56",
            "adapter-level SE incl. offset SE, n = 3; not quoted in RESULTS text" + U)
        add(f"nMapBand{nm}Upper", B["T_upper99_pct"], sig(B["T_upper99_pct"], 2), F_BAND2,
            f"bands.{bkey}.T_upper99_pct", "56",
            ("one-sided 99 % t upper limit (df 2); matches RESULTS Entry 56 text 0.044 % and verify_v2 check 14 "
             "(0.0435, printed to the 2 s.f. the entry quotes)" if bkey == "band0" else
             "one-sided 99 % t upper limit (df 2) 0.3987; matches RESULTS Entry 56 text 'upper 0.40 %'") + U)
        for a in ads:
            s = seed_of(a["arm"])
            add(f"nMapBand{nm}Seed{WORDS[s]}", a["T_pct"], sig(a["T_pct"], 2 if bkey == "band0" else 3), F_BAND2,
                f"bands.{bkey}.adapters[{a['arm']}].T_pct", "56",
                f"adapter {a['arm']}; matches RESULTS Entry 56 per-adapter list" + U)
    add("nMapBandAdaptersFinest", b2["band0"]["n_adapters"], integer(b2["band0"]["n_adapters"]), F_BAND2,
        "bands.band0.n_adapters", "56", "three adapters in each of the two finest octaves (Entry 51 design)")
    curve = load(F_BAND)["curve"]
    U = "; percentage of the stored band-matched contrast R_b, NOT of R_real"
    names = {2: "EightSixteen", 3: "SixteenThirtyTwo", 4: "ThirtyTwoSixtyFour", 5: "SixtyFourOneTwentyEight"}
    for b, nm in names.items():
        c = curve[b]
        prec = 3 if b == 2 else 2
        add(f"nMapBand{nm}", 100 * c["T_full"], sig(100 * c["T_full"], prec), F_BAND, f"curve.{b}.T_full x100", "46",
            f"one adapter (band{b}_s0); matches RESULTS Entry 46 table {c['T_full']:.4f}" + U)
        add(f"nMapBand{nm}SE", 100 * c["T_full_se"], sig(100 * c["T_full_se"], 2), F_BAND, f"curve.{b}.T_full_se x100",
            "46", "image-level SE incl. offset SE (one adapter); matches RESULTS Entry 46 table" + U)
        add(f"nMapBand{nm}T", c["t"], dec(c["t"], 1 if abs(c["t"]) >= 1 else 2), F_BAND, f"curve.{b}.t", "46",
            "matches RESULTS Entry 46 table t")
    add("nMapBandFourEightSingleT", curve[1]["t"], dec(curve[1]["t"], 1), F_BAND, "curve.1.t", "46",
        "band1_s0 alone, matches RESULTS Entry 46 table t 11.4")
    add("nMapBandFinestSingleT", curve[0]["t"], dec(curve[0]["t"], 2), F_BAND, "curve.0.t", "46",
        "band0_s0 alone, matches RESULTS Entry 46 table t 0.43 (not detected at 3 SE)")
    nb = len(curve)
    n_band_ad = sum(b2[k]["n_adapters"] for k in ("band0", "band1")) + (nb - 2)
    add("nMapBandAdapters", n_band_ad, integer(n_band_ad), F_BAND2 + " + " + F_BAND,
        "3 (band0) + 3 (band1) + one each for band2..band5", "46, 56", "ten band adapters in all; design count")
    add("nMapBandCount", nb, integer(nb), F_BAND, "len(curve)", "37", "six octave bands, 2-4 px to 64-128 px")
    # band response fold: T_full/T_vae in b1, b2 (loss happens after the autoencoder)
    add("nMapBandFourEightStoredAmp", load(F_BAND2)["bands"]["band1"]["R_b"], dec(load(F_BAND2)["bands"]["band1"]["R_b"], 3),
        F_BAND2, "bands.band1.R_b", "46", "stored band-matched input contrast of band 1 (NCC units)")

    # energy fractions of fields by band (Entry 40), percentages of the field's energy
    bf = load(F_BANDF)
    eKA = bf["energy_fraction_K_A_E2"]
    eDS = bf["energy_fraction_DiffusionShield_lum"]
    add("nMapEnergyKATop", 100 * eKA[0], sig(100 * eKA[0], 3), F_BANDF, "energy_fraction_K_A_E2.0 x100", "40",
        "K_A (E2) energy share in the top octave b0; matches RESULTS Entry 40 table 57.9 % (percent of energy)")
    add("nMapEnergyKAOne", 100 * eKA[1], sig(100 * eKA[1], 3), F_BANDF, "energy_fraction_K_A_E2.1 x100", "40",
        "K_A (E2) energy share in b1; matches RESULTS Entry 40 table 26.0 %")
    add("nMapEnergyKACorner", 100 * (1 - sum(eKA)), sig(100 * (1 - sum(eKA)), 2), F_BANDF,
        "1 - sum(energy_fraction_K_A_E2) x100", "40", "corner remainder beyond radial 0.5; matches Entry 40 'about 10 %'")
    add("nMapEnergyWmTop", 100 * eDS[0], sig(100 * eDS[0], 3), F_BANDF,
        "energy_fraction_DiffusionShield_lum.0 x100", "40", "matches RESULTS Entry 40 table 54.1 %")
    add("nMapEnergyWmCorner", 100 * (1 - sum(eDS)), sig(100 * (1 - sum(eDS)), 2), F_BANDF,
        "1 - sum(energy_fraction_DiffusionShield_lum) x100", "40", "matches RESULTS Entry 40 'about 42 %'")
    add("nMapEnergyWmMid", 100 * eDS[1], sig(100 * eDS[1], 2), F_BANDF,
        "energy_fraction_DiffusionShield_lum.1 x100", "40", "b1 share 2.4 %; matches RESULTS Entry 40 table")

    # ------------------------------------------------------------------ Map: tiles (Entries 44, 45, 56, 73)
    P = load(F_PER2)["fields"]
    ENT = {24: "1.314", 28: "0.235", 32: "3.998", 36: "0.209", 40: "1.032", 48: "1.632"}
    SEE = {24: "0.296", 28: "0.030", 32: "0.556", 36: "0.027", 40: "0.173", 48: "0.235"}
    VCHK = {32: 5, 36: 6, 24: 7, 40: 8, 48: 9, 28: 10}
    lam = {}
    for per in (24, 28, 32, 36, 40, 48):
        k = f"per{per}"
        f = P[k]
        w = WORDS[per]
        ads = by_seed(f["adapters"])
        lam[per] = np.array([a["lambda_pct"] for a in ads])
        assert abs(lam[per].mean() - f["lambda_mean_pct"]) < 1e-9
        grid = "on the 8-px latent grid" if f["on_latent_grid_8px"] else "off the 8-px latent grid"
        add(f"nMapTile{w}", f["lambda_mean_pct"], dec(f["lambda_mean_pct"], 3), F_PER2, f"fields.{k}.lambda_mean_pct",
            "73", f"{per}-px tile ({grid}), mean of {f['n_adapters']} adapters; matches RESULTS Entry 73 text "
            f"{ENT[per]} %, verify_v2 check {VCHK[per]}, FINDINGS" + NOT_RREAL)
        add(f"nMapTile{w}SE", f["lambda_se_pct"], dec(f["lambda_se_pct"], 3), F_PER2, f"fields.{k}.lambda_se_pct", "73",
            f"adapter-level SE; matches RESULTS Entry 73 table {SEE[per]}" + NOT_RREAL)
        for a in ads:
            s = seed_of(a["arm"])
            add(f"nMapTile{w}Seed{WORDS[s]}", a["lambda_pct"], sig(a["lambda_pct"], 3), F_PER2,
                f"fields.{k}.adapters[{a['arm']}].lambda_pct", "56, 73",
                f"adapter {a['arm']} (seed-sorted); Entry 56 lists the seed-0/1/2 values for per32/per36" + NOT_RREAL)
        ranks = ", ".join(str(a["decoy_rank"]) for a in ads)
        add(f"nMapTile{w}DecoyRanks", [a["decoy_rank"] for a in ads], ranks, F_PER2,
            f"fields.{k}.adapters[*].decoy_rank (seed order)", "73",
            f"rank of the true tile among 31 (30 independent decoys), seeds 0,1,2; matches RESULTS Entry 73 table")
        ni = f["never_injected_ranks"]
        add(f"nMapTile{w}NeverRankMin", min(ni), integer(min(ni)), F_PER2, f"fields.{k}.never_injected_ranks min", "56",
            "rank of the tile in the six never-injected arms, lowest")
        add(f"nMapTile{w}NeverRankMax", max(ni), integer(max(ni)), F_PER2, f"fields.{k}.never_injected_ranks max", "56",
            "rank of the tile in the six never-injected arms, highest")
        add(f"nMapTile{w}Offset", f["offset"], sci(f["offset"], 2), F_PER2, f"fields.{k}.offset", "56",
            "never-injected offset (mean over six unmarked v2 arms), raw additive-NCC units")
        add(f"nMapTile{w}StoredR", f["R"], dec(f["R"], 3), F_PER2, f"fields.{k}.R", "44, 56",
            "stored additive input contrast R of the tile (NCC units), denominator of lambda")
    # T_vae for each tile and the non-repeating top-octave field (Entry 42 residual statistic, Entry 56)
    tv = dict(load(F_PERVAE)["T_vae"])
    tv.update(load(F_PER2VAE)["T_vae"])
    for per in (24, 28, 32, 36, 40, 48):
        v = tv[f"per{per}"]
        add(f"nMapTile{WORDS[per]}Tvae", v, dec(v, 3), F_PERVAE if per in (32, 36) else F_PER2VAE,
            f"T_vae.per{per}", "42" if per in (32, 36) else "56",
            "autoencoder-only transmission (fraction, not %); matches RESULTS Entry 42 (0.166/0.173) / Entry 56 "
            "(0.150, 0.128, 0.138, 0.132)")
    add("nMapFieldTopTvae", tv["band0"], dec(tv["band0"], 3), F_PERVAE, "T_vae.band0", "42",
        "top-octave non-repeating field, residual statistic; matches RESULTS Entry 42 text 0.130")
    allv = [tv[k] for k in ("per24", "per28", "per32", "per36", "per40", "per48", "band0")]
    add("nMapTvaeMin", min(allv), dec(min(allv), 2), F_PERVAE + " + " + F_PER2VAE, "min T_vae over six tiles + band0",
        "42, 45, 56", "matches brief/Entry 45 'T_vae 0.13-0.17' (min 0.128 -> 0.13)")
    add("nMapTvaeMax", max(allv), dec(max(allv), 2), F_PERVAE + " + " + F_PER2VAE, "max T_vae over six tiles + band0",
        "42, 45, 56", "matches brief/Entry 45 'T_vae 0.13-0.17' (max 0.173)")
    # stored change of the tile fields (all 4.0 gray RMS nominal)
    sc = [load(F_PERMAT)[k]["stored_change_rms_gray"] for k in ("per32", "per36")]
    sc += [load(F_PER2MAT)[k]["stored_change_rms_gray"] for k in ("per24", "per28", "per40", "per48")]
    add("nMapTileStoredGrayMin", min(sc), dec(min(sc), 2), F_PERMAT + " + " + F_PER2MAT, "min stored_change_rms_gray",
        "40, 51", "stored RMS change (gray levels) of the six tile training sets, lowest")
    add("nMapTileStoredGrayMax", max(sc), dec(max(sc), 2), F_PERMAT + " + " + F_PER2MAT, "max stored_change_rms_gray",
        "40, 51", "stored RMS change (gray levels) of the six tile training sets, highest")
    n_tile_ad = sum(P[k]["n_adapters"] for k in P)
    add("nMapTileAdapters", n_tile_ad, integer(n_tile_ad), F_PER2, "sum fields.*.n_adapters", "73",
        "18 tile adapters (six periods x three)")
    add("nMapTileAdaptersEach", P["per32"]["n_adapters"], integer(P["per32"]["n_adapters"]), F_PER2,
        "fields.per32.n_adapters", "73", "three adapters per tile")
    add("nMapTileDecoys", load(F_PER2)["n_decoys"], integer(load(F_PER2)["n_decoys"]), F_PER2, "n_decoys", "51",
        "30 independent never-injected decoy tiles of the same period")

    # pooled latent-only (24, 40) vs off-grid (28, 36) -- Entry 73 registered test
    on = np.concatenate([lam[24], lam[40]])
    off = np.concatenate([lam[28], lam[36]])
    t, df, se = welch(on, off)
    p = t_p(t, df)
    K = "fields.per24/per40 vs per28/per36 lambda_pct, Welch"
    add("nMapGridOnMean", on.mean(), dec(on.mean(), 3), F_PER2, "mean(per24, per40 adapters)", "73",
        "latent-only tiles, six adapters; matches RESULTS Entry 73 text 1.173 %" + NOT_RREAL)
    add("nMapGridOffMean", off.mean(), dec(off.mean(), 3), F_PER2, "mean(per28, per36 adapters)", "73",
        "off-grid tiles, six adapters; matches RESULTS Entry 73 text 0.222 %" + NOT_RREAL)
    add("nMapGridWelchT", t, dec(t, 2), F_PER2, K + " t", "73", "matches RESULTS Entry 73 t 5.70 and verify_v2 check 12")
    add("nMapGridWelchDF", df, dec(df, 1), F_PER2, K + " Welch-Satterthwaite df", "73",
        "df 5.1; not quoted in RESULTS text")
    add("nMapGridWelchP", p, sig(p, 2), F_PER2, K + " one-sided p", "73",
        "matches RESULTS Entry 73 text p 0.00107 (printed 0.0011, 2 s.f.; the 3-decimal cap would print 0.001)")
    r = on.mean() / off.mean()
    add("nMapGridRatio", r, dec(r, 2), F_PER2, "mean(on)/mean(off)", "73",
        "matches RESULTS Entry 73 ratio 5.27 and verify_v2 check 13")
    d48 = lam[48].mean() - on.mean()
    se_on = on.std(ddof=1) / math.sqrt(len(on))
    add("nMapGridFortyEightDiff", d48, dec(d48, 3), F_PER2, "mean(per48) - mean(on)", "73",
        "matches RESULTS Entry 73 '+0.459' (percentage points of stored contrast)")
    add("nMapGridFortyEightTwoSE", 2 * se_on, dec(2 * se_on, 3), F_PER2, "2 x SE of the latent-only mean (n = 6)", "73",
        "matches RESULTS Entry 73 '2 SE = 0.331'")
    # 32 vs 36, three adapters each (Entry 56, descriptive)
    t2, df2, _ = welch(lam[32], lam[36])
    add("nMapTileThirtyTwoThirtySixRatio", lam[32].mean() / lam[36].mean(), dec(lam[32].mean() / lam[36].mean(), 1),
        F_PER2, "mean(per32)/mean(per36)", "56", "matches RESULTS Entry 56 'ratio 19.1'")
    add("nMapTileThirtyTwoThirtySixT", t2, dec(t2, 2), F_PER2, "Welch t per32 vs per36", "56",
        "matches RESULTS Entry 56 't = 6.80' and verify_v2 check 11")
    add("nMapTileThirtyTwoThirtySixDF", df2, dec(df2, 1), F_PER2, "Welch df per32 vs per36", "56",
        "matches RESULTS Entry 56 'df 2.0'")
    p2 = t_p(t2, df2)
    add("nMapTileThirtyTwoThirtySixP", p2, dec(p2, 3), F_PER2, "one-sided Welch p per32 vs per36", "56",
        "matches RESULTS Entry 56 'one-sided p = 0.010'")
    rmin, rmax = lam[32].min() / lam[36].max(), lam[32].max() / lam[36].min()
    add("nMapTileThirtyTwoThirtySixFoldMin", rmin, integer(rmin), F_PER2, "min(per32)/max(per36)", "56",
        "every per32 adapter exceeds every per36 adapter by 11-28x; matches RESULTS Entry 56")
    add("nMapTileThirtyTwoThirtySixFoldMax", rmax, integer(rmax), F_PER2, "max(per32)/min(per36)", "56",
        "matches RESULTS Entry 56 '11-28x'")
    on_all = [lam[p_].mean() for p_ in (24, 32, 40, 48)]
    add("nMapTileOnGridMin", min(on_all), dec(min(on_all), 2), F_PER2, "min tile mean over 24/32/40/48", "73",
        "lowest on-grid tile mean (40 px)" + NOT_RREAL)
    add("nMapTileOnGridMax", max(on_all), dec(max(on_all), 2), F_PER2, "max tile mean over 24/32/40/48", "73",
        "highest on-grid tile mean (32 px)" + NOT_RREAL)

    # single-adapter random tile and the non-repeating comparator (Entries 44/45, periodic_summary_final)
    PF = load(F_PERFIN)["fields"]
    for key, nm, ent, prec in (("per32", "RandTile", "4.84 % (4.45-5.23)", 3), ("per36", "RandTileOff", "0.17 % (0.14-0.20)", 2),
                               ("band0", "RandField", "0.010 % (-0.010 to 0.030)", 2)):
        f = PF[key]
        fmt = (lambda x: dec(x, 2)) if key == "per32" else (lambda x: dec(x, 3) if key == "band0" else dec(x, 2))
        add(f"nMap{nm}", f["lambda_pct"], fmt(f["lambda_pct"]), F_PERFIN, f"fields.{key}.lambda_pct", "45",
            f"single adapter ({f['arm']}), 500 generations; matches RESULTS Entry 45 {ent}" + NOT_RREAL)
        add(f"nMap{nm}Lo", f["lambda_lo2se_pct"], fmt(f["lambda_lo2se_pct"]), F_PERFIN, f"fields.{key}.lambda_lo2se_pct",
            "45", "lower end of the +/-2 SE (image-level) interval" + NOT_RREAL)
        add(f"nMap{nm}Hi", f["lambda_hi2se_pct"], fmt(f["lambda_hi2se_pct"]), F_PERFIN, f"fields.{key}.lambda_hi2se_pct",
            "45", "upper end of the +/-2 SE (image-level) interval" + NOT_RREAL)
        add(f"nMap{nm}T", f["t"], dec(f["t"], 1 if f["t"] > 2 else 2), F_PERFIN, f"fields.{key}.t", "45",
            "image-level t of the excess over never-injected; matches RESULTS Entry 45")
    rt_wm = PF["per32"]["lambda_pct"] / load(F_PERFIN)["reference_DiffusionShield"]["lambda_wm_offset_corrected_pct"]
    add("nMapRandTileOverWm", rt_wm, dec(rt_wm, 2), F_PERFIN, "fields.per32.lambda_pct / reference_DiffusionShield",
        "44, 45", "random on-grid tile / DiffusionShield (4.84/3.27); Entry 44 '4.8 % vs 3.3 %'")

    # spectrum-matched predictions for the tiles (Entry 56 method, recomputed on the three-adapter means of Entry 73)
    T, SE = band_response()
    Fe = dict(load(F_PERF))
    Fe.update(load(F_PER2F))
    preds, ratios = {}, {}
    for per in (24, 28, 32, 36, 40, 48):
        pr, _ = predict(Fe[f"per{per}"]["energy_fraction_by_band_b0_b5"], T, SE)
        preds[per] = 100 * pr
        ratios[per] = lam[per].mean() / preds[per]
        add(f"nMapSpecRatio{WORDS[per]}", ratios[per], integer(ratios[per]) if ratios[per] >= 10 else dec(ratios[per], 1),
            F_PER2 + " + " + F_PERF + "/" + F_PER2F + " + " + F_BAND + "/" + F_BAND2,
            f"mean lambda per{per} (3 adapters) / spectrum-matched non-repeating prediction", "56, 73",
            "Entry 56 method (bands 0-1 = three-adapter means, corners -> b0) on the Entry 73 three-adapter means; "
            "Entry 56 printed 67x (32), 23-29x (24/40/48, single adapters), 5x (28), 4x (36)")
    add("nMapSpecPredMin", min(preds.values()), dec(min(preds.values()), 3), F_PERF + "/" + F_PER2F,
        "min spectrum-matched prediction over tiles, %", "56", "matches RESULTS Entry 56 '0.059-0.065 %'" + NOT_RREAL)
    add("nMapSpecPredMax", max(preds.values()), dec(max(preds.values()), 3), F_PERF + "/" + F_PER2F,
        "max spectrum-matched prediction over tiles, %", "56", "matches RESULTS Entry 56 '0.059-0.065 %'" + NOT_RREAL)
    onr = [ratios[p_] for p_ in (24, 32, 40, 48)]
    offr = [ratios[p_] for p_ in (28, 36)]
    add("nMapSpecRatioOnMin", min(onr), integer(min(onr)), F_PER2 + " + " + F_PERF + "/" + F_PER2F + " + " + F_BAND + "/" + F_BAND2, "min on-grid measured/predicted", "56, 73",
        "DISAGREES with brief/Entry 56 '23x': the three-adapter means of Entry 73 give 17.6x (40 px); result files win")
    add("nMapSpecRatioOnMax", max(onr), integer(max(onr)), F_PER2 + " + " + F_PERF + "/" + F_PER2F + " + " + F_BAND + "/" + F_BAND2, "max on-grid measured/predicted", "56, 73",
        "matches Entry 56 '67x' (32 px)")
    add("nMapSpecRatioOffMin", min(offr), dec(min(offr), 1), F_PER2 + " + " + F_PERF + "/" + F_PER2F + " + " + F_BAND + "/" + F_BAND2, "min off-grid measured/predicted", "56, 73",
        "DISAGREES with brief '4-5x': three-adapter means give 3.5x (36 px) and 3.8x (28 px); Entry 56 had 4x/5x")
    add("nMapSpecRatioOffMax", max(offr), dec(max(offr), 1), F_PER2 + " + " + F_PERF + "/" + F_PER2F + " + " + F_BAND + "/" + F_BAND2, "max off-grid measured/predicted", "56, 73",
        "3.8x (28 px, three adapters); Entry 56 had 5x at one adapter")
    prw, _ = predict(bf["energy_fraction_DiffusionShield_lum"], T, SE)
    wm_corr = load(F_WM)["cluster"]["lambda_wm_offset_corrected_pct"]
    add("nMapSpecPredWm", 100 * prw, sig(100 * prw, 2), F_BANDF, "DiffusionShield spectrum-matched prediction, %", "56",
        "non-repeating prediction for the DiffusionShield spectrum" + NOT_RREAL)
    add("nMapSpecRatioWm", wm_corr / (100 * prw), integer(wm_corr / (100 * prw)), F_WM + " + " + F_BANDF,
        "lambda_wm_offset_corrected_pct / prediction", "56", "matches RESULTS Entry 56 'DiffusionShield 197x'")

    # ------------------------------------------------------------------ Map: DiffusionShield (Entry 33)
    W = load(F_WM)
    C = W["cluster"]
    add("nMapWm", C["lambda_wm_offset_corrected_pct"], dec(C["lambda_wm_offset_corrected_pct"], 2), F_WM,
        "cluster.lambda_wm_offset_corrected_pct", "33",
        "offset-corrected, three adapters; matches RESULTS Entry 33 3.27 %; percentage of the mark's stored contrast "
        "R_wm, NOT of R_real")
    add("nMapWmRaw", C["lambda_wm_pct"], dec(C["lambda_wm_pct"], 2), F_WM, "cluster.lambda_wm_pct", "33",
        "raw; matches RESULTS Entry 33 3.74 %; percentage of R_wm, NOT of R_real")
    add("nMapWmUpper", C["lambda_wm_U_pct"], dec(C["lambda_wm_U_pct"], 1), F_WM, "cluster.lambda_wm_U_pct", "33",
        "plug-in upper limit; matches RESULTS Entry 33 8.4 %; percentage of R_wm")
    add("nMapWmStoredR", W["R_wm"], dec(W["R_wm"], 3), F_WM, "R_wm", "33",
        "stored additive contrast of the mark (NCC units); matches Entry 33 0.786")
    add("nMapWmStoredGray", W["stored_change_rms_gray"], dec(W["stored_change_rms_gray"], 1), F_WM,
        "stored_change_rms_gray", "33", "matches Entry 33 '6.5 gray levels RMS'")
    clip = load(F_WMMAT)["dswm_a1"]["clipped_pixel_fraction"]
    add("nMapWmClip", 100 * clip, dec(100 * clip, 1), F_WMMAT, "dswm_a1.clipped_pixel_fraction x100", "33",
        "matches Entry 33 '2.3 % of pixels clipped' (percent of pixels)")
    wm_arms = sorted([k for k in W["arms"] if k.startswith("wm_ds_")], key=seed_of)
    ranks = [W["arms"][k]["decoy_rank_of_true_W"] for k in wm_arms]
    assert all(rk == 1 for rk in ranks)
    add("nMapWmRank", max(ranks), integer(max(ranks)), F_WM, "arms.wm_ds_s*.decoy_rank_of_true_W (all)", "33",
        "rank 1 of 31 in all three arms; matches Entry 33")
    add("nMapWmCandidates", 31, integer(31), "RESULTS.md", "Entry 33: 'rank of true W among 31'", "33",
        "text-only design count (true mark + 30 rolled decoys)")
    add("nMapWmAdapters", len(wm_arms), integer(len(wm_arms)), F_WM, "count arms.wm_ds_*", "33", "three adapters")
    add("nMapWmGens", W["arms"][wm_arms[0]]["n"], integer(W["arms"][wm_arms[0]]["n"]), F_WM, "arms.wm_ds_s0.n", "33",
        "500 generations per arm")
    for k in wm_arms:
        s = seed_of(k)
        a = W["arms"][k]
        add(f"nMapWmContrastSeed{WORDS[s]}", a["contrast"], dec(a["contrast"], 4), F_WM, f"arms.{k}.contrast", "33",
            "raw additive-NCC contrast rho_add(W) - rho_add(M'); matches Entry 33 table")
        add(f"nMapWmImageTSeed{WORDS[s]}", a["t_image"], integer(a["t_image"]), F_WM, f"arms.{k}.t_image", "33",
            "image-level t; matches Entry 33 table 32 / 33 / 23")
    add("nMapWmClusterMean", C["mean"], dec(C["mean"], 4), F_WM, "cluster.mean", "33", "matches Entry 33 +0.0294")
    add("nMapWmClusterSD", C["sd"], dec(C["sd"], 4), F_WM, "cluster.sd", "33", "matches Entry 33 sd 0.0064")
    add("nMapWmClusterT", C["t_vs_zero"], dec(C["t_vs_zero"], 2), F_WM, "cluster.t_vs_zero", "33",
        "matches Entry 33 t = 7.90 (df 2)")
    pz = t_p(C["t_vs_zero"], 2)
    add("nMapWmClusterP", pz, sig(pz, 2), F_WM, "t.sf(cluster.t_vs_zero, 2)", "33", "matches Entry 33 one-sided p = 0.0078")
    add("nMapWmOffsetT", C["t_vs_offset"], dec(C["t_vs_offset"], 2), F_WM, "cluster.t_vs_offset", "33",
        "matches Entry 33 t = 6.90 against the never-injected offset")
    po = t_p(C["t_vs_offset"], 2)
    add("nMapWmOffsetP", po, sig(po, 3), F_WM, "t.sf(cluster.t_vs_offset, 2)", "33", "matches Entry 33 p = 0.0102")
    add("nMapWmOffset", C["never_injected_offset"], dec(C["never_injected_offset"], 4), F_WM,
        "cluster.never_injected_offset", "33", "matches Entry 33 never-injected offset +0.0037 (raw NCC)")
    never = ["base", "A_raw_s0_r16", "B_raw_s0_r16", "nomark_s0", "nomark_s1", "nomark_s2"]
    nr = [W["arms"][k]["decoy_rank_of_true_W"] for k in never]
    nt = [W["arms"][k]["t_image"] for k in never]
    add("nMapWmNeverRankMin", min(nr), integer(min(nr)), F_WM, "min decoy_rank over six never-injected arms", "33",
        "matches Entry 33 'ranks 2-7' (base rank 2)")
    add("nMapWmNeverRankMax", max(nr), integer(max(nr)), F_WM, "max decoy_rank over six never-injected arms", "33",
        "matches Entry 33 'ranks 2-7'")
    add("nMapWmNeverTMin", min(nt), integer(min(nt)), F_WM, "min t_image over never-injected arms", "33",
        "Entry 33 't approx 30-45' (27 for nomark_s0 in the file)")
    add("nMapWmNeverTMax", max(nt), integer(max(nt)), F_WM, "max t_image over never-injected arms", "33",
        "matches Entry 33 't approx 30-45' (base 45)")
    RD = W["released_detector"]
    add("nMapWmBitTrain", RD["train_png_dswm"]["bit_accuracy_mean"], dec(RD["train_png_dswm"]["bit_accuracy_mean"], 4),
        F_WM, "released_detector.train_png_dswm.bit_accuracy_mean", "33",
        "released detector on the 50 stored training crops; matches Entry 33 0.9989")
    add("nMapWmTrainCrops", RD["train_png_dswm"]["n"], integer(RD["train_png_dswm"]["n"]), F_WM,
        "released_detector.train_png_dswm.n", "33", "50 stored crops")
    gen_keys = [k for k in RD if k != "train_png_dswm"]
    ba = [RD[k]["bit_accuracy_mean"] for k in gen_keys]
    add("nMapWmBitGen", float(np.mean(ba)), dec(float(np.mean(ba)), 4), F_WM,
        "released_detector.<9 generation arms>.bit_accuracy_mean (all equal to 4 d.p.)", "33",
        f"matches Entry 33 0.5703 in every arm (range {min(ba):.6f}-{max(ba):.6f})")
    assert max(ba) - min(ba) < 1e-5
    sa = [RD[k]["symbol_accuracy_mean"] for k in gen_keys]
    add("nMapWmSymbolGen", float(np.mean(sa)), dec(float(np.mean(sa)), 3), F_WM,
        "released_detector.<generation arms>.symbol_accuracy_mean", "33", "symbol accuracy on generations 0.297")
    add("nMapWmSymbolTrain", RD["train_png_dswm"]["symbol_accuracy_mean"],
        dec(RD["train_png_dswm"]["symbol_accuracy_mean"], 4), F_WM, "released_detector.train_png_dswm.symbol_accuracy_mean",
        "33", "symbol accuracy on the stored crops")
    sdmax = max(RD[k]["bit_accuracy_sd"] for k in gen_keys)
    add("nMapWmBitGenSDMax", sdmax, sci(sdmax, 1), F_WM, "max bit_accuracy_sd over generation arms", "33",
        "Entry 33 'with zero spread'")
    add("nMapWmDetArms", len(gen_keys), integer(len(gen_keys)), F_WM, "count released_detector generation arms", "33",
        "nine arms scored (3 marked + 6 never-injected)")
    add("nMapWmDetGensPerArm", RD[gen_keys[0]]["n"], integer(RD[gen_keys[0]]["n"]), F_WM,
        "released_detector.<arm>.n", "33", "200 generations per arm scored by the released detector")
    nat = [W["arms"][k]["natural_paired_KA_minus_KB"] for k in wm_arms]
    add("nMapWmNaturalMin", min(nat), sci(min(nat), 2), F_WM, "min arms.wm_ds_*.natural_paired_KA_minus_KB", "33",
        "natural K_A - K_B contrast in the wm arms (raw NCC), observation not tested; Entry 33 -2.1e-4")
    add("nMapWmNaturalMax", max(nat), sci(max(nat), 2), F_WM, "max arms.wm_ds_*.natural_paired_KA_minus_KB", "33",
        "Entry 33 -1.3e-4")
    A1 = load(F_A1)["DiffusionShield"]
    add("nMapWmTvaeLow", A1["predicted_retention_low"], dec(A1["predicted_retention_low"], 2), F_A1,
        "DiffusionShield.predicted_retention_low", "42", "band-weighted autoencoder retention of the mark, 0.12")
    add("nMapWmTvaeHigh", A1["predicted_retention_high"], dec(A1["predicted_retention_high"], 2), F_A1,
        "DiffusionShield.predicted_retention_high", "42", "band-weighted autoencoder retention of the mark, 0.18")

    # ------------------------------------------------------------------ Known: E1 (Entry 69)
    KS = load(F_KF)
    KF = KS["fields"]
    KM = load(F_KFM)
    U = "; percentage of the stored input contrast R of the injected K, NOT of R_real"
    a12 = KF["kinj_a12"]
    add("nKnownAlphaTwelve", a12["T_mean_pct"], sig(a12["T_mean_pct"], 3), F_KF, "fields.kinj_a12.T_mean_pct", "69",
        "matches RESULTS Entry 69 0.0174 %, verify_v2 check 22, FINDINGS" + U)
    add("nKnownAlphaTwelveSE", a12["T_se_pct"], sig(a12["T_se_pct"], 2), F_KF, "fields.kinj_a12.T_se_pct", "69",
        "adapter-level SE incl. offset SE; matches Entry 69 0.0042" + U)
    add("nKnownAlphaTwelveUpper", a12["T_upper99_pct"], sig(a12["T_upper99_pct"], 3), F_KF,
        "fields.kinj_a12.T_upper99_pct", "69", "one-sided 99 % t upper (df 2); matches Entry 69 0.0469 %" + U)
    add("nKnownAlphaTwelveLower", a12["T_lower99_pct"], sig(a12["T_lower99_pct"], 3), F_KF,
        "fields.kinj_a12.T_lower99_pct", "69", "one-sided 99 % lower (df 2); not quoted in RESULTS" + U)
    for a in by_seed(a12["adapters"]):
        s = seed_of(a["arm"])
        add(f"nKnownAlphaTwelveSeed{WORDS[s]}", a["T_pct"], sig(a["T_pct"], 3), F_KF,
            f"fields.kinj_a12.adapters[{a['arm']}].T_pct", "69", "matches Entry 69 0.0136 / 0.0171 / 0.0213 %" + U)
    ise = [a["T_image_se_pct"] for a in a12["adapters"]]
    add("nKnownAlphaTwelveImageSEMin", min(ise), sig(min(ise), 3), F_KF, "min kinj_a12 adapters T_image_se_pct", "69",
        "matches Entry 69 image SE 0.0135-0.0139" + U)
    add("nKnownAlphaTwelveImageSEMax", max(ise), sig(max(ise), 3), F_KF, "max kinj_a12 adapters T_image_se_pct", "69",
        "matches Entry 69 image SE 0.0135-0.0139" + U)
    rk = [a["decoy_rank"] for a in by_seed(a12["adapters"])]
    add("nKnownAlphaTwelveDecoyRanks", rk, ", ".join(map(str, rk)), F_KF, "fields.kinj_a12.adapters[*].decoy_rank (seed order)",
        "69", "seed order s0,s1,s2 = 3, 1, 1; Entry 69 prints '1, 1, 3' (value order); descriptive only (Entry 61.4)")
    add("nKnownAlphaTwelveAdapters", a12["n_adapters"], integer(a12["n_adapters"]), F_KF, "fields.kinj_a12.n_adapters",
        "69", "three adapters at alpha 12")
    a48 = KF["kinj_a48"]
    add("nKnownAlphaFortyEight", a48["T_mean_pct"], sig(a48["T_mean_pct"], 3), F_KF, "fields.kinj_a48.T_mean_pct", "69",
        "one adapter; matches Entry 69 0.0486 %" + U)
    add("nKnownAlphaFortyEightSE", a48["T_se_pct"], sig(a48["T_se_pct"], 3), F_KF, "fields.kinj_a48.T_se_pct", "69",
        "image-level SE (one adapter); matches Entry 69 0.0110" + U)
    d3 = KF["kinjd_a3"]
    add("nKnownDitherThree", d3["T_mean_pct"], sig(d3["T_mean_pct"], 3), F_KF, "fields.kinjd_a3.T_mean_pct", "69",
        "dithered alpha 3, one adapter; matches Entry 69 0.0137 %" + U)
    add("nKnownDitherThreeSE", d3["T_se_pct"], sig(d3["T_se_pct"], 3), F_KF, "fields.kinjd_a3.T_se_pct", "69",
        "image-level SE; matches Entry 69 0.0441" + U)
    add("nKnownPred", KS["prediction_K_spectrum_pct"], sig(KS["prediction_K_spectrum_pct"], 3), F_KF,
        "prediction_K_spectrum_pct", "61, 69, 72",
        "band-response prediction for K_B_E2's spectrum fixed before data; matches Entry 61/69/72 0.0978 %" + U)
    add("nKnownPredSE", KS["prediction_se_pct"], sig(KS["prediction_se_pct"], 2), F_KF, "prediction_se_pct", "61",
        "matches Entry 61 SE 0.0059 %")
    rr = KS["ratios"]["kinj48_over_kinj12"]
    add("nKnownAmpRatio", rr, dec(rr, 2), F_KF, "ratios.kinj48_over_kinj12", "69",
        "alpha48/alpha12; matches Entry 69 2.80 and verify_v2 check 23")
    sf = KS["prediction_K_spectrum_pct"] / a12["T_mean_pct"]
    add("nKnownShortfall", sf, dec(sf, 1), F_KF, "prediction_K_spectrum_pct / fields.kinj_a12.T_mean_pct", "69",
        "matches Entry 69 '5.6x below prediction at alpha 12'")
    for key, nm, ent in (("kinj_a12", "AlphaTwelve", "0.92"), ("kinj_a48", "AlphaFortyEight", "3.56"),
                         ("kinjd_a3", "DitherThree", "0.39")):
        v = KM[key]["stored_change_rms_gray"]
        add(f"nKnown{nm}Gray", v, dec(v, 2), F_KFM, f"{key}.stored_change_rms_gray", "72",
            f"stored RMS change (gray levels); Entry 72 quotes {ent} gray")
        add(f"nKnown{nm}StoredR", KM[key]["R"], dec(KM[key]["R"], 3), F_KFM, f"{key}.R", "69",
            "stored input contrast R of K (NCC units), the transmission denominator")
    add("nKnownOffset", a12["offset"], sci(a12["offset"], 2), F_KF, "fields.kinj_a12.offset", "61, 69",
        "never-injected offset from the three body-A unmarked arms (raw NCC)")
    add("nKnownNeverArmsK", len(KS["never_arms"]["k"]), integer(len(KS["never_arms"]["k"])), F_KF, "never_arms.k", "61",
        "three body-A unmarked arms for the K offset")
    add("nKnownNeverArmsG", len(KS["never_arms"]["g"]), integer(len(KS["never_arms"]["g"])), F_KF, "never_arms.g", "61",
        "six unmarked arms for the G offset")
    nE1 = sum(KF[k]["n_adapters"] for k in ("kinj_a12", "kinj_a48", "kinjd_a3"))
    nE2 = sum(KF[k]["n_adapters"] for k in ("gkadd_a4", "gkadd_a1", "gkmul_a4"))
    add("nKnownAdaptersEone", nE1, integer(nE1), F_KF, "sum n_adapters kinj_a12, kinj_a48, kinjd_a3", "69",
        "five E1 adapters")
    add("nKnownAdaptersEtwo", nE2, integer(nE2), F_KF, "sum n_adapters gkadd_a4, gkadd_a1, gkmul_a4", "72",
        "six E2 adapters")
    add("nKnownImagesEone", 5500, integer(5500), "RESULTS.md", "Entry 69: 'measured in 39 min over 5,500 images'", "69",
        "text-only; generations per adapter are 500 (Entry 59 protocol)")

    # ------------------------------------------------------------------ Known: E2 (Entry 72)
    U2 = "; percentage of the stored input contrast R of the spectrum-matched field G, NOT of R_real"
    for key, nm, ent, esd in (("gkadd_a4", "FieldFour", "0.0366", "0.0018"), ("gkadd_a1", "FieldOne", "0.0223", "0.0041"),
                              ("gkmul_a4", "FieldMul", "0.0502", "0.0024")):
        f = KF[key]
        add(f"nKnown{nm}", f["T_mean_pct"], sig(f["T_mean_pct"], 3), F_KF, f"fields.{key}.T_mean_pct", "72",
            f"two adapters; matches Entry 72 {ent} %" + (", verify_v2 check 24" if key == "gkadd_a4" else "") + U2)
        add(f"nKnown{nm}SE", f["T_se_pct"], sig(f["T_se_pct"], 2), F_KF, f"fields.{key}.T_se_pct", "72",
            f"adapter-level SE; matches Entry 72 {esd}" + U2)
        for a in by_seed(f["adapters"]):
            s = seed_of(a["arm"])
            add(f"nKnown{nm}Seed{WORDS[s]}", a["T_pct"], sig(a["T_pct"], 3), F_KF,
                f"fields.{key}.adapters[{a['arm']}].T_pct", "72", "per adapter (seed-sorted)" + U2)
        v = KM[key]["stored_change_rms_gray"]
        add(f"nKnown{nm}Gray", v, dec(v, 2), F_KFM, f"{key}.stored_change_rms_gray", "72",
            "stored RMS change (gray levels); matches Entry 72 table 3.95 / 1.03 / 3.91")
    lin = KS["ratios"]["gkadd1_over_gkadd4"]
    add("nKnownLinearity", lin, dec(lin, 3), F_KF, "ratios.gkadd1_over_gkadd4", "72",
        "T(1 gray)/T(4 gray); matches Entry 72 0.609 and verify_v2 check 25")
    form = KS["ratios"]["gkmul4_over_gkadd4"]
    add("nKnownFormRatio", form, dec(form, 3), F_KF, "ratios.gkmul4_over_gkadd4", "72",
        "mult/add at equal RMS; matches Entry 72 1.373 and verify_v2 check 26")
    g4 = KF["gkadd_a4"]["T_mean_pct"]
    g1 = KF["gkadd_a1"]["T_mean_pct"]
    mh = a48["T_mean_pct"] / g4
    ml = a12["T_mean_pct"] / g1
    add("nKnownMatchedHigh", mh, dec(mh, 2), F_KF, "kinj_a48.T_mean_pct / gkadd_a4.T_mean_pct", "72",
        "K alpha48 (3.56 gray) vs G add 4 gray; matches Entry 72 1.33x and verify_v2 check 28")
    add("nKnownMatchedLow", ml, dec(ml, 2), F_KF, "kinj_a12.T_mean_pct / gkadd_a1.T_mean_pct", "72",
        "K alpha12 (0.92 gray) vs G add 1 gray; matches Entry 72 0.78x")
    ex = KS["prediction_K_spectrum_pct"] / g4
    add("nKnownExtrapRatio", ex, dec(ex, 1), F_KF, "prediction_K_spectrum_pct / gkadd_a4.T_mean_pct", "72",
        "prediction over direct injection 2.677; Entry 72 'about 2.7x', FINDINGS 2.68x, verify_v2 check 27 "
        "reports 2.67 (tol 0.05); printed 2.7")
    ex1 = KS["prediction_K_spectrum_pct"] / g1
    add("nKnownExtrapRatioOne", ex1, dec(ex1, 1), F_KF, "prediction_K_spectrum_pct / gkadd_a1.T_mean_pct", "72",
        "same ratio at 1 gray (4.4x); not quoted in RESULTS")
    ps_ = KS["prediction_se_pct"]
    z4 = (KS["prediction_K_spectrum_pct"] - g4) / math.hypot(ps_, KF["gkadd_a4"]["T_se_pct"])
    z1 = (KS["prediction_K_spectrum_pct"] - g1) / math.hypot(ps_, KF["gkadd_a1"]["T_se_pct"])
    add("nKnownExtrapZFour", z4, dec(z4, 2), F_KF, "(pred - T_gkadd_a4)/hypot(pred_se, T_se)", "72",
        "matches Entry 72 z = 9.95")
    add("nKnownExtrapZOne", z1, dec(z1, 1), F_KF, "(pred - T_gkadd_a1)/hypot(pred_se, T_se)", "72",
        "matches Entry 72 z = 10.5")
    gg = g4 / g1
    add("nKnownFieldAmpGain", gg, dec(gg, 1), F_KF, "gkadd_a4.T_mean_pct / gkadd_a1.T_mean_pct", "72",
        "fourfold amplitude raises T by 1.6x for G (Entry 72); K's gain is nKnownAmpRatio (2.8x)")
    add("nKnownCalibAmpFactor", 40, "forty", "RESULTS.md",
        "Entry 72: 'the map was calibrated at about forty times the fingerprint's amplitude'", "72",
        "text-only (approximate, spelled as a word in the source)")
    eK = load(F_KFF)["energy_by_band_K"]
    eG = load(F_KFF)["energy_by_band_G"]
    add("nKnownEnergyTop", 100 * eK[0], dec(100 * eK[0], 1), F_KFF, "energy_by_band_K.0 x100", "61",
        "K_B_E2 energy share in the top octave (percent of energy); matches Entry 61 0.582")
    add("nKnownEnergyOne", 100 * eK[1], dec(100 * eK[1], 1), F_KFF, "energy_by_band_K.1 x100", "61",
        "matches Entry 61 0.237")
    add("nKnownEnergyTwo", 100 * eK[2], dec(100 * eK[2], 1), F_KFF, "energy_by_band_K.2 x100", "61",
        "matches Entry 61 0.045")
    dmax = max(abs(a - b) for a, b in zip(eK, eG))
    add("nKnownEnergyMatchMaxDiff", dmax, sci(dmax, 1), F_KFF, "max |energy_by_band_K - energy_by_band_G|", "72",
        "G matches K's band energies 'to three decimals' (Entry 61/72)")
    return M


if __name__ == "__main__":
    ms = macros()
    for m in ms:
        print(f"{m['name']:40s} {m['text']:>22s}   {m['entry']}")
    print(len(ms), "macros")
