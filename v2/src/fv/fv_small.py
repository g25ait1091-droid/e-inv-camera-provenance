"""Small quantities for the manuscript: RESULTS.md Entry 116, item 6 (descriptive, from existing files).

Registered in Entry 116 as  src/fv/fv_small.py -> out/fv_small.json ; this script carries those names.
An earlier attempt at this item (30 Sep 2026) wrote src/fv/fv_small_quantities.py -> out/fv_small_quantities.json
under the names its run asked for. That file is never overwritten and no quantity here is read from it: it is
opened only at the very end, for a side-by-side comparison block.

  (a) hostile-r2-04  Nominal max-arm limit of the primary design retrained on the local stack (G2, six per arm):
                     U_X = mean_X + t_{0.995,5} s_X / sqrt(6), U = max(U_A, U_B), as % of R_real (c = 1). Beside
                     it, labelled post hoc, the same with nomark_s5 left out (arm A at five, t_{0.995,4}), and the
                     archive's six-per-arm limit read from the ledger (history_k6, 0.249 %).
  (b) hostile-r2-14  16000-step symmetric p-values with the E2 estimation shift added:
                     SE_infl = sqrt(SE_Welch^2 + sd_est^2), t = theta_sym / SE_infl, one-sided at the Welch df of
                     each result; Entry 68 replication (seeds 3-5), Entry 68 pooled six, G1 inversion at eight per
                     arm (direction theta < 0). Each uninflated p is asserted to reproduce its entry first.
  (c) trace-r2-12    GPU-hours by GPU for the 211 adapters / 95,500 generations of Entry 112 N-A5, and for all work
                     (adding the objective arms, the designed-mark ladder and the environment chain). Training:
                     train_meta.json "minutes" where recorded, else supplement S3's per-unit time. Generation: S3's
                     per-unit range x the number of jobs. Low-high, with the recorded share. Detector, embedding,
                     scoring and autoencoder GPU time excluded. Beside it, descriptive and not registered: the same
                     totals with generation measured from the image files' write times, the L40S's occupied
                     (union) hours, and the training records against S3's per-unit times.
  (d) figures-r2-11  nDoseInvThreeT: the Welch t of the first inversion reading, seeds 0-2 per arm from
                     out/g1_ext.json. Stops unless it reproduces Entry 98's theta_sym (-0.143 %) and p (0.145).
  (e) trace-r2-09    The verifier's check count (python verify_v2.py in the repository clone, EINV_V2 unset, CPU),
                     and adapters and generations by stack exactly as Table 2(b) labels them, asserted to sum to
                     211 and 95,500. The deposit file count is not computed (waits on author actions). Beside it,
                     descriptive: the committed (HEAD) verifier's count, and the same verifier on the live workspace.

Constants typed in: only values the registration prints (R_real 0.0356703; sd_est 1.224e-05; the entries' p-values
and Entry 98's theta_sym, all used as reproduction checks) and supplement S3's per-unit times, each asserted to
appear literally in paper/fv/supp/S03_compute.tex. Generated-image folders (local, and the Drive mirror for the
archive) are read for file counts and write times only. CPU only; a few minutes, most of it Drive metadata.

Run:  python src/fv/fv_small.py
The output is opened with mode "x": the script stops rather than overwrite an existing result file.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import datetime
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys

import numpy as np
import scipy
from scipy import stats

V2 = EINV.V2
OUT = V2 + "/out"
T1 = OUT + "/t1"
FV = V2 + "/paper/fv"
REPO = EINV.REPO
LEDGER = REPO + "/analysis/FINAL_LEDGER.json"
DRIVE = (EINV.MYDRIVE + "/inv_channel")
NUMBERS = FV + "/numbers.json"
TAB2 = FV + "/tabs/tab02_devices.tex"
S3_TEX = FV + "/supp/S03_compute.tex"
LOCAL_ADAPTERS = T1 + "/adapters"
LOCAL_GENS = T1 + "/gens"
T5_GENS = OUT + "/t5/gens"
DST = OUT + "/fv_small.json"
EARLIER = OUT + "/fv_small_quantities.json"
GIT = ("git",)

R_REAL_PRINTED = 0.0356703        # Entry 116 item 6(a)
SD_EST_PRINTED = 1.224e-05        # Entry 116 item 6(b)
GAP_BREAK_S = 600.0               # descriptive timestamp check: a longer gap between consecutive images = interruption

INPUTS = {}


# ------------------------------------------------------------------------------------------------ helpers
def _key(p):
    return os.path.abspath(p).replace("\\", "/")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def note(p):
    k = _key(p)
    if k not in INPUTS:
        INPUTS[k] = sha256(p)
    return k


def jload(p):
    note(p)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def tread(p):
    note(p)
    with open(p, encoding="utf-8") as f:
        return f.read()


def close(a, b, rel=1e-10):
    a, b = float(a), float(b)
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)


def welch_sym(a, b):
    """theta_sym = (mean_A + mean_B)/2, its Welch SE and Welch df (the construction of verify_v2.welch_sym)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    th = 0.5 * (a.mean() + b.mean())
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se = 0.5 * math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    return float(th), float(se), float(df), float(va), float(vb)


def arm_limit(x):
    """One arm of the nominal max-arm construction: mean + t_{0.995, n-1} s / sqrt(n)."""
    x = np.asarray(x, float)
    n = int(x.size)
    tq = float(stats.t.ppf(0.995, n - 1))
    s = float(x.std(ddof=1))
    hw = tq * s / math.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "sd": s, "t_df": n - 1, "t_0995": tq, "half_width": hw,
            "U": float(x.mean()) + hw}


def num(nums, macro):
    v = nums[macro]["value"]
    f = float(v)
    assert f == int(round(f)), (macro, v)
    return int(round(f))


def run_git(args):
    for g in GIT:
        try:
            r = subprocess.run([g] + args, cwd=REPO, capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                return r.stdout
        except OSError:
            continue
    return None


def utc(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat(timespec="seconds")


def ledger():
    led = jload(LEDGER)
    R = float(led["denominators"]["R_real"])
    assert round(R, 7) == R_REAL_PRINTED, R
    return R, led


# ================================================================================================ (a)
def item_a(R, led):
    g2 = jload(OUT + "/g2_pooled_six.json")
    A = [float(v) for v in g2["per_adapter_A"]]
    B = [float(v) for v in g2["per_adapter_B"]]
    assert len(A) == 6 and len(B) == 6
    arms = {}
    for f in ("summary_nomark.json", "summary_nomarkrep.json", "summary_nomarkB.json"):
        arms.update(jload(T1 + "/" + f)["arms"])
    labA = [f"nomark_s{i}" for i in range(6)]
    labB = [f"nomarkB_s{i}" for i in range(6)]
    for i in range(6):     # the file's order is seeds 0-5; arm B is stored own-minus-other = -(rho(K_A) - rho(K_B))
        assert A[i] == float(arms[labA[i]]["natural_paired_KA_minus_KB"]), labA[i]
        assert B[i] == -float(arms[labB[i]]["natural_paired_KA_minus_KB"]), labB[i]
        assert int(arms[labA[i]]["n"]) == 500 and int(arms[labB[i]]["n"]) == 500
    th, se, df, _, _ = welch_sym(A, B)
    assert close(th, g2["theta_sym"]) and close(se, g2["welch_se"]) and close(df, g2["welch_df"]), "G2 file"

    uA, uB = arm_limit(A), arm_limit(B)
    U = max(uA["U"], uB["U"])
    nominal = {"construction": "U_X = mean_X + t_{0.995,5} s_X / sqrt(6); U = max(U_A, U_B); lambda_U = U / R_real",
               "c": 1.0, "U_A": uA, "U_B": uB, "U": U, "set_by_arm": "A" if uA["U"] >= uB["U"] else "B",
               "lambda_U_pct": 100.0 * U / R, "lambda_U_A_pct": 100.0 * uA["U"] / R,
               "lambda_U_B_pct": 100.0 * uB["U"] / R,
               "theta_sym": th, "theta_sym_pct": 100.0 * th / R}

    assert labA[5] == "nomark_s5"
    uA5 = arm_limit(A[:5])
    U5 = max(uA5["U"], uB["U"])
    without = {"status": "post hoc (Entry 116 item 6a labels it so)", "left_out": "nomark_s5",
               "left_out_value": A[5], "left_out_value_pct": 100.0 * A[5] / R,
               "U_A_five": uA5, "U_B_six": uB, "U": U5, "set_by_arm": "A" if uA5["U"] >= uB["U"] else "B",
               "lambda_U_pct": 100.0 * U5 / R, "lambda_U_A_pct": 100.0 * uA5["U"] / R,
               "lambda_U_B_pct": 100.0 * uB["U"] / R}

    h = led["history_k6"]
    pa = [float(v) for v in led["primary"]["per_adapter_A"]]
    pb = [float(v) for v in led["primary"]["per_adapter_B"]]
    assert len(pa) == len(pb) == 12
    kA, kB = arm_limit(pa[:6]), arm_limit(pb[:6])
    Uk = max(kA["U"], kB["U"])
    thk = welch_sym(pa[:6], pb[:6])[0]
    checks_k6 = {"U_device_within_rounding": abs(Uk - float(h["U_device"])) <= 5e-10,
                 "lambda_pct_rounds_to_ledger": round(100.0 * Uk / R, 3) == float(h["lambda_U_plugin_pct"]),
                 "theta_sym_within_rounding": abs(thk - float(h["theta_sym"])) <= 5e-9,
                 "t_crit_within_rounding": abs(kA["t_0995"] - float(h["t_crit_0995"])) <= 5e-5}
    k12A, k12B = arm_limit(pa), arm_limit(pb)
    U12 = max(k12A["U"], k12B["U"])
    assert abs(U12 - float(led["primary"]["U_device"])) <= 5e-10, U12     # the method reproduces the headline
    archive = {"ledger_history_k6": h,
               "ledger_history_k6_lambda_U_pct": float(h["lambda_U_plugin_pct"]),
               "recomputed_from_ledger_seeds_0_5": {"U_A": kA, "U_B": kB, "U": Uk, "lambda_U_pct": 100.0 * Uk / R,
                                                    "set_by_arm": "A" if kA["U"] >= kB["U"] else "B",
                                                    "theta_sym": thk, "agrees_with_ledger": checks_k6,
                                                    "note": "ledger primary.per_adapter_A/B[:6] (seeds 0-5, "
                                                            "per its _per_adapter_note); a check that history_k6 "
                                                            "is the first six seeds, not a new number"},
               "context_archive_twelve_per_arm_nominal_pct": 100.0 * U12 / R}
    return {"question": "hostile-r2-04: Table 3 row 3 (primary design retrained on the local stack), max-arm limit",
            "inputs": {"per_adapter_A": A, "per_adapter_A_labels": labA, "per_adapter_B": B,
                       "per_adapter_B_labels": labB,
                       "per_adapter_B_orientation": "own-minus-other, rho(K_B) - rho(K_A) on arm-B images",
                       "R_real": R, "file_theta_sym": g2["theta_sym"], "file_one_sided_p": g2["one_sided_p"],
                       "file_symmetric_limit99_pct": g2["limit99_pct"]},
            "nominal_six_per_arm": nominal,
            "post_hoc_without_nomark_s5": without,
            "archive_six_per_arm": archive,
            "note": "c = 1: H6's multiplier (1.25) is not transported to the local stack (Entry 116)",
            "source": "out/g2_pooled_six.json per_adapter_A/B (checked against out/t1/summary_nomark*.json); "
                      "R_real and history_k6 from analysis/FINAL_LEDGER.json of the repository clone"}


# ================================================================================================ (b)
def item_b(R):
    h6 = jload(OUT + "/h6_calibrated_limit.json")
    sd = float(h6["estimation_sd"])
    h2 = jload(OUT + "/h2_estimation_error.json")
    reps = np.array([float(r["theta_sym"]) for r in h2["replicates"]])
    assert reps.size == int(h2["n_replicates"]) == 24
    assert close(float(reps.std(ddof=1)), sd, 1e-12), "estimation SD is the SD of theta_sym over the H2 replicates"
    assert abs(sd - SD_EST_PRINTED) <= 5e-9, sd
    nrep = int(reps.size)
    sd_hi95 = sd * math.sqrt((nrep - 1) / float(stats.chi2.ppf(0.05, nrep - 1)))

    def infl(th, se, df, va, vb, nA, nB, s, lower):
        se_i = math.sqrt(se ** 2 + s ** 2)
        t_i = th / se_i
        p_i = float(stats.t.cdf(t_i, df)) if lower else float(stats.t.sf(t_i, df))
        cA, cB, cE = va / 4.0, vb / 4.0, s ** 2
        df_sat = (cA + cB + cE) ** 2 / (cA ** 2 / (nA - 1) + cB ** 2 / (nB - 1) + cE ** 2 / (nrep - 1))
        p_sat = float(stats.t.cdf(t_i, df_sat)) if lower else float(stats.t.sf(t_i, df_sat))
        return {"sd_est": s, "SE_infl": se_i, "SE_ratio": se_i / se, "t_infl": t_i, "df": df,
                "one_sided_p_infl": p_i,
                "beside_not_registered": {"df_satterthwaite_with_estimation_term": df_sat,
                                          "one_sided_p_infl_at_satterthwaite_df": p_sat}}

    arms = {}
    for f in ("summary_dose16k.json", "summary_dose16krep.json", "summary_dose16krep2.json"):
        arms.update(jload(T1 + "/" + f)["arms"])
    doses = jload(T1 + "/dose_stats.json")["doses"]
    res = {}
    spec = (("entry68_replication_new_three_per_body", "16000_new", (3, 4, 5), 0.00955, 0.0096, 0.05,
             "Entry 55 primary reading: theta_sym(new) > 0 with one-sided Welch p < 0.05"),
            ("entry68_pooled_six_per_body", "16000", tuple(range(6)), 0.00198, 0.0020, 0.01,
             "Entry 55 secondary reading: theta_sym above U_device and one-sided Welch p < 0.01"))
    for name, key, seeds, p68, p116, lvl, lvl_src in spec:
        d = doses[key]
        assert d["A_arms"] == [f"dose16k_A_s{i}" for i in seeds] and d["B_arms"] == [f"dose16k_B_s{i}" for i in seeds]
        A = [float(arms[f"dose16k_A_s{i}"]["natural_paired_KA_minus_KB"]) for i in seeds]
        B = [-float(arms[f"dose16k_B_s{i}"]["natural_paired_KA_minus_KB"]) for i in seeds]
        assert A == [float(v) for v in d["A_own"]] and B == [float(v) for v in d["B_own"]], key
        th, se, df, va, vb = welch_sym(A, B)
        assert close(th, d["theta_sym"]) and close(se, d["adapter_SE"]) and close(df, d["welch_df"]) \
            and close(th / se, d["t"]), key
        p = float(stats.t.sf(th / se, df))
        assert abs(p - p68) <= 5e-6, (key, p)          # Entry 68 prints three significant figures
        assert abs(p - p116) <= 1e-4, (key, p)         # Entry 116 restates it at two (see p_printed_note)
        res[name] = {"seeds": list(seeds), "A_own": A, "B_own": B, "theta_sym": th, "theta_sym_pct": 100.0 * th / R,
                     "SE_Welch": se, "df_Welch": df, "t": th / se, "one_sided_p": p, "direction": "theta_sym > 0",
                     "p_printed": {"Entry 68": p68, "Entry 116": p116},
                     "p_printed_note": ("Entry 116's value equals the exact p rounded to two significant figures"
                                        if round(p, 4) == p116 else
                                        f"Entry 116's {p116} is Entry 68's {p68} rounded again; the exact p "
                                        f"({p:.6f}) rounds to {round(p, 4)} at two significant figures"),
                     "inflated": infl(th, se, df, va, vb, len(A), len(B), sd, False),
                     "inflated_at_sd_est_upper95": infl(th, se, df, va, vb, len(A), len(B), sd_hi95, False),
                     "context_level": lvl, "context_level_source": lvl_src,
                     "source": f"out/t1/dose_stats.json doses['{key}']; per-adapter values equal "
                               "out/t1/summary_dose16k*.json natural_paired_KA_minus_KB (B negated)"}
        if key == "16000":
            res[name]["theta_sym_above_U_device"] = th > 5.3761e-05

    ge = jload(OUT + "/g1_ext.json")
    arms_i = {}
    for f in ("summary_inv16k.json", "summary_inv16kext.json"):
        arms_i.update(jload(T1 + "/" + f)["arms"])
    tagA = [f"inv16k_A_s{i}" for i in range(3)] + [f"inv16kext_A_s{i}" for i in range(3, 8)]
    tagB = [f"inv16k_B_s{i}" for i in range(3)] + [f"inv16kext_B_s{i}" for i in range(3, 8)]
    A8 = [float(ge["per_adapter_A"][f"s{i}"]) for i in range(8)]
    B8 = [float(ge["per_adapter_B"][f"s{i}"]) for i in range(8)]
    for i in range(8):
        assert A8[i] == float(arms_i[tagA[i]]["natural_paired_KA_minus_KB"]), tagA[i]
        assert B8[i] == -float(arms_i[tagB[i]]["natural_paired_KA_minus_KB"]), tagB[i]
    th, se, df, va, vb = welch_sym(A8, B8)
    e = ge["all_eight"]
    assert close(th, e["theta_sym"]) and close(se, e["welch_se"]) and close(df, e["welch_df"]) and close(th / se, e["t"])
    p = float(stats.t.cdf(th / se, df))
    assert close(p, e["one_sided_p_lt0"], 1e-9) and abs(p - 0.0106) <= 5e-5, p
    res["g1_inversion_eight_per_arm"] = {
        "seeds": list(range(8)), "A_own": A8, "B_own": B8, "A_tags": tagA, "B_tags": tagB,
        "theta_sym": th, "theta_sym_pct": 100.0 * th / R, "SE_Welch": se, "df_Welch": df, "t": th / se,
        "one_sided_p": p, "direction": "theta_sym < 0", "p_printed": {"Entry 110": 0.0106, "Entry 116": 0.0106},
        "inflated": infl(th, se, df, va, vb, 8, 8, sd, True),
        "inflated_at_sd_est_upper95": infl(th, se, df, va, vb, 8, 8, sd_hi95, True),
        "context_level": 0.05, "context_level_source": "Entries 77 and 99: theta_sym < 0 with one-sided Welch p < 0.05",
        "source": "out/g1_ext.json per_adapter_A/B s0-s7 (equal to out/t1/summary_inv16k*.json, B negated); "
                  "all_eight reproduced"}
    return {"question": "hostile-r2-14: 16000-step symmetric p-values with the E2 estimation shift added",
            "formula": "SE_infl = sqrt(SE_Welch^2 + sd_est^2); t = theta_sym / SE_infl; one-sided p at the Welch df "
                       "of each result (upper tail for the two Entry 68 results, lower tail for G1)",
            "sd_est": sd,
            "sd_est_source": "out/h6_calibrated_limit.json estimation_sd = SD (ddof 1) of theta_sym over the 24 "
                             "bootstrap replicates of out/h2_estimation_error.json (checked)",
            "sd_est_measured_at": {"steps": 2000, "adapters_per_arm": 12, "images_per_adapter":
                                   int(h2["images_per_adapter"]), "n_replicates": nrep,
                                   "note": "transported to 16000 steps and 250 images per adapter. Entry 116 says "
                                           "'measured at 2000 steps and 500 images'; the H2 file records 200 images "
                                           "per adapter (Entry 88 registered 200)."},
            "sd_est_upper95_beside": {"value": sd_hi95, "method": "one-sided 95 % chi-square upper limit of an SD "
                                      "from 24 replicates (23 df), normal theory; not registered"},
            "results": res,
            "not_inflated": "the within-body inverted-minus-normal contrast (Entry 110): both conditions are scored "
                            "with the same K, so the shift cancels exactly (Entry 116)",
            "reading": "none: Entry 116 items 5 and 6 are descriptive; the levels are given for orientation only"}


# ================================================================================================ (d)
def item_d(R):
    ge = jload(OUT + "/g1_ext.json")
    a3 = [float(ge["per_adapter_A"][f"s{i}"]) for i in range(3)]
    b3 = [float(ge["per_adapter_B"][f"s{i}"]) for i in range(3)]
    ch = jload(OUT + "/g_chain12.json")["G1"]
    th, se, df, _, _ = welch_sym(a3, b3)
    t = th / se
    p = float(stats.t.cdf(t, df))
    pct = 100.0 * th / R
    if not (round(pct, 3) == -0.143 and round(p, 3) == 0.145):
        raise SystemExit(f"(d) does not reproduce Entry 98: theta_sym {pct:.5f} %, p {p:.5f}")
    same = (a3 == [float(v) for v in ch["per_adapter_A"]] and b3 == [float(v) for v in ch["per_adapter_B"]]
            and close(t, ch["t"]) and close(p, ch["one_sided_p_lt0"], 1e-9))
    assert same, "g_chain12.json G1 (the file Entry 98 reported) differs from g1_ext.json seeds 0-2"
    return {"question": "figures-r2-11: nDoseInvThreeT, the Welch t of the first inversion reading (seeds 0-2)",
            "per_adapter_A": a3, "per_adapter_B": b3, "theta_sym": th, "theta_sym_pct": pct, "SE_Welch": se,
            "df_Welch": df, "t": t, "one_sided_p_lt0": p,
            "reproduces_entry_98": {"theta_sym_pct_printed": -0.143, "p_printed": 0.145,
                                    "Entry_98_table": "theta_sym -5.089e-05 = -0.1427 %, SE 4.009e-05, df 3.15, p 0.145",
                                    "also_equal_to_out/g_chain12.json_G1": True},
            "source": "out/g1_ext.json per_adapter_A/B s0-s2"}


# ================================================================================================ (e) Table 2(b)
def parse_table2b(nums):
    tex = tread(TAB2)
    blk = tex[tex.index("(b) Designs and scored generations"):]
    i_top = blk.index("\\toprule")
    i_hdr = blk.index("\\midrule", i_top)
    i_tot = blk.index("Total over the designs above")
    body = blk[i_hdr + len("\\midrule"):i_tot]
    body = body[:body.rindex("\\midrule")]
    rows = []
    for chunk in body.split("\\\\"):
        c = " ".join(chunk.split())
        if not c or c.startswith("\\multicolumn"):
            continue
        cells = [x.strip() for x in c.split("&")]
        assert len(cells) == 7, cells
        design, bodies, stack, ad, per, gens, sec = cells
        assert stack in ("archive", "local", "archive/local"), stack
        note_h = "^{\\mathrm{h}}" in design
        note_i = "^{\\mathrm{i}}" in design
        if ad == "---":
            n_ad, ad_expr = 0, "---"
        else:
            m = re.fullmatch(r"\\(n[A-Za-z]+)\{\}\s*\$\\times\$\s*(\d+)", ad)
            if m:
                n_ad, ad_expr = num(nums, m.group(1)) * int(m.group(2)), f"{m.group(1)} x {m.group(2)}"
            else:
                m = re.fullmatch(r"\\(n[A-Za-z]+)\{\}\s*in all", ad)
                assert m, ad
                n_ad, ad_expr = num(nums, m.group(1)), f"{m.group(1)} in all"
        if per == "---":
            n_per, per_expr = None, "---"
        else:
            m = re.fullmatch(r"\\(n[A-Za-z]+)", per)
            assert m, per
            n_per, per_expr = num(nums, m.group(1)), m.group(1)
        m = re.fullmatch(r"(\()?\\(n[A-Za-z]+)(\))?", gens)
        assert m and bool(m.group(1)) == bool(m.group(3)), gens
        g_macro, n_g, paren = m.group(2), num(nums, m.group(2)), bool(m.group(1))
        assert paren == note_i, design
        if n_per is not None and n_ad:
            assert n_ad * n_per == n_g, (design, n_ad, n_per, n_g)
        label = " ".join(re.sub(r"\$\^\{\\mathrm\{[a-z]\}\}\$", "", design).replace("\\quad", "").split())
        rows.append({"design": label, "stack": stack, "adapters_expr": ad_expr, "adapters": n_ad,
                     "gens_per_adapter_expr": per_expr, "gens_per_adapter": n_per, "gens_macro": g_macro, "gens": n_g,
                     "note": "h" if note_h else ("i" if note_i else ""),
                     "adapters_counted": 0 if (note_h or note_i) else n_ad, "gens_counted": 0 if note_i else n_g})
    tail = blk[i_tot:blk.index("\\bottomrule")]
    obj = None
    for chunk in tail.split("\\\\"):
        c = " ".join(chunk.split())
        if c.startswith("Training objective"):
            cells = [x.strip() for x in c.split("&")]
            m = re.fullmatch(r"\\(n[A-Za-z]+)\{\}\s*in all", cells[3])
            mp = re.fullmatch(r"\\(n[A-Za-z]+)", cells[4])
            assert m and mp and cells[2] == "archive", cells
            obj = {"design": cells[0], "stack": cells[2], "adapters": num(nums, m.group(1)),
                   "gens_per_adapter": num(nums, mp.group(1)),
                   "note": "outside the total (Table 2 note l): generations not scored"}
    assert obj is not None
    return rows, obj


def item_e_table(nums):
    rows, obj = parse_table2b(nums)
    by = {}
    for r in rows:
        s = by.setdefault(r["stack"], {"rows": 0, "adapters": 0, "generations": 0})
        s["rows"] += 1
        s["adapters"] += r["adapters_counted"]
        s["generations"] += r["gens_counted"]
    ta = sum(v["adapters"] for v in by.values())
    tg = sum(v["generations"] for v in by.values())
    assert ta == 211 and tg == 95500, (ta, tg)
    assert ta == num(nums, "nDataTotalAdapters") and tg == num(nums, "nDataTotalGens")
    na5 = jload(OUT + "/fv_derived.json")["N_A5_totals"]
    assert na5["total_adapters"] == ta and na5["total_gens"] == tg
    return {"by_stack": by, "total_adapters": ta, "total_generations": tg,
            "local_plus_archive_local": {"adapters": by["local"]["adapters"] + by["archive/local"]["adapters"],
                                         "generations": by["local"]["generations"] + by["archive/local"]["generations"]},
            "rows": rows, "outside_total_objective_row": obj,
            "counting": "a row adds its adapters unless note h (adapters counted in another row) or i (subset of the "
                        "row above), and its generations unless note i; the archive/local row's 6 adapters are the "
                        "primary seeds 0-2, counted under archive",
            "labels": "Table 2(b) Stack column: archive (Google Colab), local (one NVIDIA L40S), archive/local "
                      "(archive adapters, local generations)",
            "source": "paper/fv/tabs/tab02_devices.tex block (b), every cell evaluated through paper/fv/numbers.json; "
                      "totals equal nDataTotalAdapters, nDataTotalGens and out/fv_derived.json N_A5_totals"}


def item_e_verifier():
    note(REPO + "/verify_v2.py")
    env = {k: v for k, v in os.environ.items() if k != "EINV_V2"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    t0 = datetime.datetime.now(datetime.timezone.utc)
    r = subprocess.run([sys.executable, "verify_v2.py"], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=900)
    lines = r.stdout.splitlines()
    ok = [ln for ln in lines if ln.startswith("OK")]
    dis = [ln for ln in lines if ln.startswith("DISAGREE")]
    summ = [m for m in (re.fullmatch(r"(\d+) of (\d+) checks agree", ln.strip()) for ln in lines) if m]
    assert len(summ) == 1, lines[-5:]
    agree, total = int(summ[0].group(1)), int(summ[0].group(2))
    assert agree == len(ok) and total == len(ok) + len(dis), (agree, total, len(ok), len(dis))
    head = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--porcelain", "--", "verify_v2.py"])
    head_src = run_git(["show", "HEAD:verify_v2.py"])
    head_run = None
    if head_src is not None:
        code = ("import sys\nsrc = sys.stdin.read()\n"
                "g = {'__name__': '__main__', '__file__': " + repr(REPO + "/verify_v2.py") + "}\n"
                "exec(compile(src, 'verify_v2.py@HEAD', 'exec'), g)\n")
        rh = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, input=head_src, capture_output=True,
                            text=True, timeout=900)
        hl = rh.stdout.splitlines()
        hs = [m for m in (re.fullmatch(r"(\d+) of (\d+) checks agree", ln.strip()) for ln in hl) if m]
        head_run = {"returncode": rh.returncode,
                    "checks_executed": int(hs[0].group(2)) if hs else None,
                    "checks_agreeing": int(hs[0].group(1)) if hs else None,
                    "disagreeing": [ln for ln in hl if ln.startswith("DISAGREE")],
                    "sha256_of_HEAD_source": hashlib.sha256(head_src.encode("utf-8")).hexdigest(),
                    "stderr_tail": rh.stderr[-1500:],
                    "note": "descriptive: the committed verifier (what the public repository serves until the refresh "
                            "is pushed), run against the clone's working-tree data"}
    env_live = dict(env)
    env_live["EINV_V2"] = V2
    rl = subprocess.run([sys.executable, "verify_v2.py"], cwd=REPO, env=env_live, capture_output=True, text=True,
                        timeout=900)
    ll = rl.stdout.splitlines()
    ls_ = [m for m in (re.fullmatch(r"(\d+) of (\d+) checks agree", ln.strip()) for ln in ll) if m]
    live = {"EINV_V2": V2, "returncode": rl.returncode,
            "checks_executed": int(ls_[0].group(2)) if ls_ else None,
            "checks_agreeing": int(ls_[0].group(1)) if ls_ else None,
            "disagreeing": [ln for ln in ll if ln.startswith("DISAGREE")], "stderr_tail": rl.stderr[-1500:],
            "note": "descriptive: the same verifier against the live working directory instead of the clone's copy"}
    return {"command": "python verify_v2.py", "cwd": REPO, "python": sys.executable, "EINV_V2": "unset",
            "started_utc": t0.isoformat(timespec="seconds"), "returncode": r.returncode,
            "checks_executed": total, "checks_agreeing": agree, "checks_disagreeing": len(dis), "disagreeing": dis,
            "summary_line": summ[0].group(0), "stderr_tail": r.stderr[-1500:],
            "verify_v2_sha256": INPUTS[_key(REPO + "/verify_v2.py")],
            "git_head": None if head is None else head.strip(),
            "verify_v2_git_status": None if status is None else (status.strip() or "unmodified"),
            "committed_HEAD_verifier": head_run,
            "same_verifier_on_live_workspace": live,
            "data": "the clone's v2/workspace/out (EINV_V2 unset) and analysis/FINAL_LEDGER.json"}


# ================================================================================================ (c)
S3_UNITS = {   # key: (phrase in S03_compute.tex after whitespace normalisation, low min, high min, unit)
    "A100_adapter": ("a rank-16 adapter took about 24~min", 24.0, 24.0, "one SD-3.5 LoRA adapter, 2000 steps"),
    "A100_gen500": ("500 generations about 20~min", 20.0, 20.0, "500 SD-3.5 generations"),
    "Blackwell_FLUX_adapter": ("A FLUX.1-dev adapter took about 25~min on the Blackwell card", 25.0, 25.0,
                               "one FLUX.1-dev adapter"),
    "L40S_adapter_2000": ("A 2000-step adapter took 39~min alone and up to 87~min beside a generation job",
                          39.0, 87.0, "one 2000-step adapter"),
    "L40S_adapter_8000": ("an 8000-step adapter about 2.6~h", 156.0, 156.0, "one 8000-step adapter"),
    "L40S_adapter_16000": ("a 16000-step adapter 5.2~h alone and up to 14.8~h when sharing", 312.0, 888.0,
                           "one 16000-step adapter"),
    "L40S_gen500": ("Generating 500 images took 62--74~min", 62.0, 74.0, "500 generations"),
    "L40S_gen250": ("and 250 images about 35~min alone", 35.0, 35.0, "250 generations (S3 gives the 'alone' time only)"),
}
S3_NO_UNIT = "supplement S3 gives no per-unit time (it says no complete per-unit times were recorded)"

GPU_A100 = "NVIDIA A100-SXM4-40GB (Colab)"
GPU_A100L = "larger-memory NVIDIA A100 (Colab)"
GPU_BW = "NVIDIA RTX PRO 6000 Blackwell Server Edition (Colab)"
GPU_L40S = "NVIDIA L40S (local)"
COLAB = (GPU_A100, GPU_A100L, GPU_BW)
GROUP_COLAB = "A100 archive and Colab runs, including FLUX.1-dev"
GROUP_LOCAL = "L40S local"


def _arch_primary():
    t = [(f"E_INV_P0_v3/adapters/{b}_raw_s{s}_r16", GPU_A100) for s in range(3) for b in "AB"]
    t += [(f"E_SEEDEXT/adapters/{b}_raw_s{s}_r16", GPU_A100) for s in range(3, 6) for b in "AB"]
    t += [(f"E_SEEDEXT2/adapters/{b}_raw_s{s}_r16", GPU_A100) for s in range(6, 12) for b in "AB"]
    g = [x.replace("/adapters/", "/gens/") for x, _ in t]
    return t, g


# Archive rows of Table 2(b), keyed by their generation macro. GPU per supplement S3 (the records name the card
# only for E_INV_P0_v3 and the Blackwell FLUX seeds; where a record names one, it is checked).
ARCHIVE_ROWS = {
    "nDataPrimaryGens": dict(train=_arch_primary()[0], gen_gpu=GPU_A100, gen_folders=_arch_primary()[1]),
    "nDataBaseGens": dict(train=[], gen_gpu=GPU_A100, gen_folders=["E_INV_P0_v3/gens/base"]),
    "nDataFluxGens": dict(train=[(f"E_TRACKB2_FLUX/adapters/{b}_raw_s{s}_flux", GPU_A100L if s < 3 else GPU_BW)
                                 for s in range(6) for b in "AB"],
                          gen_gpu=None, gen_gpu_measured=lambda f: GPU_A100L if int(re.search(r"_s(\d)_", f).group(1)) < 3 else GPU_BW,
                          gen_folders=[f"E_TRACKB2_FLUX/gens/{b}_raw_s{s}_flux" for s in range(6) for b in "AB"]),
    "nDataFullFTGens": dict(train=[(f"E_FULLFT/ckpt/{b}_ft_s{s}", GPU_A100L) for s in range(3) for b in "AB"],
                            gen_gpu=None, gen_gpu_measured=lambda f: GPU_A100L,
                            gen_folders=[f"E_FULLFT/gens/{b}_ft_s{s}" for s in range(3) for b in "AB"]),
    "nDataKodakGens": dict(train=[(f"E_MULTIDEV/adapters/D{d}_s{s}", GPU_A100) for d in range(5) for s in range(2)],
                           gen_gpu=GPU_A100, gen_folders=[f"E_MULTIDEV/gens/D{d}_s{s}" for d in range(5) for s in range(2)]),
    "nDataPTwentyGens": dict(train=[(f"E_DAXING/adapters/{d}_s{s}", GPU_A100) for d in range(1101, 1106) for s in range(2)],
                             gen_gpu=GPU_A100,
                             gen_folders=[f"E_DAXING/gens/{d}_s{s}" for d in range(1101, 1106) for s in range(2)]),
}
# Local rows: adapters by folder-name pattern under out/t1/adapters; generations in the same-named folders of
# out/t1/gens unless listed. Rows with note h train nothing new.
LOCAL_ROWS = {
    "nDataPromptBankBaseGens": dict(pattern=None, gen_dirs=[T5_GENS + "/base"]),
    "nDataPromptBankGens": dict(pattern=None, gen_dirs=[T5_GENS + f"/{b}_raw_s{s}_r16" for b in "AB" for s in range(3)]),
    "nDataSecondSetGens": dict(pattern=r"alt_[AB]_s\d+"),
    "nDataSecondEnvGens": dict(pattern=r"nomarkB?_s\d+"),
    "nDataContentMatchedGens": dict(pattern=r"cm_[AB]_s\d+"),
    "nDataRandomCropGens": dict(pattern=r"rcrop_s\d+"),
    "nDataIPhoneGens": dict(pattern=r"p5c_[AB]_s\d+"),
    "nDataPromptsGens": dict(pattern=None, gen_pattern=r"p5c_[AB]_s\d+_div"),
    "nDataPTwentyPairGens": dict(pattern=r"p20b_[AB]_s\d+"),
    "nDataEightKGens": dict(pattern=r"dose8k_[AB]_s\d+"),
    "nDataSixteenKGens": dict(pattern=r"dose16k_[AB]_s\d+"),
    "nDataInvertedGens": dict(pattern=r"inv16k(ext)?_[AB]_s\d+"),
    "nMapTileGens": dict(pattern=r"per\d+_s\d+"),
    "nMapBandGens": dict(pattern=r"band\d_s\d+"),
    "nMapWmGensAll": dict(pattern=r"wm_ds_s\d+"),
    "nKnownEoneGens": dict(pattern=r"kinjd?_a\d+_s\d+"),
    "nKnownEtwoGens": dict(pattern=r"gk(add|mul)_a\d+_s\d+"),
}
EXTRA_LOCAL = {
    "ladder": dict(label="Designed-mark ladder (mark_rand_a1/a3/a12, mark_lowmid_a12)",
                   pattern=r"mark_(rand|lowmid)_a\d+_s\d+", extra_gen=[]),
    "environment_chain": dict(label="Environment chain (colab_s0-s2 trained on the local stack; local_base, "
                                    "local_A_raw_s0 and local_B_raw_s0 generated locally with no new adapter)",
                              pattern=r"colab_s\d+", extra_gen=["local_base", "local_A_raw_s0", "local_B_raw_s0"]),
}


def s3_units():
    txt = " ".join(tread(S3_TEX).split())
    out = {}
    for k, (phrase, lo, hi, unit) in S3_UNITS.items():
        assert phrase in txt, f"S3 phrase not found: {phrase}"
        out[k] = {"phrase": phrase, "low_min": lo, "high_min": hi, "unit": unit}
    return out


def train_archive(rel, gpu):
    m = jload(f"{DRIVE}/{rel}/train_meta.json")
    rec = m.get("minutes")
    g = m.get("gpu")
    if g is None and isinstance(m.get("env"), dict):
        g = m["env"].get("gpu")
    if g is not None:
        if gpu == GPU_A100:
            assert "A100-SXM4-40GB" in g, (rel, g)
        elif gpu == GPU_BW:
            assert "Blackwell" in g, (rel, g)
        elif gpu == GPU_A100L:
            assert "A100" in g and "40GB" not in g, (rel, g)
    return {"adapter": rel, "steps": m.get("steps"), "gpu": gpu, "gpu_in_record": g,
            "minutes_recorded": None if rec is None else float(rec)}


def train_local(tag):
    p = f"{LOCAL_ADAPTERS}/{tag}/train_meta.json"
    m = jload(p)
    end = os.path.getmtime(p)
    mins = float(m["minutes"])
    return {"adapter": tag, "steps": int(m["steps"]), "gpu": GPU_L40S, "minutes_recorded": mins,
            "end_utc": utc(end), "_interval": (end - 60.0 * mins, end)}


def s3_train(a, units):
    if a["gpu"] == GPU_A100 and a["steps"] == 2000:
        k = "A100_adapter"
    elif a["gpu"] == GPU_BW:
        k = "Blackwell_FLUX_adapter"
    elif a["gpu"] == GPU_L40S:
        k = {2000: "L40S_adapter_2000", 8000: "L40S_adapter_8000", 16000: "L40S_adapter_16000"}[a["steps"]]
    else:
        raise ValueError(f"no S3 per-unit training time for {a['adapter']} on {a['gpu']}")
    return k, units[k]["low_min"], units[k]["high_min"]


def s3_gen(gpu, unit, units):
    if gpu == GPU_A100 and unit == 500:
        return "A100_gen500"
    if gpu == GPU_L40S:
        return {500: "L40S_gen500", 250: "L40S_gen250"}[unit]
    return None


def measure_folder(path, expected):
    """Descriptive: a generation job's active time from its images' write times (not the registered rule)."""
    try:
        ents = [e for e in os.scandir(path) if e.name.lower().endswith(".png")]
    except OSError as exc:
        return {"folder": path, "n": 0, "valid": False, "why": f"unreadable: {exc}"}
    ents.sort(key=lambda e: e.name)
    n = len(ents)
    if n == 0:
        return {"folder": path, "n": 0, "valid": False, "why": "no images"}
    ts = [e.stat().st_mtime for e in ents]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    pos = [g for g in gaps if g > 1.0]
    step = statistics.median(pos) if pos else None
    segs, s0 = [], 0
    for i, g in enumerate(gaps):
        if g > GAP_BREAK_S or g < -1.0:
            segs.append((s0, i))
            s0 = i + 1
    segs.append((s0, n - 1))
    intervals = [(ts[i] - (step or 0.0), ts[j]) for i, j in segs]
    active = sum(b - a for a, b in intervals)
    neg = sum(g < -1.0 for g in gaps)
    why = []
    if n != expected:
        why.append(f"{n} images, expected {expected}")
    if step is None or step < 2.0:
        why.append("no inter-batch gaps of more than 2 s: the times are not generation times")
    if neg:
        why.append(f"{neg} out-of-order write times")
    return {"folder": path, "n": n, "first_utc": utc(ts[0]), "last_utc": utc(ts[-1]),
            "span_min": (ts[-1] - ts[0]) / 60.0, "step_s": step, "segments": len(segs),
            "interruptions_min": sum(g for g in gaps if g > GAP_BREAK_S) / 60.0,
            "active_min": active / 60.0, "valid": not why, "why": "; ".join(why), "_intervals": intervals}


def union_hours(ivs):
    ivs = sorted(ivs)
    tot, cur = 0.0, None
    for a, b in ivs:
        if cur is None:
            cur = [a, b]
        elif a <= cur[1]:
            cur[1] = max(cur[1], b)
        else:
            tot += cur[1] - cur[0]
            cur = [a, b]
    if cur is not None:
        tot += cur[1] - cur[0]
    return tot / 3600.0


def item_c(nums, table_rows, obj_row):
    units = s3_units()
    local_tags = sorted(d for d in os.listdir(LOCAL_ADAPTERS) if os.path.isdir(f"{LOCAL_ADAPTERS}/{d}"))
    owner = {}
    pats = {k: v["pattern"] for k, v in LOCAL_ROWS.items() if v["pattern"]}
    pats.update({k: v["pattern"] for k, v in EXTRA_LOCAL.items()})
    for key, pat in pats.items():
        for t in local_tags:
            if re.fullmatch(pat, t):
                assert t not in owner, (t, owner.get(t), key)
                owner[t] = key
    assert sorted(owner) == local_tags, sorted(set(local_tags) - set(owner))

    designs = []

    def add(key, label, stack, scope, training, gens, unit, gen_gpu, gen_folders, gen_gpu_measured=None):
        fb_lo = fb_hi = 0.0
        for a in training:
            if a["minutes_recorded"] is None:
                k, lo, hi = s3_train(a, units)
                a["s3_fallback"] = {"key": k, "low_min": lo, "high_min": hi}
                fb_lo += lo
                fb_hi += hi
        if gen_gpu is None:
            g = {"gpu": None, "generations": gens, "unit": unit, "estimated": False, "why": S3_NO_UNIT,
                 "low_min": 0.0, "high_min": 0.0}
        else:
            k = s3_gen(gen_gpu, unit, units)
            assert k is not None and gens % unit == 0, (key, gen_gpu, unit, gens)
            nj = gens // unit
            g = {"gpu": gen_gpu, "generations": gens, "unit": unit, "jobs": nj, "s3_key": k, "estimated": True,
                 "low_min": nj * units[k]["low_min"], "high_min": nj * units[k]["high_min"]}
        meas = [measure_folder(f, unit) for f in gen_folders]
        for mrec, f in zip(meas, gen_folders):
            mrec["gpu"] = gen_gpu if gen_gpu is not None else gen_gpu_measured(f)
        designs.append({"key": key, "design": label, "table2_stack": stack, "scope": scope,
                        "adapters_trained": len(training),
                        "adapters_with_record": sum(a["minutes_recorded"] is not None for a in training),
                        "training_recorded_min": sum(a["minutes_recorded"] or 0.0 for a in training),
                        "training_s3_low_min": fb_lo, "training_s3_high_min": fb_hi,
                        "training": training, "generation": g, "generation_measured": meas})

    for r in table_rows:
        if r["note"] == "i":
            continue                                   # subset of the row above: no new adapter, no new image
        gm = r["gens_macro"]
        unit = r["gens_per_adapter"] or r["gens_counted"]
        if gm in ARCHIVE_ROWS:
            spec = ARCHIVE_ROWS[gm]
            assert r["stack"] == "archive", gm
            tr = [train_archive(p, gpu) for p, gpu in spec["train"]]
            folders = [f"{DRIVE}/{f}" for f in spec["gen_folders"]]
            add(gm, r["design"], r["stack"], "N-A5", tr, r["gens_counted"], unit, spec["gen_gpu"], folders,
                spec.get("gen_gpu_measured"))
        else:
            spec = LOCAL_ROWS[gm]
            assert r["stack"] in ("local", "archive/local"), gm
            tags = [t for t in local_tags if owner[t] == gm]
            tr = [train_local(t) for t in tags]
            if spec.get("gen_dirs"):
                folders = list(spec["gen_dirs"])
            elif spec.get("gen_pattern"):
                folders = [f"{LOCAL_GENS}/{d}" for d in sorted(os.listdir(LOCAL_GENS)) if re.fullmatch(spec["gen_pattern"], d)]
            else:
                folders = [f"{LOCAL_GENS}/{t}" for t in tags]
            add(gm, r["design"], r["stack"], "N-A5", tr, r["gens_counted"], unit, GPU_L40S, folders)
        d = designs[-1]
        assert d["adapters_trained"] == r["adapters_counted"], (gm, d["adapters_trained"], r["adapters_counted"])
        nimg = sum(m["n"] for m in d["generation_measured"])
        if r["stack"] in ("local", "archive/local"):
            assert nimg == r["gens_counted"], (gm, nimg, r["gens_counted"])   # live folder count = Table 2(b)
        d["images_found_in_folders"] = nimg

    # all work: the objective arms (archive), the designed-mark ladder and the environment chain (local)
    obj_dirs = sorted(d for d in os.listdir(f"{DRIVE}/E_AMP/adapters")
                      if os.path.isfile(f"{DRIVE}/E_AMP/adapters/{d}/train_meta.json"))
    assert len(obj_dirs) == obj_row["adapters"] == num(nums, "nObjAdapters"), obj_dirs
    add("objective_arms", "Training objective, tap T2 (objective arms)", "archive", "all-work extra",
        [train_archive(f"E_AMP/adapters/{d}", GPU_A100) for d in obj_dirs],
        obj_row["adapters"] * obj_row["gens_per_adapter"], obj_row["gens_per_adapter"], GPU_A100,
        [f"{DRIVE}/E_AMP/gens/{d}" for d in obj_dirs])
    designs[-1]["images_found_in_folders"] = sum(m["n"] for m in designs[-1]["generation_measured"])
    for key, spec in EXTRA_LOCAL.items():
        tags = [t for t in local_tags if owner[t] == key]
        folders = [f"{LOCAL_GENS}/{t}" for t in tags] + [f"{LOCAL_GENS}/{t}" for t in spec["extra_gen"]]
        meas_n = [len([e for e in os.scandir(f) if e.name.lower().endswith(".png")]) for f in folders]
        assert set(meas_n) == {500}, (key, meas_n)
        add(key, spec["label"], "local", "all-work extra", [train_local(t) for t in tags], sum(meas_n), 500,
            GPU_L40S, folders)
        designs[-1]["images_found_in_folders"] = sum(meas_n)
    n_ladder = sum(1 for t in local_tags if owner[t] == "ladder")
    n_env = sum(1 for t in local_tags if owner[t] == "environment_chain")
    assert n_ladder == 6 and n_env == 3, (n_ladder, n_env)

    # ---------------------------------------------------------------- registered estimate
    def totals(scopes):
        sel = [d for d in designs if d["scope"] in scopes]
        per = {}
        for g in COLAB + (GPU_L40S,):
            tr = [a for d in sel for a in d["training"] if a["gpu"] == g]
            rec = sum(a["minutes_recorded"] for a in tr if a["minutes_recorded"] is not None) / 60.0
            flo = sum(a["s3_fallback"]["low_min"] for a in tr if a["minutes_recorded"] is None) / 60.0
            fhi = sum(a["s3_fallback"]["high_min"] for a in tr if a["minutes_recorded"] is None) / 60.0
            gs = [d["generation"] for d in sel if d["generation"]["gpu"] == g]
            per[g] = {"adapters_trained": len(tr), "adapters_with_record": sum(a["minutes_recorded"] is not None for a in tr),
                      "generations_estimated": sum(x["generations"] for x in gs),
                      "training_recorded_h": rec, "training_s3_low_h": flo, "training_s3_high_h": fhi,
                      "generation_low_h": sum(x["low_min"] for x in gs) / 60.0,
                      "generation_high_h": sum(x["high_min"] for x in gs) / 60.0}

        def agg(names):
            a = {k: sum(per[n][k] for n in names) for k in per[names[0]]}
            a["training_low_h"] = a["training_recorded_h"] + a["training_s3_low_h"]
            a["training_high_h"] = a["training_recorded_h"] + a["training_s3_high_h"]
            a["total_low_h"] = a["training_low_h"] + a["generation_low_h"]
            a["total_high_h"] = a["training_high_h"] + a["generation_high_h"]
            a["recorded_share_of_total_at_low"] = a["training_recorded_h"] / a["total_low_h"] if a["total_low_h"] else None
            a["recorded_share_of_total_at_high"] = a["training_recorded_h"] / a["total_high_h"] if a["total_high_h"] else None
            return a

        out = {"by_card": {g: agg([g]) for g in per},
               GROUP_COLAB: agg(list(COLAB)), GROUP_LOCAL: agg([GPU_L40S]),
               "all GPUs": agg(list(COLAB) + [GPU_L40S])}
        out["adapters_trained"] = sum(d["adapters_trained"] for d in sel)
        out["generations"] = sum(d["generation"]["generations"] for d in sel)
        out["generations_not_estimated"] = [{"design": d["design"], "generations": d["generation"]["generations"],
                                             "why": d["generation"]["why"]} for d in sel if not d["generation"]["estimated"]]
        return out

    na5 = totals({"N-A5"})
    assert na5["adapters_trained"] == 211 and na5["generations"] == 95500, (na5["adapters_trained"], na5["generations"])
    allw = totals({"N-A5", "all-work extra"})
    assert allw["adapters_trained"] == 211 + 7 + 6 + 3 and allw["generations"] == 95500 + 3500 + 3000 + 3000

    # ---------------------------------------------------------------- descriptive: generation measured from write times
    def measured(scopes):
        sel = [d for d in designs if d["scope"] in scopes]
        per, bad, ivs_train, ivs_gen = {}, [], [], []
        for g in COLAB + (GPU_L40S,):
            tr = [a for d in sel for a in d["training"] if a["gpu"] == g]
            ms = [m for d in sel for m in d["generation_measured"] if m["gpu"] == g]
            ok = [m for m in ms if m["valid"]]
            per[g] = {"training_recorded_h": sum(a["minutes_recorded"] or 0.0 for a in tr) / 60.0,
                      "training_unrecorded_adapters": [a["adapter"] for a in tr if a["minutes_recorded"] is None],
                      "training_s3_fallback_h": sum(a["s3_fallback"]["low_min"] for a in tr
                                                    if a["minutes_recorded"] is None) / 60.0,
                      "generation_jobs": len(ms), "generation_jobs_valid": len(ok),
                      "generation_images_valid": sum(m["n"] for m in ok),
                      "generation_measured_h": sum(m["active_min"] for m in ok) / 60.0}
            assert all(a["s3_fallback"]["low_min"] == a["s3_fallback"]["high_min"] for a in tr
                       if a["minutes_recorded"] is None)
            per[g]["total_h"] = (per[g]["training_recorded_h"] + per[g]["training_s3_fallback_h"]
                                 + per[g]["generation_measured_h"])
            bad += [{"folder": m["folder"], "n": m["n"], "why": m["why"]} for m in ms if not m["valid"]]
            if g == GPU_L40S:
                ivs_train += [a["_interval"] for a in tr]
                ivs_gen += [iv for m in ok for iv in m["_intervals"]]
        colab = {k: sum(per[g][k] for g in COLAB) for k in ("training_recorded_h", "training_s3_fallback_h",
                                                               "generation_measured_h", "total_h",
                                                               "generation_jobs", "generation_jobs_valid",
                                                               "generation_images_valid")}
        occ = {"training_jobs": len(ivs_train), "generation_segments": len(ivs_gen),
               "sum_of_job_hours": sum(b - a for a, b in ivs_train + ivs_gen) / 3600.0,
               "union_hours_training_only": union_hours(ivs_train),
               "union_hours_generation_only": union_hours(ivs_gen),
               "union_hours_training_and_generation": union_hours(ivs_train + ivs_gen)}
        return {"by_card": per, GROUP_COLAB: colab, GROUP_LOCAL: per[GPU_L40S], "L40S_occupancy": occ,
                "folders_not_measured": bad}

    meas_na5 = measured({"N-A5"})
    meas_all = measured({"N-A5", "all-work extra"})

    # per-job comparison of measured generation times with S3's units
    def jobstats(gpu, unit):
        xs = [m["active_min"] for d in designs for m in d["generation_measured"]
              if m["gpu"] == gpu and m["valid"] and m["n"] == unit]
        if not xs:
            return None
        return {"jobs": len(xs), "min": min(xs), "median": statistics.median(xs), "max": max(xs),
                "mean": statistics.fmean(xs)}
    per_job = {"L40S_500": jobstats(GPU_L40S, 500), "L40S_250": jobstats(GPU_L40S, 250),
               "A100_40GB_500": jobstats(GPU_A100, 500), "larger_A100_500": jobstats(GPU_A100L, 500),
               "Blackwell_500": jobstats(GPU_BW, 500),
               "S3_units_min": {k: [v["low_min"], v["high_min"]] for k, v in units.items()}}
    if per_job["L40S_250"]:
        xs = [m["active_min"] for d in designs for m in d["generation_measured"]
              if m["gpu"] == GPU_L40S and m["valid"] and m["n"] == 250]
        per_job["L40S_250"]["share_above_S3_35_min_by_more_than_10pct"] = sum(x > 38.5 for x in xs) / len(xs)
    if per_job["L40S_500"]:
        xs = [m["active_min"] for d in designs for m in d["generation_measured"]
              if m["gpu"] == GPU_L40S and m["valid"] and m["n"] == 500]
        per_job["L40S_500"]["share_above_S3_74_min"] = sum(x > 74.0 for x in xs) / len(xs)
        per_job["L40S_500"]["share_below_S3_62_min"] = sum(x < 62.0 for x in xs) / len(xs)

    # training records against S3's per-unit times
    tr_vs = {}
    for name, gpu, steps, k in (("L40S_2000", GPU_L40S, 2000, "L40S_adapter_2000"),
                                ("L40S_8000", GPU_L40S, 8000, "L40S_adapter_8000"),
                                ("L40S_16000", GPU_L40S, 16000, "L40S_adapter_16000"),
                                ("A100_40GB_2000", GPU_A100, 2000, "A100_adapter"),
                                ("Blackwell_FLUX_2000", GPU_BW, 2000, "Blackwell_FLUX_adapter")):
        recs = [(a["adapter"], a["minutes_recorded"]) for d in designs for a in d["training"]
                if a["gpu"] == gpu and a["steps"] == steps and a["minutes_recorded"] is not None]
        xs = [m for _, m in recs]
        tr_vs[name] = {"n": len(xs), "min": min(xs), "median": statistics.median(xs), "max": max(xs),
                       "s3_low_min": units[k]["low_min"], "s3_high_min": units[k]["high_min"],
                       "above_s3_high": sorted(([a, m] for a, m in recs if m > units[k]["high_min"] * 1.005),
                                               key=lambda r: -r[1]),
                       "below_s3_low": sorted(([a, m] for a, m in recs if m < units[k]["low_min"] * 0.995),
                                              key=lambda r: r[1])}
    others_l = {}
    for name, gpu, steps in (("larger_A100_FLUX_2000", GPU_A100L, 2000), ("larger_A100_fullFT_4000", GPU_A100L, 4000)):
        xs = [a["minutes_recorded"] for d in designs for a in d["training"]
              if a["gpu"] == gpu and a["steps"] == steps and a["minutes_recorded"] is not None]
        others_l[name] = {"n": len(xs), "min": min(xs), "max": max(xs), "s3": "no per-unit time in S3"}
    tr_vs.update(others_l)

    # archive training records in neither scope (pilot and unreported arms): listed, not added
    in_scope = {a["adapter"] for d in designs for a in d["training"] if a["gpu"] in COLAB}
    others = []
    for root, dirs, files in os.walk(DRIVE):
        if "train_meta.json" in files:
            rel = os.path.relpath(root, DRIVE).replace("\\", "/")
            if rel not in in_scope:
                m = jload(os.path.join(root, "train_meta.json"))
                others.append({"adapter": rel, "steps": m.get("steps"), "minutes_recorded": m.get("minutes")})
        dirs[:] = [x for x in dirs if x not in ("gens", "train_png", "cache_crops", "fingerprints", "csv", "figs")]

    for d in designs:                                   # per-design summary of the descriptive measurement
        ok = [m for m in d["generation_measured"] if m["valid"]]
        d["generation_measured_summary"] = {
            "jobs": len(d["generation_measured"]), "jobs_valid": len(ok), "images_valid": sum(m["n"] for m in ok),
            "measured_h": sum(m["active_min"] for m in ok) / 60.0,
            "registered_s3_low_h": d["generation"]["low_min"] / 60.0,
            "registered_s3_high_h": d["generation"]["high_min"] / 60.0,
            "registered_estimated": d["generation"]["estimated"]}
    for d in designs:                                   # private fields out of the record
        for a in d["training"]:
            a.pop("_interval", None)
        for m in d["generation_measured"]:
            m.pop("_intervals", None)

    def brief(t):
        keep = ("training_recorded_h", "training_s3_low_h", "training_s3_high_h", "generation_low_h",
                "generation_high_h", "total_low_h", "total_high_h", "recorded_share_of_total_at_low",
                "recorded_share_of_total_at_high", "adapters_trained", "adapters_with_record", "generations_estimated")
        return {k: t[k] for k in keep}

    return {"question": "trace-r2-12: total GPU-hours, split by GPU",
            "definition": ("GPU-hours = summed wall-clock hours of the training and generation jobs on each card "
                           "(job-hours). Training: train_meta.json 'minutes' (the training loop, model loading "
                           "excluded) where recorded, else supplement S3's per-unit time. Generation: S3's per-unit "
                           "range x the number of jobs of 500 or 250 images (Table 2(b) generations per adapter). "
                           "Card per supplement S3 (A100-SXM4-40GB for the SD-3.5 archive adapters; a "
                           "larger-memory A100 for FLUX.1-dev seeds 0-2 and full fine-tuning; RTX PRO 6000 Blackwell "
                           "for FLUX.1-dev seeds 3-5; the L40S for every local and archive/local job). Excluded: "
                           "detector, embedding, scoring and autoencoder round-trip GPU time, model loading, and jobs "
                           "that were interrupted and re-run (for example about 11 GPU-hours of inv16kext_B_s3 lost "
                           "in D13, Entry 104). Jobs that shared the L40S are each counted in full."),
            "s3_per_unit_minutes": units,
            "registered_estimate": {
                "N-A5 (211 adapters, 95,500 generations)": na5,
                "all work (N-A5 + objective arms + designed-mark ladder + environment chain)": allw,
                "headline": {"N-A5": {GROUP_COLAB: brief(na5[GROUP_COLAB]), GROUP_LOCAL: brief(na5[GROUP_LOCAL]),
                                      "all GPUs": brief(na5["all GPUs"])},
                             "all work": {GROUP_COLAB: brief(allw[GROUP_COLAB]), GROUP_LOCAL: brief(allw[GROUP_LOCAL]),
                                          "all GPUs": brief(allw["all GPUs"])}}},
            "beside_not_registered": {
                "what": "generation measured from the write times of the generated images (local folders for the "
                        "L40S; the Drive mirror for Colab, where the files keep the time Colab wrote them), "
                        "training from the records; a folder counts only if it holds the expected images and its "
                        "inter-batch gaps exceed 2 s; gaps over 600 s are interruptions and not counted",
                "N-A5": meas_na5, "all work": meas_all,
                "per_job_generation_minutes_vs_S3": per_job,
                "training_records_vs_S3": tr_vs},
            "designs": designs,
            "archive_training_records_in_neither_scope": {
                "n": len(others), "recorded_hours": sum((o["minutes_recorded"] or 0.0) for o in others) / 60.0,
                "adapters": others, "note": "trained on Colab with a record, in neither registered scope; not added"},
            "environment_chain_count": ("Entry 112 lists the environment chain as 5 adapters; 3 were trained "
                                        "(colab_s0-s2). local_A_raw_s0 and local_B_raw_s0 are the archived primary "
                                        "seed-0 adapters generating on the local stack (Entries 25, 30), whose "
                                        "training is counted in the primary row; all work is therefore 227 trained "
                                        "adapters and 105,000 generations")}


# ================================================================================================ comparison
def compare_earlier(res):
    if not os.path.exists(EARLIER):
        return {"exists": False}
    e = jload(EARLIER)
    out = {"exists": True, "file": "out/fv_small_quantities.json", "run_utc": (e.get("run") or {}).get("utc"),
           "note": "an earlier attempt at this item; nothing above is read from it"}
    pairs = []

    def g(d, path):
        for k in path:
            d = d[k]
        return d
    try:
        pairs += [("a nominal lambda_U_pct", res["a"]["nominal_six_per_arm"]["lambda_U_pct"],
                   g(e, ["a_local_stack_maxarm", "nominal_six_per_arm", "lambda_U_pct"])),
                  ("a without nomark_s5 lambda_U_pct", res["a"]["post_hoc_without_nomark_s5"]["lambda_U_pct"],
                   g(e, ["a_local_stack_maxarm", "post_hoc_without_nomark_s5", "lambda_U_pct"]))]
        for mine, theirs in (("entry68_replication_new_three_per_body", "entry68_replication_new_three_per_body"),
                             ("entry68_pooled_six_per_body", "entry68_pooled_six_per_body"),
                             ("g1_inversion_eight_per_arm", "g1_inversion_eight_per_arm")):
            pairs.append((f"b {mine} p_infl", res["b"]["results"][mine]["inflated"]["one_sided_p_infl"],
                          g(e, ["b_inflated_p", "results", theirs, "one_sided_p_infl"])))
        pairs.append(("d t", res["d"]["t"], g(e, ["d_inversion_first_reading", "t"])))
        ev = g(e, ["e_counts", "verifier"])
        pairs.append(("e verifier checks executed", res["e"]["verifier"]["checks_executed"], ev["checks_executed"]))
        pairs.append(("e verifier checks agreeing", res["e"]["verifier"]["checks_agreeing"], ev["checks_agreeing"]))
        for st in ("archive", "local", "archive/local"):
            pairs.append((f"e {st} adapters", res["e"]["table2b_by_stack"]["by_stack"][st]["adapters"],
                          g(e, ["e_counts", "table2b_by_stack", "by_stack", st, "adapters"])))
            pairs.append((f"e {st} generations", res["e"]["table2b_by_stack"]["by_stack"][st]["generations"],
                          g(e, ["e_counts", "table2b_by_stack", "by_stack", st, "gens"])))
        ce = g(e, ["c_gpu_hours", "N-A5 (211 adapters, 95,500 generations)", "by_gpu"])
        cm = res["c"]["registered_estimate"]["N-A5 (211 adapters, 95,500 generations)"]
        for grp_m, grp_e in ((GROUP_COLAB, "Colab (archive stack), all cards"), (GROUP_LOCAL, GPU_L40S),
                             ("all GPUs", "all GPUs")):
            for k in ("total_low_h", "total_high_h"):
                pairs.append((f"c N-A5 {grp_m} {k}", cm[grp_m][k], ce[grp_e][k]))
    except (KeyError, TypeError) as exc:
        out["incomplete"] = f"key not found in the earlier file: {exc}"
    out["pairs"] = [{"quantity": q, "this_run": a, "earlier": b,
                     "abs_diff": (abs(float(a) - float(b)) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None)}
                    for q, a, b in pairs]
    out["all_equal_within_1e-9_relative"] = all(
        p["abs_diff"] is not None and p["abs_diff"] <= 1e-9 * max(1.0, abs(float(p["this_run"]))) for p in out["pairs"])
    return out


# ================================================================================================ main
def main():
    if os.path.exists(DST):
        raise SystemExit(f"{DST} exists; result files are never overwritten")
    t0 = datetime.datetime.now(datetime.timezone.utc)
    R, led = ledger()
    nums = jload(NUMBERS)
    res = {"entry": "RESULTS.md Entry 116, item 6 (small quantities; descriptive, from existing files)",
           "script": "src/fv/fv_small.py", "output": "out/fv_small.json",
           "names": {"registered": "src/fv/fv_small.py -> out/fv_small.json (Entry 116, item 6 heading)",
                     "this_run": "the registered names",
                     "why_not_the_requested_names": "the run asked for src/fv/fv_small_quantities.py -> "
                                                    "out/fv_small_quantities.json, which an earlier attempt already "
                                                    "wrote (30 Sep 2026); result files are never overwritten"},
           "status": "descriptive; no reading attaches (Entry 116: items 5 and 6 are descriptive)",
           "run": {"started_utc": t0.isoformat(timespec="seconds"), "python": sys.version.split()[0],
                   "numpy": np.__version__, "scipy": scipy.__version__, "gpu": "not used"},
           "settings": {"R_real": R, "R_real_printed": R_REAL_PRINTED, "max_arm_quantile": 0.995,
                        "sd_est_printed": SD_EST_PRINTED, "timestamp_gap_break_s": GAP_BREAK_S,
                        "image_folders": "generated-image folders are read for file counts and write times only; "
                                         "the images are not opened or hashed (inputs_sha256 lists the files read)"}}
    res["a"] = item_a(R, led)
    res["b"] = item_b(R)
    res["d"] = item_d(R)
    rows, obj = parse_table2b(nums)
    res["e"] = {"verifier": item_e_verifier(), "table2b_by_stack": item_e_table(nums),
                "deposit_file_count": "not computed: waits on author actions (Entry 116 item 6e)"}
    res["c"] = item_c(nums, rows, obj)
    res["comparison_with_earlier_attempt"] = compare_earlier(res)
    res["inputs_sha256"] = dict(sorted(INPUTS.items()))
    res["run"]["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    ordered = {k: res[k] for k in ("entry", "script", "output", "names", "status", "run", "settings")}
    ordered["a_hostile_r2_04_local_stack_max_arm"] = res["a"]
    ordered["b_hostile_r2_14_inflated_p"] = res["b"]
    ordered["c_trace_r2_12_gpu_hours"] = res["c"]
    ordered["d_figures_r2_11_nDoseInvThreeT"] = res["d"]
    ordered["e_trace_r2_09_counts"] = res["e"]
    ordered["comparison_with_earlier_attempt"] = res["comparison_with_earlier_attempt"]
    ordered["inputs_sha256"] = res["inputs_sha256"]
    with open(DST, "x", encoding="utf-8") as f:
        json.dump(ordered, f, indent=1)
    print(f"[fv_small] wrote {DST}")


if __name__ == "__main__":
    main()
