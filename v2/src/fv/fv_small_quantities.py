"""Small quantities for the manuscript: RESULTS.md Entry 116, item 6 (descriptive, from existing files).

CPU only; about a minute (the repository verifier runs as a subprocess for a few seconds). Entry 116 registered
this item as src/fv/fv_small.py -> out/fv_small.json. The run that executed it asked for the names
src/fv/fv_small_quantities.py -> out/fv_small_quantities.json; that is the only deviation, and the output
records both names.

  (a) hostile-r2-04  Nominal max-arm limit of the primary design retrained on the local stack (G2, six per arm):
                     U_X = mean + t_{0.995,5} s / sqrt(6), U = max(U_A, U_B), as % of R_real (c = 1). Beside it,
                     post hoc: the same with nomark_s5 left out (arm A at five, t_{0.995,4}); and the archive's
                     six-per-arm limit (ledger history_k6, 0.249 %), recomputed from the ledger's first six.
  (b) hostile-r2-14  16000-step symmetric p-values with the E2 estimation shift added to the Welch SE:
                     SE_infl = sqrt(SE_Welch^2 + sd_est^2), t = theta_sym / SE_infl, one-sided at the Welch df,
                     for the Entry 68 replication (three per body), the pooled six per body, and the G1 inversion
                     at eight per arm (direction theta < 0). Each uninflated p is asserted to reproduce its entry.
  (c) trace-r2-12    GPU-hours by GPU for the 211 adapters / 95,500 generations of Entry 112 N-A5, and for all
                     work (adding the objective arms, the designed-mark ladder and the environment chain).
                     Training: recorded wall time (train_meta.json "minutes") where a record exists, otherwise
                     the per-unit times of supplement S3. Generation: S3's per-unit times x the number of units.
                     Reported low-high with the recorded share. Detector, embedding and scoring GPU time excluded.
  (d) figures-r2-11  nDoseInvThreeT: Welch t of the first inversion reading (seeds 0-2 per arm, out/g1_ext.json).
                     Stops unless it reproduces Entry 98's theta_sym (-0.143 %) and p (0.145).
  (e) trace-r2-09    The verifier's check count (python verify_v2.py in the repository clone), and adapters and
                     generations by stack exactly as Table 2(b) labels them, asserted to sum to 211 and 95,500.

Constants typed in are only those the registration fixes (R_real 0.0356703; the estimation SD 1.224e-05, read
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
from out/h6_calibrated_limit.json and checked against it; the p-values printed in Entries 68, 98 and 110, used
as reproduction checks) and the per-unit wall-clock times of supplement S3, each asserted to appear literally in
paper/fv/supp/S03_compute.tex.

Run:  python src/fv/fv_small_quantities.py [--out PATH]
The default output (out/fv_small_quantities.json) is never overwritten: the script stops if it exists.
"""
import argparse
import datetime
import glob
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
REPO = EINV.REPO
LEDGER = REPO + "/analysis/FINAL_LEDGER.json"
DRIVE = (EINV.MYDRIVE + "/inv_channel")
FV = V2 + "/paper/fv"
NUMBERS = FV + "/numbers.json"
TAB2 = FV + "/tabs/tab02_devices.tex"
S3_TEX = FV + "/supp/S03_compute.tex"
INV_GENS = FV + "/work/inv_gens.json"
LOCAL_ADAPTERS = T1 + "/adapters"
LOGS = V2 + "/logs"
DST_DEFAULT = OUT + "/fv_small_quantities.json"

R_REAL_REGISTERED = 0.0356703      # Entry 116 item 6(a), printed to seven digits
SD_EST_REGISTERED = 1.224e-05      # Entry 116 item 6(b), printed to four significant figures

INPUTS = {}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def note_input(path):
    key = path.replace("\\", "/")
    if key not in INPUTS:
        INPUTS[key] = sha256(path)


def load(path):
    note_input(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_text(path):
    note_input(path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def close(a, b, rel=1e-12, abs_=0.0):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def welch_sym(a, b):
    """theta_sym = (mean_A + mean_B) / 2 with its Welch SE and df (as verify_v2.welch_sym)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    th = 0.5 * (a.mean() + b.mean())
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = 0.5 * math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(th), float(se), float(df)


def arm_upper(x, mult=1.0):
    """One arm of the max-arm construction: mean + mult * t_{0.995, n-1} * s / sqrt(n)."""
    x = np.asarray(x, float)
    n = len(x)
    tq = float(stats.t.ppf(0.995, n - 1))
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "t_0995": tq,
            "half_width": float(mult * tq * x.std(ddof=1) / math.sqrt(n)),
            "U": float(x.mean() + mult * tq * x.std(ddof=1) / math.sqrt(n))}


def ledger_R():
    R = float(load(LEDGER)["denominators"]["R_real"])
    assert abs(R - R_REAL_REGISTERED) < 5e-8, R
    return R


# =================================================================================================== (a)
def item_a(R):
    g = load(OUT + "/g2_pooled_six.json")
    A = [float(v) for v in g["per_adapter_A"]]
    B = [float(v) for v in g["per_adapter_B"]]
    S = {}
    for f in ("summary_nomark.json", "summary_nomarkrep.json", "summary_nomarkB.json"):
        S.update(load(T1 + "/" + f)["arms"])
    names_A = [f"nomark_s{i}" for i in range(6)]
    names_B = [f"nomarkB_s{i}" for i in range(6)]
    # the file's order is seeds 0-5; arm B is stored as own-minus-other, the summaries as rho(K_A) - rho(K_B)
    assert A == [float(S[n]["natural_paired_KA_minus_KB"]) for n in names_A], "arm A order"
    assert B == [-float(S[n]["natural_paired_KA_minus_KB"]) for n in names_B], "arm B order"
    th, se, df = welch_sym(A, B)
    assert close(th, g["theta_sym"]) and close(se, g["welch_se"]) and close(df, g["welch_df"]), "G2 file"

    uA, uB = arm_upper(A), arm_upper(B)
    U = max(uA["U"], uB["U"])
    nominal = {"U_A": uA, "U_B": uB, "U": U, "set_by_arm": "A" if uA["U"] >= uB["U"] else "B",
               "lambda_U_pct": 100 * U / R, "lambda_U_A_pct": 100 * uA["U"] / R,
               "lambda_U_B_pct": 100 * uB["U"] / R, "c": 1.0,
               "construction": "U_X = mean_X + t_{0.995,5} s_X / sqrt(6); U = max(U_A, U_B); / R_real"}

    # post hoc: nomark_s5 (seed 5 of body A, index 5) left out; arm B keeps its six
    assert names_A[5] == "nomark_s5"
    uA5 = arm_upper(A[:5])
    U5 = max(uA5["U"], uB["U"])
    without_s5 = {"left_out": "nomark_s5", "left_out_value": A[5], "left_out_value_pct": 100 * A[5] / R,
                  "U_A_five": uA5, "U_B_six": uB, "U": U5,
                  "set_by_arm": "A" if uA5["U"] >= uB["U"] else "B",
                  "lambda_U_pct": 100 * U5 / R, "lambda_U_A_pct": 100 * uA5["U"] / R,
                  "status": "post hoc (Entry 116 item 6a: labelled post hoc)"}

    led = load(LEDGER)
    h = led["history_k6"]
    pa = [float(v) for v in led["primary"]["per_adapter_A"][:6]]
    pb = [float(v) for v in led["primary"]["per_adapter_B"][:6]]
    kA, kB = arm_upper(pa), arm_upper(pb)
    Uk6 = max(kA["U"], kB["U"])
    pct_k6 = 100 * Uk6 / R
    assert round(pct_k6, 3) == round(float(h["lambda_U_plugin_pct"]), 3), (pct_k6, h["lambda_U_plugin_pct"])
    assert abs(Uk6 - float(h["U_device"])) <= 5e-10, (Uk6, h["U_device"])
    archive = {"ledger_history_k6_lambda_U_pct": h["lambda_U_plugin_pct"],
               "ledger_history_k6_U_device": h["U_device"], "ledger_status": h.get("_status"),
               "recomputed_from_ledger_first_six": {"U_A": kA, "U_B": kB, "U": Uk6, "lambda_U_pct": pct_k6,
                                                     "set_by_arm": "A" if kA["U"] >= kB["U"] else "B",
                                                     "note": "ledger primary.per_adapter_A/B, seeds 0-5 "
                                                             "(numeric seed order, per _per_adapter_note)"}}
    return {"question": "hostile-r2-04: Table 3 row 3 (primary design retrained on the local stack) max-arm limit",
            "inputs": {"per_adapter_A": A, "per_adapter_A_labels": names_A,
                       "per_adapter_B": B, "per_adapter_B_labels": names_B,
                       "per_adapter_B_sign": "own-minus-other, i.e. -(rho(K_A) - rho(K_B)) of the summary",
                       "R_real": R, "file_theta_sym": g["theta_sym"],
                       "file_symmetric_limit99_pct": g["limit99_pct"]},
            "nominal_six_per_arm": nominal,
            "post_hoc_without_nomark_s5": without_s5,
            "archive_six_per_arm": archive,
            "note": "c = 1: H6's multiplier (1.25) is not transported to the local stack (Entry 116)",
            "source": "out/g2_pooled_six.json per_adapter_A/B; order checked against out/t1/summary_nomark*.json; "
                      "R_real and history_k6 from analysis/FINAL_LEDGER.json"}


# =================================================================================================== (b)
def item_b(R):
    h6 = load(OUT + "/h6_calibrated_limit.json")
    sd_est = float(h6["estimation_sd"])
    h2 = load(OUT + "/h2_estimation_error.json")
    reps = np.array([r["theta_sym"] for r in h2["replicates"]], float)
    assert close(float(reps.std(ddof=1)), sd_est, rel=1e-12), "estimation SD vs H2 replicates"
    assert abs(sd_est - SD_EST_REGISTERED) < 5e-9, sd_est

    dose = load(T1 + "/dose_stats.json")["doses"]
    S = {}
    for f in ("summary_dose16k.json", "summary_dose16krep.json", "summary_dose16krep2.json"):
        S.update(load(T1 + "/" + f)["arms"])

    def own(i):
        return (float(S[f"dose16k_A_s{i}"]["natural_paired_KA_minus_KB"]),
                -float(S[f"dose16k_B_s{i}"]["natural_paired_KA_minus_KB"]))

    def inflate(th, se, df, direction):
        se_i = math.sqrt(se ** 2 + sd_est ** 2)
        t_i = th / se_i
        p_i = float(stats.t.sf(t_i, df)) if direction == "theta > 0" else float(stats.t.cdf(t_i, df))
        return {"SE_infl": se_i, "t_infl": t_i, "one_sided_p_infl": p_i, "SE_ratio": se_i / se}

    out = {}
    # (i) Entry 68 replication (new three per body) and (ii) pooled six per body
    for name, key, seeds, printed, printed_txt, thr, thr_src in (
            ("entry68_replication_new_three_per_body", "16000_new", (3, 4, 5), 0.00955, "0.00955 (Entry 68)",
             0.05, "Entry 55 registered reading, one-sided 0.05"),
            ("entry68_pooled_six_per_body", "16000", tuple(range(6)), 0.00198, "0.00198 (Entry 68)",
             0.01, "Entry 68 secondary reading, one-sided 0.01")):
        a, b = zip(*[own(i) for i in seeds])
        d = dose[key]
        assert list(a) == [float(v) for v in d["A_own"]] and list(b) == [float(v) for v in d["B_own"]], key
        assert d["A_arms"] == [f"dose16k_A_s{i}" for i in seeds] and d["B_arms"] == [f"dose16k_B_s{i}" for i in seeds]
        th, se, df = welch_sym(a, b)
        assert close(th, d["theta_sym"]) and close(se, d["adapter_SE"]) and close(df, d["welch_df"]) \
            and close(th / se, d["t"]), key
        p = float(stats.t.sf(th / se, df))
        assert abs(p - printed) <= 5e-6, (key, p, printed)
        out[name] = {"seeds": list(seeds), "A_own": list(a), "B_own": list(b), "theta_sym": th,
                     "theta_sym_pct": 100 * th / R, "SE_Welch": se, "df_Welch": df, "t": th / se,
                     "one_sided_p": p, "p_printed_in_entry": printed_txt, "direction": "theta > 0",
                     **inflate(th, se, df, "theta > 0"),
                     "context_threshold": thr, "context_threshold_source": thr_src,
                     "source": f"out/t1/dose_stats.json doses['{key}'] (per-adapter values equal the Entry 68 "
                               "summaries out/t1/summary_dose16k*.json natural_paired_KA_minus_KB)"}
        out[name]["p_infl_below_context_threshold"] = out[name]["one_sided_p_infl"] < thr

    # (iii) G1 inversion at eight per arm (Entry 110), direction theta < 0
    ge = load(OUT + "/g1_ext.json")
    a8 = [float(ge["per_adapter_A"][f"s{i}"]) for i in range(8)]
    b8 = [float(ge["per_adapter_B"][f"s{i}"]) for i in range(8)]
    th, se, df = welch_sym(a8, b8)
    e = ge["all_eight"]
    assert close(th, e["theta_sym"]) and close(se, e["welch_se"]) and close(df, e["welch_df"]) \
        and close(th / se, e["t"]), "g1_ext all_eight"
    p = float(stats.t.cdf(th / se, df))
    assert close(p, e["one_sided_p_lt0"], rel=1e-9) and abs(p - 0.0106) <= 5e-5, p
    out["g1_inversion_eight_per_arm"] = {
        "seeds": list(range(8)), "A_own": a8, "B_own": b8, "theta_sym": th, "theta_sym_pct": 100 * th / R,
        "SE_Welch": se, "df_Welch": df, "t": th / se, "one_sided_p": p, "p_printed_in_entry": "0.0106 (Entry 110)",
        "direction": "theta < 0", **inflate(th, se, df, "theta < 0"),
        "context_threshold": 0.05, "context_threshold_source": "Entries 77 and 99 registered reading, one-sided 0.05",
        "source": "out/g1_ext.json per_adapter_A/B s0-s7; all_eight reproduced"}
    out["g1_inversion_eight_per_arm"]["p_infl_below_context_threshold"] = \
        out["g1_inversion_eight_per_arm"]["one_sided_p_infl"] < 0.05

    return {"question": "hostile-r2-14: 16000-step symmetric p-values with the E2 estimation shift added",
            "estimation_sd": sd_est,
            "estimation_sd_source": "out/h6_calibrated_limit.json estimation_sd = SD (ddof 1) of theta_sym over "
                                    "the 24 H2 bootstrap replicates (out/h2_estimation_error.json), checked",
            "estimation_sd_measured_at": {"steps": 2000, "images_per_adapter": h2["images_per_adapter"],
                                          "adapters_per_arm": 12, "n_replicates": h2["n_replicates"],
                                          "note": "transported to 16000 steps and 250 images per adapter "
                                                  "(stated); Entry 116 says it was measured at 500 images, but "
                                                  "H2 used 200 per adapter (images_per_adapter in its file)"},
            "formula": "SE_infl = sqrt(SE_Welch^2 + sd_est^2); t = theta_sym / SE_infl; one-sided p at the Welch "
                       "df of each result",
            "results": out,
            "not_inflated": "the within-body inverted-minus-normal contrast (Entry 110): both conditions are "
                            "scored with the same K, so the shift cancels exactly (Entry 116)",
            "reading": "none: Entry 116 item 6 is descriptive; context thresholds are the registered levels of "
                       "the three readings, reported for orientation only"}


# =================================================================================================== (d)
def item_d(R):
    ge = load(OUT + "/g1_ext.json")
    a3 = [float(ge["per_adapter_A"][f"s{i}"]) for i in range(3)]
    b3 = [float(ge["per_adapter_B"][f"s{i}"]) for i in range(3)]
    ch = load(OUT + "/g_chain12.json")["G1"]
    assert a3 == [float(v) for v in ch["per_adapter_A"]] and b3 == [float(v) for v in ch["per_adapter_B"]], \
        "g1_ext seeds 0-2 differ from g_chain12 G1"
    th, se, df = welch_sym(a3, b3)
    t = th / se
    p = float(stats.t.cdf(t, df))
    pct = 100 * th / R
    if not (round(pct, 3) == -0.143 and round(p, 3) == 0.145):
        raise SystemExit(f"(d) does not reproduce Entry 98: theta_sym {pct:.4f} %, p {p:.4f}")
    assert close(th, ch["theta_sym"]) and close(se, ch["welch_se"]) and close(df, ch["welch_df"]) \
        and close(t, ch["t"]) and close(p, ch["one_sided_p_lt0"], rel=1e-9), "g_chain12 G1"
    return {"question": "figures-r2-11: nDoseInvThreeT, the Welch t of the first inversion reading (seeds 0-2)",
            "per_adapter_A": a3, "per_adapter_B": b3, "theta_sym": th, "theta_sym_pct": pct, "SE_Welch": se,
            "df_Welch": df, "t": t, "one_sided_p_lt0": p,
            "reproduces_entry_98": {"theta_sym_pct_printed": -0.143, "p_printed": 0.145,
                                    "also_equal_to": "out/g_chain12.json G1 (theta_sym, welch_se, welch_df, t, p)"},
            "source": "out/g1_ext.json per_adapter_A/B s0-s2"}


# =================================================================================================== (e)
def parse_table2b(nums):
    """Rows of Table 2(b) with their stack label, evaluated through numbers.json."""
    tex = read_text(TAB2)
    i = tex.index("(b) Designs and scored generations")
    body = tex[i:]
    j = body.index("\\midrule")                       # the rule under the header row
    k = body.index("Total over the designs above")
    seg = body[j + len("\\midrule"):k]
    seg = seg[:seg.rindex("\\midrule")]                # the rule above the total row
    outside = body[k:body.index("\\bottomrule")]

    def val(m):
        v = nums[m]["value"]
        assert float(v) == int(round(float(v))), (m, v)
        return int(round(float(v)))

    def cells_of(chunk):
        c = " ".join(chunk.split())
        if not c or c.startswith("\\multicolumn") or c.startswith("\\midrule"):
            return None
        cells = [x.strip() for x in c.split("&")]
        assert len(cells) == 7, cells
        return cells

    def eval_adapters(cell):
        if cell == "---":
            return 0, "---"
        m = re.fullmatch(r"\\(n[A-Za-z]+)\{\}\s*\$\\times\$\s*(\d+)", cell)
        if m:
            return val(m.group(1)) * int(m.group(2)), f"{m.group(1)} x {m.group(2)}"
        m = re.fullmatch(r"\\(n[A-Za-z]+)\{\}\s*in all", cell)
        if m:
            return val(m.group(1)), f"{m.group(1)} in all"
        raise ValueError(cell)

    def eval_per(cell):
        if cell == "---":
            return None, "---"
        m = re.fullmatch(r"\\(n[A-Za-z]+)", cell)
        assert m, cell
        return val(m.group(1)), m.group(1)

    rows = []
    for chunk in seg.split("\\\\"):
        cells = cells_of(chunk)
        if cells is None:
            continue
        design, bodies, stack, ad, per, gens, sec = cells
        assert stack in ("archive", "local", "archive/local"), stack
        flag_h = "^{\\mathrm{h}}" in design
        flag_i = "^{\\mathrm{i}}" in design
        n_ad, ad_expr = eval_adapters(ad)
        n_per, per_expr = eval_per(per)
        m = re.fullmatch(r"(\()?\\(n[A-Za-z]+)(\))?", gens)
        assert m, gens
        n_g = val(m.group(2))
        in_parens = bool(m.group(1))
        assert in_parens == flag_i, design
        rows.append({"design": design, "stack": stack, "adapters_expr": ad_expr, "adapters": n_ad,
                     "adapters_counted": 0 if (flag_h or flag_i) else n_ad,
                     "gens_per_adapter_expr": per_expr, "gens_per_adapter": n_per,
                     "gens_macro": m.group(2), "gens": n_g, "gens_counted": 0 if flag_i else n_g,
                     "note": ("adapters counted in another row (note h)" if flag_h else
                              "same adapters and images as the row above; not added (note i)" if flag_i else "")})
    obj = None
    for chunk in outside.split("\\\\"):
        if not " ".join(chunk.split()).startswith("Training objective"):
            continue                                   # the total row spans columns; only the objective row is read
        cells = cells_of(chunk)
        if cells:
            n_ad, ad_expr = eval_adapters(cells[3])
            n_per, per_expr = eval_per(cells[4])
            obj = {"design": cells[0], "stack": cells[2], "adapters": n_ad, "adapters_expr": ad_expr,
                   "gens_per_adapter": n_per, "gens_per_adapter_expr": per_expr, "gens": n_ad * n_per,
                   "note": "outside the total (note l): generations not scored"}
    assert obj is not None
    return rows, obj


def item_e_table(nums):
    rows, obj = parse_table2b(nums)
    by = {}
    for r in rows:
        s = by.setdefault(r["stack"], {"adapters": 0, "gens": 0, "rows": 0})
        s["adapters"] += r["adapters_counted"]
        s["gens"] += r["gens_counted"]
        s["rows"] += 1
    tot_ad = sum(v["adapters"] for v in by.values())
    tot_g = sum(v["gens"] for v in by.values())
    assert tot_ad == 211 and tot_g == 95500, (tot_ad, tot_g)
    assert tot_ad == int(nums["nDataTotalAdapters"]["value"]) and tot_g == int(nums["nDataTotalGens"]["value"])
    # cross-check against Entry 112 N-A5 (out/fv_derived.json): same totals
    na5 = load(OUT + "/fv_derived.json")["N_A5_totals"]
    assert na5["total_adapters"] == tot_ad and na5["total_gens"] == tot_g
    return {"by_stack": by, "total_adapters": tot_ad, "total_gens": tot_g, "rows": rows, "objective_row": obj,
            "labels": "Table 2(b) Stack column: archive (Google Colab), local (one NVIDIA L40S), archive/local "
                      "(archive adapters, local generations)",
            "source": "paper/fv/tabs/tab02_devices.tex block (b), each cell evaluated through paper/fv/numbers.json; "
                      "totals equal nDataTotalAdapters/nDataTotalGens and out/fv_derived.json N_A5_totals"}


def item_e_verifier():
    env = dict(os.environ)
    env.pop("EINV_V2", None)                          # the repository's own shipped copy of the results
    env["CUDA_VISIBLE_DEVICES"] = ""
    note_input(REPO + "/verify_v2.py")
    t0 = datetime.datetime.now(datetime.timezone.utc)
    r = subprocess.run([sys.executable, "verify_v2.py"], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=600)
    lines = r.stdout.splitlines()
    ok = [l for l in lines if l.startswith("OK")]
    dis = [l for l in lines if l.startswith("DISAGREE")]
    summ = [l for l in lines if re.fullmatch(r"\d+ of \d+ checks agree", l.strip())]
    assert len(summ) == 1, lines[-3:]
    n_agree, n_total = map(int, re.findall(r"\d+", summ[0]))
    assert n_agree == len(ok) and n_total == len(ok) + len(dis), (n_agree, n_total, len(ok), len(dis))
    return {"command": "python verify_v2.py", "cwd": REPO, "python": sys.executable,
            "started_utc": t0.isoformat(timespec="seconds"), "returncode": r.returncode,
            "checks_executed": n_total, "checks_agreeing": n_agree, "checks_disagreeing": len(dis),
            "disagreeing": dis, "summary_line": summ[0].strip(), "stderr_tail": r.stderr[-2000:],
            "verify_v2_sha256": INPUTS[REPO + "/verify_v2.py"],
            "data": "v2/workspace/out of the repository clone (EINV_V2 unset)"}


# =================================================================================================== (c)
S3_UNITS = {
    # key: (phrase that must appear in S03_compute.tex after whitespace normalisation, low min, high min, unit)
    "A100_adapter": ("a rank-16 adapter took about 24~min", 24.0, 24.0, "one SD-3.5 LoRA adapter (2000 steps)"),
    "A100_gen500": ("500 generations about 20~min", 20.0, 20.0, "500 generations"),
    "Blackwell_FLUX_adapter": ("A FLUX.1-dev adapter took about 25~min on the Blackwell card", 25.0, 25.0,
                               "one FLUX.1-dev adapter"),
    "L40S_adapter_2000": ("A 2000-step adapter took 39~min alone and up to 87~min beside a generation job",
                          39.0, 87.0, "one 2000-step adapter"),
    "L40S_adapter_8000": ("an 8000-step adapter about 2.6~h", 156.0, 156.0, "one 8000-step adapter"),
    "L40S_adapter_16000": ("a 16000-step adapter 5.2~h alone and up to 14.8~h when sharing", 312.0, 888.0,
                           "one 16000-step adapter"),
    "L40S_gen500": ("Generating 500 images took 62--74~min", 62.0, 74.0, "500 generations"),
    "L40S_gen250": ("and 250 images about 35~min alone", 35.0, 35.0, "250 generations"),
}

GPU_A100_40 = "NVIDIA A100-SXM4-40GB (Colab)"
GPU_A100_BIG = "larger-memory NVIDIA A100 (Colab)"
GPU_BLACKWELL = "NVIDIA RTX PRO 6000 Blackwell Server Edition (Colab)"
GPU_L40S = "NVIDIA L40S (local)"
COLAB_GPUS = (GPU_A100_40, GPU_A100_BIG, GPU_BLACKWELL)

# Table 2(b) rows (by their generation macro) -> adapters trained (archive paths or local regex) and GPUs.
ARCHIVE_ADAPTERS = {
    "nDataPrimaryGens": ([f"E_INV_P0_v3/adapters/{b}_raw_s{s}_r16" for s in range(3) for b in "AB"]
                         + [f"E_SEEDEXT/adapters/{b}_raw_s{s}_r16" for s in range(3, 6) for b in "AB"]
                         + [f"E_SEEDEXT2/adapters/{b}_raw_s{s}_r16" for s in range(6, 12) for b in "AB"]),
    "nDataFluxGens": [f"E_TRACKB2_FLUX/adapters/{b}_raw_s{s}_flux" for s in range(6) for b in "AB"],
    "nDataFullFTGens": [f"E_FULLFT/ckpt/{b}_ft_s{s}" for s in range(3) for b in "AB"],
    "nDataKodakGens": [f"E_MULTIDEV/adapters/D{d}_s{s}" for d in range(5) for s in range(2)],
    "nDataPTwentyGens": [f"E_DAXING/adapters/{d}_s{s}" for d in range(1101, 1106) for s in range(2)],
}
LOCAL_PATTERNS = {
    "nDataSecondSetGens": r"alt_[AB]_s\d+", "nDataSecondEnvGens": r"nomarkB?_s\d+",
    "nDataContentMatchedGens": r"cm_[AB]_s\d+", "nDataRandomCropGens": r"rcrop_s\d+",
    "nDataIPhoneGens": r"p5c_[AB]_s\d+", "nDataPTwentyPairGens": r"p20b_[AB]_s\d+",
    "nDataEightKGens": r"dose8k_[AB]_s\d+", "nDataSixteenKGens": r"dose16k_[AB]_s\d+",
    "nDataInvertedGens": r"inv16k(ext)?_[AB]_s\d+", "nMapTileGens": r"per(24|28|32|36|40|48)_s\d+",
    "nMapBandGens": r"band\d_s\d+", "nMapWmGensAll": r"wm_ds_s\d+", "nKnownEoneGens": r"kinjd?_a\d+_s\d+",
    "nKnownEtwoGens": r"gk(add|mul)_a\d+_s\d+",
    "ladder": r"mark_(rand|lowmid)_a\d+_s\d+", "environment_chain": r"colab_s\d+",
}
NO_S3_GENERATION = {"nDataFluxGens": "FLUX.1-dev", "nDataFullFTGens": "full fine-tuning"}
ARCHIVE_GEN_ROWS = {"nDataPrimaryGens", "nDataBaseGens", "nDataKodakGens", "nDataPTwentyGens"}


def s3_units():
    txt = " ".join(read_text(S3_TEX).split())
    for key, (phrase, lo, hi, unit) in S3_UNITS.items():
        assert phrase in txt, f"S3 phrase not found: {phrase}"
    return {k: {"phrase": v[0], "low_min": v[1], "high_min": v[2], "unit": v[3]} for k, v in S3_UNITS.items()}


def archive_training(path_rel, gpu):
    f = f"{DRIVE}/{path_rel}/train_meta.json"
    m = load(f)
    rec = m.get("minutes")
    g = m.get("gpu") or (m.get("env") or {}).get("gpu")
    if g is not None:
        if gpu == GPU_A100_40:
            assert "A100-SXM4-40GB" in g, (path_rel, g)
        if gpu == GPU_BLACKWELL:
            assert "Blackwell" in g, (path_rel, g)
    return {"adapter": path_rel, "steps": m.get("steps"), "gpu": gpu, "gpu_in_record": g,
            "minutes_recorded": None if rec is None else float(rec)}


def local_training(tag):
    m = load(f"{LOCAL_ADAPTERS}/{tag}/train_meta.json")
    rec = m.get("minutes")
    return {"adapter": tag, "steps": m.get("steps"), "gpu": GPU_L40S,
            "minutes_recorded": None if rec is None else float(rec)}


def s3_fallback(a, units):
    """Per-unit S3 time for a training job without a record: (key, low, high)."""
    if a["gpu"] == GPU_A100_40 and a["steps"] == 2000:
        k = "A100_adapter"
    elif a["gpu"] == GPU_BLACKWELL:
        k = "Blackwell_FLUX_adapter"
    elif a["gpu"] == GPU_L40S:
        k = {2000: "L40S_adapter_2000", 8000: "L40S_adapter_8000", 16000: "L40S_adapter_16000"}[a["steps"]]
    else:
        raise ValueError(f"no S3 per-unit training time for {a}")
    return k, units[k]["low_min"], units[k]["high_min"]


def gen_units(gpu, n_gens, unit, units):
    """(S3 key, number of units, low min, high min) for n_gens generations made in jobs of `unit` images."""
    assert n_gens % unit == 0, (n_gens, unit)
    n_units = n_gens // unit
    if gpu == GPU_A100_40:
        assert unit == 500
        k = "A100_gen500"
    elif gpu == GPU_L40S:
        k = {500: "L40S_gen500", 250: "L40S_gen250"}[unit]
    else:
        raise ValueError(gpu)
    return k, n_units, n_units * units[k]["low_min"], n_units * units[k]["high_min"]


def item_c(nums, table_rows, obj_row):
    units = s3_units()
    inv = load(INV_GENS)[(EINV.V2 + "/out/t1/gens")]
    local_tags = sorted(d for d in os.listdir(LOCAL_ADAPTERS) if os.path.isdir(f"{LOCAL_ADAPTERS}/{d}"))
    matched = {}
    for key, pat in LOCAL_PATTERNS.items():
        tags = [t for t in local_tags if re.fullmatch(pat, t)]
        for t in tags:
            assert t not in matched, (t, key, matched.get(t))
            matched[t] = key
    assert sorted(matched) == local_tags, sorted(set(local_tags) - set(matched))

    designs = []

    def add_design(key, label, stack, scope, adapters, gens_spec):
        """adapters: list of training dicts; gens_spec: (gpu or None, n_gens, unit, why_excluded)."""
        tr_rec = sum(a["minutes_recorded"] for a in adapters if a["minutes_recorded"] is not None)
        fb = []
        for a in adapters:
            if a["minutes_recorded"] is None:
                k, lo, hi = s3_fallback(a, units)
                a["s3_fallback"] = {"key": k, "low_min": lo, "high_min": hi}
                fb.append((a["gpu"], lo, hi))
        gpu_g, n_g, unit, why = gens_spec
        if gpu_g is None:
            g = {"gpu": None, "n_gens": n_g, "unit": unit, "excluded": True, "why": why,
                 "low_min": 0.0, "high_min": 0.0}
        elif n_g == 0:
            g = {"gpu": gpu_g, "n_gens": 0, "unit": unit, "excluded": False, "low_min": 0.0, "high_min": 0.0}
        else:
            k, nu, lo, hi = gen_units(gpu_g, n_g, unit, units)
            g = {"gpu": gpu_g, "n_gens": n_g, "unit": unit, "n_units": nu, "s3_key": k, "excluded": False,
                 "low_min": lo, "high_min": hi}
        designs.append({"key": key, "design": label, "table2_stack": stack, "scope": scope,
                        "adapters_trained": len(adapters),
                        "adapters_with_record": sum(a["minutes_recorded"] is not None for a in adapters),
                        "training_recorded_min": tr_rec,
                        "training_s3_low_min": sum(x[1] for x in fb), "training_s3_high_min": sum(x[2] for x in fb),
                        "training": adapters, "generation": g})

    for r in table_rows:
        gm = r["gens_macro"]
        if r["note"].startswith("same adapters"):
            continue                                            # note i: a subset of the row above
        label = " ".join(re.sub(r"\$\^\{\\mathrm\{[a-z]\}\}\$", "", r["design"]).split())
        n_new = r["adapters_counted"]
        if gm in ARCHIVE_ADAPTERS:
            ads = []
            for p in ARCHIVE_ADAPTERS[gm]:
                if gm == "nDataFluxGens":
                    s = int(re.search(r"_s(\d+)_flux", p).group(1))
                    gpu = GPU_A100_BIG if s < 3 else GPU_BLACKWELL
                elif gm == "nDataFullFTGens":
                    gpu = GPU_A100_BIG
                else:
                    gpu = GPU_A100_40
                ads.append(archive_training(p, gpu))
        elif gm in LOCAL_PATTERNS:
            ads = [local_training(t) for t in local_tags if matched[t] == gm]
        else:
            ads = []                                            # base-model rows and note-h rows: no training
        assert len(ads) == n_new, (gm, len(ads), n_new)
        unit = r["gens_per_adapter"] if r["gens_per_adapter"] else r["gens_counted"]
        if gm in NO_S3_GENERATION:
            spec = (None, r["gens_counted"], unit,
                    f"{NO_S3_GENERATION[gm]} generation: supplement S3 gives no per-unit generation time")
        elif gm in ARCHIVE_GEN_ROWS:
            assert r["stack"] == "archive", gm
            spec = (GPU_A100_40, r["gens_counted"], unit, "")
        else:
            assert r["stack"] in ("local", "archive/local"), gm
            spec = (GPU_L40S, r["gens_counted"], unit, "")
        add_design(gm, label, r["stack"], "N-A5", ads, spec)

    # all work: objective arms (archive), designed-mark ladder and environment chain (local)
    obj_dirs = sorted(os.path.dirname(p).replace("\\", "/").split("/")[-1]
                      for p in glob.glob(f"{DRIVE}/E_AMP/adapters/*/train_meta.json"))
    assert len(obj_dirs) == obj_row["adapters"] == int(nums["nObjAdapters"]["value"]), obj_dirs
    add_design("objective_arms", "Training objective, tap T2 (objective arms)", "archive", "all-work extra",
               [archive_training(f"E_AMP/adapters/{d}", GPU_A100_40) for d in obj_dirs],
               (GPU_A100_40, obj_row["gens"], obj_row["gens_per_adapter"], ""))
    lad = [t for t in local_tags if matched[t] == "ladder"]
    lad_g = [inv[t]["n_img"] for t in lad]
    assert len(lad) == 6 and set(lad_g) == {500}, (lad, lad_g)
    add_design("ladder", "Designed-mark ladder (mark_rand_a1/a3/a12, mark_lowmid_a12)", "local", "all-work extra",
               [local_training(t) for t in lad], (GPU_L40S, sum(lad_g), 500, ""))
    env_tr = [t for t in local_tags if matched[t] == "environment_chain"]
    env_gen = [t for t in inv if re.fullmatch(r"colab_s\d+|local_[AB]_raw_s\d+|local_base", t)]
    env_g = [inv[t]["n_img"] for t in env_gen]
    assert len(env_tr) == 3 and len(env_gen) == 6 and set(env_g) == {500}, (env_tr, env_gen, env_g)
    add_design("environment_chain", "Environment chain (colab_s0-s2 trained; local_base, local_A_raw_s0, "
               "local_B_raw_s0 generated with no new adapter)", "local", "all-work extra",
               [local_training(t) for t in env_tr], (GPU_L40S, sum(env_g), 500, ""))

    def totals(scopes):
        sel = [d for d in designs if d["scope"] in scopes]
        groups = {}
        for g in COLAB_GPUS + (GPU_L40S,):
            tr_rec = sum(a["minutes_recorded"] for d in sel for a in d["training"]
                         if a["gpu"] == g and a["minutes_recorded"] is not None)
            tr_lo = sum(a["s3_fallback"]["low_min"] for d in sel for a in d["training"]
                        if a["gpu"] == g and a["minutes_recorded"] is None)
            tr_hi = sum(a["s3_fallback"]["high_min"] for d in sel for a in d["training"]
                        if a["gpu"] == g and a["minutes_recorded"] is None)
            ge_lo = sum(d["generation"]["low_min"] for d in sel if d["generation"]["gpu"] == g)
            ge_hi = sum(d["generation"]["high_min"] for d in sel if d["generation"]["gpu"] == g)
            n_ad = sum(1 for d in sel for a in d["training"] if a["gpu"] == g)
            n_rec = sum(1 for d in sel for a in d["training"] if a["gpu"] == g and a["minutes_recorded"] is not None)
            n_gen = sum(d["generation"]["n_gens"] for d in sel
                        if d["generation"]["gpu"] == g and not d["generation"]["excluded"])
            groups[g] = {"adapters_trained": n_ad, "adapters_with_record": n_rec, "generations_estimated": n_gen,
                         "training_recorded_h": tr_rec / 60, "training_s3_low_h": tr_lo / 60,
                         "training_s3_high_h": tr_hi / 60, "generation_low_h": ge_lo / 60,
                         "generation_high_h": ge_hi / 60}

        def agg(names):
            a = {k: sum(groups[n][k] for n in names) for k in groups[names[0]]}
            a["training_low_h"] = a["training_recorded_h"] + a["training_s3_low_h"]
            a["training_high_h"] = a["training_recorded_h"] + a["training_s3_high_h"]
            a["total_low_h"] = a["training_low_h"] + a["generation_low_h"]
            a["total_high_h"] = a["training_high_h"] + a["generation_high_h"]
            a["recorded_share_of_total_at_low"] = a["training_recorded_h"] / a["total_low_h"] if a["total_low_h"] else None
            a["recorded_share_of_total_at_high"] = a["training_recorded_h"] / a["total_high_h"] if a["total_high_h"] else None
            a["recorded_share_of_training_at_low"] = a["training_recorded_h"] / a["training_low_h"] if a["training_low_h"] else None
            a["recorded_share_of_training_at_high"] = a["training_recorded_h"] / a["training_high_h"] if a["training_high_h"] else None
            return a

        res = {g: agg([g]) for g in groups}
        res["Colab (archive stack), all cards"] = agg(list(COLAB_GPUS))
        res["all GPUs"] = agg(list(COLAB_GPUS) + [GPU_L40S])
        excl = [{"design": d["design"], "generations": d["generation"]["n_gens"], "why": d["generation"]["why"]}
                for d in sel if d["generation"]["excluded"]]
        n_ad = sum(d["adapters_trained"] for d in sel)
        n_g = sum(d["generation"]["n_gens"] for d in sel)
        return {"adapters_trained": n_ad, "generations": n_g, "by_gpu": res, "generation_not_estimated": excl}

    na5 = totals({"N-A5"})
    assert na5["adapters_trained"] == 211 and na5["generations"] == 95500, (na5["adapters_trained"], na5["generations"])
    allw = totals({"N-A5", "all-work extra"})
    assert allw["adapters_trained"] == 211 + 7 + 6 + 3 and allw["generations"] == 95500 + 3500 + 3000 + 3000

    # ---------------------------------------------------------------------------------------- checks
    # 1. local training jobs that ran concurrently: union of [end - minutes, end] (end = train_meta.json mtime)
    iv = []
    for t in local_tags:
        f = f"{LOCAL_ADAPTERS}/{t}/train_meta.json"
        end = os.path.getmtime(f)
        mins = float(json.load(open(f, encoding="utf-8"))["minutes"])
        iv.append((end - 60 * mins, end, t, mins))

    def union_h(ivs):
        ivs = sorted((a, b) for a, b, *_ in ivs)
        tot, cur = 0.0, None
        for a, b in ivs:
            if cur is None:
                cur = [a, b]
            elif a <= cur[1]:
                cur[1] = max(cur[1], b)
            else:
                tot += cur[1] - cur[0]
                cur = [a, b]
        return (tot + cur[1] - cur[0]) / 3600

    na5_local = {a["adapter"] for d in designs if d["scope"] == "N-A5" for a in d["training"] if a["gpu"] == GPU_L40S}
    iso = lambda x: datetime.datetime.fromtimestamp(x).isoformat(timespec="minutes")
    overlap = {
        "all_local_adapters": {"n": len(iv), "sum_of_job_hours": sum(x[3] for x in iv) / 60,
                               "union_hours": union_h(iv), "first_start_local": iso(min(x[0] for x in iv)),
                               "last_end_local": iso(max(x[1] for x in iv))},
        "N-A5_local_adapters": {"n": len(na5_local),
                                "sum_of_job_hours": sum(x[3] for x in iv if x[2] in na5_local) / 60,
                                "union_hours": union_h([x for x in iv if x[2] in na5_local])},
        "method": "each training job's interval is [mtime(train_meta.json) - recorded minutes, mtime]; the union "
                  "merges jobs that ran at the same time on the one L40S. Training-with-generation overlap is not "
                  "in this check (generation has no per-job record); descriptive, not pre-specified",
    }
    # 2. generation per-unit times printed in the local generation logs (progress lines), against S3's units
    prog = re.compile(r"^\[(\d\d:\d\d:\d\d)\] (\S+): (\d+)/(\d+)\s+\(([\d.]+) min\)")
    seqs = {}
    for f in sorted(glob.glob(LOGS + "/t1_*generate.log") + [LOGS + "/f5_generate.log", LOGS + "/t5_generate.log"]):
        note_input(f)
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = prog.match(line)
                if m:
                    seqs.setdefault((os.path.basename(f), m.group(2)), []).append(
                        (int(m.group(3)), int(m.group(4)), float(m.group(5))))
    per_unit, not_used = {}, []
    for (f, tag), seq in sorted(seqs.items()):
        seq = list(dict.fromkeys(seq))                     # a log that repeats lines verbatim counts once
        ns, ms = [x[0] for x in seq], [x[2] for x in seq]
        fresh = ns[0] == 100 and all(b > a for a, b in zip(ns, ns[1:])) and all(b > a for a, b in zip(ms, ms[1:]))
        if not fresh:                                      # resumed or partial: the minutes restart mid-job
            not_used.append({"log": f, "tag": tag, "lines": seq})
            continue
        n, N, mins = seq[-1]
        per_unit.setdefault(N, []).append({"log": f, "tag": tag, "images_at_last_line": n,
                                           "minutes_at_last_line": mins, "extrapolated_minutes": mins * N / n})
    log_check = {}
    for N, lst in sorted(per_unit.items()):
        xs = [x["extrapolated_minutes"] for x in lst]
        k = {500: "L40S_gen500", 250: "L40S_gen250"}.get(N)
        lo, hi = (units[k]["low_min"], units[k]["high_min"]) if k else (None, None)
        log_check[str(N)] = {"jobs_in_logs": len(xs), "min": min(xs), "median": statistics.median(xs),
                             "max": max(xs), "sum_h": sum(xs) / 60, "s3_low_min": lo, "s3_high_min": hi,
                             "share_above_s3_high": (sum(x > hi for x in xs) / len(xs)) if k else None,
                             "sum_over_s3_low_x_jobs": (sum(xs) / (lo * len(xs))) if k else None,
                             "sum_over_s3_high_x_jobs": (sum(xs) / (hi * len(xs))) if k else None,
                             "largest": sorted(lst, key=lambda x: -x["extrapolated_minutes"])[:5]}
    log_check["not_used_resumed_or_partial"] = not_used
    log_check["method"] = ("per adapter and log, the progress lines '[hh:mm:ss] tag: n/N (m min)'; a job is used "
                           "only if its lines start at n = 100 and both n and minutes increase (a resumed job "
                           "restarts its minutes); its time is the last line's minutes x N/n (250-image jobs print "
                           "their last line at 200); pipeline loading is not included; the logs are incomplete "
                           "(some were overwritten by later runs). A check of S3's per-unit inputs, not an "
                           "estimate; descriptive, not pre-specified")

    # 3. S3's per-unit training times against the training records they summarise
    s3_train = {}
    for steps, k in ((2000, "L40S_adapter_2000"), (8000, "L40S_adapter_8000"), (16000, "L40S_adapter_16000")):
        recs = [(a["adapter"], a["minutes_recorded"]) for d in designs for a in d["training"]
                if a["gpu"] == GPU_L40S and a["steps"] == steps]
        xs = [m for _, m in recs]
        s3_train[f"L40S_{steps}"] = {"n": len(xs), "min": min(xs), "max": max(xs), "median": statistics.median(xs),
                                     "s3_low_min": units[k]["low_min"], "s3_high_min": units[k]["high_min"],
                                     "above_s3_high": sorted([r for r in recs if r[1] > units[k]["high_min"]],
                                                             key=lambda r: -r[1])}
    arch = [(a["adapter"], a["minutes_recorded"]) for d in designs for a in d["training"]
            if a["gpu"] == GPU_A100_40 and a["minutes_recorded"] is not None]
    s3_train["A100_40GB_2000"] = {"n": len(arch), "min": min(m for _, m in arch), "max": max(m for _, m in arch),
                                  "s3_min": units["A100_adapter"]["low_min"]}
    s3_train["note"] = ("S3's times are 'about' values; its 2000-step 'up to 87 min' does not cover the local records "
                        "listed under above_s3_high. Only one adapter (A_raw_s0_r16) uses an S3 training time here")

    # 4. archive adapters with a training record that neither scope includes (pilot and unreported arms)
    in_scope = {a["adapter"] for d in designs for a in d["training"] if a["gpu"] != GPU_L40S}
    others = []
    for p in sorted(glob.glob(f"{DRIVE}/*/adapters/*/train_meta.json") + glob.glob(f"{DRIVE}/*/ckpt/*/train_meta.json")):
        rel = os.path.dirname(p).replace("\\", "/")[len(DRIVE) + 1:]
        if rel not in in_scope:
            m = load(p)
            others.append({"adapter": rel, "steps": m.get("steps"), "minutes_recorded": m.get("minutes")})

    return {"question": "trace-r2-12: total GPU-hours by GPU",
            "s3_per_unit_minutes": units,
            "definition": ("GPU-hours = summed wall-clock hours of the training and generation jobs on each card. "
                           "Training: train_meta.json 'minutes' where recorded, else S3's per-unit time; generation: "
                           "S3's per-unit time x number of jobs (500 or 250 images per job, as in Table 2(b)). "
                           "Excluded: detector, embedding and scoring GPU time, autoencoder round trips, and jobs "
                           "that were discarded, interrupted or re-run (for example the ~11 GPU-hours of "
                           "inv16kext_B_s3 lost in D13, Entry 104). Jobs that shared the L40S are each counted in "
                           "full, so these are job-hours; the card's occupied time is lower (see checks)"),
            "N-A5 (211 adapters, 95,500 generations)": na5,
            "all work (N-A5 + objective arms + designed-mark ladder + environment chain)": allw,
            "designs": designs,
            "checks": {"local_training_concurrency": overlap, "generation_logs_vs_S3": log_check,
                       "training_records_vs_S3": s3_train,
                       "archive_adapters_outside_both_scopes": {
                           "n": len(others),
                           "recorded_hours": sum(o["minutes_recorded"] or 0 for o in others) / 60,
                           "adapters": others,
                           "note": "trained on Colab, recorded, but in neither registered scope; not added"}},
            "environment_chain_count_note": ("Entry 112 / out/fv_derived.json list the environment chain as 5 "
                                             "adapters; only colab_s0-s2 were trained. local_A_raw_s0 and "
                                             "local_B_raw_s0 are v1's archived adapters generating on the local "
                                             "stack (Entry 25), so the all-work count here is 211+7+6+3 = 227 "
                                             "trained adapters (105,000 generations)")}


# =================================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DST_DEFAULT)
    args = ap.parse_args()
    dst = args.out.replace("\\", "/")
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists; result files are never overwritten")
    for p in (OUT + "/fv_small.json",):
        if dst == DST_DEFAULT and os.path.exists(p):
            raise SystemExit(f"{p} exists (the registered name); stop rather than write a second copy")

    R = ledger_R()
    nums = load(NUMBERS)
    res = {
        "entry": "RESULTS.md Entry 116, item 6 (small quantities; descriptive, from existing files)",
        "script": "src/fv/fv_small_quantities.py",
        "registered_names": {"script": "src/fv/fv_small.py", "output": "out/fv_small.json"},
        "name_deviation": "written as fv_small_quantities.py/.json, the names the executing run asked for; the "
                          "analysis is as registered",
        "status": "descriptive; no reading attaches (Entry 116: items 5 and 6 are descriptive)",
        "run": {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
        "settings": {"R_real": R, "R_real_registered": R_REAL_REGISTERED, "max_arm_quantile": 0.995,
                     "estimation_sd_registered": SD_EST_REGISTERED},
    }
    res["a_local_stack_maxarm"] = item_a(R)
    res["b_inflated_p"] = item_b(R)
    res["d_inversion_first_reading"] = item_d(R)
    table = item_e_table(nums)
    res["c_gpu_hours"] = item_c(nums, table["rows"], table["objective_row"])
    res["e_counts"] = {"verifier": item_e_verifier(), "table2b_by_stack": table,
                       "deposit_file_count": "not computed: waits on author actions (Entry 116 item 6e)"}
    res["inputs_sha256"] = dict(sorted(INPUTS.items()))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"[fv_small_quantities] wrote {dst}")


if __name__ == "__main__":
    main()
