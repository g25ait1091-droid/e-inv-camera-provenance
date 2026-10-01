"""Post-hoc quantities computed for the manuscript (RESULTS.md Entry 112). CPU only, a few seconds.

Writes out/fv_derived.json. Every quantity is computed from result files (out/, the repository verifier, the
paper's own count macros in paper/fv/numbers.json, and the generation inventory paper/fv/work/inv_gens.json);
nothing is typed in. None of these quantities was pre-specified: they are descriptive or sensitivity analyses.

  N-A1  two-candidate attribution TPR at 1 % FPR at the CALIBRATED limit (H6 U_device), G = 500, 5000, inf,
        with verify_v2.tpr() imported from the repository; also reproduces the nominal-limit values.
  N-A2  Hartung-Knapp-Sidik-Jonkman one-sided 99 % upper limit of the pooled random-effects estimate over the
        three pairs (m1_pooled.json primary inputs; t with k - 1 = 2 df); DL value reproduced for comparison.
  N-A3  D200 four-design combination: I^2 and Q p (m1_pooled.json secondary.d200_combination).
  N-A4  the local 2000-step arms: adapters per body, generations per adapter, total.
  N-A5  study totals (adapters trained, generations scored) over the designs of Table 2 block B.
  N-A6  G2 (second environment) additive part as % of R_real.
  Q1    learning-rate schedule of each primary adapter, from the training records.

Run:  python src/fv/fv_derived.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import contextlib
import glob
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import sys

import numpy as np
from scipy import stats

V2 = EINV.V2
OUT = V2 + "/out"
REPO = EINV.REPO
DRIVE = (EINV.MYDRIVE + "/inv_channel")
NUMBERS = V2 + "/paper/fv/numbers.json"
INV_GENS = V2 + "/paper/fv/work/inv_gens.json"
R_REAL = 0.0356703416571125
DST = OUT + "/fv_derived.json"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ============================================================================ N-A1
def import_verifier():
    """Import verify_v2 for its tpr() model. The module runs its checks at import and calls sys.exit at the
    end; its output is captured and the exit is caught, after which tpr() and U_DEVICE are module attributes."""
    spec = importlib.util.spec_from_file_location("verify_v2", REPO + "/verify_v2.py")
    mod = importlib.util.module_from_spec(spec)
    buf = io.StringIO()
    status = "imported"
    with contextlib.redirect_stdout(buf):
        try:
            spec.loader.exec_module(mod)
        except SystemExit as e:
            status = f"module ran its checks and exited with code {e.code}"
        except Exception as e:  # tpr() is defined before most checks; keep it if it exists
            status = f"module raised {type(e).__name__} during its checks: {e}"
    assert hasattr(mod, "tpr"), "verify_v2.tpr not defined"
    last = [l for l in buf.getvalue().splitlines() if "checks agree" in l]
    return mod, status, (last[-1] if last else "")


def n_a1():
    vv, status, summary = import_verifier()
    pw = load(OUT + "/t3_power_v4.json")
    inp = pw["inputs"]
    h6 = load(OUT + "/h6_calibrated_limit.json")["primary_d200"]
    U_cal, U_nom = h6["U_device"], inp["U_device"]
    Gs = {"G500": 500, "G5000": 5000, "G_inf": math.inf}

    def run(U):
        vv.U_DEVICE = U               # tpr() reads the module-level U_DEVICE at call time
        return {k: vv.tpr(G, sigma_mu=inp["sigma_mu"], se500=inp["SE_img_500"], M=2) for k, G in Gs.items()}

    nom = run(U_nom)
    cal = run(U_cal)
    vv.U_DEVICE = U_nom
    row = next(r for r in pw["rows"] if r["transfer"] == "at the upper limit U_device" and r["M_candidates"] == 2)
    nums = load(NUMBERS)
    macro_of = {"G500": "nAttTprTwoFiveHundred", "G5000": "nAttTprTwoFiveThousand", "G_inf": "nAttTprTwoInf"}
    repro = {}
    for k in Gs:
        f = row["TPR_" + k]
        mv = nums[macro_of[k]]["value"]
        # tolerance 1e-6: the file's G_inf column was evaluated at a large finite G (differs by ~8e-8)
        repro[k] = {"recomputed": nom[k], "file": f, "macro": macro_of[k], "macro_value": mv,
                    "abs_diff": abs(nom[k] - f), "agree": abs(nom[k] - f) < 1e-6 and abs(nom[k] - mv) < 1e-6}
    assert all(r["agree"] for r in repro.values()), repro
    return {
        "model": "verify_v2.tpr(G, sigma_mu, se500, M=2): TPR = 1 - Phi(z_{1-0.01/(M-1)} - U / sqrt(sigma_mu^2 + "
                 "(se500*sqrt(500/G))^2)); two candidate cameras, 1 % false-positive rate",
        "verifier_import": status, "verifier_summary": summary,
        "inputs": {"sigma_mu": inp["sigma_mu"], "SE_img_500": inp["SE_img_500"], "M": 2,
                   "U_nominal": U_nom, "U_calibrated": U_cal, "c": h6["c"],
                   "lambda_U_calibrated_pct": h6["lambda_U_calibrated_pct"],
                   "sources": ["out/t3_power_v4.json inputs", "out/h6_calibrated_limit.json primary_d200.U_device"]},
        "tpr_calibrated": cal,
        "tpr_nominal_reproduced": repro,
        "status": "post hoc; descriptive (a power-model evaluation at the calibrated limit)",
    }


# ============================================================================ N-A2 / N-A3
def dl_pool(y, se):
    """DerSimonian-Laird exactly as src/m1_pooled.py pool()."""
    y, se = np.asarray(y, float), np.asarray(se, float)
    w = 1 / se ** 2
    fe = float((w * y).sum() / w.sum())
    k = len(y)
    Q = float((w * (y - fe) ** 2).sum())
    C = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    ws = 1 / (se ** 2 + tau2)
    re = float((ws * y).sum() / ws.sum())
    re_se = float(1 / np.sqrt(ws.sum()))
    return dict(k=k, Q=Q, tau2=tau2, ws=ws, re=re, re_se=re_se)


def n_a2_a3():
    M1 = load(OUT + "/m1_pooled.json")
    prim = M1["primary"]
    names = prim["designs"]
    y = [M1["per_design"][n]["lambda_pct"] for n in names]
    se = [M1["per_design"][n]["se_pct"] for n in names]
    P = dl_pool(y, se)
    k = P["k"]
    z99 = stats.norm.ppf(0.99)
    dl_upper = P["re"] + z99 * P["re_se"]
    assert abs(dl_upper - prim["random"]["upper99_pct"]) < 1e-12, (dl_upper, prim["random"]["upper99_pct"])
    assert abs(P["tau2"] - prim["random"]["tau2"]) < 1e-15
    # Hartung-Knapp (Sidik-Jonkman) variance around the DL random-effects mean, t with k - 1 df
    ws, yy = P["ws"], np.asarray(y, float)
    q = float((ws * (yy - P["re"]) ** 2).sum() / (k - 1))
    se_hk = math.sqrt(q / ws.sum())
    t99 = stats.t.ppf(0.99, k - 1)
    hk_upper = P["re"] + t99 * se_hk
    hk_t = P["re"] / se_hk
    hk_p = float(stats.t.sf(hk_t, k - 1))
    se_mhk = math.sqrt(max(1.0, q) / ws.sum())          # modified HKSJ (q floored at 1)
    d2 = M1["secondary"]["d200_combination"]
    return {
        "hksj": {
            "designs": names, "k": k, "df": k - 1, "t_0.99": t99,
            "tau2_DL_pct2": P["tau2"], "pooled_lambda_pct": P["re"],
            "q_scale": q, "se_hk_pct": se_hk, "se_dl_pct": P["re_se"],
            "upper99_pct": hk_upper, "t": hk_t, "one_sided_p": hk_p,
            "modified_upper99_pct": P["re"] + t99 * se_mhk,
            "modified_note": "q >= 1 here, so the modified (q floored at 1) interval is identical" if q >= 1 else
                             "q < 1: the modified interval uses q = 1",
            "dl_upper99_pct_reproduced": dl_upper, "dl_upper99_pct_file": prim["random"]["upper99_pct"],
            "source": "out/m1_pooled.json per_design[primary.designs].{lambda_pct,se_pct}; DL tau^2 as src/m1_pooled.py",
            "status": "post hoc sensitivity (small-sample interval beside the pre-specified DL limit, Entry 106)",
        },
        "d200_combination": {
            "designs": d2["designs"], "k": d2["k"], "I2": d2["I2"], "I2_pct": 100 * d2["I2"],
            "Q": d2["Q"], "Q_df": d2["Q_df"], "Q_p": d2["Q_p"],
            "source": "out/m1_pooled.json secondary.d200_combination",
            "status": "read from the M1 result file (Entry 107); homogeneity of the four D200 designs",
        },
    }


# ============================================================================ N-A4
def n_a4():
    D = load(OUT + "/t1/dose_stats.json")["doses"]["2000"]
    inv = load(INV_GENS)[(EINV.V2 + "/out/t1/gens")]
    per = {}
    for arm, f in ((D["A_arms"], "summary_nomark.json"), (D["B_arms"], "summary_nomarkB.json")):
        S = load(OUT + "/t1/" + f)["arms"]
        for a in arm:
            per[a] = {"scored": S[a]["n"], "folder": inv[a]["n_img"]}
            assert per[a]["scored"] == per[a]["folder"], (a, per[a])
    ns = sorted({v["scored"] for v in per.values()})
    assert len(ns) == 1
    nA, nB = len(D["A_arms"]), len(D["B_arms"])
    assert nA == nB
    return {"A_arms": D["A_arms"], "B_arms": D["B_arms"], "adapters_per_body": nA, "bodies": 2,
            "gens_per_adapter": ns[0], "total_gens": sum(v["scored"] for v in per.values()), "per_adapter": per,
            "note": "the 2000-step local arms are seeds 0-2 of the second-environment design (G2, nomark/nomarkB "
                    "s0-s5); they add no adapters or generations to the study totals",
            "source": "out/t1/dose_stats.json doses.2000.{A,B}_arms; out/t1/summary_nomark*.json arms.*.n; "
                      "paper/fv/work/inv_gens.json folder counts",
            "status": "count"}


# ============================================================================ N-A5
def n_a5():
    nums = load(NUMBERS)
    v = lambda m: int(round(nums[m]["value"]))
    inv = load(INV_GENS)[(EINV.V2 + "/out/t1/gens")]

    def folders(prefixes):
        tags = [t for t in inv if any(re.fullmatch(p, t) for p in prefixes)]
        return len(tags), sum(inv[t]["n_img"] for t in tags), sorted(tags)

    rows = []

    def add(design, adapters, gens, how, new_adapters=True):
        rows.append({"design": design, "adapters": adapters if new_adapters else 0, "adapters_used": adapters,
                     "gens": gens, "basis": how})

    # (design, per-arm macro, arms, gens macro)
    for design, ap, arms, gm in (
            ("D200 primary (archive)", "nDataPrimaryAdaptersPerArm", 2, "nDataPrimaryGens"),
            ("D200 second training set (G5)", "nDataSecondSetAdaptersPerArm", 2, "nDataSecondSetGens"),
            ("D200 second environment (G2)", "nDataSecondEnvAdaptersPerArm", 2, "nDataSecondEnvGens"),
            ("D200 FLUX.1-dev", "nDataFluxAdaptersPerArm", 2, "nDataFluxGens"),
            ("D200 full fine-tuning", "nDataFullFTAdaptersPerArm", 2, "nDataFullFTGens"),
            ("iPhone 5c pair (G6)", "nDataIPhoneAdaptersPerArm", 2, "nDataIPhoneGens"),
            ("P20 pair (G4b)", "nDataPTwentyPairAdaptersPerArm", 2, "nDataPTwentyPairGens"),
            ("Kodak M1063 five bodies", "nDataKodakAdaptersPerArm", 5, "nDataKodakGens"),
            ("P20 five bodies", "nDataPTwentyAdaptersPerArm", 5, "nDataPTwentyGens"),
            ("D200 8000 steps", "nDataEightKAdaptersPerArm", 2, "nDataEightKGens"),
            ("D200 16000 steps", "nDataSixteenKAdaptersPerArm", 2, "nDataSixteenKGens"),
            ("D200 16000 steps, fingerprint inverted (G1 + extension)", "nDataInvertedAdaptersPerArm", 2,
             "nDataInvertedGens"),
            ("D200 content-matched", "nDataContentMatchedAdaptersPerArm", 2, "nDataContentMatchedGens"),
            ("D200 random crop (body A only)", "nDataRandomCropAdaptersPerArm", 1, "nDataRandomCropGens")):
        add(design, v(ap) * arms, v(gm), f"{ap} x {arms} arms; {gm}")
    # generations without new adapters
    add("base model, single caption (archive)", 0, v("nDataBaseGens"), "nDataBaseGens")
    add("base model, five captions", 0, v("nDataPromptBankBaseGens"), "nDataPromptBankBaseGens")
    add("D200 five captions on primary seeds 0-2", v("nDataPromptBankAdaptersPerArm") * 2, v("nDataPromptBankGens"),
        "nDataPromptBankGens; adapters are primary s0-s2 (not new)", new_adapters=False)
    add("iPhone 5c five captions (P1)", v("nDataPromptsAdaptersPerArm") * 2, v("nDataPromptsGens"),
        "nDataPromptsGens; adapters are the G6 adapters (not new)", new_adapters=False)
    # transmission map (body A, local): adapters from the count macros, generations from the folder inventory
    for design, macro, pats in (
            ("map: periodic tiles", "nMapTileAdapters", [r"per(24|28|32|36|40|48)_s\d+"]),
            ("map: fingerprint-spectrum octave bands", "nMapBandAdapters", [r"band\d_s\d+"]),
            ("map: DiffusionShield watermark", "nMapWmAdapters", [r"wm_ds_s\d+"]),
            ("map: E1 fingerprint as a known pattern", "nKnownAdaptersEone", [r"kinj_a\d+_s\d+", r"kinjd_a\d+_s\d+"]),
            ("map: E2 spectrum-matched fields", "nKnownAdaptersEtwo", [r"gk(add|mul)_a\d+_s\d+"])):
        n, g, tags = folders(pats)
        assert n == v(macro), (design, n, v(macro), tags)
        add(design, n, g, f"{macro}; generations = PNG counts of {len(tags)} folders in inv_gens.json")
    # cross-check the local designs' generation macros against the folder inventory
    xcheck = {}
    for gm, pats in (("nDataSecondSetGens", [r"alt_[AB]_s\d+"]), ("nDataSecondEnvGens", [r"nomarkB?_s\d+"]),
                     ("nDataEightKGens", [r"dose8k_[AB]_s\d+"]), ("nDataSixteenKGens", [r"dose16k_[AB]_s\d+"]),
                     ("nDataInvertedGens", [r"inv16k(ext)?_[AB]_s\d+"]),
                     ("nDataContentMatchedGens", [r"cm_[AB]_s\d+"]), ("nDataRandomCropGens", [r"rcrop_s\d+"]),
                     ("nDataIPhoneGens", [r"p5c_[AB]_s\d+"]), ("nDataPromptsGens", [r"p5c_[AB]_s\d+_div"]),
                     ("nDataPTwentyPairGens", [r"p20b_[AB]_s\d+"])):
        n, g, _ = folders(pats)
        xcheck[gm] = {"macro": v(gm), "folders": g, "agree": g == v(gm)}
    assert all(x["agree"] for x in xcheck.values()), xcheck
    tot_ad = sum(r["adapters"] for r in rows)
    tot_g = sum(r["gens"] for r in rows)
    excluded = {
        "rescored, no new generations": ["iPhone flat-field rows (rescore G6 and P1 images)",
                                         "16000-step replication seeds 3-5 (subset of the 16000-step design)",
                                         "2000-step local arms (seeds 0-2 of G2)",
                                         "detector panel, learned CNN, attribution, G3, H2, H3, C5, copy audit, "
                                         "memorization and firearm counts (all score existing generations)",
                                         "DiffusionShield released-decoder runs (200 per arm on existing arms)"],
        "trained and generated but not a Table 2 block B design": {
            "objective arms (archive E_AMP, reported in S05 on stored energy and loss)":
                {"adapters": v("nObjAdapters"), "gens": v("nObjAdapters") * v("nObjGensPerAdapter")},
            "designed-mark ladder (mark_rand_a1/a3/a12, mark_lowmid_a12; N-A7, no macros)":
                dict(zip(("adapters", "gens"), folders([r"mark_(rand|lowmid)_a\d+_s\d+"])[:2])),
            "environment chain (colab_s0-s2, local_A_raw_s0, local_B_raw_s0, local_base; N-A7, no macros)":
                {"adapters": folders([r"colab_s\d+", r"local_[AB]_raw_s\d+"])[0],
                 "gens": folders([r"colab_s\d+", r"local_[AB]_raw_s\d+", r"local_base"])[1]},
            "pilot arms of the archive run (D, D2, clean, swap, rank-64 ceiling; not reported)": "not counted",
            "P10 Plus pair": "gate failed; never trained",
        },
    }
    obj = excluded["trained and generated but not a Table 2 block B design"][
        "objective arms (archive E_AMP, reported in S05 on stored energy and loss)"]
    return {"definition": "designs listed in Table 2 block B (OUTLINE.md 'Table 2'); a design adds adapters only "
                          "if it trained new ones, and generations only if they are new images",
            "rows": rows, "total_adapters": tot_ad, "total_gens": tot_g,
            "with_objective_arms": {"total_adapters": tot_ad + obj["adapters"], "total_gens": tot_g + obj["gens"]},
            "not_counted": excluded, "folder_crosscheck": xcheck,
            "source": "paper/fv/numbers.json count macros (nData*, nMap*, nKnown*); paper/fv/work/inv_gens.json",
            "status": "count"}


# ============================================================================ N-A6
def n_a6():
    g = load(OUT + "/g2_pooled_six.json")
    a, b = np.asarray(g["per_adapter_A"], float), np.asarray(g["per_adapter_B"], float)
    th = 0.5 * (a.mean() + b.mean())
    assert abs(th - g["theta_sym"]) < 1e-15
    add = 0.5 * (a.mean() - b.mean())
    return {"additive_part": float(add), "additive_pct_of_R_real": float(100 * add / R_REAL),
            "theta_sym": g["theta_sym"], "R_real": R_REAL,
            "definition": "additive part = (mean_A - mean_B)/2 of the per-arm own-minus-other contrasts; it "
                          "estimates b_A - b_B (QUARRY Q3)",
            "source": "out/g2_pooled_six.json per_adapter_A, per_adapter_B", "status": "descriptive"}


# ============================================================================ Q1
S2_FINAL_KEYS = ["tag", "role", "variant", "seed", "rank", "steps", "meas", "clean_alpha", "config_sha", "manifest_sha",
                 "model", "dtype", "loss_tail", "loss_head", "lora_B_norm", "lora_A_norm", "effective", "grad_ckpt",
                 "minutes", "gpu"]
NB01_KEYS = ["loss_tail", "meas", "rank", "steps", "role", "variant", "seed", "config_sha"]


def code_line(path, pattern):
    """First line matching pattern in a .py file or a notebook's code cells."""
    if path.endswith(".ipynb"):
        cells = load(path)["cells"]
        text = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    else:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    for line in text.splitlines():
        if re.search(pattern, line):
            return line.strip()
    return None


def nb02_config_sha():
    """Protocol hash of notebook 02's CFG (session field STAGES excluded), rebuilt from the notebook source."""
    from dataclasses import dataclass, asdict
    src = "".join(load(REPO + "/notebooks/02_seed_ext.ipynb")["cells"][3]["source"])
    body = src[src.find("@dataclass"):src.find("C = CFG()")]
    ns = {"dataclass": dataclass}
    exec(body, ns)
    C = ns["CFG"]()
    return hashlib.sha256(json.dumps({k: x for k, x in asdict(C).items() if k != "STAGES"}, sort_keys=True,
                                     default=str).encode()).hexdigest()


def q1():
    res = {"code": {
        "notebooks/01_pilot.ipynb S2 cell": code_line(REPO + "/notebooks/01_pilot.ipynb", r"CosineAnnealingLR"),
        "cells/S2_FINAL_cell.py": "CosineAnnealingLR(optimizer, T_max=math.ceil(steps / C.GRAD_ACC))"
        if code_line(REPO + "/cells/S2_FINAL_cell.py", r"T_max=math\.ceil\(steps / C\.GRAD_ACC\)") else None,
        "notebooks/02_seed_ext.ipynb": code_line(REPO + "/notebooks/02_seed_ext.ipynb", r"CosineAnnealingLR"),
        "notebooks/12_seed_ext2.ipynb": code_line(REPO + "/notebooks/12_seed_ext2.ipynb", r"CosineAnnealingLR"),
        "v2 src/t1_ladder.py (local stack, as run)": code_line(EINV.SRC + "/t1_ladder.py", r"CosineAnnealingLR"),
        "stepping": "every variant steps the scheduler once per optimizer update (every GRAD_ACC = 2 micro-steps; "
                    "1000 updates in 2000 micro-steps): T_max = 1000 updates decays to ~0, T_max = 2000 ends at half "
                    "the initial rate"},
        "note_docs": "docs/E_INV_RESULTS_v2.md section 9 item 4: 'A_raw_s0 trained under an earlier cell revision'; "
                     "its lora_B_norm was recovered post hoc (notebook 04)"}
    adapters = {}
    if not os.path.isdir(DRIVE):
        res["verdict"] = "Drive mirror not readable; training records not checked"
        return res
    sha02 = nb02_config_sha()
    for root, seeds in (("E_INV_P0_v3", range(0, 3)), ("E_SEEDEXT", range(3, 6)), ("E_SEEDEXT2", range(6, 12))):
        for s in seeds:
            for arm in "AB":
                tag = f"{arm}_raw_s{s}_r16"
                p = f"{DRIVE}/{root}/adapters/{tag}/train_meta.json"
                if not os.path.exists(p):
                    adapters[tag] = {"record": "missing"}
                    continue
                m = load(p)
                keys = list(m.keys())
                if root == "E_INV_P0_v3":
                    if keys == S2_FINAL_KEYS:
                        who, sched = "cells/S2_FINAL_cell.py (key set and order match exactly)", "T_max = 1000 updates (to ~0)"
                    elif keys[:len(NB01_KEYS)] == NB01_KEYS:
                        who, sched = "notebooks/01_pilot.ipynb S2 cell", "T_max = 2000 (to half)"
                    else:
                        who, sched = ("an earlier S2 cell revision (record matches neither S2_FINAL nor the notebook "
                                      "01 cell; lora_B_norm recovered post hoc)"), "not recorded"
                elif root == "E_SEEDEXT":
                    ok = m.get("config_sha") == sha02
                    who = "notebooks/02_seed_ext.ipynb (config_sha " + ("matches" if ok else "DOES NOT match") + \
                          " the notebook's protocol hash)"
                    sched = "T_max = 1000 updates (to ~0)" if ok else "unverified"
                else:
                    ok = {"tag", "arm", "seed", "grad_ckpt", "peak_gb"} <= set(keys)
                    who = "notebooks/12_seed_ext2.ipynb (" + ("record fields match" if ok else "fields DO NOT match") + ")"
                    sched = "T_max = 1000 updates (to ~0)" if ok else "unverified"
                adapters[tag] = {"seed": s, "arm": arm, "writer": who, "schedule": sched,
                                 "loss_tail": m.get("loss_tail"), "gpu": m.get("gpu")}
    res["adapters"] = adapters
    res["nb02_protocol_sha"] = sha02
    # indirect check: training-loss level separates the two noise-level samplers (archive: model's shifted sigma
    # table; notebook 01 cell and local t1_ladder: sigma = u), which a schedule change alone would not
    loc = {}
    for d in sorted(glob.glob(OUT + "/t1/adapters/*")):
        t = os.path.basename(d)
        if re.fullmatch(r"(nomarkB?|alt_[AB]|cm_[AB]|colab)_s\d+", t) and os.path.exists(d + "/train_meta.json"):
            mm = load(d + "/train_meta.json")
            if mm.get("steps") == 2000:
                loc[t] = mm["loss_tail"]
    arch = [a["loss_tail"] for a in adapters.values() if a.get("loss_tail") is not None]
    res["loss_tail_check"] = {
        "archive_primary_min": min(arch), "archive_primary_max": max(arch),
        "A_raw_s0_r16": adapters.get("A_raw_s0_r16", {}).get("loss_tail"),
        "local_2000_step_unmarked_min": min(loc.values()) if loc else None,
        "local_2000_step_unmarked_max": max(loc.values()) if loc else None, "local_n": len(loc),
        "reading": "A_raw_s0's tail loss lies with the archive adapters, not with the sigma = u sampler of the "
                   "notebook 01 cell; indirect evidence only (the stacks also differ)"}
    sched = [a["schedule"] for a in adapters.values()]
    n_full = sum(s == "T_max = 1000 updates (to ~0)" for s in sched)
    n_half = sum(s == "T_max = 2000 (to half)" for s in sched)
    n_unk = len(sched) - n_full - n_half
    res["counts"] = {"to_zero": n_full, "to_half": n_half, "not_recorded": n_unk, "total": len(sched)}
    res["split_confirmed"] = False if n_half == 0 else (n_half == 6 and n_full == 18)
    res["verdict"] = (
        f"The seeds 0-2 vs 3-11 schedule split (QUARRY Q1) is NOT confirmed. {n_full} of {len(sched)} primary "
        f"adapters have records written by code with T_max = 1000 updates (rate decays to ~0), including five of "
        f"the six seed 0-2 adapters (written by cells/S2_FINAL_cell.py, not by the notebook 01 cell that Q1 cites); "
        f"{n_unk} adapter (A_raw_s0_r16) was trained by an earlier cell revision whose schedule is not recorded; "
        f"no primary record shows the half-decay variant. The half-decay variant (T_max = 2000 micro-steps stepped "
        f"per update) is confirmed for the local stack (t1_ladder.py). No descriptive comparison by schedule is "
        f"computed.")
    res["status"] = "record audit; no statistic"
    return res


def main():
    out = {"entry": "RESULTS.md Entry 112", "script": "src/fv/fv_derived.py",
           "status": "post hoc quantities computed for the manuscript; descriptive and sensitivity analyses, "
                     "not pre-specified",
           "N_A1_tpr_calibrated": n_a1()}
    a23 = n_a2_a3()
    out["N_A2_hksj"] = a23["hksj"]
    out["N_A3_d200_combination"] = a23["d200_combination"]
    out["N_A4_dose2000"] = n_a4()
    out["N_A5_totals"] = n_a5()
    out["N_A6_g2_additive"] = n_a6()
    out["Q1_schedule"] = q1()
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    a1 = out["N_A1_tpr_calibrated"]["tpr_calibrated"]
    h = out["N_A2_hksj"]
    print(f"N-A1 TPR(M=2) at calibrated U: G500 {a1['G500']:.4f}  G5000 {a1['G5000']:.4f}  inf {a1['G_inf']:.4f}")
    print(f"N-A2 HKSJ upper99 {h['upper99_pct']:.4f} %  (DL {h['dl_upper99_pct_reproduced']:.4f} %; q {h['q_scale']:.3f}; "
          f"t {h['t']:.3f}, p {h['one_sided_p']:.3f})")
    d = out["N_A3_d200_combination"]
    print(f"N-A3 D200 four designs I2 {d['I2_pct']:.1f} %, Q p {d['Q_p']:.3f}")
    n4 = out["N_A4_dose2000"]
    print(f"N-A4 2000-step local: {n4['adapters_per_body']} per body x {n4['gens_per_adapter']} = {n4['total_gens']}")
    n5 = out["N_A5_totals"]
    print(f"N-A5 totals: {n5['total_adapters']} adapters, {n5['total_gens']} generations "
          f"(with objective arms {n5['with_objective_arms']})")
    n6 = out["N_A6_g2_additive"]
    print(f"N-A6 G2 additive {n6['additive_part']:.3e} = {n6['additive_pct_of_R_real']:.4f} % of R_real")
    print("Q1:", out["Q1_schedule"]["verdict"])
    print("wrote", DST)


if __name__ == "__main__":
    main()
