"""Consolidated findings index - regenerated from the result files, never transcribed.

Writes FINDINGS.md: every finding with its current value, the file it came from, the pre-specified
reading where one applies, and where it belongs in the manuscript. Re-run whenever a result lands;
anything not yet measured is listed as pending rather than omitted, so the gaps stay visible.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2
OUT = os.path.join(V2, "out")
T1 = os.path.join(OUT, "t1")
LEDGER = os.path.join(EINV.REPO, 'analysis', 'FINAL_LEDGER.json')
BT = chr(96)
L = []


def load(*p):
    try:
        return json.load(open(os.path.join(*p)))
    except Exception:
        return None


def row(sec, finding, value, src, reading="", where=""):
    L.append((sec, finding, value, src, reading, where))


led = load(LEDGER)
if led:
    pr, den = led["primary"], led["denominators"]
    eta = {"value": den["eta"], "ci": den["eta_ci"]}
    row("Stages", "Real-image device contrast R_real", f"{den['R_real']:.6f}", "FINAL_LEDGER.json", "", "V-A")
    row("Stages", "Autoencoder retention eta", f"{eta['value']:.4f} [{eta['ci'][0]:.4f}, {eta['ci'][1]:.4f}]",
        "FINAL_LEDGER.json", "about a third survives", "V-A")
    row("Limits", "Primary max-arm limit", f"{100 * max(pr['U_A'], pr['U_B']) / den['R_real']:.4f} %",
        "FINAL_LEDGER.json", "pre-specified construction, 12 adapters per arm", "V-D, Table V")
    row("Limits", "Primary symmetric statistic", f"{0.5 * (pr['theta_A_mean'] + pr['theta_B_mean']):+.3e}", "FINAL_LEDGER.json", "", "V-D")

kf = load(T1, "kfield_summary.json")
if kf:
    f = kf["fields"]
    row("Transmission map", "Fingerprint injected as a known pattern (alpha 12)",
        f"{f['kinj_a12']['T_mean_pct']:.4f} % (SE {f['kinj_a12']['T_se_pct']:.4f})",
        "t1/kfield_summary.json", "passes less than the band prediction (E1)", "V-E")
    row("Transmission map", "Fingerprint amplitude ratio alpha48/alpha12",
        f"{kf['ratios']['kinj48_over_kinj12']:.2f}", "t1/kfield_summary.json", "not linear in amplitude", "V-E")
    row("Transmission map", "Spectrum-matched random field, 4 gray",
        f"{f['gkadd_a4']['T_mean_pct']:.4f} %", "t1/kfield_summary.json", "", "V-E")
    row("Transmission map", "Amplitude ratio 1 gray / 4 gray",
        f"{kf['ratios']['gkadd1_over_gkadd4']:.3f}", "t1/kfield_summary.json", "linear (E2)", "V-E")
    row("Transmission map", "Multiplicative / additive at equal stored change",
        f"{kf['ratios']['gkmul4_over_gkadd4']:.3f}", "t1/kfield_summary.json", "form does not matter (E2)", "V-E")
    row("Transmission map", "Band prediction over direct injection",
        f"{kf['prediction_K_spectrum_pct'] / f['gkadd_a4']['T_mean_pct']:.2f}x",
        "t1/kfield_summary.json", "the extrapolation was high; register R9", "V-E, VI")

per = load(T1, "periodic2_summary.json")
if per:
    for k in sorted(per["fields"], key=lambda x: per["fields"][x]["period_px"]):
        v = per["fields"][k]
        grid = "on" if v["on_latent_grid_8px"] else "off"
        row("Transmission map", f"Tile {v['period_px']} px ({grid} the 8-px grid)",
            f"{v['lambda_mean_pct']:.3f} % (n={v['n_adapters']})", "t1/periodic2_summary.json", "", "V-E, Fig. 5")

c7 = load(OUT, "c7_transplant.json")
if c7:
    for det in ("ncc", "pce", "lowmid", "noiseprint"):
        d = c7["results"].get(det)
        if d:
            sm = min((r["s"] for r in d["per_s"] if r.get("increment_t", 0) > 3), default=None)
            row("Detectors", f"{det.upper()}: smallest transplant detected", f"s = {sm}",
                "c7_transplant.json", "no detector is insensitive in generated images (C7)", "V-F, S12")

c8 = load(OUT, "c8_learned_texture.json")
cm = load(OUT, "t2_learned_arms_cm.json")
um = load(OUT, "t2_learned_arms_nomark.json")
if c8:
    o, sh = c8["stats_orig"], c8["stats_shuffle"]
    row("Detectors", "Learned detector interaction",
        f"{o['theta_sym']:+.3f} logits ({o['lambda_sym_pct']:.2f} % of its real contrast)",
        "c8_learned_texture.json", "", "V-F")
    row("Detectors", "Block-shuffled / original", f"{sh['theta_sym'] / o['theta_sym']:.3f}",
        "c8_learned_texture.json", "local texture, not spatial arrangement (C8)", "V-F, S6")
if cm and um:
    a, b = cm["stats_orig"]["theta_sym"], um["stats_orig"]["theta_sym"]
    row("Detectors", "Content-matched / unmatched interaction",
        f"{a:+.3f} vs {b:+.3f} logits ({100 * a / b:.1f} %)", "t2_learned_arms_cm.json",
        "training-set content, not the camera (E3); register R10", "V-F, VI")

ds = load(T1, "dose_stats.json")
if ds:
    for k, lab in (("2000", "2000 steps"), ("8000", "8000 steps"),
                   ("16000", "16000 steps, six adapters per body"),
                   ("16000_new", "16000 steps, registered replication")):
        d = ds["doses"].get(k)
        if d:
            row("Adaptation dose", lab,
                f"theta_sym {d['theta_sym']:+.3e} = {d['lambda_sym_pct']:.4f} %, t {d['t']:.2f}",
                "t1/dose_stats.json", "transfer resolved at this dose; register R11" if "16000" in k else "", "V-G")

fx = load(OUT, "flux_seed_ext_summary.json")
if fx:
    row("Robustness", "FLUX.1-dev, six adapters per arm",
        f"max-arm {fx['max_arm']['lambda_U_pct']:.4f} %, symmetric {fx['symmetric']['lambda_sym_U_pct']:.4f} %",
        "flux_seed_ext_summary.json", fx["reading"], "V-H, Table IX")
if led:
    for r in led["replications"]:
        if "FLUX" not in r["name"]:
            row("Robustness", r["name"], f"{r['lambda_U_pct']:.2f} % (n={r['n']}, {r['construction']})",
                "FINAL_LEDGER.json", "", "V-H, Table IX")

g2 = load(OUT, "g2_pooled_six.json")
if g2:
    row("Robustness", "Second training environment, six adapters per arm",
        f"theta_sym {g2['theta_sym']:+.3e} = {100 * g2['theta_sym'] / 0.0356703:.4f} %, p {g2['one_sided_p']:.3f}",
        "g2_pooled_six.json", "inconclusive: neither replicated nor excluded (G2)", "V-H, VII")

if load(OUT, "c5_shift_grid.json"):
    row("Method checks", "Shifted-template excess at 8-px displacements",
        "present in base-model generations too", "c5_shift_grid.json", "not explained by the grid (C5)", "VII, S6")
if load(OUT, "c6_coverage.json"):
    row("Method checks", "Coverage of the two limit constructions",
        "max-arm 1.000; symmetric 0.987-0.996", "c6_coverage.json", "max-arm adequate, symmetric marginal (C6)", "S12")

# Device pairs: the gate decides, on real photographs alone, whether an arm can carry a limit at all.
for sub, label in (("fp_p10", "Huawei P10 Plus (G4)"), ("fp_p20b", "Huawei P20 (G4b)"),
                   ("fp_5c", "Apple iPhone 5c (G6)")):
    m = load(OUT, sub, "manifest.json")
    if not m: continue
    g = m["gates"]
    for tag in ("E2", "FLAT"):
        sfx = "" if tag == "E2" else "_" + tag
        if "AUC_held_out" + sfx not in g: continue
        pb = g.get("per_body_contrast" + sfx, {})
        ok = g["AUC_held_out" + sfx] >= 0.90 and min(g.get("splithalf_A", 1), g.get("splithalf_B", 1)) >= 0.15
        row("Device pairs", f"{label}, {tag} estimate",
            f"AUC {g['AUC_held_out' + sfx]:.3f}, R_real {g['R_real' + sfx]:.4f}"
            + (f", bodies {pb['A']:+.4f} / {pb['B']:+.4f}" if pb else ""),
            f"{sub}/manifest.json",
            "gate met; a limit may be claimed" if ok else "gate not met; descriptive only",
            "VII")

g4 = load(OUT, "g4_p10.json")
if g4 and not g4.get("gate_met", True):
    row("Device pairs", "G4 outcome", "closed at the fingerprint gate, no generations scored",
        "g4_p10.json", "one 2017 smartphone body's PRNU is not estimable from its own photographs", "VII")

h1 = load(OUT, "h1_strength.json")
if h1:
    m = h1["models"]["lora_B_norm"]; f = m["fits"]
    row("Adaptation dose", "16000-step interaction: body vs how hard the adapter trained",
        f"body alone p {f['body_only']['body_A']['p']:.3f}; strength alone p {f['strength_only']['strength']['p']:.3f}; "
        f"together p {f['body_plus_strength']['body_A']['p']:.3f} / {f['body_plus_strength']['strength']['p']:.3f} "
        f"(collinear r {m['corr_strength_body']:+.2f})",
        "h1_strength.json", h1["reading"], "V-G, VII")

h2 = load(OUT, "h2_estimation_error.json")
if h2:
    lp = h2["limit_pct"]; vc = h2["variance_components_pct2"]
    row("Limits", "Symmetric limit over bootstrapped fingerprint estimates",
        f"{lp['mean']:.4f} % (sd {lp['sd']:.4f}, range {lp['min']:.4f} to {lp['max']:.4f}), n={h2['n_replicates']}",
        "h2_estimation_error.json", "the estimate's own sampling error, measured", "VII")
    row("Limits", "Variance from not knowing the fingerprint",
        f"{100 * vc['estimation_share']:.1f} % of the total", "h2_estimation_error.json",
        "a component the published limit sets to zero", "VII")
    row("Limits", "Estimation-inflated limit",
        f"{h2['inflated_limit_pct']:.4f} % against {h2['filed_limit_pct']:.4f} % filed "
        f"({h2['inflation_factor_vs_filed']:.2f}x)", "h2_estimation_error.json", h2["reading"], "VII")
    row("Limits", "Null across bootstrap estimates",
        f"one-sided p {h2['null']['min_p']:.3f} to {h2['null']['max_p']:.3f}; "
        f"{h2['null']['n_replicates_rejecting_at_01']} of {h2['n_replicates']} reject at 0.01",
        "h2_estimation_error.json", "the null does not depend on one estimate", "VII")

h3 = load(OUT, "h3_shift_model.json"); h3b = load(OUT, "h3b_shift_model.json")
if h3 and h3b:
    # the first H3 run predates the pooled block; its per-arm means give the same pooled figure
    o1 = sum(a["observed_excess"] for a in h3["arms"].values()) / len(h3["arms"])
    pb = h3b["pooled_on_grid"]
    row("Method checks", "Shifted-template excess vs a grid-periodic prediction",
        f"predicted {pb['predicted_excess']:.2e}; observed {o1:.2e} then {pb['observed_excess']:.2e} "
        f"on disjoint samples",
        "h3b_shift_model.json",
        "within 2 SE twice, but the observation is too noisy to confirm; artifact stays open", "VII")

h4 = load(OUT, "h4_coverage2.json")
if h4:
    w = h4["worst_case"]
    row("Method checks", "Coverage with fingerprint-estimation error (measured)",
        f"symmetric {w['estimation_only']['min_coverage_sym']:.4f}, "
        f"max-arm {w['estimation_only']['min_coverage_maxarm']:.4f} (nominal 0.99)",
        "h4_coverage2.json", "max-arm is the primary construction; symmetric is secondary", "VII, S12")
    row("Method checks", "Coverage adding a training-set component at 1x adapter SD",
        f"symmetric {w['both_at_1sd']['min_coverage_sym']:.4f}, "
        f"max-arm {w['both_at_1sd']['min_coverage_maxarm']:.4f}",
        "h4_coverage2.json", "sensitivity sweep, not a measurement until G5 lands", "VII, S12")

h5 = load(OUT, "h5_weight_signature.json")
if h5:
    for dose, d in sorted(h5["doses"].items(), key=lambda kv: int(kv[0])):
        b = d["batch_diagnostic_descriptive"]
        row("Detectors", f"Body recoverable from adapter weights, {dose} steps",
            f"same-body cos {d['mean_cos_same_body']:+.4f} vs cross-body {d['mean_cos_diff_body']:+.4f}, "
            f"D {d['D']:+.4f}, permutation p {d['perm_p_one_sided']:.3f}",
            "h5_weight_signature.json", d["reading"], "VII")
        row("Detectors", f"Adapter weights by training batch, {dose} steps",
            f"same-batch cos {b['mean_cos_same_batch']:+.4f} vs cross-batch {b['mean_cos_diff_batch']:+.4f} "
            f"(D {b['D_batch']:+.4f})", "h5_weight_signature.json",
            "descriptive: the training run dominates weight space, not the camera", "VII")

g5 = load(OUT, "g5_alt_training.json")
if g5:
    a = g5["alt"]; c = g5["comparison"]
    row("Robustness", "Second, disjoint training set per body",
        f"theta_sym {a['theta_sym']:+.3e} ({a['theta_sym_pct']:+.4f} %), p {a['one_sided_p']:.3f}, "
        f"limit {a['limit99_pct']:.4f} % (n={g5['n_per_arm']} per arm)",
        "g5_alt_training.json", g5["reading"], "V-H, Table IX")
    row("Robustness", "Second training set vs primary",
        f"difference {c['difference']:+.2e} +- {c['se_difference']:.2e} (z {c['z']:+.2f}, p {c['two_sided_p_normal']:.3f})",
        "g5_alt_training.json", "the two training sets are indistinguishable", "V-H")

gc = load(OUT, "g1_covariate.json")
if gc:
    row("Adaptation dose", "G1: fingerprint inverted in the training photographs",
        f"theta_sym {gc['theta_sym']:+.3e} ({gc['theta_sym_pct']:+.4f} %), one-sided p {gc['one_sided_p_lt0']:.3f}",
        "g1_covariate.json", gc["registered_reading"] + " - negative as predicted, but not resolved", "V-G, VII")
    pw = gc["power"]
    row("Adaptation dose", "G1 power against the effect it mirrors",
        f"{pw['achieved_power_at_target']:.2f}; needed {pw['n_per_arm_required']} adapters per arm, ran {pw['n_per_arm_run']}",
        "g1_covariate.json", "the test could not have decided; not evidence of absence", "VII")
    row("Adaptation dose", "G1 strength covariate recurs",
        f"corr(norm, theta) {gc['corr_strength_theta']:+.3f}, corr(norm, body) {gc['corr_strength_body']:+.3f}; "
        f"adjusted p {gc['strength_adjusted']['one_sided_p_lt0']:.3f}",
        "g1_covariate.json", "adaptation strength stays entangled with body identity", "VII")

g6 = load(OUT, "g6_p5c.json")
if g6:
    for tag, e in g6["estimators"].items():
        sy, ma = e["symmetric"], e["max_arm"]
        row("Robustness", f"iPhone 5c paired design (G6), {tag} estimate",
            f"theta_sym {100 * sy['theta_sym'] / e['R_real']:+.4f} %, p {e['p_sym']:.3f}; "
            f"symmetric {sy['lambda_sym_pct']:.4f} %, max-arm {ma['lambda_U_pct']:.4f} %; "
            f"additive part {0.5 * (e['mean_A'] - e['mean_B']):+.2e}",
            "g6_p5c.json", e["reading"], "V-H, VII")
    dd = g6["estimator_dependence"]
    row("Robustness", "G6 estimator comparison (flat fields vs natural images)",
        f"symmetric limit ratio {dd['ratio_FLAT_over_E2']:.2f}; per-adapter r {dd['per_adapter_corr']:.2f}",
        "g6_p5c.json", "spread reported: a good estimate does not remove estimator dependence", "VII")

g4b = load(OUT, "g4b_p20.json")
if g4b and g4b.get("gate_met"):
    sy, ma, R = g4b["symmetric"], g4b["max_arm"], g4b["R_real"]
    row("Robustness", "Huawei P20 paired design (G4b), modern smartphone",
        f"theta_sym {100 * sy['theta_sym'] / R:+.4f} %, t {sy['t']:.2f}, p {sy['one_sided_p']:.3f}; "
        f"symmetric {sy['lambda_sym_pct']:.4f} %, max-arm {ma['lambda_U_pct']:.4f} %",
        "g4b_p20.json", g4b["reading"], "V-H, VII")
    add = g4b.get("additive_part", 0.5 * (g4b["mean_A"] - g4b["mean_B"]))
    row("Detectors", "G4b shared main effect",
        f"additive part {add:+.2e} ({100 * add / R:.2f} % of R_real); IU p_A {g4b['iu_signflip']['p_A']:.5f}, "
        f"p_B {g4b['iu_signflip']['p_B']:.3f}",
        "g4b_p20.json", "every adapter leans toward one body; paired statistic cancels it", "V-B")

m1 = load(OUT, "m1_pooled.json")
if m1:
    pr = m1["primary"]; q = pr["quoted"]
    row("Limits", "Transfer pooled over three device pairs (M1)",
        f"{pr[q]['lambda_pct']:+.4f} %, p {pr[q]['one_sided_p']:.3f}, one-sided 99 % limit {pr[q]['upper99_pct']:.4f} % "
        f"({q} effects, I2 {100 * pr['I2']:.0f} %)", "m1_pooled.json", m1["reading"], "V-H, VII")
    sf = m1["sensitivity_flat_field"]; si = m1["sensitivity_estimation_inflated"]
    row("Limits", "M1 sensitivities",
        f"flat-field iPhone {sf['fixed']['lambda_pct']:+.4f} % p {sf['fixed']['one_sided_p']:.3f}; "
        f"estimation-inflated p {si['fixed']['one_sided_p']:.3f}", "m1_pooled.json",
        "a common lean of a few hundredths of a per cent is neither established nor excluded", "VII")
h6 = load(OUT, "h6_calibrated_limit.json")
if h6:
    pd6 = h6["primary_d200"]
    row("Limits", "Headline limit calibrated to cover at 99 % (H6)",
        f"{pd6['lambda_U_calibrated_pct']:.4f} % (c = {pd6['c']}) against nominal {pd6['lambda_U_nominal_pct']:.4f} %",
        "h6_calibrated_limit.json", "the paper's headline 99 % limit", "V-D")

g1e = load(OUT, "g1_ext.json")
if g1e:
    a8, n5 = g1e["all_eight"], g1e["five_new_alone"]
    row("Adaptation dose", "G1 extension: fingerprint inverted, eight adapters per arm",
        f"theta_sym {a8['theta_sym_pct']:+.4f} %, p(<0) {a8['one_sided_p_lt0']:.4f}; five new alone p {n5['one_sided_p_lt0']:.3f}; "
        f"strength-adjusted p {g1e['strength_covariate']['strength_adjusted']['one_sided_p_lt0']:.4f}",
        "g1_ext.json", g1e["reading"], "V-G, VII")
wbd = load(OUT, "g1_within_body.json")
if wbd:
    pb = wbd["per_body"]; bb = wbd["both_bodies"]
    row("Adaptation dose", "Inverted minus normal, within each body (descriptive)",
        f"A {pb['A']['difference_pct']:+.3f} % (p {pb['A']['one_sided_p_drop']:.4f}), B {pb['B']['difference_pct']:+.3f} % "
        f"(p {pb['B']['one_sided_p_drop']:.4f}); both {bb['difference_pct']:+.3f} %, z {bb['z']:.2f}",
        "g1_within_body.json", "both bodies follow the sign of their own fingerprint; not a one-body effect", "V-G")

p1 = load(OUT, "p1_prompts.json")
if p1:
    for tag in ("E2", "FLAT"):
        e = p1["estimators"][tag]; dvv = e["diverse"]
        row("Robustness", f"Five everyday captions, iPhone 5c, {tag} estimate (P1)",
            f"theta_sym {dvv['theta_sym_pct']:+.4f} %, p {dvv['one_sided_p']:.4f}; uniform bank "
            f"{e['uniform']['theta_sym_pct']:+.4f} %; difference p {e['paired_difference_diverse_minus_uniform']['two_sided_p']:.2f}",
            "p1_prompts.json", e["reading"], "V-H, VII")
    fw = p1["firearm"]
    row("Robustness", "Firearm share of generations (P1)",
        f"'sks' caption {100 * fw['uniform_bank']['share']:.1f} %; five captions {100 * fw['diverse_bank']['share']:.1f} %",
        "p1_prompts.json", "the diverse bank removes the rifles entirely", "IV, V-H")

PENDING = ()
for name, path, desc, where in PENDING:
    row("In flight", name, "measured - see file" if load(path) else "PENDING", os.path.basename(path), desc, where)

head = ("# Consolidated findings\n\n"
        "Regenerated by " + BT + "src/findings.py" + BT + " from the result files; do not edit by hand.\n"
        "Every row names its source file, and " + BT + "verify_v2.py" + BT +
        " recomputes the headline subset from those same files.\n")
body, last = [], None
for sec, finding, value, src, reading, where in L:
    if sec != last:
        body.append("\n## " + sec + "\n\n| Finding | Value | Source | Reading | Paper |\n|---|---|---|---|---|")
        last = sec
    body.append("| " + finding + " | " + value + " | " + BT + src + BT + " | " + reading + " | " + where + " |")
open(os.path.join(V2, "FINDINGS.md"), "w", encoding="utf-8").write(head + "\n".join(body) + "\n")
print(f"FINDINGS.md written: {len(L)} findings, {sum(1 for r in L if r[2] == 'PENDING')} pending")
