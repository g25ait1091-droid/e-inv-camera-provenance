# %% X — RE-ANALYSIS (X0 bound, X1 shift structure, X2 offset attribution)
#    Implements the review's corrections. Pure re-analysis of existing CSVs: no GPU, ~1 min.
#    Run in a CPU runtime in parallel with GPU work in another session.
#
#    X0 replaces "lambda_min from a power calculation" with a real inferential bound:
#      - one-sided upper confidence limits on the device contrast, Bonferroni-simultaneous
#        across the two symmetric arms;
#      - a DEVICE-SPECIFIC denominator (paired real-image contrast), not rho(Y_A,K_A);
#      - a device-specific VAE ceiling computed with the SAME statistic;
#      - both an image-level (conditional-on-adapters) and an adapter-CLUSTERED bound.
#    The clustered bound is the one that generalises; the image-level one is conditional.

import pandas as pd, numpy as np, json, os, math
from scipy import stats as sps

ROOT_ = ROOT
df5 = pd.read_csv(os.path.join(ROOT_, "csv", "s5_measure.csv"))
s1  = pd.read_csv(os.path.join(ROOT_, "csv", "s1_positive_control.csv"))
RAW_A = [f"A_raw_s{s}_r16" for s in C.TRAIN_SEEDS]
RAW_B = [f"B_raw_s{s}_r16" for s in C.TRAIN_SEEDS]
RAW_D = [f"D_raw_s{s}_r16" for s in C.TRAIN_SEEDS]
X = {}

def gen_rho(tag, K):
    return df5[(df5.tag == tag) & (df5.K == K)].sort_values("gen_idx")["rho_mult"].to_numpy(float)

# ── denominator: device-specific REAL contrast (review §3) ───────────────────────────
def real_contrast(role, other):
    a = s1[(s1.role_img == role) & (s1.role_K == role)].sort_values("idx")["rho_mult"].to_numpy(float)
    b = s1[(s1.role_img == role) & (s1.role_K == other)].sort_values("idx")["rho_mult"].to_numpy(float)
    n = min(len(a), len(b))
    return a[:n] - b[:n]

RA, RB = real_contrast("A", "B"), real_contrast("B", "A")
R_real = 0.5 * (RA.mean() + RB.mean())
X["R_real"] = {"A_arm": float(RA.mean()), "B_arm": float(RB.mean()), "mean": float(R_real),
               "naive_rho_real_A": 0.0381}
print("── X0: device-specific denominator ──")
print(f"  R_real (paired real contrast) = {R_real:.5e}")
print(f"  vs naive rho(Y_A,K_A)=0.0381  ratio {R_real/0.0381:.3f}")

# ── device-specific VAE ceiling with the SAME statistic (review §3) ──────────────────
vae_csv = os.path.join(ROOT_, "csv", "s1b_vae_roundtrip.csv")
if os.path.exists(vae_csv):
    v = pd.read_csv(vae_csv)
    def vae_contrast(role, stage):
        g = v[(v.role == role) & (v.stage == stage)].sort_values("idx")
        own, oth = ("rho_A", "rho_B") if role == "A" else ("rho_B", "rho_A")
        return (g[own].to_numpy(float) - g[oth].to_numpy(float))
    R_vae_pre  = 0.5*(vae_contrast("A","pre").mean()  + vae_contrast("B","pre").mean())
    R_vae_post = 0.5*(vae_contrast("A","post").mean() + vae_contrast("B","post").mean())
    eta_vae = R_vae_post / R_vae_pre
    X["R_vae"] = {"pre": float(R_vae_pre), "post": float(R_vae_post), "eta_device": float(eta_vae)}
    print(f"  R_VAE pre={R_vae_pre:.5e} post={R_vae_post:.5e}")
    print(f"  DEVICE-SPECIFIC VAE retention eta = {eta_vae:.4f}  (scalar retention was 0.358)")
else:
    R_vae_post, eta_vae = None, None
    print("  [!] s1b_vae_roundtrip.csv not found — device-specific VAE ceiling unavailable")

# ── X0: upper confidence limits, image-level AND adapter-clustered ───────────────────
print("\n── X0: upper confidence limits on device transfer ──")
ALPHA_SIM = 0.01                      # 99% SIMULTANEOUS over the two arms
z_bonf = sps.norm.ppf(1 - ALPHA_SIM/2)          # Bonferroni: each arm at 99.5%
res = {}
for name, tags, ks, ko in (("A", RAW_A, "A", "B"), ("B", RAW_B, "B", "A")):
    per_seed = [gen_rho(t, ks) - gen_rho(t, ko) for t in tags]
    allv = np.concatenate(per_seed)
    theta = float(allv.mean())
    # (i) image-level: conditional on the adapters trained
    se_img = float(allv.std(ddof=1) / np.sqrt(len(allv)))
    U_img = theta + z_bonf * se_img
    # (ii) adapter-clustered: the unit of analysis is the adapter (review §6)
    means = np.array([p.mean() for p in per_seed])
    k = len(means)
    se_cl = float(means.std(ddof=1) / np.sqrt(k))
    t_bonf = sps.t.ppf(1 - ALPHA_SIM/2, df=k - 1)
    U_cl = float(means.mean() + t_bonf * se_cl)
    res[name] = {"theta": theta, "seed_means": means.tolist(),
                 "se_image": se_img, "U_image": float(U_img),
                 "se_cluster": se_cl, "t_crit": float(t_bonf), "U_cluster": U_cl}
    print(f"  arm {name}: theta={theta:+.3e}  seeds={np.array2string(means, precision=2)}")
    print(f"           U_image  ={U_img:+.3e}   (conditional on these adapters)")
    print(f"           U_cluster={U_cl:+.3e}   (n={k} adapters, t_crit={t_bonf:.2f})")

U_dev_img = max(res["A"]["U_image"], res["B"]["U_image"])
U_dev_cl  = max(res["A"]["U_cluster"], res["B"]["U_cluster"])
theta_sym = 0.5 * (res["A"]["theta"] + res["B"]["theta"])          # review §4
X["theta_sym"] = theta_sym
X["arms"] = res
X["U_device"] = {"image_level": U_dev_img, "adapter_clustered": U_dev_cl}

lam_hat = theta_sym / R_real
lam_U_img = U_dev_img / R_real
lam_U_cl  = U_dev_cl / R_real
X["lambda"] = {"hat_symmetric": float(lam_hat), "U_image": float(lam_U_img),
               "U_cluster": float(lam_U_cl)}
print(f"\n  theta_sym (symmetric point estimate) = {theta_sym:+.3e}")
print(f"  lambda_hat            = {lam_hat:+.4%}")
print(f"  lambda_U (image-level, 99% simultaneous)      = {lam_U_img:.4%}")
print(f"  lambda_U (adapter-clustered, 99% simultaneous)= {lam_U_cl:.4%}   <- the generalising one")
if R_vae_post:
    X["tau"] = {"U_image": float(U_dev_img/R_vae_post), "U_cluster": float(U_dev_cl/R_vae_post)}
    print(f"  tau_U  = fraction of VAE-TRANSMITTED device signal appearing in generation:")
    print(f"           image-level {U_dev_img/R_vae_post:.4%}   clustered {U_dev_cl/R_vae_post:.4%}")

# leave-one-adapter-out sensitivity (review §6)
print("\n  leave-one-adapter-out (arm A):")
for i, t in enumerate(RAW_A):
    keep = [gen_rho(x, "A") - gen_rho(x, "B") for j, x in enumerate(RAW_A) if j != i]
    m = np.concatenate(keep).mean()
    print(f"    drop {t}: theta={m:+.3e}  lambda={m/R_real:+.4%}")

# ── X1: shift structure by fingerprint and by arm (review §15) ───────────────────────
print("\n── X1: mod-8 argmax structure by fingerprint ──")
X["x1"] = {}
for tags, lbl in ((RAW_A, "A-gens"), (RAW_D, "D-gens"), (["base"], "base")):
    for K in ROLES:
        s = df5[df5.tag.isin(tags) & (df5.K == K)]
        m8 = float((((s.sh_r % 8) == 0) & ((s.sh_c % 8) == 0)).mean())
        m16 = float((((s.sh_r % 16) == 0) & ((s.sh_c % 16) == 0)).mean())
        X["x1"][f"{lbl}->K_{K}"] = {"mod8": m8, "mod16": m16, "n": int(len(s))}
        print(f"  {lbl:8s} -> K_{K}: mod8={m8:.4f} mod16={m16:.4f}  (chance {1/64:.4f}/{1/256:.4f})")
vals = [v["mod8"] for k, v in X["x1"].items()]
print(f"  spread across cells: {min(vals):.4f}–{max(vals):.4f}")
print("  READ: uniform across cells -> generic VAE-grid artifact, cancels in paired contrasts.")
print("        elevated ONLY for A-gens->K_A -> grid-quantised learning; zero-lag was wrong.")

# ── X2: is the D offset generic across all fingerprints? (review §12) ────────────────
print("\n── X2: generic offset by fingerprint (association only, no causal claim) ──")
X["x2"] = {}
for K in ROLES:
    a = float(np.concatenate([gen_rho(t, K) for t in RAW_A]).mean())
    b = float(np.concatenate([gen_rho(t, K) for t in RAW_B]).mean())
    d = float(np.concatenate([gen_rho(t, K) for t in RAW_D]).mean())
    X["x2"][K] = {"A_gens": a, "B_gens": b, "D_gens": d}
    print(f"  K_{K}: A-gens {a:+.3e}  B-gens {b:+.3e}  D-gens {d:+.3e}")
print("  READ: D elevated against EVERY K -> generic residual-statistics difference (one paragraph).")
print("        D elevated only against K_A -> needs a real explanation.")

atomic_json_dump(X, os.path.join(ROOT_, "X_reanalysis.json"), indent=2)
print("\n[X] written to X_reanalysis.json")
print("\nWORDING: report lambda_U (adapter-clustered) as THE bound. lambda_min from the power")
print("         calculation is a DESIGN quantity and must be labelled as such, never as a bound.")
