# %% SANITY BLOCK — run between S5 and S6/S7. ~5 min.
#    SG1 and SG2 can invalidate the power calculation, so they run before any verdict.
import pandas as pd
from collections import Counter

_SG = {}
df5 = pd.read_csv(os.path.join(ROOT, "csv", "s5_measure.csv"))
df4 = pd.read_csv(os.path.join(ROOT, "csv", "s4_copy.csv"))
RAW_A = [f"A_raw_s{s}_r16" for s in C.TRAIN_SEEDS]
RAW_B = [f"B_raw_s{s}_r16" for s in C.TRAIN_SEEDS]
RAW_D = [f"D_raw_s{s}_r16" for s in C.TRAIN_SEEDS]
KEY_TAGS = RAW_A + RAW_D

# ─────────────────────────────────────────────────────────────────────────────
# SG2 — does the null SE scale as 1/sqrt(G)?  THE power-critical check.
# The off-peak of the mean surface IS the circular-shift null of rho_bar, so the
# independent question is whether averaging over images actually buys sqrt(G).
# Split one tag into two interleaved halves; SD(half-mean off-peak) must be
# sqrt(2) x SD(full-mean off-peak) if the 500 images contribute independently.
# ─────────────────────────────────────────────────────────────────────────────
print("── SG2: sqrt(G) scaling of the empirical null ──")
_tag = RAW_A[0]
KA = torch.from_numpy(np.load(K_path("A", "E2"))).to(_MEASURE_DEV)
acc = [torch.zeros((MEAS, MEAS), dtype=torch.float32, device=_MEASURE_DEV) for _ in range(2)]
cnt = [0, 0]
for i in range(C.G_PER_ADAPTER):
    Z_np = load_lum_crop(os.path.join(gen_dir(_tag), f"{i:05d}.png"), use_cache=False)
    W_np = wavelet_residual(Z_np)
    Z = torch.from_numpy(Z_np).to(_MEASURE_DEV)
    W = torch.from_numpy(W_np).to(_MEASURE_DEV)
    S = _xcorr_batch(W, (Z * KA).unsqueeze(0))[0]
    h = i % 2
    acc[h] += S
    cnt[h] += 1
mask = _offpeak_t((MEAS, MEAS))
full = ((acc[0] + acc[1]) / (cnt[0] + cnt[1]))[mask]
half = (acc[0] / cnt[0])[mask]
sd_full, sd_half = float(full.std()), float(half.std())
ratio = sd_half / max(sd_full, 1e-30)
_SG["sg2"] = {"sd_full": sd_full, "sd_half": sd_half, "ratio": ratio,
              "expected": 2 ** 0.5, "n_eff_implied": C.G_PER_ADAPTER / max(ratio ** 2 / 2, 1e-9)}
print(f"  SD(off-peak) full G={cnt[0]+cnt[1]}: {sd_full:.4e}")
print(f"  SD(off-peak) half G={cnt[0]}:  {sd_half:.4e}")
print(f"  ratio {ratio:.3f}  (expect 1.414 under independence)")
print(f"  -> implied n_eff ≈ {_SG['sg2']['n_eff_implied']:.0f} of {C.G_PER_ADAPTER}")
print("  VERDICT:", "OK — sqrt(G) scaling holds" if 1.30 <= ratio <= 1.55
      else "WARNING — images do not contribute independently; recompute lambda_min with n_eff")
del acc, KA
gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None

# ─────────────────────────────────────────────────────────────────────────────
# SG1 — mode collapse / within-tag near-duplicates
# ─────────────────────────────────────────────────────────────────────────────
print("\n── SG1: within-tag generation diversity ──")
_dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).to(DEV).eval()
from torchvision import transforms as _T
_pre = _T.Compose([_T.Resize(256), _T.CenterCrop(224), _T.ToTensor(),
                   _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
@torch.no_grad()
def _emb(paths):
    out = []
    for s in range(0, len(paths), 32):
        b = torch.stack([_pre(_open_rgb(p)) for p in paths[s:s + 32]]).to(DEV)
        out.append(tF.normalize(_dino(b), dim=-1).cpu())
    return torch.cat(out)

_SG["sg1"] = {}
NSUB = 200
for tag in KEY_TAGS:
    paths = [os.path.join(gen_dir(tag), f"{i:05d}.png")
             for i in np.linspace(0, C.G_PER_ADAPTER - 1, NSUB).astype(int)]
    E = _emb(paths)
    Sim = (E @ E.T).numpy()
    np.fill_diagonal(Sim, -1.0)
    dup = int((Sim >= 0.95).sum() // 2)
    # participation ratio of the Gram spectrum: a blunt effective-dimension proxy
    ev = np.linalg.eigvalsh((E @ E.T).numpy())
    ev = np.clip(ev, 0, None)
    pr = float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))
    _SG["sg1"][tag] = {"max_sim": float(Sim.max()), "mean_sim": float(Sim[Sim > -1].mean()),
                       "n_pairs_ge_0.95": dup, "participation_ratio": pr, "n_sub": NSUB}
    print(f"  {tag:18s} max={Sim.max():.3f} mean={Sim[Sim>-1].mean():.3f} "
          f"dup_pairs={dup:4d} PR={pr:5.1f}/{NSUB}")
worst_pr = min(v["participation_ratio"] for v in _SG["sg1"].values())
print("  VERDICT:", "OK — no mode collapse" if worst_pr > NSUB * 0.25
      else "WARNING — low effective diversity; n_eff may be << G")
del _dino
gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None

# ─────────────────────────────────────────────────────────────────────────────
# SG3 — PCE argmax shift histogram. The VAE downsamples x8 and the DiT patchifies
# x2, so clustering at multiples of 8 or 16 would mean a learned but grid-quantised
# pattern — a FINDING, not noise.
# ─────────────────────────────────────────────────────────────────────────────
print("\n── SG3: shift structure ──")
sub = df5[df5.tag.isin(RAW_A) & (df5.K == "A")]
sr, sc = sub["sh_r"].to_numpy(), sub["sh_c"].to_numpy()
at_zero = int(((sr == 0) & (sc == 0)).sum())
mod8 = float((((sr % 8) == 0) & ((sc % 8) == 0)).mean())
mod16 = float((((sr % 16) == 0) & ((sc % 16) == 0)).mean())
_SG["sg3"] = {"n": len(sub), "at_zero_lag": at_zero, "frac_mod8": mod8, "frac_mod16": mod16,
              "chance_mod8": 1 / 64, "chance_mod16": 1 / 256}
print(f"  n={len(sub)}  argmax exactly at (0,0): {at_zero}")
print(f"  both shifts ≡0 mod 8 : {mod8:.4f}  (chance {1/64:.4f})")
print(f"  both shifts ≡0 mod 16: {mod16:.4f}  (chance {1/256:.4f})")
print("  VERDICT:", "uniform — ordinary null" if mod8 < 3 / 64
      else "INVESTIGATE — grid-quantised pattern would change the conclusion")

# ─────────────────────────────────────────────────────────────────────────────
# SG4 — do generated and real images sit in the same measurement regime?
# ─────────────────────────────────────────────────────────────────────────────
print("\n── SG4: gen vs real measurement regime ──")
s1 = pd.read_csv(os.path.join(ROOT, "csv", "s1_positive_control.csv"))
real_cross = s1[(s1.role_img == "A") & (s1.role_K == "C")]["pce0"].to_numpy()
gen_null = df5[df5.tag.isin(RAW_D) & (df5.K == "A")]["pce0"].to_numpy()
_SG["sg4"] = {"real_cross_pce_sd": float(np.std(real_cross)),
              "gen_null_pce_sd": float(np.std(gen_null)),
              "real_cross_pce_med": float(np.median(real_cross)),
              "gen_null_pce_med": float(np.median(gen_null))}
print(f"  real cross-device PCE: med={np.median(real_cross):+.2f} sd={np.std(real_cross):.2f}")
print(f"  gen null (D→K_A) PCE : med={np.median(gen_null):+.2f} sd={np.std(gen_null):.2f}")
print("  (a large sd gap means the denominator of lambda comes from a different regime "
      "than the numerator — state it, it does not invalidate the gen-vs-gen tests)")

# ─────────────────────────────────────────────────────────────────────────────
# SG5 — copy-flag rate. If high, gate G3 runs on much less data.
# ─────────────────────────────────────────────────────────────────────────────
print("\n── SG5: copy-flag rates ──")
_SG["sg5"] = {}
for tag in all_tags():
    g = df4[df4.tag == tag]
    r = float(g["copy_flag"].mean())
    _SG["sg5"][tag] = {"rate": r, "n_flagged": int(g["copy_flag"].sum()),
                       "by": {c: int((g[c] >= th).sum()) if c != "phash_min"
                              else int((g[c] <= C.COPY_PHASH).sum())
                              for c, th in (("dino_sim", C.COPY_DINO), ("crop_sim", C.COPY_CROP),
                                            ("phash_min", 0), ("resid_z", C.COPY_RESID_Z))}}
    if tag in KEY_TAGS or r > 0.05:
        print(f"  {tag:18s} {r*100:5.1f}%  {_SG['sg5'][tag]['by']}")
worst = max(_SG["sg5"][t]["rate"] for t in RAW_A)
print("  VERDICT:", "OK" if worst < 0.20 else
      "WARNING — G3's copy-excluded rerun loses substantial data; report n remaining")

# ─────────────────────────────────────────────────────────────────────────────
# Descriptive peek at the pre-registered primary (S7 computes it properly)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── descriptive peek: primary contrast (not a test) ──")
def _pick(tag, K):
    return df5[(df5.tag == tag) & (df5.K == K)].sort_values("gen_idx")["rho_mult"].to_numpy()
for name, tags, ks, ko in (("S_A", RAW_A, "A", "B"), ("S_B", RAW_B, "B", "A")):
    d = np.concatenate([_pick(t, ks) - _pick(t, ko) for t in tags])
    print(f"  {name}: mean={d.mean():+.3e}  sd={d.std():.3e}  n={len(d)}  "
          f"z≈{d.mean()/(d.std()/np.sqrt(len(d))):+.2f}")
XA = np.concatenate([_pick(t, "A") for t in RAW_A])
XD = np.concatenate([_pick(t, "A") for t in RAW_D])
print(f"  A-gens→K_A mean={XA.mean():+.3e}   D-gens→K_A mean={XD.mean():+.3e}")

atomic_json_dump(_SG, os.path.join(ROOT, "sanity_block.json"), indent=2)
print("\n[SANITY] written to sanity_block.json — review SG1/SG2 before trusting lambda_min")
