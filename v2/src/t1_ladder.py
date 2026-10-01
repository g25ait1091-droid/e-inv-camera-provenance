"""Tier 1 — designed-mark ladder. Faithful port of v1 S1c/S2/S3 (notebook 01) for the arms
registered in RESULTS.md Entry 01. Resumable: every stage skips completed work.

    python t1_ladder.py fields      # build M_rand, M_lowmid, M' and log their statistics
    python t1_ladder.py materialise # injected training PNGs per arm + clipped-pixel fraction + R_mark
    python t1_ladder.py train       # LoRA adapters, v1 recipe
    python t1_ladder.py generate    # 500 images per arm from the v1 paired seed bank
    python t1_ladder.py all
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json, glob, time, random
import numpy as np, torch
from PIL import Image
sys.path.insert(0, EINV.SRC)
from fingerprints import load_lum_crop, wavelet_residual, MEAS, dv, splits
from scipy.fft import dctn, idctn

V2 = EINV.V2; OUT = os.path.join(V2, "out", "t1"); FP = os.path.join(V2, "out", "fp")
for d in ("fields", "train_png", "adapters", "gens"): os.makedirs(os.path.join(OUT, d), exist_ok=True)
DEV = "cuda"
HF_MODEL = "stabilityai/stable-diffusion-3.5-medium"
CAPTION = "a photograph, sks style"
LORA_RANK, LORA_TARGETS = 16, ["to_q", "to_k", "to_v", "to_out.0"]
LR, STEPS, BATCH, GRAD_ACC = 1e-4, 2000, 2, 2
# 500 per adapter throughout; T1_GENS lowers it for arms registered at a smaller budget
# (Entry 86: the iPhone 5c and P20 pairs at 250, decided before either arm generated anything).
G_PER_ADAPTER = int(os.environ.get("T1_GENS", "500"))
GEN_SEED_BASE, GEN_STEPS, CFG_SCALE, GEN_BATCH = 770000, 28, 4.5, 4
PROMPT_PT = os.path.join(V2, "data", "adapters", "_prompt_embeds_a1759453b6c6cd42.pt")

# ---- registered arms ---------------------------------------------------------------------------
ARMS = [("mark_rand_a1_s0",    "rand",   1.0,  0),
        ("mark_rand_a3_s0",    "rand",   3.0,  0),
        ("mark_rand_a3_s1",    "rand",   3.0,  1),
        ("mark_rand_a3_s2",    "rand",   3.0,  2),
        ("mark_rand_a12_s0",   "rand",  12.0,  0),
        ("mark_lowmid_a12_s0", "lowmid", 12.0, 0)]
# F4 (RESULTS.md Entry 20): unmarked local-decode control arms, selected with T1_ARMSET=nomark.
# field "none" is the zero field, so the stored PNGs are the decoded crops themselves.
if os.environ.get("T1_ARMSET") == "nomark":
    ARMS = [("nomark_s0", "none", 0.0, 0), ("nomark_s1", "none", 0.0, 1), ("nomark_s2", "none", 0.0, 2)]
# Entry 26: DiffusionShield released watermark, additive at released strength (alpha is a flag only)
if os.environ.get("T1_ARMSET") == "wm":
    ARMS = [("wm_ds_s0", "dswm", 1.0, 0), ("wm_ds_s1", "dswm", 1.0, 1), ("wm_ds_s2", "dswm", 1.0, 2)]
# Entry 27: random-crop training from the native frames (no stored crops; field/alpha are flags only)
if os.environ.get("T1_ARMSET") == "rcrop":
    ARMS = [("rcrop_s0", "rcrop", 0.0, 0), ("rcrop_s1", "rcrop", 0.0, 1), ("rcrop_s2", "rcrop", 0.0, 2)]
# Entry 35 (F7): unmarked B-body arms in the v2 environment — the symmetric partner of nomark_s0..s2
if os.environ.get("T1_ARMSET") == "nomarkB":
    ARMS = [("nomarkB_s0", "noneB", 0.0, 0), ("nomarkB_s1", "noneB", 0.0, 1), ("nomarkB_s2", "noneB", 0.0, 2)]
# Entry 31: v2 stack on v1's Colab-decoded training crops (out/t1/train_png/colab_a0, fetched from the archive)
if os.environ.get("T1_ARMSET") == "colab":
    ARMS = [("colab_s0", "colab", 0.0, 0), ("colab_s1", "colab", 0.0, 1), ("colab_s2", "colab", 0.0, 2)]
# Entry 37: band-limited additive fields (crops materialised by t1_band.py into train_png/band{b}_a4)
if os.environ.get("T1_ARMSET") == "band":
    ARMS = [(f"band{b}_s0", f"band{b}", 4.0, 0) for b in range(6)]
# Entry 40: periodic fields, grid-aligned (32 px) and non-aligned (36 px) (crops from t1_periodic.py)
if os.environ.get("T1_ARMSET") == "periodic":
    ARMS = [("per32_s0", "per32", 4.0, 0), ("per36_s0", "per36", 4.0, 0)]
# Entry 39: adaptation dose, paired by body (crops already in train_png/none_a0 and noneB_a0)
if os.environ.get("T1_ARMSET") == "dose8k":
    ARMS = [("dose8k_A_s0", "none", 0.0, 0), ("dose8k_A_s1", "none", 0.0, 1), ("dose8k_B_s0", "noneB", 0.0, 0), ("dose8k_B_s1", "noneB", 0.0, 1)]
    STEPS, G_PER_ADAPTER = 8000, 250
if os.environ.get("T1_ARMSET") == "dose16k":
    ARMS = [("dose16k_A_s0", "none", 0.0, 0), ("dose16k_B_s0", "noneB", 0.0, 0)]
    STEPS, G_PER_ADAPTER = 16000, 250
# Entry 48 (F9): 16000-step replication, two more adapters per body
if os.environ.get("T1_ARMSET") == "dose16krep":
    ARMS = [("dose16k_A_s1", "none", 0.0, 1), ("dose16k_B_s1", "noneB", 0.0, 1), ("dose16k_A_s2", "none", 0.0, 2), ("dose16k_B_s2", "noneB", 0.0, 2)]
    STEPS, G_PER_ADAPTER = 16000, 250
# Entry 59 (chain 11): E1 fingerprint as a known pattern; E3 content-matched sets; E2 strength/form; E4 tile replicates.
# Training crops come from t1_kfield.py (kinj*, gk*), t1_content_match.py (cmA, cmB) and t1_periodic2.py (per*).
if os.environ.get("T1_ARMSET") == "kinj":
    ARMS = [("kinj_a12_s0", "kinj", 12.0, 0), ("kinj_a12_s1", "kinj", 12.0, 1), ("kinj_a12_s2", "kinj", 12.0, 2),
            ("kinj_a48_s0", "kinj", 48.0, 0), ("kinjd_a3_s0", "kinjd", 3.0, 0)]
if os.environ.get("T1_ARMSET") == "cm":
    ARMS = [(f"cm_{b}_s{s}", f"cm{b}", 0.0, s) for s in range(3) for b in ("A", "B")]
if os.environ.get("T1_ARMSET") == "gk":
    ARMS = [("gkadd_a4_s0", "gkadd", 4.0, 0), ("gkadd_a4_s1", "gkadd", 4.0, 1), ("gkadd_a1_s0", "gkadd", 1.0, 0),
            ("gkadd_a1_s1", "gkadd", 1.0, 1), ("gkmul_a4_s0", "gkmul", 4.0, 0), ("gkmul_a4_s1", "gkmul", 4.0, 1)]
if os.environ.get("T1_ARMSET") == "periodic3":
    ARMS = [(f"per{p}_s{s}", f"per{p}", 4.0, s) for p in (24, 28, 40, 48) for s in (1, 2)]
# Entry 76 (chain 12): G2 second-environment replication at the primary dose; G1 fingerprint-suppressed arms
if os.environ.get("T1_ARMSET") == "nomarkrep":
    ARMS = [(f"nomark{b}_s{s}", f"none{b}", 0.0, s) for s in (3, 4, 5) for b in ("", "B")]
if os.environ.get("T1_ARMSET") == "inv16kext":
    # Entry 99: five more adapters per body on the same inverted crops, to reach eight per arm
    ARMS = [(f"inv16kext_{b}_s{s}", f"inv{b}", 0.0, s) for s in (3, 4, 5, 6, 7) for b in ("A", "B")]
    STEPS, G_PER_ADAPTER = 16000, 250
if os.environ.get("T1_ARMSET") == "inv16k":
    ARMS = [(f"inv16k_{b}_s{s}", f"inv{b}", 0.0, s) for s in (0, 1, 2) for b in ("A", "B")]
    STEPS, G_PER_ADAPTER = 16000, 250
# Entry 78 (chain 13): G5 second training set (D200), G4 modern-smartphone pair (P10 Plus)
if os.environ.get("T1_ARMSET") == "alt":
    ARMS = [(f"alt_{b}_s{s}", f"alt{b}", 0.0, s) for s in (0, 1, 2) for b in ("A", "B")]
if os.environ.get("T1_ARMSET") == "p10":
    ARMS = [(f"p10_{b}_s{s}", f"p10{b}", 0.0, s) for s in range(12) for b in ("A", "B")]
# Entry 80 (lane 3): G6 iPhone 5c pair, VISION
if os.environ.get("T1_ARMSET") == "p20b":
    ARMS = [(f"p20b_{b}_s{s}", f"p20b{b}", 0.0, s) for s in range(12) for b in ("A", "B")]
if os.environ.get("T1_ARMSET") == "p5c":
    ARMS = [(f"p5c_{b}_s{s}", f"p5c{b}", 0.0, s) for s in range(12) for b in ("A", "B")]
# Entry 55: second 16000-step replication, three more adapters per body (seeds 3-5)
if os.environ.get("T1_ARMSET") == "dose16krep2":
    ARMS = [("dose16k_A_s3", "none", 0.0, 3), ("dose16k_B_s3", "noneB", 0.0, 3), ("dose16k_A_s4", "none", 0.0, 4),
            ("dose16k_B_s4", "noneB", 0.0, 4), ("dose16k_A_s5", "none", 0.0, 5), ("dose16k_B_s5", "noneB", 0.0, 5)]
    STEPS, G_PER_ADAPTER = 16000, 250
# Entry 51 (review follow-up): tile replicates and grid-separating tiles (crops from t1_periodic2.py),
# and replicates of the two finest octave bands (crops already in train_png/band{0,1}_a4)
if os.environ.get("T1_ARMSET") == "periodic2":
    ARMS = [("per32_s1", "per32", 4.0, 1), ("per32_s2", "per32", 4.0, 2), ("per36_s1", "per36", 4.0, 1), ("per36_s2", "per36", 4.0, 2),
            ("per24_s0", "per24", 4.0, 0), ("per40_s0", "per40", 4.0, 0), ("per48_s0", "per48", 4.0, 0), ("per28_s0", "per28", 4.0, 0)]
if os.environ.get("T1_ARMSET") == "band2":
    ARMS = [("band0_s1", "band0", 4.0, 1), ("band0_s2", "band0", 4.0, 2), ("band1_s1", "band1", 4.0, 1), ("band1_s2", "band1", 4.0, 2)]
DS_REPO = os.path.join(EINV.EXT, 'DiffusionShield')

def build_dswm():
    """Tile DiffusionShield's released 4x4 patches by its released 64-symbol message (8x8 blocks =
    one 32-px tile) over the 1024^2 crop, as the authors' add_watermark.py lays them out. Returns
    the RGB field in [0,1] units (HxWx3) and its zero-mean luminance."""
    pW = os.path.join(OUT, "fields", "W_ds.npy"); pL = os.path.join(OUT, "fields", "W_ds_lum.npy")
    if os.path.exists(pW) and os.path.exists(pL): return np.load(pW), np.load(pL)
    patches = np.array(torch.load(os.path.join(DS_REPO, "trained_patches", "wm_patches.pt"), weights_only=False), np.float32)  # (3,4,4,3)
    msg = np.array(torch.load(os.path.join(DS_REPO, "example.pt"), weights_only=False)).astype(int)                       # (64,)
    bw, count = 4, 8
    tile = np.zeros((32, 32, 3), np.float32)
    for i in range(count):
        for j in range(count):
            s = msg[count * i + j]
            if s != 0: tile[i*bw:(i+1)*bw, j*bw:(j+1)*bw] += patches[s - 1]
    W = np.tile(tile, (MEAS // 32, MEAS // 32, 1)).astype(np.float32)
    L = lum(W); L = (L - L.mean()).astype(np.float32)
    np.save(pW, W); np.save(pL, L)
    log("built W_ds", W.shape, "abs max", float(np.abs(W).max()), "RMS gray", float(np.sqrt((255*W)**2).mean()) if False else float(np.sqrt(((255*W)**2).mean())))
    return W, L
TARGET_RMS = json.load(open(os.path.join(FP, "K_stats_A.json")))["rms_K_A_E1"]   # 1.0437e-3

def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)

# ---- fields ------------------------------------------------------------------------------------
def band_limit(M):
    m = np.zeros((8, 8), np.float32)
    for i in range(8):
        for j in range(8):
            if 1 <= i + j <= 5 and not (i == 0 and j == 0): m[i, j] = 1.0
    n = MEAS // 8
    B = M.reshape(n, 8, n, 8).transpose(0, 2, 1, 3)
    D = dctn(B, axes=(2, 3), norm="ortho") * m[None, None]
    return idctn(D, axes=(2, 3), norm="ortho").transpose(0, 2, 1, 3).reshape(MEAS, MEAS).astype(np.float32)

def scale(M):
    M = M - M.mean(); return (M * (TARGET_RMS / np.sqrt((M**2).mean()))).astype(np.float32)

def stage_fields():
    p = os.path.join(OUT, "fields")
    if os.path.exists(os.path.join(p, "M_rand.npy")): log("fields exist"); return
    r = np.random.default_rng(20260908); M = scale(r.choice([-1.0, 1.0], size=(MEAS, MEAS)).astype(np.float32))
    r2 = np.random.default_rng(20260909); Mp = scale(r2.choice([-1.0, 1.0], size=(MEAS, MEAS)).astype(np.float32))
    ML = scale(band_limit(M))
    KA1 = np.load(os.path.join(FP, "K_A_E1.npy")); KA2 = np.load(os.path.join(FP, "K_A_E2.npy")); KB2 = np.load(os.path.join(FP, "K_B_E2.npy"))
    def ncc(a, b):
        a = a - a.mean(); b = b - b.mean(); return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
    st = {"target_rms": TARGET_RMS, "rms": {"M_rand": float(np.sqrt((M**2).mean())), "M_lowmid": float(np.sqrt((ML**2).mean())), "M_prime": float(np.sqrt((Mp**2).mean()))},
          "ncc": {"M_rand-M_prime": ncc(M, Mp), "M_rand-M_lowmid": ncc(M, ML), "M_rand-K_A_E1": ncc(M, KA1),
                  "M_rand-K_A_E2": ncc(M, KA2), "M_rand-K_B_E2": ncc(M, KB2), "M_lowmid-K_A_E2": ncc(ML, KA2)}}
    np.save(os.path.join(p, "M_rand.npy"), M); np.save(os.path.join(p, "M_lowmid.npy"), ML); np.save(os.path.join(p, "M_prime.npy"), Mp)
    json.dump(st, open(os.path.join(p, "field_stats.json"), "w"), indent=1); log("fields:", json.dumps(st))

# ---- materialise -------------------------------------------------------------------------------
def load_rgb_crop(fp):
    with Image.open(fp) as im:
        im = im.convert("RGB"); W, H = im.size
        return np.asarray(im.crop(((W-MEAS)//2, (H-MEAS)//2, (W-MEAS)//2+MEAS, (H-MEAS)//2+MEAS)), np.float32)

def inject(rgb, M, alpha):
    """v1 injection operator: Y_alpha = clip[Y * (1 + alpha*M)], luminance field applied per channel."""
    return np.clip(np.rint(rgb * (1.0 + alpha * M[..., None])), 0, 255).astype(np.float32)   # rint: astype() truncates and would record only sign(M)

def lum(rgb): return (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).astype(np.float32)

def train_dir(field, alpha): return os.path.join(OUT, "train_png", f"{field}_a{alpha:g}")

def stage_materialise():
    body = "Nikon_D200_0" if os.environ.get("T1_ARMSET") == "nomarkB" else "Nikon_D200_1"
    T = splits(dv[body])["T"]                                # device A (or B for F7), T split, 50 crops
    fields = {k: np.load(os.path.join(OUT, "fields", f"M_{k}.npy")) for k in ("rand", "lowmid", "prime")}
    Mp = fields["prime"]; fields["none"] = np.zeros_like(Mp); fields["noneB"] = np.zeros_like(Mp)
    from fingerprints import zero_mean  # noqa (import parity with v1 core)
    def ncc0(a, b):
        a = a - a.mean(); b = b - b.mean(); return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
    meta_p = os.path.join(OUT, "train_png", "materialise.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    KA, KB = np.load(os.path.join(FP, "K_A_E2.npy")), np.load(os.path.join(FP, "K_B_E2.npy"))
    for field, alpha in sorted({(a[1], a[2]) for a in ARMS}):
        if field in ("rcrop", "colab"): continue               # rcrop: native frames; colab: archived PNGs already in place
        d = train_dir(field, alpha); os.makedirs(d, exist_ok=True); key = f"{field}_a{alpha:g}"
        if len(glob.glob(os.path.join(d, "*.png"))) >= len(T) and key in meta: log("skip", key); continue
        clip_frac = []; r_mark = []; r_nat = []; rms_gray = []
        if field == "dswm": Wrgb, Wlum = build_dswm()
        else: M = fields[field]
        for i, fp in enumerate(T):
            rgb = load_rgb_crop(fp)
            if field == "dswm":                                    # DiffusionShield rule: additive in [0,1], clip; rounded once
                Y = rgb + 255.0 * Wrgb
                clip_frac.append(float(((Y < 0) | (Y > 255)).mean()))
                inj = np.clip(np.rint(Y), 0, 255).astype(np.float32)
                rms_gray.append(float(np.sqrt(((inj - np.rint(rgb)) ** 2).mean())))
            else:
                Y = rgb * (1.0 + alpha * M[..., None])
                clip_frac.append(float(((Y < 0) | (Y > 255)).mean()))
                inj = inject(rgb, M, alpha)
            Image.fromarray(inj.astype(np.uint8)).save(os.path.join(d, f"{i:04d}.png"), compress_level=1)
            # what the mark contributed in the inputs (registration statistic 4), measured on the stored 8-bit PNG
            Yq = lum(np.asarray(Image.open(os.path.join(d, f"{i:04d}.png")).convert("RGB"), np.float32))
            W = wavelet_residual(Yq)
            if field == "dswm": r_mark.append(ncc0(W, Wlum) - ncc0(W, Mp))          # additive statistic (Entry 26)
            else: r_mark.append(ncc0(W, Yq*M) - ncc0(W, Yq*Mp))
            r_nat.append(ncc0(W, Yq*KA) - ncc0(W, Yq*KB))
        meta[key] = {"n": len(T), "clipped_pixel_fraction": float(np.mean(clip_frac)),
                     "R_mark": float(np.mean(r_mark)), "R_mark_sd": float(np.std(r_mark, ddof=1)),
                     "natural_paired_contrast_on_injected": float(np.mean(r_nat))}
        if rms_gray: meta[key]["stored_change_rms_gray"] = float(np.mean(rms_gray))
        json.dump(meta, open(meta_p, "w"), indent=1); log("materialised", key, meta[key])

# ---- train (v1 S2, verbatim recipe) -----------------------------------------------------------
def adapter_dir(tag): return os.path.join(OUT, "adapters", tag)

def load_prompt():
    d = torch.load(PROMPT_PT, map_location="cpu", weights_only=False)
    if isinstance(d, dict):
        pe = d.get("pe", d.get("prompt_embeds")); pp = d.get("pp", d.get("pooled", d.get("pooled_prompt_embeds")))
    else: pe, pp = d[0], d[1]
    return pe.to(DEV, torch.bfloat16), pp.to(DEV, torch.bfloat16)

def train_arm(tag, field, alpha, seed):
    out = adapter_dir(tag)
    if os.path.exists(os.path.join(out, "pytorch_lora_weights.safetensors")): log("skip existing", tag); return
    from diffusers import StableDiffusion3Pipeline
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    import torch.nn.functional as tF
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    pipe = StableDiffusion3Pipeline.from_pretrained(HF_MODEL, torch_dtype=torch.bfloat16,
                                                    text_encoder=None, text_encoder_2=None, text_encoder_3=None,
                                                    tokenizer=None, tokenizer_2=None, tokenizer_3=None).to(DEV)
    pe, pp = load_prompt()
    vae, tr = pipe.vae, pipe.transformer
    vae.requires_grad_(False); tr.requires_grad_(False)
    tr.add_adapter(LoraConfig(r=LORA_RANK, lora_alpha=LORA_RANK, init_lora_weights="gaussian", target_modules=LORA_TARGETS))
    tr.enable_gradient_checkpointing()
    params = [p for p in tr.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    if field == "rcrop":                                          # Entry 27: native frames, a random 1024^2 crop per step
        frames = [np.asarray(Image.open(fp).convert("RGB"), np.uint8) for fp in splits(dv["Nikon_D200_1"])["T"]]
        assert len(frames) == 50, f"{tag}: expected 50 native frames, found {len(frames)}"
        cache = None; log(f"{tag}: 50 native frames {frames[0].shape} -> random {MEAS}px crops, rank{LORA_RANK} {STEPS} steps")
    else:
        files = sorted(glob.glob(os.path.join(train_dir(field, alpha), "*.png")))
        assert len(files) == 50, f"{tag}: expected 50 training PNGs, found {len(files)}"
        cache = torch.stack([torch.from_numpy(np.asarray(Image.open(f).convert("RGB"), np.float32)/127.5 - 1.0).permute(2, 0, 1) for f in files])
        log(f"{tag}: {len(files)} imgs @ {MEAS}px rank{LORA_RANK} {STEPS} steps")
    sf, sh = vae.config.scaling_factor, vae.config.shift_factor
    g = torch.Generator().manual_seed(seed); losses = []; t0 = time.time()
    for st in range(STEPS):
        if cache is None:
            idx = torch.randint(0, len(frames), (BATCH,), generator=g)
            crops = []
            for k in idx.tolist():
                H, Wd = frames[k].shape[:2]
                r0 = int(torch.randint(0, H - MEAS + 1, (1,), generator=g)); c0 = int(torch.randint(0, Wd - MEAS + 1, (1,), generator=g))
                crops.append(torch.from_numpy(frames[k][r0:r0+MEAS, c0:c0+MEAS].astype(np.float32)/127.5 - 1.0).permute(2, 0, 1))
            px = torch.stack(crops).to(DEV, torch.bfloat16)
        else:
            idx = torch.randint(0, cache.shape[0], (BATCH,), generator=g)
            px = cache[idx].to(DEV, torch.bfloat16)
        with torch.no_grad():
            lat = vae.encode(px).latent_dist.sample(); lat = (lat - sh) * sf
        noise = torch.randn_like(lat)
        u = torch.sigmoid(torch.randn(lat.shape[0], device=DEV)); t = u.view(-1, 1, 1, 1)
        # diffusers 0.39 does not cast inputs; the float32 t would promote the bf16 latents (v1's Colab stack cast internally)
        x_t = ((1.0 - t)*lat + t*noise).to(torch.bfloat16)
        pred = tr(hidden_states=x_t, timestep=(u*1000).to(torch.bfloat16),
                  encoder_hidden_states=pe.repeat(lat.shape[0], 1, 1),
                  pooled_projections=pp.repeat(lat.shape[0], 1), return_dict=False)[0]
        loss = tF.mse_loss(pred.float(), (noise - lat).float()) / GRAD_ACC
        loss.backward()
        if (st + 1) % GRAD_ACC == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss.item() * GRAD_ACC)
        if (st + 1) % 200 == 0: log(f"    {tag} {st+1}/{STEPS} loss {np.mean(losses[-200:]):.4f}  ({(time.time()-t0)/60:.1f} min)")
    os.makedirs(out, exist_ok=True)
    StableDiffusion3Pipeline.save_lora_weights(out, transformer_lora_layers=get_peft_model_state_dict(tr))
    bnorm = float(np.sqrt(sum((p.detach().float()**2).sum().item() for n, p in tr.named_parameters() if p.requires_grad and "lora_B" in n)))
    json.dump({"loss_tail": float(np.mean(losses[-100:])), "meas": MEAS, "rank": LORA_RANK, "steps": STEPS,
               "field": field, "alpha": alpha, "seed": seed, "lora_B_norm": bnorm, "minutes": (time.time()-t0)/60},
              open(os.path.join(out, "train_meta.json"), "w"), indent=1)
    del pipe, tr, vae, opt; torch.cuda.empty_cache(); log("saved", tag, "lora_B_norm", round(bnorm, 3))

def stage_train():
    for tag, field, alpha, seed in ARMS: train_arm(tag, field, alpha, seed)

# ---- generate (v1 S3, paired seed bank) ---------------------------------------------------------
# Entry 106 (P1): an opt-in five-caption bank - v1's DIVERSE_PROMPTS, caption j mod 5 with the same seed per j
# as the uniform bank, so image j pairs across banks - and an output-folder suffix so its images can never mix
# with the uniform-bank ones. Both default off; every other armset is unchanged.
DIVERSE_PROMPTS = ["a photograph of a street", "a photograph of a room interior", "a photograph of trees",
                   "a photograph of a building facade", "a photograph of a table with objects"]
PROMPT_BANK = os.environ.get("T1_PROMPTBANK", "uniform")
GEN_SUFFIX = os.environ.get("T1_GEN_SUFFIX", "")
assert PROMPT_BANK in ("uniform", "diverse"), PROMPT_BANK
assert PROMPT_BANK == "uniform" or GEN_SUFFIX, "the diverse bank must write to suffixed folders (T1_GEN_SUFFIX)"

def gen_dir(tag): return os.path.join(OUT, "gens", tag + GEN_SUFFIX)

def generate(tag):
    outd = gen_dir(tag); os.makedirs(outd, exist_ok=True)
    if PROMPT_BANK == "diverse":
        bank = [(DIVERSE_PROMPTS[j % 5], GEN_SEED_BASE + j) for j in range(G_PER_ADAPTER)]
    else:
        bank = [(CAPTION, GEN_SEED_BASE + j) for j in range(G_PER_ADAPTER)]
    have = len(glob.glob(os.path.join(outd, "*.png")))
    if have >= len(bank): log("skip", tag, f"({have}/{len(bank)})"); return
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(HF_MODEL, torch_dtype=torch.bfloat16).to(DEV)
    pipe.load_lora_weights(adapter_dir(tag))
    pipe.set_progress_bar_config(disable=True)
    i = have; t0 = time.time()
    while i < len(bank):
        chunk = bank[i:i+GEN_BATCH]
        # Entry 87: lanes share one card, so a transient squeeze must not end the run. Each image carries its
        # own seeded generator, so a retry reproduces exactly the same images - waiting costs time, nothing else.
        for attempt in range(72):
            try:
                gs = [torch.Generator(device=DEV).manual_seed(s) for _, s in chunk]
                with torch.no_grad():
                    imgs = pipe(prompt=[p for p, _ in chunk], num_inference_steps=GEN_STEPS,
                                guidance_scale=CFG_SCALE, height=MEAS, width=MEAS, generator=gs).images
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                log(f"{tag}: CUDA OOM at {i}/{len(bank)}, waiting 5 min (attempt {attempt + 1}/72)")
                time.sleep(300)
        else:
            raise RuntimeError(f"{tag}: out of memory at image {i} after 72 attempts over six hours")
        for j, im in enumerate(imgs): im.save(os.path.join(outd, f"{i+j:05d}.png"), format="PNG")
        i += len(chunk)
        if i % 100 < GEN_BATCH: log(f"{tag}: {i}/{len(bank)}  ({(time.time()-t0)/60:.1f} min)")
    del pipe; torch.cuda.empty_cache()

def stage_generate():
    for tag, *_ in ARMS: generate(tag)

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("fields", "all"): stage_fields()
    if what in ("materialise", "all"): stage_materialise()
    if what in ("train", "all"): stage_train()
    if what in ("generate", "all"): stage_generate()
    log("done", what)
