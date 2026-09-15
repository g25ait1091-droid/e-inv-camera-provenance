"""Entry 26 — measurement for the DiffusionShield watermark arm.

Arms: wm_ds_s0..s2 (out/t1/gens), never-injected: base / A_raw_s0 / B_raw_s0 (data/gens) and
nomark_s0..s2 (out/t1/gens). Per image:
  rho_add(W_lum), rho_add(M'), 30 decoys = circular rolls of W_lum with row and column shifts that
  are both non-multiples of the 32-px period; rho_mult against K_A / K_B (natural contrast);
  DiffusionShield bit accuracy: 1,024 tiles of 32x32, 64 blocks of 4x4 each, released classifier
  argmax -> two bits per block, compared with the tiled released message.
Also scores the 50 stored training crops with the released classifier (should be ~1).
Outputs out/t1/wm_rows.csv, out/t1/wm_summary.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json, glob, csv, time
import numpy as np, torch
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, EINV.SRC)

V2 = EINV.V2; T1 = os.path.join(V2, "out", "t1"); FP = os.path.join(V2, "out", "fp")
DS = os.path.join(EINV.EXT, 'DiffusionShield')
ROWS = os.path.join(T1, "wm_rows.csv"); SUMM = os.path.join(T1, "wm_summary.json")
WM_ARMS = ["wm_ds_s0", "wm_ds_s1", "wm_ds_s2"]
NEVER_ARCH = ["base", "A_raw_s0_r16", "B_raw_s0_r16"]; NEVER_V2 = ["nomark_s0", "nomark_s1", "nomark_s2"]
N_DECOY = 30; DECOY_SEED = 20260911; WORKERS = int(os.environ.get("T1_W", "6")); PERIOD = 32

_G = {}
def _init():
    import torch as _t; _t.set_num_threads(1)
    from fingerprints import wavelet_residual, MEAS
    rng = np.random.default_rng(DECOY_SEED)
    Wl = np.load(os.path.join(T1, "fields", "W_ds_lum.npy"))
    shifts = []
    while len(shifts) < N_DECOY:
        dr, dc = int(rng.integers(64, MEAS-64)), int(rng.integers(64, MEAS-64))
        if dr % PERIOD and dc % PERIOD: shifts.append((dr, dc))
    _G.update(wr=wavelet_residual, MEAS=MEAS, Wl=Wl, Mp=np.load(os.path.join(T1, "fields", "M_prime.npy")),
              KA=np.load(os.path.join(FP, "K_A_E2.npy")), KB=np.load(os.path.join(FP, "K_B_E2.npy")),
              decoys=[np.roll(np.roll(Wl, dr, 0), dc, 1) for dr, dc in shifts], shifts=shifts)

def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

def _measure(task):
    arm, path = task; g = _G
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    if a.shape[0] != g["MEAS"]: return None
    Y = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(np.float32); W = g["wr"](Y)
    row = [arm, os.path.basename(path), _ncc(W, g["Wl"]), _ncc(W, g["Mp"]), _ncc(W, Y*g["KA"]), _ncc(W, Y*g["KB"])]
    row += [_ncc(W, D) for D in g["decoys"]]
    return row

# ---- the released detector ------------------------------------------------------------------
def load_classifier(dev):
    sys.path.insert(0, DS)
    from resnet import ResNet_modified
    net = ResNet_modified(); ck = torch.load(os.path.join(DS, "trained_model", "classifier.pt"), map_location=dev, weights_only=False)
    net = torch.nn.DataParallel(net); net.load_state_dict(ck["net"]); net.to(dev).eval(); return net

def tiled_message():
    msg = np.array(torch.load(os.path.join(DS, "example.pt"), weights_only=False)).astype(int)   # (64,) symbols 0..3 on an 8x8 grid
    return msg

@torch.no_grad()
def bit_accuracy(net, path, msg, dev):
    a = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0).permute(2, 0, 1)  # (3,1024,1024)
    C, H, Wd = a.shape; bw = 4
    # blocks in raster order over the whole image: (H/4)*(W/4) blocks of 4x4
    blocks = a.unfold(1, bw, bw).unfold(2, bw, bw)                      # (3, H/4, W/4, 4, 4)
    nb_r, nb_c = blocks.shape[1], blocks.shape[2]
    blocks = blocks.permute(1, 2, 0, 3, 4).reshape(-1, C, bw, bw).to(dev)
    preds = []
    for k in range(0, blocks.shape[0], 8192):
        preds.append(torch.argmax(net(blocks[k:k+8192]), dim=1).cpu())
    preds = torch.cat(preds).numpy().reshape(nb_r, nb_c)
    # expected symbol at block (i, j): message index ((i mod 8) * 8 + (j mod 8))
    ii, jj = np.meshgrid(np.arange(nb_r) % 8, np.arange(nb_c) % 8, indexing="ij")
    expect = msg[ii * 8 + jj]
    sym_acc = float((preds == expect).mean())
    bits_p = np.stack([preds // 2, preds % 2]); bits_e = np.stack([expect // 2, expect % 2])
    return float((bits_p == bits_e).mean()), sym_acc

def main():
    hdr = ["arm", "image", "rho_W", "rho_Mp", "rho_KA", "rho_KB"] + [f"decoy{i}" for i in range(N_DECOY)]
    done = set()
    if os.path.exists(ROWS):
        for r in csv.DictReader(open(ROWS)): done.add((r["arm"], r["image"]))
    tasks = []
    for arm in WM_ARMS + NEVER_V2:
        for f in sorted(glob.glob(os.path.join(T1, "gens", arm, "*.png"))):
            if (arm, os.path.basename(f)) not in done: tasks.append((arm, f))
    for arm in NEVER_ARCH:
        for f in sorted(glob.glob(os.path.join(V2, "data", "gens", arm, "*.png"))):
            if (arm, os.path.basename(f)) not in done: tasks.append((arm, f))
    print(f"[wm] {len(done)} present, {len(tasks)} to measure", flush=True)
    new = not os.path.exists(ROWS); fh = open(ROWS, "a", newline=""); wr = csv.writer(fh)
    if new: wr.writerow(hdr)
    t0 = time.time(); n = 0
    with Pool(WORKERS, initializer=_init) as pool:
        for row in pool.imap_unordered(_measure, tasks, chunksize=4):
            if row is None: continue
            wr.writerow(row); n += 1
            if n % 200 == 0: fh.flush(); print(f"[wm] {n}/{len(tasks)} {(time.time()-t0)/60:.1f} min", flush=True)
    fh.close()
    # ---- released detector ----
    dev = "cuda" if torch.cuda.is_available() else "cpu"; net = load_classifier(dev); msg = tiled_message()
    det = {}
    sets = {"train_png_dswm": sorted(glob.glob(os.path.join(T1, "train_png", "dswm_a1", "*.png")))}
    for arm in WM_ARMS + NEVER_V2: sets[arm] = sorted(glob.glob(os.path.join(T1, "gens", arm, "*.png")))[:200]
    for arm in NEVER_ARCH: sets[arm] = sorted(glob.glob(os.path.join(V2, "data", "gens", arm, "*.png")))[:200]
    for k, files in sets.items():
        acc = np.array([bit_accuracy(net, f, msg, dev) for f in files])
        det[k] = {"n": len(files), "bit_accuracy_mean": float(acc[:, 0].mean()), "bit_accuracy_sd": float(acc[:, 0].std(ddof=1)) if len(files) > 1 else None,
                  "symbol_accuracy_mean": float(acc[:, 1].mean())}
        print(f"[wm] detector {k}: {det[k]}", flush=True)
    # ---- statistics ----
    import pandas as pd
    from scipy import stats as sps
    d = pd.read_csv(ROWS); mat = json.load(open(os.path.join(T1, "train_png", "materialise.json")))
    R = mat["dswm_a1"]["R_mark"]
    S = {"R_wm": R, "stored_change_rms_gray": mat["dswm_a1"].get("stored_change_rms_gray"), "arms": {}, "released_detector": det}
    for arm in WM_ARMS + NEVER_ARCH + NEVER_V2:
        g = d[d.arm == arm]
        if not len(g): continue
        c = g.rho_W - g.rho_Mp; dec = g[[f"decoy{i}" for i in range(N_DECOY)]].mean().values
        S["arms"][arm] = {"n": int(len(g)), "rho_W": float(g.rho_W.mean()), "rho_Mp": float(g.rho_Mp.mean()),
                          "contrast": float(c.mean()), "contrast_se": float(c.std(ddof=1)/np.sqrt(len(g))), "t_image": float(c.mean()/(c.std(ddof=1)/np.sqrt(len(g)))),
                          "decoy_rank_of_true_W": int(1 + (dec > g.rho_W.mean()).sum()), "decoy_mean": float(dec.mean()), "decoy_max": float(dec.max()),
                          "natural_paired_KA_minus_KB": float((g.rho_KA - g.rho_KB).mean())}
    never = [S["arms"][a]["contrast"] for a in NEVER_ARCH + NEVER_V2 if a in S["arms"]]
    off = float(np.mean(never)) if never else 0.0
    wm = [S["arms"][a]["contrast"] for a in WM_ARMS if a in S["arms"]]
    if len(wm) == 3:
        m, sd = float(np.mean(wm)), float(np.std(wm, ddof=1)); tcrit = float(sps.t.ppf(0.995, 2))
        S["cluster"] = {"per_seed": wm, "mean": m, "sd": sd, "t_vs_zero": m/(sd/np.sqrt(3)), "t_vs_offset": (m-off)/(sd/np.sqrt(3)), "t_crit_0995_df2": tcrit,
                        "never_injected_offset": off, "lambda_wm_pct": 100*m/R, "lambda_wm_offset_corrected_pct": 100*(m-off)/R,
                        "lambda_wm_U_pct": 100*(m + tcrit*sd/np.sqrt(3))/R}
    json.dump(S, open(SUMM, "w"), indent=1); print(json.dumps({k: v for k, v in S.items() if k != "arms"}, indent=1))
    for a, e in S["arms"].items(): print(a, e)

if __name__ == "__main__":
    main()
