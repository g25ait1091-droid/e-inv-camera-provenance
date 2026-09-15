"""Entry 18, follow-up F3 (descriptive) and F2 (E-PROMPT) — the saved learned detector applied to
new generation sets.

  python t2_learned_f3.py ladder   -> out/t2_learned_f3.json  (six Tier 1 arms, all trained on body A)
  python t2_learned_f3.py prompt   -> out/t2_learned_f2.json  (E-PROMPT: base + A/B s0-s2 on the
                                      five-caption bank, out/t5/gens/<arm>)
Score per image = patch-mean logit(A) - logit(B) from out/t2_learned_net.pt; per arm mean, SE,
n; for the prompt set also theta_A, theta_B, theta_sym and the additive part over the three
seeds, and the shift relative to the prompt-bank base arm. Nothing is trained here.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json, glob, numpy as np, torch
sys.path.insert(0, EINV.SRC)
import t2_learned as L

V2 = EINV.V2; NET = os.path.join(V2, "out", "t2_learned_net.pt")
mode = sys.argv[1] if len(sys.argv) > 1 else "ladder"
if mode == "ladder":
    arms = {a: os.path.join(V2, "out", "t1", "gens", a) for a in
            ("mark_rand_a1_s0", "mark_rand_a3_s0", "mark_rand_a3_s1", "mark_rand_a3_s2", "mark_rand_a12_s0", "mark_lowmid_a12_s0")}
    OUT = os.path.join(V2, "out", "t2_learned_f3.json")
else:
    arms = {a: os.path.join(V2, "out", "t5", "gens", a) for a in
            ("base", "A_raw_s0_r16", "A_raw_s1_r16", "A_raw_s2_r16", "B_raw_s0_r16", "B_raw_s1_r16", "B_raw_s2_r16")}
    OUT = os.path.join(V2, "out", "t2_learned_f2.json")

net = L.Net().to(L.DEV); net.load_state_dict(torch.load(NET, map_location=L.DEV)); net.eval()

@torch.no_grad()
def score(W):
    x = torch.from_numpy(L.patches(W, np.random.default_rng(L.SEED)))[:, None].to(L.DEV)
    lg = net(x); return float((lg[:, 0] - lg[:, 1]).mean())

res = {"model": NET, "arms": {}}
for arm, d in arms.items():
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    s = np.array([score(L.resid(f, f"gen_{mode}_{arm}_{os.path.basename(f)}")) for f in files])
    res["arms"][arm] = {"n": len(s), "mean": float(s.mean()), "se": float(s.std(ddof=1) / np.sqrt(len(s))),
                        "mean_first_250": float(s[:250].mean()) if len(s) >= 250 else None}
    print(f"[{mode}] {arm}: n={len(s)} mean {s.mean():+.3f} ± {s.std(ddof=1)/np.sqrt(len(s)):.3f}", flush=True)
if mode == "prompt" and all(k in res["arms"] for k in arms):
    A = [res["arms"][f"A_raw_s{i}_r16"]["mean"] for i in range(3)]; B = [res["arms"][f"B_raw_s{i}_r16"]["mean"] for i in range(3)]
    thA, thB = float(np.mean(A)), float(-np.mean(B)); base = res["arms"]["base"]["mean"]
    vA, vB = np.var(A, ddof=1) / 3, np.var(B, ddof=1) / 3
    res["summary"] = {"theta_A": thA, "theta_B": thB, "theta_sym": 0.5 * (thA + thB), "additive_part": 0.5 * (np.mean(A) + np.mean(B)),
                      "adapter_level_SE_theta_sym": 0.5 * float(np.sqrt(vA + vB)), "base": base,
                      "A_minus_base": float(np.mean(A) - base), "B_minus_base": float(np.mean(B) - base)}
    print("[prompt]", json.dumps(res["summary"], indent=1))
json.dump(res, open(OUT, "w"), indent=1); print("written", OUT)
