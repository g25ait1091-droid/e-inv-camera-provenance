# %% C4 — mixed model: separate between-device from adapter-training variance
#     FIXES vs the shipped version:
#       1. `for dv in dm` iterated the Series VALUES, not its index -> KeyError. Now dm.index.
#       2. Added a DENOMINATOR. The shipped cell reported theta and U in raw contrast units with
#          nothing to divide by, so there was no lambda_U for the multi-device experiment and
#          nothing comparable to the headline. R_real for the Kodak set is computed from
#          c0_positive_control.csv using C4's OWN statistic (own vs mean-of-the-other-four).
#       3. Per-device sign agreement across seeds is printed explicitly — it is the thing the
#          crossed design exists to detect.
def c4():
    import pandas as pd
    rng = np.random.default_rng(0)

    # ---- denominator: real-image contrast on the H split, SAME statistic as theta ----
    pc = pd.read_csv(os.path.join(ROOT, "csv", "c0_positive_control.csv"))
    per_dev_real = []
    for d in DEVS():
        g = pc[pc.img_dev == d]
        own = g[g.K_dev == d].sort_values("idx")["rho"].to_numpy(float)
        oth = np.mean([g[g.K_dev == o].sort_values("idx")["rho"].to_numpy(float)
                       for o in DEVS() if o != d], axis=0)
        n = min(len(own), len(oth))
        per_dev_real.append(float((own[:n] - oth[:n]).mean()))
    per_dev_real = np.array(per_dev_real)
    R_real = float(per_dev_real.mean())
    se_R = float(per_dev_real.std(ddof=1) / np.sqrt(len(per_dev_real)))
    R_low = R_real - sps.norm.ppf(0.99) * se_R          # conservative (lower 99%) denominator
    print(f"R_real (Kodak, own vs mean-of-others, n={len(DEVS())} devices) = {R_real:.5e}")
    print(f"   per-device: {np.array2string(per_dev_real, precision=5)}")
    print(f"   SE {se_R:.3e}   lower99 {R_low:.5e}   <- conservative denominator\n")

    out = {"config_sha": CFG_SHA, "R_real": R_real, "R_real_lower99": R_low,
           "per_device_real": per_dev_real.tolist(), "variants": {}}

    for variant in ("raw", "lodo"):
        d = pd.read_csv(os.path.join(ROOT, "csv", f"c3_measure_{variant}.csv"))
        rows = []
        for tag, dev, seed in all_arms():
            g = d[d.tag == tag]
            own = g[g.K == dev].sort_values("gen_idx")["rho"].to_numpy(float)
            oth = [g[g.K == o].sort_values("gen_idx")["rho"].to_numpy(float)
                   for o in DEVS() if o != dev]
            n = min([len(own)] + [len(x) for x in oth])
            th = own[:n] - np.mean([x[:n] for x in oth], axis=0)
            m = json.load(open(os.path.join(ROOT, "adapters", tag, "train_meta.json")))
            rows.append({"tag": tag, "dev": dev, "seed": seed, "theta": float(th.mean()),
                         "se_img": float(th.std(ddof=1)/np.sqrt(len(th))),
                         "lora_B": m.get("lora_B_norm"), "tail": m.get("loss_tail")})
        R = pd.DataFrame(rows)
        print(f"\n{'='*74}\n=== {variant.upper()} ===")
        print(R.to_string(index=False, float_format=lambda x: f"{x:.5g}"))

        dm = R.groupby("dev")["theta"].mean()
        gm = float(R.theta.mean()); k = len(dm); nps = len(C.SEEDS)

        # per-device sign agreement across seeds — what the crossed design is FOR
        print(f"\n  per-device means and seed agreement:")
        n_agree = 0
        for dv in dm.index:                                   # <-- FIX 1: .index, not the Series
            vals = R[R.dev == dv].sort_values("seed")["theta"].to_numpy()
            agree = bool(np.all(np.sign(vals) == np.sign(vals[0])) and vals[0] != 0)
            n_agree += agree
            print(f"    {dv}: [{', '.join(f'{v:+.3e}' for v in vals)}]"
                  f"  mean {dm[dv]:+.3e}  seeds agree in sign: {agree}")
        print(f"  {n_agree}/{k} devices have both seeds on the same side "
              f"(chance would be ~{k/2:.1f})")

        # variance decomposition theta_ds = mu + u_d + v_ds
        ms_dev = float(nps * ((dm - gm)**2).sum() / (k - 1))
        within = float(sum(((R[R.dev == dv].theta - dm[dv])**2).sum()
                           for dv in dm.index) / (k * (nps - 1)))   # <-- FIX 1 again
        var_u = max((ms_dev - within) / nps, 0.0)
        icc = float(var_u / (var_u + within)) if (var_u + within) > 0 else 0.0

        # device-level inference: n = number of DEVICES, not adapters
        se_dev = float(dm.std(ddof=1) / np.sqrt(k))
        tcrit = float(sps.t.ppf(1 - C.ALPHA/2, df=k - 1))
        U = float(gm + tcrit * se_dev)
        obs = dm.mean(); cnt = 0
        for _ in range(C.N_PERM):
            cnt += ((dm.to_numpy() * rng.choice((-1., 1.), k)).mean() >= obs)
        p = (cnt + 1) / (C.N_PERM + 1)

        out["variants"][variant] = {
            "per_adapter": rows, "device_means": dm.to_dict(), "grand_mean": gm,
            "n_devices": k, "seeds_per_device": nps, "n_sign_agree": int(n_agree),
            "var_between_device": var_u, "var_within_device": within, "icc": icc,
            "t_crit": tcrit, "U_device_level": U, "p_signflip_devices": float(p),
            "lambda_hat": gm / R_real, "lambda_U": U / R_low,
            "device_mean_spread": float(dm.max() - dm.min())}

        print(f"\n  grand mean theta   = {gm:+.4e}      lambda_hat = {gm/R_real:+.4%}")
        print(f"  between-device var = {var_u:.4e}   within-device (seed) var = {within:.4e}")
        print(f"  ICC (device share) = {icc:.3f}")
        print(f"  device-mean spread = {dm.max()-dm.min():.4e}")
        print(f"  device-level U(99%, n={k}, t={tcrit:.2f}) = {U:+.4e}")
        print(f"  lambda_U (device level, conservative denominator) = {U/R_low:.4%}")
        print(f"  sign-flip p over the {k} device means = {p:.4g}"
              f"{'   <-- SIGNIFICANT' if p < C.ALPHA else ''}")

    raw, lodo = out["variants"]["raw"], out["variants"]["lodo"]
    print(f"\n{'='*74}\nRAW vs LODO")
    print(f"  device-mean spread {raw['device_mean_spread']:.4e} -> "
          f"{lodo['device_mean_spread']:.4e}  "
          f"({100*(1-lodo['device_mean_spread']/max(raw['device_mean_spread'],1e-30)):+.1f}%)")
    print(f"  ICC               {raw['icc']:.3f} -> {lodo['icc']:.3f}")
    print(f"  lambda_U          {raw['lambda_U']:.4%} -> {lodo['lambda_U']:.4%}")
    print("="*74)
    print("READ:")
    print("  - ICC near 0 => adapter-training variance dominates and device identity contributes")
    print("    little. ICC well above 0 with MIXED SIGNS => stable PER-DEVICE BIAS, not leakage:")
    print("    leakage predicts every device favours its OWN fingerprint, i.e. all theta > 0.")
    print("  - The sign-flip p over device means is the actual test.")
    print("  - If LODO shrinks the device-mean spread, the per-device offsets were shared")
    print("    camera-model structure — that is what five bodies bought you.")
    print("  - This bound generalises over DEVICES (n=5); the seed-level headline (0.1507% at k=12) generalises over")
    print("    SEEDS on two devices. Different claims; report both, neither substitutes.")
    ajson(out, os.path.join(ROOT, "C4_multidev.json"), indent=2)

if stage("C4"): c4()
