# %% H4 — the dose-response extended, plotted on EFFECTIVE amplitude
def h4():
    rng=np.random.default_rng(0)
    d=pd.read_csv(os.path.join(ROOT,"csv","h3_measure.csv"))
    fid=json.load(open(os.path.join(ROOT,"h0_fidelity.json")))
    eff={r[0]:r[1] for r in fid["rows"]}; psnr={r[0]:r[2] for r in fid["rows"]}
    def v(tag,K): return d[(d.tag==tag)&(d.K==K)].sort_values("gen_idx")["rho"].to_numpy(float)
    def sf(x,n=C.N_PERM):
        o=x.mean(); c=sum((x*rng.choice((-1.,1.),len(x))).mean()>=o for _ in range(n))
        return (c+1)/(n+1),float(o)
    # the E_AMP arms already measured, for one continuous curve
    PRIOR={1.0:-1.520e-05,1.6:-2.862e-05,3.0:-6.868e-05,6.0:-3.233e-05,12.0:-2.101e-05}
    out={"config_sha":CFG_SHA,"levels":{}}
    print(f"{'alpha_nom':>10s} {'alpha_eff':>10s} {'PSNR':>7s} {'contrast':>12s} {'t':>7s} {'p':>9s}")
    for a_prev,c_prev in sorted(PRIOR.items()):
        print(f"{a_prev:10.1f} {'(E_AMP)':>10s} {'':>7s} {c_prev:+12.3e} {'':>7s} {'':>9s}")
    for tag in all_tags():
        a=alpha_of(tag); ctr=v(tag,"B")-v(tag,"A")
        p_,o_=sf(ctr); t_=o_/(ctr.std(ddof=1)/np.sqrt(len(ctr)))
        out["levels"][a]={"alpha_effective":eff.get(a),"psnr":psnr.get(a),
                          "contrast":o_,"t":float(t_),"p":p_,"n":int(len(ctr))}
        print(f"{a:10.1f} {eff.get(a,float('nan')):10.2f} {psnr.get(a,float('nan')):7.1f} "
              f"{o_:+12.3e} {t_:7.2f} {p_:9.4g}{'  <== DETECTED' if p_<C.ALPHA else ''}")
    det=[a for a,r in out["levels"].items() if r["p"]<C.ALPHA]
    hi=max(eff.get(a,0) for a in out["levels"])
    print("\n"+"="*72)
    if det:
        print(f"TRANSFER DETECTED at nominal {min(det):g} "
              f"(~{eff.get(min(det)):.1f}x effective) — this changes the headline.")
    else:
        print(f"NO TRANSFER up to ~{hi:.1f}x EFFECTIVE amplitude (nominal {max(out['levels']):g}),")
        print(f"extending the previous ceiling of ~3.5x. Fidelity at that level: "
              f"{psnr.get(max(out['levels'])):.1f} dB.")
        print("Report alongside the fidelity ceiling: reaching 12x effective needs nominal ~41 at")
        print("~34 dB, i.e. VISIBLE damage. That bounds ANY injection-based verification scheme")
        print("using an estimated fingerprint, not just this one.")
    print("="*72)
    ajson(out,os.path.join(ROOT,"E_AMPHI_results.json"),indent=2)
if stage("H4"): h4()
