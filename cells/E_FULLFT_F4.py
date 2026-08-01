# %% F4 — adapter-level IUT + the SYMMETRIC statistic. Quality-gated.
def f4():
    rng=np.random.default_rng(0)
    d=pd.read_csv(os.path.join(ROOT,"csv","f3_measure.csv"))
    def th(tag,own,oth):
        a=d[(d.tag==tag)&(d.K==own)].sort_values("gen_idx")["rho"].to_numpy(float)
        b=d[(d.tag==tag)&(d.K==oth)].sort_values("gen_idx")["rho"].to_numpy(float)
        n=min(len(a),len(b)); return a[:n]-b[:n]
    def sf(x,n=C.N_PERM):
        o=x.mean(); c=sum((x*rng.choice((-1.,1.),len(x))).mean()>=o for _ in range(n))
        return (c+1)/(n+1)
    out={"config_sha":CFG_SHA,"arms":{}}
    print(f"{'arm':4s} {'run':14s} {'theta':>12s} {'tail loss':>10s} {'drift':>11s}")
    per={}
    for arm,own,oth in (("A","A","B"),("B","B","A")):
        v=[]
        for s in C.SEEDS:
            t=f"{arm}_ft_s{s}"; x=th(t,own,oth)
            if not len(x): continue
            v.append(x)
            m=json.load(open(os.path.join(cdir(t),"train_meta.json")))
            print(f"{arm:4s} {t:14s} {x.mean():+12.3e} {m['tail_loss']:10.5f} "
                  f"{m['drift_probe']:11.3e}")
        mm=np.array([x.mean() for x in v]); per[arm]=mm; k=len(mm)
        tc=float(sps.t.ppf(1-C.ALPHA/2,df=k-1)); U=float(mm.mean()+tc*mm.std(ddof=1)/np.sqrt(k))
        out["arms"][arm]={"n":k,"means":mm.tolist(),"theta":float(mm.mean()),
                          "sd":float(mm.std(ddof=1)),"U":U,"p_adapter":sf(mm)}
        print(f"     -> n={k} theta={mm.mean():+.3e} sd={mm.std(ddof=1):.3e} U={U:+.3e} "
              f"p={out['arms'][arm]['p_adapter']:.4g}\n")
    # SYMMETRIC statistic — the additive term cancels (see RESULTS 7.1)
    sym=(per["A"]+per["B"])/2; k=len(sym)
    tc=float(sps.t.ppf(1-C.ALPHA/2,df=k-1)); Usym=float(sym.mean()+tc*sym.std(ddof=1)/np.sqrt(k))
    R_real=3.56703e-02
    out["symmetric"]={"per_seed":sym.tolist(),"theta_sym":float(sym.mean()),
                      "sd":float(sym.std(ddof=1)),"U":Usym,"lambda_U":Usym/R_real,
                      "t":float(sym.mean()/(sym.std(ddof=1)/np.sqrt(k)))}
    IUT=all(out["arms"][a]["p_adapter"]<C.ALPHA for a in ("A","B"))
    out["IUT_pass"]=bool(IUT)
    print("="*72)
    print(f"FULL FINE-TUNING — IUT: {'PASS — the null WAS a LoRA artefact' if IUT else 'FAIL — the null holds under full fine-tuning'}")
    print(f"  theta_sym = {sym.mean():+.3e}  t = {out['symmetric']['t']:+.2f}")
    print(f"  lambda_U (symmetric, n={k}) = {Usym/R_real:.4%}   [FLAG AS n=3, full FT]")
    print("="*72)
    print("BEFORE BELIEVING THIS: run the base study's S4 copy audit on these generations.")
    print("LoRA gave 0.0%. Full FT on 50 images overfits; if the copy rate is high, memorisation")
    print("is doing the work and the device statistic is not interpretable. Also eyeball ~20")
    print("generations per arm — a bound from degraded generations is worse than no bound.")
    ajson(out,os.path.join(ROOT,"E_FULLFT_results.json"),indent=2)
if stage("F4"): f4()
