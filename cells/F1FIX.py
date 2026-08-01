# %% F0b — REPLACEMENT train_full: fp32 master weights, bf16 compute, RELATIVE drift.
#     Run this cell after F0. It redefines train_full; everything downstream picks it up.
#
#     WHY: AdamW's step magnitude is ~LR regardless of gradient scale, so each update is ~2e-6.
#     bf16 spacing at |theta| is |theta|/128, so updates survive ONLY where |theta| < ~2.6e-4.
#     Every larger weight was frozen; the 5.66e-03 drift came from the smallest weights alone —
#     a magnitude-dependent distortion, not under-training. fp32 masters fix it.
#     Footprint: weights 8.3 + grads 8.3 + AdamW m,v 16.7 = 33.4 GB + activations ~= 45 GB.

from diffusers import StableDiffusion3Pipeline

def load_base(with_text=False):
    kw=dict(torch_dtype=DT, token=True)
    if not with_text: kw.update(text_encoder=None, text_encoder_2=None, text_encoder_3=None,
                                tokenizer=None, tokenizer_2=None, tokenizer_3=None)
    try:    return StableDiffusion3Pipeline.from_pretrained(C.MODEL, **kw).to(DEV)
    except Exception:
        p=StableDiffusion3Pipeline.from_pretrained(C.MODEL, torch_dtype=DT, token=True).to(DEV)
        p.text_encoder=p.text_encoder_2=p.text_encoder_3=None; return p

N_PROBE = 60          # parameters sampled across the whole model for the drift probe

def train_full(tag, role, seed, steps, pe, pp, save=True):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    pipe=load_base(); vae,tr,sch = pipe.vae, pipe.transformer, pipe.scheduler
    vae.requires_grad_(False)
    tr = tr.float()                                  # <-- fp32 MASTER WEIGHTS
    tr.requires_grad_(True)
    if C.GRAD_CKPT: tr.enable_gradient_checkpointing()
    tr.train()

    names=[n for n,_ in tr.named_parameters()]
    probe=[names[i] for i in np.linspace(0,len(names)-1,min(N_PROBE,len(names))).astype(int)]
    P=dict(tr.named_parameters())
    theta0={k:P[k].detach().clone() for k in probe}
    norm0=float(np.sqrt(sum(float((v**2).sum()) for v in theta0.values())))

    ps=[p for p in tr.parameters() if p.requires_grad]
    n_par=sum(p.numel() for p in ps)
    print(f"[FT] {tag}: {n_par/1e9:.2f}B trainable (fp32 masters, bf16 compute), LR {C.LR}")
    opt=torch.optim.AdamW(ps, lr=C.LR, weight_decay=0.0)
    opt.zero_grad(set_to_none=True)

    files=sorted(glob.glob(os.path.join(tdir(role),"*.png")))
    if not files: HALT(f"no training PNGs in {tdir(role)}")
    cache=torch.stack([torch.from_numpy(
        np.asarray(Image.open(f).convert("RGB"),np.float32).copy()/127.5-1.0).permute(2,0,1)
        for f in files])
    sf=float(vae.config.scaling_factor); shf=float(vae.config.shift_factor or 0.0)
    ts,sg=sch.timesteps.to(DEV), sch.sigmas.to(DEV); nts=int(sch.config.num_train_timesteps)
    if len(sg)!=nts or len(ts)!=nts: HALT("scheduler array length mismatch — pin diffusers")
    g=torch.Generator().manual_seed(seed); losses=[]; t0=time.time()
    W=max(20, steps//10)                              # window for the divergence check

    for st in range(steps):
        idx=torch.randint(0,cache.shape[0],(C.BATCH,),generator=g)
        px=cache[idx].to(DEV, dtype=DT)
        with torch.no_grad():
            lat=vae.encode(px).latent_dist.sample()
            lat=((lat-shf)*sf).float()                # fp32 target
        noise=torch.randn_like(lat)
        u=torch.sigmoid(torch.randn(lat.shape[0],device=DEV,dtype=torch.float32))
        ii=(u*nts).long().clamp_(0,nts-1)
        sig=sg[ii].float().view(-1,1,1,1)
        noisy=((1.0-sig)*lat + sig*noise)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred=tr(hidden_states=noisy, timestep=ts[ii],
                    encoder_hidden_states=pe.repeat(lat.shape[0],1,1).float(),
                    pooled_projections=pp.repeat(lat.shape[0],1).float(),
                    return_dict=False)[0]
        loss=tF.mse_loss(pred.float(), (noise-lat).float())/C.GRAD_ACC
        loss.backward()
        if (st+1)%C.GRAD_ACC==0:
            torch.nn.utils.clip_grad_norm_(ps,1.0); opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach())*C.GRAD_ACC)
        if not np.isfinite(losses[-1]): HALT(f"{tag}: non-finite loss at step {st+1}")
        if (st+1)%max(20,steps//10)==0:
            P=dict(tr.named_parameters())
            dn=float(np.sqrt(sum(float(((P[k].detach()-v)**2).sum()) for k,v in theta0.items())))
            el=time.time()-t0
            print(f"    {st+1}/{steps} loss {np.mean(losses[-W:]):.5f} "
                  f"rel-drift {dn/norm0:.3e} peak {torch.cuda.max_memory_allocated()/2**30:.0f}GB "
                  f"eta {(el/(st+1))*(steps-st-1)/60:.1f}min")

    open_l=float(np.mean(losses[:W])); tail_l=float(np.mean(losses[-W:]))
    if tail_l > C.MAX_LOSS_RATIO*open_l:
        HALT(f"{tag}: loss diverged over {W}-step windows ({open_l:.4f} -> {tail_l:.4f}). Lower LR.")
    P=dict(tr.named_parameters())
    dnorm=float(np.sqrt(sum(float(((P[k].detach()-v)**2).sum()) for k,v in theta0.items())))
    rel=dnorm/norm0
    print(f"[FT] {tag}: loss {open_l:.5f} -> {tail_l:.5f} | RELATIVE drift {rel:.4e} "
          f"| {(time.time()-t0)/60:.1f} min")
    if rel < 1e-5:
        print(f"[FT] *** WARNING: relative drift {rel:.2e} is negligible — the model barely moved. "
              f"Raise LR before trusting any null from this run. ***")
    if save:
        os.makedirs(cdir(tag),exist_ok=True)
        tr.to(torch.bfloat16).save_pretrained(os.path.join(cdir(tag),"transformer"))  # 4.5 GB
        ajson({"tag":tag,"role":role,"seed":seed,"lr":C.LR,"steps":steps,"config_sha":CFG_SHA,
               "open_loss":open_l,"tail_loss":tail_l,"drift_relative":rel,"drift_abs":dnorm,
               "probe_norm":norm0,"n_probe":len(probe),"precision":"fp32_master_bf16_compute",
               "minutes":(time.time()-t0)/60,"effective":True},
              os.path.join(cdir(tag),"train_meta.json"),indent=2)
    del pipe,tr,vae,cache,theta0; gc.collect(); torch.cuda.empty_cache()
    return open_l, tail_l, rel

# 30-step re-smoke on the fixed path — confirms fp32 masters actually move the weights
pe,pp = cached_prompt()
o,t_,rel = train_full("SMOKE2","A",0,30,pe,pp,save=False)
print(f"\n[F0b] peak {torch.cuda.max_memory_allocated()/2**30:.0f} GB | relative drift {rel:.3e}")
print("[F0b] Expect rel-drift ~1e-5..1e-3 for 30 steps. If still ~0, RAISE C.LR before F1.")
