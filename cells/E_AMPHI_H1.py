# %% H1 — train one adapter per amplitude
def _expected(tag):
    return {"tag":tag,"alpha":alpha_of(tag),"rank":C.RANK,"steps":C.STEPS,"meas":MEAS,
            "seed":C.SEED,"config_sha":CFG_SHA,"model":C.MODEL,"dtype":str(DT)}
def adapter_is_valid(tag):
    p=os.path.join(adir(tag),"train_meta.json")
    if not os.path.exists(os.path.join(adir(tag),"pytorch_lora_weights.safetensors")): return False
    m=json.load(open(p)) if os.path.exists(p) else None
    return m is not None and all(m.get(k)==v for k,v in _expected(tag).items()) and m.get("effective",False)
def _prompt_embeds():
    from diffusers import StableDiffusion3Pipeline
    p=os.path.join(ROOT,"adapters","_prompt.pt")
    if os.path.exists(p):
        z=torch.load(p,map_location=DEV); return z["pe"].to(DEV),z["pp"].to(DEV)
    pipe=StableDiffusion3Pipeline.from_pretrained(C.MODEL,torch_dtype=DT,token=True).to(DEV)
    with torch.no_grad():
        pe,_,pp,_=pipe.encode_prompt(prompt=C.CAPTION,prompt_2=C.CAPTION,prompt_3=C.CAPTION,
                                     device=DEV,num_images_per_prompt=1,
                                     do_classifier_free_guidance=False)
    torch.save({"pe":pe.cpu(),"pp":pp.cpu()},p); del pipe
    gc.collect(); torch.cuda.empty_cache()
    return pe.to(DEV),pp.to(DEV)

def train_arm(tag,pe,pp):
    if adapter_is_valid(tag): print(f"[H1] skip valid {tag}"); return
    if DEV!="cuda": HALT("H1 needs a GPU")
    from diffusers import StableDiffusion3Pipeline
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    out=adir(tag)
    if os.path.exists(out): shutil.move(out,out+f"__stale_{int(time.time())}")
    torch.manual_seed(C.SEED); np.random.seed(C.SEED); random.seed(C.SEED)
    try:
        pipe=StableDiffusion3Pipeline.from_pretrained(
            C.MODEL,torch_dtype=DT,token=True,text_encoder=None,text_encoder_2=None,
            text_encoder_3=None,tokenizer=None,tokenizer_2=None,tokenizer_3=None).to(DEV)
    except Exception:
        pipe=StableDiffusion3Pipeline.from_pretrained(C.MODEL,torch_dtype=DT,token=True).to(DEV)
        pipe.text_encoder=pipe.text_encoder_2=pipe.text_encoder_3=None
    sch,vae,tr=pipe.scheduler,pipe.vae,pipe.transformer
    vae.requires_grad_(False); tr.requires_grad_(False)
    tr.add_adapter(LoraConfig(r=C.RANK,lora_alpha=C.RANK,init_lora_weights="gaussian",
                              target_modules=list(C.TARGETS)))
    if C.GRAD_CKPT: tr.enable_gradient_checkpointing()
    tr.train()
    ps=[p for p in tr.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(ps,lr=C.LR,weight_decay=1e-4)
    lrs=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=math.ceil(C.STEPS/C.GRAD_ACC))
    opt.zero_grad(set_to_none=True)
    files=sorted(glob.glob(os.path.join(tdir(tag),"*.png")))
    if len(files)!=C.N_T: HALT(f"{tag}: {len(files)} training PNGs, expected {C.N_T}")
    cache=torch.stack([torch.from_numpy(
        np.asarray(Image.open(f).convert("RGB"),np.float32).copy()/127.5-1.0).permute(2,0,1)
        for f in files])
    sf=float(vae.config.scaling_factor); sh=float(vae.config.shift_factor or 0.0)
    ts,sg=sch.timesteps.to(DEV),sch.sigmas.to(DEV); nts=int(sch.config.num_train_timesteps)
    if len(sg)!=nts or len(ts)!=nts: HALT("scheduler array length mismatch — pin diffusers")
    g=torch.Generator().manual_seed(C.SEED); losses=[]; t0=time.time()
    print(f"\n[H1] {tag}: alpha={alpha_of(tag)}, rank {C.RANK}, {C.STEPS} steps")
    for st in range(C.STEPS):
        idx=torch.randint(0,cache.shape[0],(C.BATCH,),generator=g)
        px=cache[idx].to(DEV,dtype=DT)
        with torch.no_grad():
            lat=vae.encode(px).latent_dist.sample(); lat=((lat-sh)*sf).to(DT)
        noise=torch.randn_like(lat)
        u=torch.sigmoid(torch.randn(lat.shape[0],device=DEV,dtype=torch.float32))
        ii=(u*nts).long().clamp_(0,nts-1); sig=sg[ii].to(lat.dtype).view(-1,1,1,1)
        pred=tr(hidden_states=((1.0-sig)*lat+sig*noise).to(DT),timestep=ts[ii],
                encoder_hidden_states=pe.repeat(lat.shape[0],1,1).to(DT),
                pooled_projections=pp.repeat(lat.shape[0],1).to(DT),return_dict=False)[0]
        loss=tF.mse_loss(pred.float(),(noise-lat).float())/C.GRAD_ACC
        loss.backward()
        if (st+1)%C.GRAD_ACC==0:
            torch.nn.utils.clip_grad_norm_(ps,1.0); opt.step(); lrs.step()
            opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach())*C.GRAD_ACC)
        if not np.isfinite(losses[-1]): HALT(f"{tag}: non-finite loss at step {st+1}")
        if (st+1)%400==0:
            el=time.time()-t0
            print(f"    {st+1}/{C.STEPS} loss {np.mean(losses[-400:]):.5f} "
                  f"eta {(el/(st+1))*(C.STEPS-st-1)/60:.1f}min")
    sd=get_peft_model_state_dict(tr)
    bn=float(sum(float(v.float().norm()) for k,v in sd.items() if "lora_B" in k))
    if not (bn>1e-6 and all(bool(torch.isfinite(v).all()) for v in sd.values())):
        HALT(f"{tag}: DEAD adapter (||lora_B||={bn:.3e})")
    os.makedirs(out,exist_ok=True)
    StableDiffusion3Pipeline.save_lora_weights(save_directory=out,transformer_lora_layers=sd)
    m=_expected(tag); m.update({"loss_tail":float(np.mean(losses[-100:])),"lora_B_norm":bn,
                                "effective":True,"minutes":(time.time()-t0)/60})
    ajson(m,os.path.join(out,"train_meta.json"),indent=2)
    print(f"[H1] saved {tag}: ||lora_B||={bn:.2f} tail={m['loss_tail']:.5f} "
          f"({m['minutes']:.1f} min)")
    del pipe,tr,vae,cache; gc.collect(); torch.cuda.empty_cache()

if stage("H1"):
    pe,pp=_prompt_embeds()
    for i,t in enumerate(all_tags(),1):
        print(f"[H1] adapter {i}/{len(all_tags())}: {t}"); train_arm(t,pe,pp)
    bad=[t for t in all_tags() if not adapter_is_valid(t)]
    if bad: HALT(f"H1 incomplete: {bad}")
    print(f"[H1] PASS — {len(all_tags())} adapters")
