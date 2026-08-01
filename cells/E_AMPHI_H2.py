# %% H2 — generation, same paired seed bank as every other experiment
def h2():
    if DEV!="cuda": HALT("H2 needs a GPU")
    from diffusers import StableDiffusion3Pipeline
    bank=[(C.CAPTION,C.GEN_SEED_BASE+j) for j in range(C.G_PER)]
    pipe=StableDiffusion3Pipeline.from_pretrained(C.MODEL,torch_dtype=DT,token=True).to(DEV)
    pipe.set_progress_bar_config(disable=True); prev=False
    try:
        for tag in all_tags():
            d_=gdir(tag); os.makedirs(d_,exist_ok=True)
            for t_ in glob.glob(os.path.join(d_,"*.tmp")): os.remove(t_)
            miss=[i for i in range(C.G_PER)
                  if not os.path.exists(os.path.join(d_,f"{i:05d}.png"))]
            if not miss: print(f"[H2] skip complete {tag}"); continue
            if prev: pipe.unload_lora_weights(); prev=False
            pipe.load_lora_weights(adir(tag)); prev=True
            print(f"[H2] {tag}: {len(miss)} missing"); t0=time.time()
            for s in range(0,len(miss),C.GEN_BATCH):
                ix=miss[s:s+C.GEN_BATCH]
                gs=[torch.Generator(device=DEV).manual_seed(bank[i][1]) for i in ix]
                with torch.inference_mode():
                    imgs=pipe(prompt=[bank[i][0] for i in ix],num_inference_steps=C.GEN_STEPS,
                              guidance_scale=C.CFG_SCALE,height=MEAS,width=MEAS,
                              generator=gs).images
                for i,im in zip(ix,imgs):
                    t_=os.path.join(d_,f"{i:05d}.png.tmp")
                    im.save(t_,format="PNG"); os.replace(t_,os.path.join(d_,f"{i:05d}.png"))
                if (s//C.GEN_BATCH)%25==0: print(f"    {s+len(ix)}/{len(miss)}")
            print(f"[H2] complete {tag} ({(time.time()-t0)/60:.0f} min)")
    finally:
        del pipe; gc.collect(); torch.cuda.empty_cache()
    print("[H2] PASS")
if stage("H2"): h2()
