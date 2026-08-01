# %% F2 — generation from each fine-tuned transformer (same paired seed bank)
def f2():
    if DEV!="cuda": HALT("F2 needs a GPU")
    from diffusers import SD3Transformer2DModel
    bank=[(C.CAPTION,C.GEN_SEED_BASE+j) for j in range(C.G_PER)]
    for tag in all_tags():
        d_=gdir(tag); os.makedirs(d_,exist_ok=True)
        for t_ in glob.glob(os.path.join(d_,"*.tmp")): os.remove(t_)
        miss=[i for i in range(C.G_PER)
              if not os.path.exists(os.path.join(d_,f"{i:05d}.png"))]
        if not miss: print(f"[F2] skip complete {tag}"); continue
        tr=SD3Transformer2DModel.from_pretrained(os.path.join(cdir(tag),"transformer"),
                                                 torch_dtype=DT)
        pipe=StableDiffusion3Pipeline.from_pretrained(C.MODEL,transformer=tr,torch_dtype=DT,
                                                      token=True).to(DEV)
        pipe.set_progress_bar_config(disable=True)
        print(f"[F2] {tag}: {len(miss)} missing"); t0=time.time()
        try:
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
        finally:
            del pipe,tr; gc.collect(); torch.cuda.empty_cache()
        print(f"[F2] complete {tag} ({(time.time()-t0)/60:.0f} min)")
        # DISK: uncomment to delete the checkpoint once its generations exist
        # shutil.rmtree(os.path.join(cdir(tag),"transformer"), ignore_errors=True)
    print("[F2] PASS")
if stage("F2"): f2()
