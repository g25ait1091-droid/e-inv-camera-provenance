# %% F1b — ADAPTATION-STRENGTH GATE. Run between F1 and F2. ~4 min.
#     The comparison we want is MATCHED ADAPTATION, different parameterisation.
#     LoRA shifted generated images by |delta| 31-59 / 255 against base-model generations.
#     If full FT lands far below that it barely adapted; far above and it is degraded.
#     Either way you want to know BEFORE generating 3000 images.
from diffusers import SD3Transformer2DModel
N_GATE = 16
BASE_GENS = os.path.join(DRIVE,"inv_channel/E_INV_P0_v3","gens","base")
bank=[(C.CAPTION,C.GEN_SEED_BASE+j) for j in range(C.G_PER)]
if not os.path.isdir(BASE_GENS): HALT(f"base generations not found: {BASE_GENS}")

rows=[]
for tag in all_tags():
    tr=SD3Transformer2DModel.from_pretrained(os.path.join(cdir(tag),"transformer"),torch_dtype=DT)
    pipe=StableDiffusion3Pipeline.from_pretrained(C.MODEL,transformer=tr,torch_dtype=DT,
                                                  token=True).to(DEV)
    pipe.set_progress_bar_config(disable=True)
    d=[]
    try:
        for s in range(0,N_GATE,C.GEN_BATCH):
            ix=list(range(s,min(s+C.GEN_BATCH,N_GATE)))
            gs=[torch.Generator(device=DEV).manual_seed(bank[i][1]) for i in ix]
            with torch.inference_mode():
                imgs=pipe(prompt=[bank[i][0] for i in ix],num_inference_steps=C.GEN_STEPS,
                          guidance_scale=C.CFG_SCALE,height=MEAS,width=MEAS,generator=gs).images
            for i,im in zip(ix,imgs):
                bp=os.path.join(BASE_GENS,f"{i:05d}.png")
                if not os.path.exists(bp): continue
                a=np.asarray(im.convert("RGB"),np.float32)
                with Image.open(bp) as b_:
                    b_=b_.convert("RGB"); W_,H_=b_.size
                    b=np.asarray(b_.crop(((W_-MEAS)//2,(H_-MEAS)//2,
                                          (W_-MEAS)//2+MEAS,(H_-MEAS)//2+MEAS)),np.float32)
                d.append(float(np.abs(a-b).mean()))
    finally:
        del pipe,tr; gc.collect(); torch.cuda.empty_cache()
    rows.append((tag,float(np.mean(d)),float(np.std(d))))
    print(f"[F1b] {tag:14s} mean |delta| vs base = {np.mean(d):6.2f} / 255  (sd {np.std(d):.2f})")

m=np.array([r[1] for r in rows])
print("\n" + "="*72)
print(f"  full FT: {m.min():.1f} - {m.max():.1f} / 255      LoRA reference: 31 - 59 / 255")
if m.max() < 15:
    print("  VERDICT: BARELY ADAPTED. A null from these runs says nothing about full fine-tuning.")
    print("           Raise C.LR (try 1e-5) or C.STEPS, re-run F1, and re-check this gate.")
elif m.min() > 90:
    print("  VERDICT: LIKELY DEGRADED. Inspect images before proceeding; a bound from broken")
    print("           generations is worse than no bound.")
else:
    print("  VERDICT: COMPARABLE to LoRA adaptation. Proceed to F2.")
print("="*72)
ajson({"per_run":rows,"lora_reference":[31,59]},os.path.join(ROOT,"F1b_adaptation_gate.json"),indent=2)
