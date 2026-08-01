# %% H3 — measurement against the same 5-fingerprint bank
def h3():
    K=torch.stack([torch.from_numpy(K_amp(r)) for r in BANK]).to(_MD)
    p=os.path.join(ROOT,"csv","h3_measure.csv"); done=done_keys(p,["tag","gen_idx","K"])
    ex=ThreadPoolExecutor(max_workers=4)
    try:
        for tag in all_tags():
            paths=[os.path.join(gdir(tag),f"{i:05d}.png") for i in range(C.G_PER)]
            todo=[i for i in range(C.G_PER) if not all((tag,str(i),r) in done for r in BANK)]
            if not todo: print(f"[H3] {tag}: complete"); continue
            fut={i:ex.submit(load_lum,paths[i],False) for i in todo[:6]}; t0=time.time()
            for pos,i in enumerate(todo):
                Z_np=fut.pop(i).result(); nx=pos+6
                if nx<len(todo): fut[todo[nx]]=ex.submit(load_lum,paths[todo[nx]],False)
                W=torch.from_numpy(wres(Z_np)).to(_MD); Z=torch.from_numpy(Z_np).to(_MD)
                v=nccb(W,Z.unsqueeze(0)*K).cpu().numpy()
                for j,r in enumerate(BANK):
                    if (tag,str(i),r) in done: continue
                    append_row(p,["tag","gen_idx","K","rho"],[tag,i,r,f"{v[j]:.6e}"])
            print(f"[H3] {tag}: {len(todo)} measured ({time.time()-t0:.0f}s)")
    finally:
        ex.shutdown(wait=True); gc.collect()
        if DEV=="cuda": torch.cuda.empty_cache()
    n=len(done_keys(p,["tag","gen_idx","K"])); exp=len(all_tags())*C.G_PER*len(BANK)
    if n!=exp: HALT(f"H3 incomplete: {n}/{exp}")
    print(f"[H3] PASS — {n}/{exp} rows")
if stage("H3"): h3()
