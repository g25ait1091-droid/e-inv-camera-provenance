# %% F1 — six full fine-tuning runs (3 seeds x 2 arms)
def ft_valid(tag):
    p=os.path.join(cdir(tag),"train_meta.json")
    if not os.path.isdir(os.path.join(cdir(tag),"transformer")): return False
    m=json.load(open(p)) if os.path.exists(p) else None
    return m is not None and m.get("config_sha")==CFG_SHA and m.get("effective",False)
if stage("F1"):
    import shutil as _sh
    free=_sh.disk_usage(ROOT).free/2**30
    print(f"[F1] free space at ROOT: {free:.0f} GB (need ~4 GB per run, {4*len(all_arms())} GB total)")
    if free < 4*len(all_arms())+10:
        print("[F1] WARNING: tight. Generate and delete each checkpoint as you go (see F2 note).")
    pe,pp=cached_prompt()
    for i,(tag,role,seed) in enumerate(all_arms(),1):
        if ft_valid(tag): print(f"[F1] {i}/{len(all_arms())} skip {tag}"); continue
        print(f"\n[F1] {i}/{len(all_arms())}: {tag}")
        train_full(tag,role,seed,C.STEPS,pe,pp,save=True)
    bad=[t for t in all_tags() if not ft_valid(t)]
    if bad: HALT(f"F1 incomplete: {bad}")
    print(f"\n[F1] PASS — {len(all_tags())} runs")
    for t in all_tags():
        m=json.load(open(os.path.join(cdir(t),"train_meta.json")))
        print(f"   {t:14s} open {m['open_loss']:.5f} -> tail {m['tail_loss']:.5f}  "
              f"drift {m['drift_probe']:.3e}")
    print("   ^ if tail_loss or drift vary wildly across runs, the runs are NOT comparable.")
