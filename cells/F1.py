# %% F1 — six full fine-tuning runs (3 seeds x 2 arms). Uses the fp32-master train_full.
def ft_valid(tag):
    p=os.path.join(cdir(tag),"train_meta.json")
    if not os.path.isdir(os.path.join(cdir(tag),"transformer")): return False
    m=json.load(open(p)) if os.path.exists(p) else None
    return (m is not None and m.get("config_sha")==CFG_SHA and m.get("effective",False)
            and m.get("precision")=="fp32_master_bf16_compute")

import shutil as _sh
free=_sh.disk_usage(ROOT).free/2**30
print(f"[F1] free at ROOT: {free:.0f} GB | need ~4.5 GB/run x {len(all_arms())} "
      f"= {4.5*len(all_arms()):.0f} GB")
if free < 4.5*len(all_arms())+10:
    print("[F1] TIGHT — uncomment the rmtree in F2 so each checkpoint is deleted once its "
          "generations exist.")
pe,pp=cached_prompt()
for i,(tag,role,seed) in enumerate(all_arms(),1):
    if ft_valid(tag): print(f"[F1] {i}/{len(all_arms())} skip valid {tag}"); continue
    print(f"\n[F1] {i}/{len(all_arms())}: {tag}")
    train_full(tag,role,seed,C.STEPS,pe,pp,save=True)
bad=[t for t in all_tags() if not ft_valid(t)]
if bad: HALT(f"F1 incomplete: {bad}")
print(f"\n[F1] PASS — {len(all_tags())} runs\n")
print(f"{'run':14s} {'open':>9s} {'tail':>9s} {'rel drift':>11s} {'min':>6s}")
_r=[]
for t in all_tags():
    m=json.load(open(os.path.join(cdir(t),"train_meta.json"))); _r.append(m["drift_relative"])
    print(f"{t:14s} {m['open_loss']:9.5f} {m['tail_loss']:9.5f} "
          f"{m['drift_relative']:11.3e} {m['minutes']:6.1f}")
print(f"\n  relative drift across runs: mean {np.mean(_r):.3e} sd {np.std(_r,ddof=1):.3e} "
      f"({100*np.std(_r,ddof=1)/np.mean(_r):.0f}% CV)")
print("  ^ if the CV is large the runs are NOT comparable and the adapter-level test is invalid.")
