# %% S2 — SD3.5 LoRA training (final: dtype-keyed validation, dead-adapter detection,
#         cached prompt embeddings, auto grad-checkpointing, non-finite loss halt)
import random, gc, shutil

# grad checkpointing is a memory/compute trade only — it does NOT change the result, so it
# is deliberately NOT part of the adapter validation key (turning it off must not invalidate
# an already-trained adapter).
def _use_grad_ckpt():
    if getattr(C, "GRAD_CKPT", None) is not None: return bool(C.GRAD_CKPT)
    free = torch.cuda.get_device_properties(0).total_memory / 2**30
    return free < 24.0          # A100-40G / L40S / H100 -> off; 16 GB cards -> on


def adapter_dir(tag): return os.path.join(ROOT, "adapters", tag)


def _adapter_expected(tag):
    _, role, variant, seed, capacity = arm_of(tag)
    return {
        "tag": tag, "role": role, "variant": variant, "seed": seed,
        "rank":  C.LORA_RANK if capacity == "r16" else C.CEIL_RANK,
        "steps": C.STEPS     if capacity == "r16" else C.CEIL_STEPS,
        "meas": MEAS, "clean_alpha": float(C.CLEAN_ALPHA),
        "config_sha": CFG_SHA, "manifest_sha": manifest_sha(),
        "model": C.HF_MODEL,
        "dtype": str(TRAIN_DTYPE),      # fp16 vs bf16 must invalidate a cached adapter
    }


def adapter_is_valid(tag, verbose=False):
    if tag == "base": return True
    out = adapter_dir(tag)
    meta = json_or_none(os.path.join(out, "train_meta.json"))
    exp = _adapter_expected(tag)
    ok = (os.path.exists(os.path.join(out, "pytorch_lora_weights.safetensors"))
          and meta is not None
          and all(meta.get(k) == v for k, v in exp.items())
          and bool(meta.get("effective", True)))     # dead adapters never count as valid
    if verbose and not ok: print(f"[S2] invalid/incomplete adapter: {tag}")
    return ok


def _archive_stale_dir(path):
    if not os.path.exists(path): return
    stamp = time.strftime("%Y%m%d_%H%M%S"); target = f"{path}__stale_{stamp}"; n = 1
    while os.path.exists(target): target = f"{path}__stale_{stamp}_{n}"; n += 1
    shutil.move(path, target); print("[S2] archived stale output:", target)


# ---- prompt embeddings: computed once, then the 11 GB of text encoders are never loaded again
def _prompt_embed_path():
    key = hashlib.sha256(f"{C.HF_MODEL}|{C.CAPTION}|{TRAIN_DTYPE}".encode()).hexdigest()[:16]
    return os.path.join(ROOT, "adapters", f"_prompt_embeds_{key}.pt")


def _get_prompt_embeds():
    p = _prompt_embed_path()
    if os.path.exists(p):
        d = torch.load(p, map_location=DEV)
        return d["prompt_embeds"].to(DEV), d["pooled"].to(DEV)
    from diffusers import StableDiffusion3Pipeline
    print("[S2] encoding the caption once (text encoders loaded this time only)")
    pipe = StableDiffusion3Pipeline.from_pretrained(C.HF_MODEL, torch_dtype=TRAIN_DTYPE, token=True).to(DEV)
    with torch.no_grad():
        pe, _, pp, _ = pipe.encode_prompt(prompt=C.CAPTION, prompt_2=C.CAPTION, prompt_3=C.CAPTION,
                                          device=DEV, num_images_per_prompt=1,
                                          do_classifier_free_guidance=False)
    torch.save({"prompt_embeds": pe.cpu(), "pooled": pp.cpu(), "caption": C.CAPTION}, p)
    del pipe; gc.collect(); torch.cuda.empty_cache()
    return pe.to(DEV), pp.to(DEV)


def _load_training_pipe():
    """Load VAE + transformer + scheduler only. Falls back to the full pipeline if this
    diffusers version refuses a text-encoder-free construction."""
    from diffusers import StableDiffusion3Pipeline
    try:
        return StableDiffusion3Pipeline.from_pretrained(
            C.HF_MODEL, torch_dtype=TRAIN_DTYPE, token=True,
            text_encoder=None, text_encoder_2=None, text_encoder_3=None,
            tokenizer=None, tokenizer_2=None, tokenizer_3=None).to(DEV)
    except Exception as e:
        print(f"[S2] slim load unavailable ({type(e).__name__}); loading full pipeline")
        pipe = StableDiffusion3Pipeline.from_pretrained(C.HF_MODEL, torch_dtype=TRAIN_DTYPE, token=True).to(DEV)
        pipe.text_encoder = pipe.text_encoder_2 = pipe.text_encoder_3 = None
        gc.collect(); torch.cuda.empty_cache()
        return pipe


def train_arm(tag, prompt_embeds=None, pooled_embeds=None):
    _, role, variant, seed, capacity = arm_of(tag)
    if role is None: return
    if adapter_is_valid(tag): print(f"[S2] skip valid adapter {tag}"); return
    if DEV != "cuda": HALT("S2 requires a CUDA GPU")
    if prompt_embeds is None: prompt_embeds, pooled_embeds = _get_prompt_embeds()

    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    out = adapter_dir(tag)
    if os.path.exists(out): _archive_stale_dir(out)

    rank  = C.LORA_RANK if capacity == "r16" else C.CEIL_RANK
    tgts  = list(C.LORA_TARGETS if capacity == "r16" else C.CEIL_TARGETS)
    steps = C.STEPS if capacity == "r16" else C.CEIL_STEPS
    files = sorted(glob.glob(os.path.join(train_dir(role, variant), "*.png")))
    if len(files) != C.N_T:
        HALT(f"{tag}: {len(files)} training PNGs, expected {C.N_T}. Run S1c.")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); torch.cuda.manual_seed_all(seed)
    ckpt = _use_grad_ckpt()
    print("\n" + "="*70)
    print(f"[S2] {tag}: {len(files)} imgs, rank={rank}, steps={steps}, dtype={TRAIN_DTYPE}, "
          f"grad_ckpt={ckpt}")
    print("="*70)

    pipe = vae = transformer = optimizer = image_cache = None
    try:
        pipe = _load_training_pipe()
        noise_scheduler = pipe.scheduler
        vae, transformer = pipe.vae, pipe.transformer
        vae.requires_grad_(False); transformer.requires_grad_(False)
        transformer.add_adapter(LoraConfig(r=rank, lora_alpha=rank,
                                           init_lora_weights="gaussian", target_modules=tgts))
        if ckpt: transformer.enable_gradient_checkpointing()
        transformer.train()

        params = [p for p in transformer.parameters() if p.requires_grad]
        print(f"[S2] trainable LoRA parameters: {sum(p.numel() for p in params):,}")
        optimizer = torch.optim.AdamW(params, lr=C.LR, weight_decay=1e-4)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=math.ceil(steps / C.GRAD_ACC))
        optimizer.zero_grad(set_to_none=True)

        image_cache = torch.stack([
            torch.from_numpy((np.asarray(Image.open(p).convert("RGB"), np.float32).copy()/127.5 - 1.0))
                 .permute(2, 0, 1) for p in files])

        sf = float(vae.config.scaling_factor); sh = float(vae.config.shift_factor or 0.0)
        ts_all = noise_scheduler.timesteps.to(DEV)
        sg_all = noise_scheduler.sigmas.to(DEV)
        n_ts   = int(noise_scheduler.config.num_train_timesteps)
        if len(sg_all) != n_ts or len(ts_all) != n_ts:
            HALT(f"scheduler arrays len(sigmas)={len(sg_all)} len(timesteps)={len(ts_all)} "
                 f"but this loop indexes 0..{n_ts-1}. Pin diffusers.")

        cpu_gen = torch.Generator(device="cpu").manual_seed(seed)
        losses = []; t_start = time.time()
        for step in range(steps):
            idx = torch.randint(0, image_cache.shape[0], (C.BATCH,), generator=cpu_gen)
            px = image_cache[idx].to(DEV, dtype=TRAIN_DTYPE, non_blocking=True)
            with torch.no_grad():
                lat = vae.encode(px).latent_dist.sample()
                lat = ((lat - sh) * sf).to(dtype=TRAIN_DTYPE)
            noise = torch.randn_like(lat)
            # logit-normal timestep density, mapped through the model's own FlowMatch arrays
            u  = torch.sigmoid(torch.randn(lat.shape[0], device=DEV, dtype=torch.float32))
            ii = (u * n_ts).long().clamp_(0, n_ts - 1)
            sig = sg_all[ii].to(dtype=lat.dtype).view(-1, 1, 1, 1)
            noisy = ((1.0 - sig) * lat + sig * noise).to(dtype=TRAIN_DTYPE)
            pred = transformer(hidden_states=noisy, timestep=ts_all[ii],
                               encoder_hidden_states=prompt_embeds.repeat(lat.shape[0],1,1).to(TRAIN_DTYPE),
                               pooled_projections=pooled_embeds.repeat(lat.shape[0],1).to(TRAIN_DTYPE),
                               return_dict=False)[0]
            loss = tF.mse_loss(pred.float(), (noise - lat).float()) / C.GRAD_ACC
            loss.backward()
            if (step + 1) % C.GRAD_ACC == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step(); lr_sched.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()) * C.GRAD_ACC)
            if not np.isfinite(losses[-1]):
                HALT(f"{tag}: non-finite loss at step {step+1} (dtype={TRAIN_DTYPE}). "
                     f"Discard this adapter; do not continue.")
            if (step + 1) % 200 == 0:
                el = time.time() - t_start
                print(f"[S2] {tag}: {step+1}/{steps} loss={np.mean(losses[-200:]):.5f} "
                      f"GPU={torch.cuda.memory_allocated()/2**30:.1f}GB "
                      f"eta={(el/(step+1))*(steps-step-1)/60:.1f}min")

        sd = get_peft_model_state_dict(transformer)
        # DEAD-ADAPTER DETECTION: peft initialises lora_B at zero, so a training run that did
        # nothing leaves it at zero. The loss curve cannot reveal this (it is dominated by
        # timestep variance); an all-zero B would sail through S3/S5 and read as a C0 null.
        bnorm = float(sum(float(v.float().norm()) for k, v in sd.items() if "lora_B" in k))
        anorm = float(sum(float(v.float().norm()) for k, v in sd.items() if "lora_A" in k))
        finite = all(bool(torch.isfinite(v).all()) for v in sd.values())
        effective = finite and bnorm > 1e-6
        print(f"[S2] ||lora_B||={bnorm:.4f}  ||lora_A||={anorm:.4f}  finite={finite}")
        if not effective:
            HALT(f"{tag}: adapter is DEAD (||lora_B||={bnorm:.3e}, finite={finite}). "
                 f"Training produced no usable update.")

        from diffusers import StableDiffusion3Pipeline
        os.makedirs(out, exist_ok=True)
        StableDiffusion3Pipeline.save_lora_weights(save_directory=out, transformer_lora_layers=sd)
        meta = _adapter_expected(tag)
        meta.update({"loss_tail": float(np.mean(losses[-100:])),
                     "loss_head": float(np.mean(losses[:100])),
                     "lora_B_norm": bnorm, "lora_A_norm": anorm, "effective": True,
                     "grad_ckpt": ckpt, "minutes": (time.time()-t_start)/60,
                     "gpu": torch.cuda.get_device_name(0)})
        atomic_json_dump(meta, os.path.join(out, "train_meta.json"), indent=2)
        if not adapter_is_valid(tag): HALT(f"adapter validation failed right after saving {tag}")
        print(f"[S2] saved {tag}: tail loss={meta['loss_tail']:.6f} "
              f"({meta['minutes']:.1f} min)")

    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(f"CUDA OOM training {tag}. Set C.GRAD_CKPT=True and retry.") from exc
    finally:
        optimizer = image_cache = transformer = vae = pipe = None
        gc.collect(); torch.cuda.empty_cache()


if stage("S2"):
    arms = [a[0] for a in all_arms() if a[1] is not None]
    pending = [t for t in arms if not adapter_is_valid(t)]
    print(f"[S2] {len(pending)} of {len(arms)} adapters pending: {pending}")
    if pending:
        pe, pp = _get_prompt_embeds()
        for i, tag in enumerate(arms, 1):
            print(f"[S2] adapter {i}/{len(arms)}: {tag}")
            train_arm(tag, pe, pp)
    incomplete = [t for t in arms if not adapter_is_valid(t)]
    if incomplete: HALT(f"S2 incomplete: {incomplete}")
    print(f"[S2] PASS — all {len(arms)} adapters valid")
