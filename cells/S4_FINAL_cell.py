# %% S4 — copy ensemble (DINOv2 full/quadrants, pHash, residual peakedness)
#     v2: (a) skip rebuilding a reference bank when every tag in its group is already
#         audited — saves ~10-15 min on a resume; (b) back the CSV up before any reset.
def _open_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _generation_run_sha():
    metas = {tag: json_or_none(_generation_meta_path(tag)) for tag in all_tags()}
    return sha_obj(metas)


def s4_copy_ensemble():
    import imagehash

    incomplete = [tag for tag in all_tags() if not generation_is_complete(tag)]
    if incomplete:
        HALT(f"S4 requires complete S3 generations: {incomplete}")

    csvp = os.path.join(ROOT, "csv", "s4_copy.csv")
    metap = os.path.join(ROOT, "meta", "s4_copy.json")
    expected_meta = {
        "config_sha": CFG_SHA,
        "manifest_sha": manifest_sha(),
        "generation_run_sha": _generation_run_sha(),
    }
    if json_or_none(metap) != expected_meta:
        if os.path.exists(csvp):
            shutil.copy(csvp, csvp + ".bak")          # never discard hours of work silently
            os.remove(csvp)
            print(f"[S4] reset stale copy-audit CSV (previous rows kept at {csvp}.bak)")
        else:
            print("[S4] no prior copy-audit CSV; starting fresh")
        atomic_json_dump(expected_meta, metap)

    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).to(DEV).eval()
    from torchvision import transforms as T2
    pre = T2.Compose([
        T2.Resize(256), T2.CenterCrop(224), T2.ToTensor(),
        T2.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    @torch.no_grad()
    def enc_imgs(images):
        outputs = []
        for start in range(0, len(images), 16):
            batch = torch.stack([pre(image) for image in images[start:start+16]]).to(DEV)
            outputs.append(tF.normalize(dino(batch), dim=-1).cpu())
        return torch.cat(outputs)

    def enc_paths(paths):
        outputs = []
        for start in range(0, len(paths), 16):
            outputs.append(enc_imgs([_open_rgb(path) for path in paths[start:start+16]]))
        return torch.cat(outputs)

    def quadrants(image):
        width, height = image.size
        return [
            image.crop((0,0,width//2,height//2)),
            image.crop((width//2,0,width,height//2)),
            image.crop((0,height//2,width//2,height)),
            image.crop((width//2,height//2,width,height)),
        ]

    M = read_manifest()
    header = ["tag","gen_idx","dino_sim","crop_sim","phash_min","resid_z","copy_flag"]
    done = done_keys(csvp, ["tag", "gen_idx"])

    # Group tags by the exact training reference so each residual bank is released promptly.
    groups = {}
    for tag in all_tags():
        _, role, variant, _, _ = arm_of(tag)
        key = (role or "A", variant or "raw")
        groups.setdefault(key, []).append(tag)

    for (role, variant), tags in groups.items():
        # RESUME GUARD: building a reference bank costs 50 wavelet residuals at MEAS^2 plus
        # DINOv2 + pHash over ~200 images. Don't pay it for a group that is already finished.
        if all((tag, str(index)) in done
               for tag in tags for index in range(C.G_PER_ADAPTER)):
            print(f"[S4] skip reference {role}_{variant} — all {len(tags)} tag(s) complete")
            continue

        train_paths = sorted(glob.glob(os.path.join(train_dir(role, variant), "*.png")))
        held_paths = M[(role, "E2")]
        if len(train_paths) != C.N_T:
            HALT(f"S4 reference {role}_{variant}: expected {C.N_T} training PNGs")

        hashes = []
        for path in train_paths + held_paths:
            image = _open_rgb(path)
            hashes.append(imagehash.phash(image))
        residuals = np.stack([
            wavelet_residual(load_lum_crop(path, use_cache=False)).astype(np.float16)
            for path in train_paths
        ])
        reference_embeddings = enc_paths(train_paths + held_paths)
        print(f"[S4] reference {role}_{variant}: {len(train_paths)} train + {len(held_paths)} held-out")

        for tag in tags:
            paths = [os.path.join(gen_dir(tag), f"{index:05d}.png") for index in range(C.G_PER_ADAPTER)]
            todo = [(index, path) for index, path in enumerate(paths) if (tag, str(index)) not in done]
            if not todo:
                print(f"[S4] {tag}: already complete")
                continue
            for start in range(0, len(todo), 32):
                chunk = todo[start:start+32]
                images = [_open_rgb(path) for _, path in chunk]
                dino_sim = (enc_imgs(images) @ reference_embeddings.T).max(dim=1).values.numpy()
                crop_sim = (
                    enc_imgs([q for image in images for q in quadrants(image)]) @ reference_embeddings.T
                ).max(dim=1).values.numpy().reshape(len(chunk), 4).max(axis=1)

                for local_index, (gen_index, path) in enumerate(chunk):
                    phash_min = min(int(imagehash.phash(images[local_index]) - ref_hash) for ref_hash in hashes)
                    W = wavelet_residual(load_lum_crop(path, use_cache=False))
                    correlations = np.array([ncc0(W, residual.astype(np.float32)) for residual in residuals])
                    median = float(np.median(correlations))
                    mad = float(np.median(np.abs(correlations - median))) / 0.6745 + 1e-12
                    resid_z = float((correlations.max() - median) / mad)
                    flag = int(
                        dino_sim[local_index] >= C.COPY_DINO
                        or crop_sim[local_index] >= C.COPY_CROP
                        or phash_min <= C.COPY_PHASH
                        or resid_z >= C.COPY_RESID_Z
                    )
                    append_row(csvp, header, [
                        tag, gen_index, f"{dino_sim[local_index]:.4f}", f"{crop_sim[local_index]:.4f}",
                        phash_min, f"{resid_z:.2f}", flag,
                    ])
            print(f"[S4] {tag}: {len(todo)} newly audited")

        del residuals, reference_embeddings, hashes
        gc.collect()
        if DEV == "cuda":
            torch.cuda.empty_cache()

    del dino
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()

    total = len(done_keys(csvp, ["tag", "gen_idx"]))
    expected_total = len(all_tags()) * C.G_PER_ADAPTER
    if total != expected_total:
        HALT(f"S4 incomplete: {total}/{expected_total} rows")
    print(f"[S4] PASS — {total}/{expected_total} generations audited")


if stage("S4"):
    s4_copy_ensemble()
