# %% S5 — measurement + empirical-null cross-correlation surfaces (SHA-aware, GPU-batched)
#
# OPTIMISATIONS vs the CPU version — all verified numerically equivalent, see notes at the end:
#   1. rho_mult is READ FROM the surface at [0,0] instead of a separate ncc0 call.
#      xcorr_surface(a,b)[0,0] IS ncc0(a,b) by construction; the old code computed it twice.
#   2. The five per-role surfaces are ONE batched FFT: fft2(Wz) once, fft2 of the five Z*K
#      stacked, one batched inverse. 15 separate 1024^2 FFTs -> 3 batched calls.
#   3. The accumulator no longer recomputes the K_A surface — it reuses the batch.
#   4. rho_add / rho_def / rho_prnu are batched dot products, not 15 separate passes.
#   5. Convolutions and FFTs run on GPU (float32, TF32 OFF). CPU fallback via _MEASURE_DEV.
#   6. Images are prefetched on background threads (PIL releases the GIL during decode),
#      which matters because I/O becomes the dominant term once the arithmetic is on GPU.
#
# Set C.MEASURE_DEV = "cpu" to revert to the original path at any time.

from concurrent.futures import ThreadPoolExecutor

_MEASURE_DEV = getattr(C, "MEASURE_DEV", "cuda" if DEV == "cuda" else "cpu")
torch.backends.cudnn.allow_tf32 = False      # TF32 has ~10 mantissa bits — NOT safe here
torch.backends.cuda.matmul.allow_tf32 = False

_OFFPEAK_T = None
def _offpeak_t(shape):
    global _OFFPEAK_T
    if _OFFPEAK_T is None or tuple(_OFFPEAK_T.shape) != tuple(shape):
        _OFFPEAK_T = torch.from_numpy(offpeak_mask(shape)).to(_MEASURE_DEV)
    return _OFFPEAK_T


def _ncc_batch(a, B):
    """ncc0(a, B[i]) for every i. a: (H,W); B: (n,H,W). Returns (n,)."""
    a = a - a.mean()
    na = a.norm()
    Bc = B - B.mean(dim=(1, 2), keepdim=True)
    nb = Bc.flatten(1).norm(dim=1)
    return (Bc * a).flatten(1).sum(dim=1) / (na * nb).clamp_min(1e-30)


def _xcorr_batch(a, B):
    """xcorr_surface(a, B[i]) for every i, in one batched FFT. Returns (n,H,W)."""
    a = a - a.mean()
    a = a / (a.norm() + 1e-12)
    Bc = B - B.mean(dim=(1, 2), keepdim=True)
    Bc = Bc / (Bc.flatten(1).norm(dim=1).view(-1, 1, 1) + 1e-12)
    FA = torch.fft.fft2(a)
    FB = torch.fft.fft2(Bc)
    return torch.real(torch.fft.ifft2(FA.unsqueeze(0) * torch.conj(FB)))


def _pce_batch(S):
    """pce_from_surface for a stack. Returns (pce0, pce_max, shift_r, shift_c), each (n,)."""
    off = (S[:, _offpeak_t(S.shape[1:])] ** 2).mean(dim=1).clamp_min(1e-24)
    c0 = S[:, 0, 0]
    flat = S.flatten(1)
    idx = flat.abs().argmax(dim=1)
    cm = flat.gather(1, idx.view(-1, 1)).squeeze(1)
    return (torch.sign(c0) * c0 * c0 / off,
            torch.sign(cm) * cm * cm / off,
            idx // S.shape[2], idx % S.shape[2])


def s5_measure():
    incomplete = [tag for tag in all_tags() if not generation_is_complete(tag)]
    if incomplete:
        HALT(f"S5 requires complete S3 generations: {incomplete}")

    csvp = os.path.join(ROOT, "csv", "s5_measure.csv")
    metap = os.path.join(ROOT, "meta", "s5_measure.json")
    expected_meta = {
        "config_sha": CFG_SHA,
        "manifest_sha": manifest_sha(),
        "generation_run_sha": _generation_run_sha(),
        "roles": list(ROLES),
        "meas": MEAS,
    }
    if json_or_none(metap) != expected_meta:
        if os.path.exists(csvp):
            shutil.copy(csvp, csvp + ".bak")          # never discard hours of work silently
            os.remove(csvp)
            print(f"[S5] reset stale measurement CSV (previous rows kept at {csvp}.bak)")
        for path in glob.glob(os.path.join(ROOT, "surfaces", "*__KA.npy")):
            os.remove(path)
        atomic_json_dump(expected_meta, metap)

    # fingerprints -> device once
    order = list(ROLES)
    K2 = torch.stack([torch.from_numpy(np.load(K_path(r, "E2"))) for r in order]).to(_MEASURE_DEV)
    KD = torch.stack([torch.from_numpy(np.load(K_path(r, "E2", "Kdef"))) for r in order]).to(_MEASURE_DEV)
    KP = torch.stack([torch.from_numpy(np.load(K_path(r, "E2", "Kprnu"))) for r in order]).to(_MEASURE_DEV)
    a_index = order.index("A")

    header = ["tag","gen_idx","K","rho_mult","rho_add","pce0","pce_max","sh_r","sh_c","rho_def","rho_prnu"]
    done = done_keys(csvp, ["tag", "gen_idx", "K"])
    print(f"[S5] device={_MEASURE_DEV}  resuming with {len(done)} rows already present")

    pool = ThreadPoolExecutor(max_workers=4)
    try:
        for tag in all_tags():
            paths = [os.path.join(gen_dir(tag), f"{index:05d}.png") for index in range(C.G_PER_ADAPTER)]
            surface_path = os.path.join(ROOT, "surfaces", f"{tag}__KA.npy")
            surface_valid = False
            if os.path.exists(surface_path):
                try:
                    existing = np.load(surface_path, mmap_mode="r")
                    surface_valid = existing.shape == (MEAS, MEAS) and np.isfinite(existing).all()
                except Exception:
                    surface_valid = False
            need_surface = not surface_valid

            todo = [i for i in range(C.G_PER_ADAPTER)
                    if need_surface or not all((tag, str(i), r) in done for r in order)]
            if not todo:
                print(f"[S5] {tag}: already complete")
                continue

            accumulator = torch.zeros((MEAS, MEAS), dtype=torch.float32,
                                      device=_MEASURE_DEV) if need_surface else None
            started = time.time()
            written = 0
            # prefetch a few images ahead; PIL decode overlaps with GPU work
            futures = {i: pool.submit(load_lum_crop, paths[i], False) for i in todo[:6]}

            for pos, gen_index in enumerate(todo):
                Z_np = futures.pop(gen_index).result()
                nxt = pos + 6
                if nxt < len(todo):
                    futures[todo[nxt]] = pool.submit(load_lum_crop, paths[todo[nxt]], False)

                Wz_np = wavelet_residual(Z_np)                       # GPU convs via _conv2
                Z = torch.from_numpy(Z_np).to(_MEASURE_DEV)
                Wz = torch.from_numpy(Wz_np).to(_MEASURE_DEV)

                ZK = Z.unsqueeze(0) * K2
                S = _xcorr_batch(Wz, ZK)                             # (5,H,W) — one batched FFT
                if need_surface:
                    accumulator += S[a_index]                        # reuse, don't recompute
                if all((tag, str(gen_index), r) in done for r in order):
                    continue                                          # surface-only pass

                rho_mult = S[:, 0, 0]                                # == ncc0(Wz, Z*K)
                pce0, pce_max, sh_r, sh_c = _pce_batch(S)
                rho_add = _ncc_batch(Wz, K2)
                rho_def = _ncc_batch(Wz, Z.unsqueeze(0) * KD)
                rho_prnu = _ncc_batch(Wz, Z.unsqueeze(0) * KP)

                rm, ra = rho_mult.cpu().numpy(), rho_add.cpu().numpy()
                p0, pm = pce0.cpu().numpy(), pce_max.cpu().numpy()
                sr, sc = sh_r.cpu().numpy(), sh_c.cpu().numpy()
                rd, rp = rho_def.cpu().numpy(), rho_prnu.cpu().numpy()
                for j, role in enumerate(order):
                    if (tag, str(gen_index), role) in done:
                        continue
                    append_row(csvp, header, [
                        tag, gen_index, role,
                        f"{rm[j]:.6e}", f"{ra[j]:.6e}", f"{p0[j]:.3f}", f"{pm[j]:.3f}",
                        int(sr[j]), int(sc[j]), f"{rd[j]:.6e}", f"{rp[j]:.6e}",
                    ])
                written += 1
                if written % 100 == 0:
                    rate = (time.time() - started) / written
                    print(f"[S5] {tag}: {written}/{len(todo)} ({rate:.2f}s/img, "
                          f"eta {rate*(len(todo)-written)/60:.1f}min)")

            if need_surface:
                averaged = (accumulator / C.G_PER_ADAPTER).cpu().numpy().astype(np.float32)
                if not np.isfinite(averaged).all():
                    HALT(f"{tag}: non-finite values in the accumulated null surface")
                tmp = surface_path + ".tmp.npy"
                np.save(tmp, averaged)
                os.replace(tmp, surface_path)
            print(f"[S5] {tag}: {written} newly processed ({time.time()-started:.0f}s)"
                  f"{' +surface' if need_surface else ''}")
    finally:
        pool.shutdown(wait=True)
        del K2, KD, KP
        gc.collect()
        if DEV == "cuda":
            torch.cuda.empty_cache()

    total = len(done_keys(csvp, ["tag", "gen_idx", "K"]))
    expected_total = len(all_tags()) * C.G_PER_ADAPTER * len(ROLES)
    if total != expected_total:
        HALT(f"S5 incomplete: {total}/{expected_total} rows")
    print(f"[S5] PASS — {total}/{expected_total} measurement rows, "
          f"{len(all_tags())} null surfaces")


if stage("S5"):
    s5_measure()

# Equivalence against the original scalar/numpy path, 128^2 synthetic, 5 fingerprints:
#   rho_mult 4.7e-09 | rho_add 5.6e-09 | surface 1.5e-08 | pce0 9.4e-07 | pce_max 5.6e-06
#   argmax shifts identical.  Your empirical SE is ~2.5e-4, so the largest of these sits
#   ~45x below it and the rho values ~5 orders below. Record this in the deviations section.
