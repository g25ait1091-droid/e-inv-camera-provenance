"""F5 (RESULTS.md Entry 25) — is the positive natural contrast of the v2-trained adapters a
property of the adapters or of the local generation environment?

Generates, with the local pipeline (same code path as the ladder, uniform caption, v1 paired seed
bank, 500 images), three arms that involve NO v2 training:
  local_base        no adapter
  local_A_raw_s0    v1's archived A_raw_s0_r16 adapter (data/adapters/A_raw_s0_r16)
  local_B_raw_s0    v1's archived B_raw_s0_r16 adapter (data/adapters/B_raw_s0_r16)
Outputs to out/t1/gens/<arm>; measured afterwards by  T1_ARMSET=f5 python t1_measure.py .
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, glob, time, torch
sys.path.insert(0, EINV.SRC)
import t1_ladder as L

V2 = EINV.V2
ARMS = [("local_base", None),
        ("local_A_raw_s0", os.path.join(V2, "data", "adapters", "A_raw_s0_r16")),
        ("local_B_raw_s0", os.path.join(V2, "data", "adapters", "B_raw_s0_r16"))]

def generate(tag, adapter):
    outd = L.gen_dir(tag); os.makedirs(outd, exist_ok=True)
    bank = [(L.CAPTION, L.GEN_SEED_BASE + j) for j in range(L.G_PER_ADAPTER)]
    have = len(glob.glob(os.path.join(outd, "*.png")))
    if have >= len(bank): L.log("skip", tag, f"({have}/{len(bank)})"); return
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(L.HF_MODEL, torch_dtype=torch.bfloat16).to(L.DEV)
    if adapter:
        assert os.path.exists(os.path.join(adapter, "pytorch_lora_weights.safetensors")), adapter
        pipe.load_lora_weights(adapter)
    pipe.set_progress_bar_config(disable=True)
    i = have; t0 = time.time()
    while i < len(bank):
        chunk = bank[i:i+L.GEN_BATCH]
        gs = [torch.Generator(device=L.DEV).manual_seed(s) for _, s in chunk]
        with torch.no_grad():
            imgs = pipe(prompt=[p for p, _ in chunk], num_inference_steps=L.GEN_STEPS, guidance_scale=L.CFG_SCALE,
                        height=L.MEAS, width=L.MEAS, generator=gs).images
        for j, im in enumerate(imgs): im.save(os.path.join(outd, f"{i+j:05d}.png"), format="PNG")
        i += len(chunk)
        if i % 100 < L.GEN_BATCH: L.log(f"{tag}: {i}/{len(bank)}  ({(time.time()-t0)/60:.1f} min)")
    del pipe; torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    for tag, ad in ARMS: generate(tag, ad)
    L.log("done f5 generate")
