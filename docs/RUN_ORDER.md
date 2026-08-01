# E-INV — FINAL RUN ORDER
**2026-08-01.** Everything else is complete. This is what is left.

---

## GPU track — E_FULLFT (~3.5 h total)

### Step 0 · one config edit, before anything else
In the **FT-CFG** cell change:

```python
    STEPS: int = 1000        ->        STEPS: int = 4000
```

**Why:** LoRA saw 2000 × batch 2 = **4000 image presentations / 1000 optimizer updates**. Full FT at
1000 × batch 1 sees 1000 presentations / 250 updates — a quarter of both. A null from an
under-exposed run proves nothing. With `BATCH=1, GRAD_ACC=4`, `STEPS=4000` matches LoRA on *both*
counts.

Cost at the measured 0.4 s/step: **~27 min/run, ~2.7 h for six.**

**Do this now.** Nothing is saved yet, so the `CFG_SHA` change is free. After F1 starts it
invalidates every completed run.

### Step 1 · re-run these three cells, in order
1. **FT-CFG** — picks up `STEPS=4000`, recomputes `CFG_SHA`
2. **forensic core** (`load_lum`, `wres`, `nccb`, `cached_prompt`)
3. **`F1FIX.py`** — redefines `train_full` with fp32 masters; its 30-step re-smoke costs 12 s

> **Do NOT re-run the original F0 cell.** It defines the *bf16* `train_full` and would silently undo
> the fix. F1FIX supersedes it.

### Step 2 · run in sequence
| | cell | wall | gate |
|---|---|---|---|
| 2a | **`F1.py`** | ~2.7 h | relative drift consistent across the six runs — a large CV means the runs are not comparable |
| 2b | **`F1b_GATE.py`** | ~4 min | **mean \|Δ\| vs base generations must land near LoRA's 31–59 / 255.** Below ~15 → barely adapted, raise LR and redo F1. Above ~90 → degraded, inspect before proceeding |
| 2c | `E_FULLFT_F2.py` | ~2 h | 3000 generations |
| 2d | `E_FULLFT_F3.py` | ~15 min | 6000 measurement rows |
| 2e | `E_FULLFT_F4.py` | ~5 min | **the verdict** — IUT plus the symmetric statistic |

**Before believing F4:** run the base study's S4 copy audit on these generations. LoRA gave 0.0%;
full FT on 50 images overfits by construction, and if the copy rate is high then memorisation is
doing the work and the device statistic is not interpretable. Also eyeball ~20 images per arm.
**If the generations are degraded, report the degradation and call the arm inconclusive.**

---

## CPU track — run in parallel, no GPU, no contention

**`E_PAPERPREP.ipynb`** — P1 reference verification (put a real address in `MAILTO`),
P2 amplitude reconciliation (already resolved — the intercept was the culprit), P3 frozen numbers.

These are the two genuinely blocking items for submission. Neither needs a runtime.

---

## When Daxing lands

**`E_DAXING.ipynb`**, stage **D0 only** at first. Read the device-code printout and fix `RX` /
`MODEL_OF` before running D1. D0 also halts on cross-split scene leakage, which is the failure mode
Daxing invites.

---

## Do not run
| | why |
|---|---|
| H1–H4 | H0 already delivered the result: effective amplitude saturates at ≈4× at any nominal amplitude |
| the better-injection variant | would push the ceiling to ~6.4× and re-test the same null; the ceiling *is* the contribution |
| Kodak low/mid | reliability 0.081 on Dresden; Kodak bodies would be noisier — more caveats than information |
| E-PROMPT, E2, E3 | generality already covered by two systems, five devices, six seeds, two representations |
| the original F0 cell, after F1FIX | it redefines `train_full` in bf16 and undoes the fix |

---

## After F4 and Daxing
Re-run **`E_INV_CONSOLIDATE.ipynb`** — it recomputes every number from the per-row CSVs and
self-audits against the cached JSONs. Then draft: **arXiv → TIFS**, with IH&MMSec 2027 as the
named fallback.
