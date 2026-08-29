# Adversarial review, written against myself

**2 August 2026.** Four external audits have run against this manuscript. Each found things
I had not. This is an attempt to do the exercise properly, which means it is written under
one rule I did not apply the first three times:

> **A finding only counts if I cannot fix it in the same sitting.**

My earlier "adversarial review" produced four findings and I closed all four immediately.
That is not adversarial. It is looking for objections I already had answers to. What follows
is restricted to things that are either unfixable without new experiments, or that I do not
know how to answer.

---

## Why the external reviewers found what I did not

Worth recording, because it predicts where the next miss will be.

**I read what I meant, not what I wrote.** The Table II contradiction — "each simultaneous
at 99 %" two lines above "BCa falls to 98.2 %" — I introduced it and read past it four
times across separate audit passes.

**I checked inside categories I was already thinking in.** Type 3 fonts: I verified the
figures rendered, had no collisions, and fit the column measure. Opening the PDF's font
table never occurred to me, because "production compliance" was not a category in my head.
Same shape of error on the sign-flip floor: I computed the exact *p*-values and never asked
what the smallest attainable one was. The reviewer asked, and the answer — $2^{-6} = 0.0156$
— invalidated the registered test.

**I reconstructed instead of reading.** Twice. The D1 cache stamp, and then E-FLOATCAL,
where I rebuilt the calibration from memory while `E_POST_LOW_cell.py` sat in the
repository. That produced a confident, internally consistent, entirely wrong result — a
2.03× slope discrepancy that would have entered the paper as a finding about quantization.
The error was in the device, the pool and the estimator, none of which I would have got
wrong by opening the file.

**And the structural one:** I optimised for objections I could answer in the same turn.

---

## Findings I cannot close

### A1. Six clusters is a design limit, not a presentation problem

Everything downstream inherits it. BCa is marginal at this count; the cluster distribution
cannot be estimated so the coverage simulation has to test assumed shapes; one adapter moves
the limit by 72 %; and the registered test is structurally non-rejectable.

The coverage simulation answers "does the construction cover under these assumptions." It
cannot answer "are these the right assumptions," and with six points nothing can.

**Only more adapters closes this.** k = 12 makes BCa valid, halves single-adapter leverage,
makes the registered test capable of rejecting, and lets the cluster distribution be tested
rather than assumed. Twelve new adapters, about eleven GPU-hours. Everything short of that
is presentation.

### A2. The calibration and the bound live at different inferential levels

The calibration injects an exactly aligned template into already-generated images and
analyses 2500 of them. The bound generalises over independently trained adapters, in which a
learned signal need not preserve phase, scale or position.

So "the detector is sensitive enough" is established for a template that is a perfect
spatial match to the measurement fingerprint, and *assumed* for whatever a trained adapter
would emit. The manuscript now says this. It does not fix it.

**Closing it requires injecting known amplitudes across all adapter clusters and checking
that the cluster-level analysis recovers them.** That is a new experiment, not a rewording.

### A3. There is no positive control at the level of the claim

Related to A2 but distinct. Nowhere does the study demonstrate that *if* device transfer
existed at some amplitude, this design would find it. The amplification arm comes closest,
but it injects into training images and its effective amplitude saturates near 4×, so it
cannot probe the regime the bound describes.

A null result with a calibrated detector and no design-level positive control is weaker than
it looks. I do not know how to build that control without a generator that provably does
transfer a fingerprint — which does not exist, because if it did the paper's question would
already be answered.

### A4. The shifted-template artifact is unexplained, and I stopped looking

The manuscript localises it experimentally — the template ranks first in arms that never
received it, including base-model generations at rank 7 of 36 — and then argues it cannot
affect the primary result because uniform elevation is a main effect that the paired
statistic cancels.

That argument is correct. It is also a reason to stop investigating, and I took it.

The honest position is that a measurement pipeline contains an effect whose mechanism is
unknown, and the defence is structural rather than mechanistic. If the mechanism turned out
to interact with arm identity in some configuration not tested, the defence fails. I have no
evidence that it does, and no evidence that it does not.

### A5. Effective amplitude cannot exceed ~4×, so a whole regime is untested

Because the injected field is an estimate, its correlation with the measurement estimate
caps the achievable contrast at roughly 4× natural regardless of nominal amplitude. The
paper reports this as a contribution — a ceiling on injection-based verification using
estimated fingerprints — which it is.

It is also a hole. The amplification arm is silent above 4× effective, and no experiment in
the study probes it. Injecting a *true* fingerprint rather than an estimate would lift the
ceiling, but no true fingerprint exists to inject.

### A6. Scene independence is asserted at a threshold nobody calibrated

The cross-split audit reports a worst similarity of 0.789 against a halting threshold of
0.95. The threshold was fixed before measurement, which protects against post-hoc
adjustment, but it was never calibrated against known same-scene and different-scene pairs.

The Daxing work made this concrete in a way the Dresden work did not: on that dataset, raw
thumbnail cosine saturates at 1.000 between splits because the scenes include sky and blank
walls, and perceptual hashing gives distance 0 for the same reason. Three appearance metrics
failed there before an EXIF-timestamp test worked.

Dresden has more varied scenes, so the audit is probably fine. **Probably** is doing work in
that sentence, and the supplement now reports the full distribution so a reader can judge.

### A7. The bound is one detector's bound

Every number is conditional on zero-lag normalised correlation against a wavelet-denoised
residual, at a fixed center crop, with a specific fingerprint estimator. The second
extractor agrees within 3 %, and the low/mid band is a second representation — but both are
variations on the same detection philosophy.

A learned detector, a phase-invariant statistic, or a matched filter tuned to what a
diffusion decoder actually emits could all behave differently. The manuscript scopes this.
Scoping is not the same as testing.

---

## Findings I can close but have not

**B1. The repository claim.** §VIII asserts the per-row records are in the repository; they
are in a linked Drive archive. DATA.md now makes the split explicit, but the manuscript
sentence should say "repository and data archive." Four words.

**B2. No qualitative figure.** For a text-to-image personalization study the paper shows no
generated images. A reader cannot see what regime the residual analysis operates in — whether
the generations are photographic, degenerate, or something in between. One figure with
training crops, base-model generations, LoRA generations and full-fine-tuning generations at
matched seeds would settle it. I did not make it because the figures are drawn from archived
numbers and the images live in Drive.

**B3. Fingerprint variance decomposition is under-specified.** The device/noise split of
0.355 / 0.638 appears without a formula, without stated assumptions, and without an interval,
and the components do not sum to one.

---

## What I would tell a reviewer, if asked what worries me most

Not the statistics. The statistics are now either verified by simulation or explicitly
scoped, and the numbers reproduce from raw data 36 times out of 36.

**A3 worries me most.** The paper demonstrates that its detector responds to a template it
was handed, and then reports that it found nothing when the template had to be learned. The
step between those two is the entire claim, and nothing in the study tests it directly. The
additive decomposition, the cross-system replication, the five-device design and the full
fine-tuning arm all constrain the *nuisance* structure. None of them establishes that a real
transferred fingerprint would have been visible.

That is the honest shape of the result: a well-controlled null with a calibrated instrument
and no proof of end-to-end sensitivity. It is worth publishing. It is not worth
overstating, and the current draft, after four rounds of correction, is close to the line
where it stops overstating.
