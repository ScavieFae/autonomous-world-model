# JEPA Scaling Notes — Batch Size × Dataset Size × Encoder Capacity

Planning framework for the JEPA lineage (e030+). Written during e030b's first epoch, before any scaling experiments have run — captures the reasoning so future experiments don't re-derive it and so we have a shared language for why we picked a specific knob combination in e030d, e030e, etc.

**TL;DR**: batch size and dataset size are entangled for JEPA in a way they aren't for Mamba2. The interaction is mediated by SIGReg's statistical power (scales with batch-to-dataset ratio), encoder capacity (likely caps somewhere 2K-20K games), and wall-clock feasibility (data-loader-bound at small batches). Three concrete regimes worth running, one to avoid.

## Why JEPA is different from Mamba2 on scaling

Mamba2 trains a supervised per-field loss — 16 weighted MSE/cross-entropy heads, each predicting a specific game state field. The gradient signal is per-sample and per-field. Batch size affects wall clock and statistical efficiency, but not the character of the learning signal.

JEPA trains two losses:
1. **`pred_loss`**: MSE between predicted next embedding and actual next embedding. Per-sample, local, like Mamba2.
2. **`sigreg_loss`**: Epps-Pulley characteristic function test on the distribution of embeddings in the batch. **Distribution-level, sample-size-dependent**.

`sigreg_loss` is the only thing preventing encoder collapse. It's computed by projecting the batch embeddings onto M=1024 random unit-norm directions and testing whether the resulting 1D marginals match N(0,1). The test statistic's variance scales as ~1/N where N is the batch size. **At small batches, the anti-collapse gradient is noisy; the encoder can collapse in the noise and SIGReg won't catch it reliably.**

This creates a coupling Mamba2 doesn't have:
- Mamba2: bigger batch = faster wall clock, better gradient averaging, that's it
- JEPA: bigger batch = faster wall clock, better gradient averaging, **AND stronger anti-collapse regularization**

It also creates a data-scale coupling:
- **Mamba2's regularization doesn't care about dataset size.** Cross-entropy loss on fields is a per-sample, local signal.
- **JEPA's SIGReg effectively measures the encoder's output distribution.** When the underlying dataset gets bigger, the true distribution has more modes and more variance. A fixed-batch SIGReg sample becomes a smaller fraction of the true distribution, and the regularizer's gradient signal degrades.

Concretely: **SIGReg's effective strength scales with `batch_size / dataset_size`**, not with batch_size alone.

## Pure batch-size tradeoffs (fixed dataset)

Against e030b's setup (2K FD+top-5 games, 15.6M training examples, bs=1024):

| Target | VRAM peak (est) | Batches/epoch | Epoch time (est) | Cost @ 10 epochs | Linear-scaled LR | Grad steps @ 10 epochs |
|---|---|---|---|---|---|---|
| bs=1024 (e030b) | 2.54 GB measured | 15K | 14 min | $4.80 | 4e-4 | 150K |
| bs=2048 | ~4-5 GB | 7.6K | ~10 min | $3.50 | 8e-4 | 76K |
| bs=4096 | ~8 GB | 3.8K | ~7-8 min | $2.50 | 1.6e-3 ⚠️ | 38K |
| bs=8192 | ~16 GB | 1.9K | ~5 min | $1.80 | 3.2e-3 ⚠️⚠️ | 19K |

### What gets better

- **SIGReg statistical power.** This is the JEPA-specific one and it's load-bearing. 2× batch → half the SIGReg gradient variance → stronger anti-collapse signal. LeWM picked bs=128 because their environments are tiny (~10K frames); at our scale bs=1024 is probably already marginal for SIGReg.
- **Wall clock.** e030a at bs=128 was 99% data-loader-bound. e030b at bs=1024 is still ~6% VRAM utilization → still data-loader-bound, just less. bs=2048-4096 cuts data loader invocations proportionally.
- **BatchNorm stats.** Both JEPA projectors end in `BatchNorm1d`. At bs=1024 × T=4 = 4096 samples per BN update we're already fine. More is free.

### What gets worse

- **LR scaling rule breaks down.** Linear scaling (lr ∝ batch) holds empirically up to ~batch 4K for Transformer-family training, then gets fuzzy. At bs=4096 the linear rule gives lr=1.6e-3 — **3.2× b002's proven 5e-4 LR on this data**, uncomfortably high. At bs=4096 cap lr at **1e-3** (sub-linear) and accept the drift from clean e030b comparison.
- **Extended warmup needed.** First-step instability is worse at large batches. bs=2048 wants 5-8% warmup, bs=4096 wants 8-10%.
- **Generalization gap** (Keskar et al., Goyal et al.). Starts biting at batch >4-8K for most vision/Transformer models. At 2048 clear, at 4096 boundary, beyond risky without LARS/LAMB-style tricks.
- **Fewer gradient steps at fixed epochs.** bs=4096 × 10 epochs = 38K steps — only 25% of LeWM's 150K. Needs more epochs to compensate OR acceptance that we're training less.

### JEPA-specific recommendation

| Target | When to use | Caveats |
|---|---|---|
| **bs=1024** | Direct comparison to e030b baseline | The current pinned default |
| **bs=2048** | Free throughput, stronger SIGReg, use anywhere not pinned to e030b | LR 8e-4 linear scale, warmup 8% |
| **bs=4096** | Larger datasets (see below) or explicit throughput sweep | LR 1e-3 (sub-linear), warmup 10%, drift from e030b comparison |
| **bs=8192+** | Large dataset regime (20K+ games) | Needs encoder capacity bump first, LR gets sketchy |

## Scaling with more data — four regimes

Two axes: **dataset size** (2K → 7.7K → 20K → 100K games) and **batch size** (1024 → 2048 → 4096 → 8192). Fix one, vary the other, and you get four distinct behaviors. Three are useful; one is wasteful.

### Regime A: scale batch proportional to data (matched SIGReg ratio)

Keep `batch_size / dataset_size` roughly constant as data grows. Preserves SIGReg's effective sample ratio, keeps wall clock near constant, isolates the "more data" variable cleanly.

| Dataset | batch | epochs | Grad steps | Wall clock | Cost |
|---|---|---|---|---|---|
| 2K (e030b) | 1024 | 10 | 150K | 2.3h | $4.80 |
| 7.7K | 4096 | 10 | 150K | ~2.5h | $5.25 |
| 20K | 8192 | 15 | 380K | ~4h | $8.40 |
| 100K | 8192 | 30 | 3.8M | ~20h | $42 |

**Use this when**: you want a clean apples-to-apples "does more data help" experiment. e030d should be this regime.

**Caveat**: at bs=8192 you hit the LR scaling ceiling and may need sub-linear scaling + longer warmup. At 100K + 3.8M steps you're in "real training run" territory, not a quick experiment.

### Regime B: fixed batch, let more data mean fewer effective passes

Keep bs=1024. More data → each example seen fewer times, but SIGReg weakens.

| Dataset | batch | epochs | Grad steps | Wall clock | Cost |
|---|---|---|---|---|---|
| 2K | 1024 | 10 | 150K | 2.3h | $4.80 |
| 7.7K | 1024 | 3 | 180K | 2.8h | $5.80 |
| 20K | 1024 | 1 | 200K | 3h | $6.30 |

**SIGReg ratio at 7.7K**: `1024 / 60M = 0.0017%`, vs e030b's `1024 / 15.6M = 0.0066%`. That's 4× weaker — enough to matter for anti-collapse.

**Use this when**: you want to test "does fewer passes + more variety help?" as a specific question. The weaker SIGReg is itself a variable you're testing.

**Caveat**: conflates "more data" with "weaker regularization." Don't use for clean scaling experiments.

### Regime C: fixed batch AND fixed epochs — naive scaling

Keep both bs=1024 and 10 epochs. Just feed more data.

| Dataset | batch | epochs | Grad steps | Wall clock | Cost |
|---|---|---|---|---|---|
| 2K | 1024 | 10 | 150K | 2.3h | $4.80 |
| 7.7K | 1024 | 10 | 600K | 9h | $19 |
| 20K | 1024 | 10 | 1.5M | 23h | $48 |

**Modal 24h timeout hits around 20K games in this regime.** Also over-trains: 1.5M gradient steps on 200M training examples means each example gets ~7.5 passes — fine but wasteful, and burns 4-10× compute to answer the same question Regime A would answer cheaper.

**Don't use this regime.** It's the "I forgot to think about scaling" regime. Flagged for avoidance.

### Regime D: fixed wall clock, scale batch + data together

Maximize the "more data per unit wall clock" axis. Requires careful LR tuning at each scale.

| Dataset | batch | epochs | Grad steps | Wall clock | Cost |
|---|---|---|---|---|---|
| 2K | 1024 | 10 | 150K | 2.3h | $4.80 |
| 7.7K | 4096 | 15 | 220K | ~3h | $6.30 |
| 20K | 8192 | 25 | 610K | ~4.5h | $9.45 |
| 100K | 8192 | 50 | 6.3M | ~12h | $25 |

**Use this when**: you're pushing the data axis hard and want to know "how much data can I process in a bounded wall clock." Useful for the "is diversity the bottleneck?" question once e030d establishes a baseline.

**Caveat**: gradient step count grows with dataset, which means this regime over-trains slightly compared to Regime A. Not as clean an experiment but a more realistic one.

## Encoder capacity as a separate ceiling

All regimes above assume the 2-layer MLP encoder has enough capacity for the larger dataset. **Probably true at 7.7K, probably false at 20K+.**

Reasoning:
- Our MLP encoder: ~1.5M params, 288 → 512 → 512 → 2048 → 192 + BatchNorm
- LeWM's ViT-tiny encoder: ~5M params, 12 Transformer layers
- LeWM trained on environments with 10K-50K total frames
- We're training on 15.6M frames (2K games) with a smaller encoder
- At 7.7K games (60M frames) we're 4× that — encoder is probably tight but OK
- At 20K games (200M frames) we're 13× — encoder is likely underfit relative to data
- At 100K games (1B frames) we're 65× — encoder is definitely underfit

**Encoder capacity is probably the ceiling we hit before batch scaling.** e030e or later should have `encoder_hidden_dim` and `encoder depth` as explicit levers. For 20K+ scale, plan to bump to 3-4 trunk layers and/or 1024-2048 hidden_dim **in the same experiment** that scales the data. Don't expect 2-layer MLP to benefit from 20× more data.

Alternative framing: there's probably a "capacity-matched scaling" sweet spot where each 10× more data wants roughly 2× more encoder params. e030b → e030d is within the flat region; anything beyond probably needs a capacity bump.

## Broader data contract is orthogonal

"Bring in more chars and stages" is a separate axis from "bring in more games." It's about distribution width, not volume.

- **More characters**: more animation priors, more action distributions. Forces the encoder to learn abstract "what's happening" features rather than character-specific shortcuts. Good for JEPA (bad for supervised Mamba2, which benefits from narrow predictable distributions).
- **More stages**: more position distributions, more geometry, more edge-guarding patterns. Forces the latent to encode position geometrically.
- **LeWM's Two-Room failure case** is explicitly about low-diversity environments being hard for JEPA. Our FD + top-5 restriction might be our Two-Room.

This axis can be combined with any of regimes A-D. The blocker is **encoded-data prerequisite**: we need a `.pt` file on the Modal volume encoded with the broader scope, and right now `encoded-e012-fd-top5.pt` is what exists. Generating a broader-scope encoded file is its own work item (parse more raw replays, re-run encoding pipeline, push to volume). Scope separately from the modeling work.

## Concrete per-experiment recommendations

**e030c (v1-minimal encoding ablation)**: stay at bs=1024 like e030b. We're isolating the encoding-flags axis, don't move two things at once. Same LR, same warmup, same batch. Only the encoding flags change.

**e030d (7.7K data scaling)**: **Regime A with bs=4096**, lr=1e-3 (sub-linear), warmup 8%, 10 epochs. Matches e030b's gradient step count (150K) and SIGReg sample ratio (~0.0068%). Cleanest apples-to-apples "only data distribution changed" comparison. Cost ~$5.25.

**e030e+ (LR/WD sweep on e030b best)**: stay at bs=1024. Moving LR and batch at the same time is useless. Do the sweep at fixed batch, pick the winner, *then* consider a batch scan.

**e031+ (broader data contract)**: build the encoded file first (separate work). When it lands, run at bs=2048-4096 with matched SIGReg ratio depending on the dataset size. This is where diversity actually helps.

**20K+ dataset regime**: don't run until encoder capacity is addressed. Mint a new base build (b003-jepa?) with wider/deeper encoder trunk first. Probably the right moment for a "LeWM-scale training run" dedicated experiment.

**Never**: Regime C at any scale above 2K. It's the "I forgot about scaling" regime and it wastes compute.

## Prediction-shortcut collapse (observed in e030b)

**Added after e030b completed.** A failure mode specific to JEPA + our dataset that we didn't see in LeWM's environments and wasn't predicted by the pre-launch reasoning.

### What it is

The encoder progressively drops information about input frames as training proceeds, despite loss curves looking healthy. Specifically: fields that cause large frame-to-frame MSE spikes (percent on hits, action state on transitions, shield on breaks) get compressed out of the latent, while fields that change smoothly from frame to frame (position, velocity) get preserved.

SIGReg prevents **distribution-level collapse** (the embedding distribution stays isotropic Gaussian, swap test stays healthy, rank stays full). But it has no mechanism to prevent **information-level compression** — the encoder can drop whatever information it wants as long as what's left is still approximately Gaussian. And the loss actively rewards dropping information about discrete events, because those events are the ones that make the "predict a small delta" target expensive.

### Signature pattern in training metrics

From e030b, per-epoch:

| Metric | Direction | What it looked like |
|---|---|---|
| `pred_loss` | ↓ monotonically | 0.080 → 0.003 |
| `sigreg_loss` | ↓ monotonically | 2.07 → 0.69 |
| `val_pred_loss` | ↓ monotonically (tracks train) | 0.048 → 0.003 |
| `emergent/straightness` | ↑ monotonically | 0.51 → 0.78 |
| `swap/mean`, `swap/ditto` | stable | ~0.035-0.045, ~0.19-0.21 |
| **`probe/p0_x_r2`** | **↓ monotonically** | **0.904 → 0.305** |
| **`probe/p0_action_acc`** | **↓ monotonically** | **0.753 → 0.151** |

**Every loss-side signal and the straightness diagnostic said "healthy training."** Every probe signal said "encoder is degrading." Both were internally consistent trajectories. Only the probes were aligned with what we actually care about (decodable representation).

### How to detect it

- **Linear probes must be held-out and reported every epoch.** e030a's in-batch-holdout probe would have shown R²=1.000 and missed this entirely. Stride-sampled disjoint probe batches from `training/jepa_diagnostics.py::run_diagnostic_suite_holdout` are the minimum.
- **Run a nonlinear probe diagnostic on the final checkpoint** to confirm information is genuinely lost, not just linearly opaque. Script: `scripts/nonlinear_probe_diagnostic.py`. If nonlinear R² > linear R² by more than ~0.3 on any target, the linear probe is under-measuring. If they agree, the information really is gone.
- **Watch per-epoch probe direction, not just magnitude.** A single epoch of R²=0.7 can look "Promising" but if the trend is -0.1 per epoch, the next run with the same config will be "Not viable."

### What it isn't

- **Not distribution-level collapse**: SIGReg prevents that, and swap test confirms it didn't happen
- **Not identity collapse**: P0 vs P1 stayed distinguishable throughout
- **Not a probe methodology bug**: confirmed by the nonlinear probe test
- **Not training instability**: loss curves were textbook
- **Not overfitting**: `val_pred_loss` tracked `train_pred_loss`
- **Not LR-scaling fallout**: 4e-4 linear-scaled from LeWM's 5e-5 is in-range; no oscillation; warmup worked as designed

It's specifically and narrowly: **MSE in latent space + isotropic Gaussian regularization + extended training = encoder specialization on smooth physics at the cost of discrete events**.

### What to try against it

Full treatment in `docs/jepa-direction-notes/smooth-physics-vs-game-rules.md`. Quick list:

1. **Reduce SIGReg λ** (cheapest, ~$5 next run): 0.1 → 0.01 or 0.03. Less pressure toward isotropic Gaussian → less pressure to drop information. This is the hypothesis e030c should test, per "move slowly" — one lever at a time.
2. **Early stopping**: epoch 1 was Competitive; every later epoch was worse. Stop at epoch 1-2 if all the probe numbers look good.
3. **Checkpoint every epoch** (required to do early stopping well, see below).
4. **Add an auxiliary decoder loss** (medium cost): reconstruct game state fields from the latent, train jointly. Not pure JEPA anymore — flag as a divergence.
5. **Hybrid latent with explicit event dimension** (structural): extend latent to have continuous + discrete components trained with different losses. New experiment lineage.

## Checkpoint-saving strategy

**Problem surfaced by e030b**: `best.pt` is saved on lowest `val_total_loss`. In e030b, `val_total_loss` monotonically decreased across all 10 epochs. So `best.pt = latest.pt = final.pt = epoch 10`. But **epoch 1 was the best-probe checkpoint** (R² 0.904 vs epoch 10's R² 0.305). We do not have the epoch-1 checkpoint on disk. **The actually-good encoder is gone.**

### The fix (not yet implemented)

Add `--save-every-epoch` flag to `scripts/modal_train_jepa.py` and `training/jepa_trainer.py`. When set, save `epoch_1.pt`, `epoch_2.pt`, ..., `epoch_N.pt` alongside the usual `best.pt`/`latest.pt`/`final.pt`. Each checkpoint is ~155 MB for the current 13.5M-param model, so 10 epochs = ~1.5 GB on the Modal volume — cheap.

**Scope**: ~15 lines of code.

```python
# In JEPATrainer.train(), inside the per-epoch loop, after _save_checkpoint(...):
if self._save_every_epoch:
    self._save_checkpoint(f"epoch_{epoch+1}.pt", epoch, val_loss)
```

Plus a kwarg in `__init__` and a config field in the yaml.

### Why it matters beyond e030b

Any JEPA run is at risk of prediction-shortcut collapse. Without per-epoch checkpoints, "best.pt by val_loss" can be meaningfully different from "best.pt by probe R²" — and we have no recourse once training is done. Saving every epoch is cheap insurance.

**Corollary**: add a post-training "find best epoch by probe R²" step. Run the diagnostic suite against each `epoch_N.pt` and pick the highest probe R². This becomes our actual `best.pt` for the run.

**Not blocking e030c** — we can use the feature when it lands. Queued as a pre-e030c code change.

## What the e031a profile showed us (cross-architecture infrastructure data)

**Added after e031a completed**, 2026-04-12. The profile was on Mamba2, not JEPA, but the infrastructure findings apply to both lineages — anything that uses `training/trainer.py` or `training/jepa_trainer.py` with the shared data path.

### The breakdown (Mamba2, b002 config, batch 512, 2K games on A100)

| Phase | % of wall clock | ms/batch | Notes |
|---|---|---|---|
| data loading | **0.0%** | 0.02 | GPU-resident data completely eliminated it |
| forward pass | 16.1% | 64.87 | Per-TF-batch |
| backward pass | 27.5% | 110.44 | ~1.7× forward (normal ratio) |
| optimizer | 1.1% | 3.40 | Negligible |
| self-forcing | **55.3%** | 889.62 | Per-SF-batch, 5× a TF batch |

### Finding 1 — GPU-resident data works, and was not a JEPA-specific win

Moving `dataset.floats` and `dataset.ints` to GPU at startup eliminates the data-loading phase to **0.02 ms/batch**. This is on Mamba2 but the same 2 lines apply to the JEPA trainer. When we come back to JEPA, adding `gpu_resident: true` to the config is the first ~2× speedup we get without touching anything else.

The gpu_resident path required ~4 small fixes to downstream code that was assuming `dataset.floats` lived on CPU: the SF path in `training/trainer.py` (Mamba2 only, doesn't affect JEPA), and the rollout eval path in `scripts/eval_rollout.py` (Mamba2 only). JEPA's trainer uses a different data path (`JEPAFrameDataset.get_batch`) and a different diagnostic suite (`training/jepa_diagnostics.py`) — so for JEPA, the gpu_resident change should be a pure win with no downstream fixes needed.

### Finding 2 — 12 GB GPU memory cost limits in-memory scaling

Moving the 2K FD top-5 dataset to GPU consumed **12.02 GB** (17.4M frames × 138 floats × 4 bytes for the float tensor, plus 17.4M × 17 × 8 bytes for the int tensor). Took 7.5s. Model + activations peak at **20.21 GB**. A100-40GB has ~20 GB free after this, plenty for training.

**But the 7.7K dataset would be ~48 GB — that doesn't fit on A100-40GB.** For e031+ Mamba2 scaling and any JEPA e031+ data-scaling experiment, we have three options:

1. **Stay on 2K with gpu_resident** — what e031a/b/c are doing now. Fast but bounded.
2. **Accept the cost on 7.7K** — fall back to CPU-resident data (the `mmap=True` path) at 7.7K. Adds back the data loader overhead. Per e031a's profile this was 0.0% of cost WHEN data was already on GPU; with CPU tensors it was probably the dominant cost per the old e030a profile data (hypothesized ~50-80%). Back to the old speed regime.
3. **Larger GPU** — H100-80GB fits a 7.7K dataset at ~48 GB with headroom. Requires explicit sign-off per the budget gate but resolves the memory issue cleanly.

This is a real scaling wall we didn't predict in the original scaling notes. **Logged as a hard constraint.** Cross-reference: the "encoder capacity as a separate ceiling" section in this doc remains true, but there's now a second ceiling (data-on-GPU memory) that bites earlier.

### Finding 3 — Self-forcing is 5× the cost of a TF batch, not 3×

SF at 20% ratio with N=3 unroll was always advertised as "~60% overhead." The profile shows it's closer to **~100% overhead** (SF + TF = 2× what pure TF would cost). This is Mamba2-specific — JEPA's `JEPATrainer` doesn't have SF at all. But it matters for anyone reading the scaling notes and wondering "how much does self-forcing cost."

Per SF batch breakdown (Mamba2):
- N=3 autoregressive forward passes through the model
- Per pass: one forward + one frame reconstruction
- Frame reconstruction: CPU argmax on ~12 categorical heads (400-class action state, etc.)
- Per-step CPU↔GPU roundtrip (move preds to CPU, reconstruct, move next frame back to GPU)
- `batch_floats = torch.cat([batch_floats, next_float.unsqueeze(1)], dim=1)` — grows the context buffer each step

The CPU argmax is the suspected hot spot — the model runs on GPU, but argmax on categoricals is done on CPU because the reconstruction logic was written assuming CPU tensors. **If we wanted to keep SF and speed it up, moving the argmax + reconstruction to GPU would probably be the biggest win.** But that's ~50-100 lines of code to rewrite `reconstruct_frame` and isn't worth it until we've confirmed SF is still load-bearing (e031b).

### What it means for the JEPA direction specifically

1. **gpu_resident: true is free speedup** — port when we come back to JEPA. No downstream fixes expected because JEPA's data/eval paths are separate.
2. **The 12 GB memory wall applies to JEPA too** — a JEPA training run on 7.7K won't fit in `gpu_resident` mode on A100. Same three options as above.
3. **The profile validated the infrastructure plan** — the four-step speed optimization (GPU-resident, bigger batch, SSD chunked scan, torch.compile) holds conceptually. GPU-resident is the biggest per-step win. SSD chunked scan is already on in b002. Bigger batch and torch.compile are the next levers.
4. **JEPA doesn't have the SF overhead problem** (because it doesn't have SF), so the 55% wall-clock tax on Mamba2 doesn't apply. JEPA's speed bottleneck is the predictor's sequential forward + backward, which is where batch-scaling helps most.

## What to remember when planning

1. **Think in gradient steps, not epochs.** LeWM's "100 epochs" assumed ~1K batches/epoch. Ours varies by 3 orders of magnitude depending on dataset × batch. Target ~150-200K total steps unless you have a reason to go bigger.
2. **Match SIGReg sample ratio when scaling data.** `batch / dataset_size` should stay roughly constant. Doubling data means doubling batch for SIGReg to see the same fraction of the distribution.
3. **Encoder capacity is a separate ceiling.** Don't scale data more than ~10× without scaling encoder params. The MLP is already small.
4. **Wall clock is roughly constant in Regime A.** That's the magic — more data, same runtime, same cost. It's the regime to push.
5. **Linear LR scaling breaks at bs=4096+.** Cap at lr=1e-3 and extend warmup beyond that point.
6. **Never move two knobs at once in a first experiment.** Isolate the variable you're testing. Combined knob-moves come after you have a baseline.
7. **Broader data contract is diversity, game-count is volume.** Different questions. Can be combined but think of them separately.

## References

- e030a run card: `docs/run-cards/e030a-jepa-baseline.md` — the discarded LeWM-defaults baseline
- e030b run card: `docs/run-cards/e030b-jepa-rescale.md` — the rescale + probe fix (currently running)
- JEPA direction: `docs/jepa-direction.md` — architecture overview and future-levers table
- Data flow: `docs/jepa-data-flow.md` — how P0/P1 reach the latent
- LeWM paper summary: `research/sources/2603.19312-summary.md`
- LeJEPA / SIGReg paper summary: `research/sources/vjepa-ijepa-summary.md`
