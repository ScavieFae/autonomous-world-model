---
id: e030b
created: 2026-04-11
status: discarded
type: hyperparameter
base_build: null
built_on: [e030a]
source_paper: "2603.19312"
rollout_coherence: null
prior_best_rc: 4.798
---

# Run Card: e030b-jepa-rescale

> **Lineage note**: Originally planned e030b was "v1-minimal encoding ablation on 2K" — that experiment now bumps to **e030c**. This card fills the e030b slot because scaling-to-data-size blocks every downstream e030 experiment and has to come first.

## Goal

Rerun the e030a JEPA baseline with hyperparameters appropriate for our dataset size, not LeWM's. Two specific changes: **batch size 128 → 1024** (GPU was 1.25% utilized at LeWM's default) and **num_epochs 50 → 10** (LeWM's epoch count assumed ~1K gradient steps per epoch; ours was 122K). Also fix the linear probe methodology so the probe numbers are actually held-out.

Everything else stays at LeWM defaults.

This is **not** an encoding ablation, not a new architecture, not a different paradigm — it's e030a with the scale knobs turned to match our data regime. The "does JEPA work on structured game state" question is still the core ask; we just can't answer it when the run Modal-timeouts at epoch 18 of 50 and the probes are overfitting to their own holdout.

## Context — what e030a showed

The e030a run (started 2026-04-11 12:07 PDT, ran for ~4h, killed after epoch 3) surfaced four findings:

**What worked:**
- Pipeline ran end-to-end. Modal + JEPATrainer + diagnostic suite + wandb all intact.
- Losses descended as LeWM predicts. By epoch 2: `pred_loss=0.0076` (from ~1.9 smoke-test baseline), `sigreg_loss=0.67` (sharp early drop, plateau), `total_loss=0.075`. Textbook JEPA training curve.
- Swap test healthy: `swap/mean_cosine_sim=0.102` — encoder is identity-aware, no collapse warning.
- Temporal straightness high: `0.626` at epoch 2 — LeWM's emergent diagnostic is firing positive.

**What didn't work:**
- **~80 minutes per epoch.** Estimated total runtime: 50 × 80 min = ~67 hours. Modal's 24h function timeout would have killed the run at epoch ~18, before the curve matured.
- **VRAM peak: 0.50 GB on a 40 GB A100.** 1.25% utilization. The GPU sat idle 99% of the time waiting on the data loader.
- **Probe methodology artifact.** `probe/p0_x_r2 = 1.000, p1_x_r2 = 1.000, rel_x_r2 = 1.000` at epoch 2. Too perfect. Root cause: the 256-sample diagnostic batch is split 204/52 train/val internally by `linear_probe_regression`, a 193-param (192-dim + bias) linear fit to 204 samples is underdetermined, and the 52 "val" samples are temporally adjacent frames from the same games. The probe is interpolating, not generalizing. Result: probe R² is uninformative at this batch size; we can't use it for the go/no-go call.

## What changes

| Axis | e030a (LeWM defaults) | e030b (data-scale fit) | Why |
|------|-----------------------|------------------------|-----|
| `batch_size` | 128 | **1024** (8×) | LeWM's 128 was sized for a 1-GPU-hours budget on 10K-frame environments. Our 15.6M training examples need bigger batches to finish in reasonable wall clock. VRAM headroom: ~40× (0.5 → ~4 GB est). |
| `num_epochs` | 50 | **10** | LeWM's 100 assumed ~1K gradient steps/epoch. Ours is 122K at bs=128 (16K at bs=1024). 10 epochs at bs=1024 is 160K gradient steps total — more than LeWM's entire training budget of 100K at their scale. |
| `lr` | 5e-5 | **4e-4** (linear scale by batch ratio) | 8× batch → 8× LR is the textbook linear scaling rule. Lands us within 20% of b002's 5e-4 (which was tuned for this data). ScavieFae's review already marked LR as a probable lever. |
| `weight_decay` | 1e-3 | 1e-3 (unchanged) | Leaving WD at LeWM default for now. b002's 1e-5 is a separate lever — don't move two things at once. |
| `warmup_pct` | 0.05 | 0.05 (unchanged) | |
| `diagnostic_every` | 1 | 1 (unchanged) | |
| `diagnostic_batch_size` | 256 | **1024** | More samples → proper probe fit, ~4× less noisy ditto test, probe R² confidence interval shrinks ~2×. Costs <200ms/epoch extra. |
| Diagnostic batch sampling | first 256 val indices (one game) | **stride sample across all val games** | e030a's diagnostic batch was 256 consecutive frames from one val game, which is why `ditto=nan` and probe numbers were unreliable. Stride sampling at `stride = 900K // 1024 ≈ 880` hits ~all 200 val games. |
| Probe methodology | in-batch 80/20 split | **fit + eval on two disjoint stride-sampled batches** | See the Probe methodology fix section below. ~25 lines of code in `training/jepa_trainer.py` and `training/jepa_diagnostics.py`. |

**All other values unchanged from e030a**: embed_dim=192, history_size=3, encoder_hidden_dim=512, predictor 6/16/64/2048, SIGReg λ=0.1 / 1024 projections / 17 knots. AMP float16. Data contract unchanged.

## Divergences from LeWM (flagged per `models/jepa/CLAUDE.md` rule 2)

Three deliberate departures from the reference:

1. **`batch_size = 1024` (not 128)**. LeWM's default was tuned for their infrastructure (single GPU, ~15M examples is unimaginable at their scale). Our A100 sits at 1.25% VRAM utilization at 128. The inductive argument for changing: LeWM says "batch size" doesn't appear in their ablations as a sensitivity axis, so it's a tuning knob, not a load-bearing architectural choice.

2. **`num_epochs = 10` (not 100)**. LeWM's 100 was a wall-clock choice for their data scale. We'd need 100 × 80 min = 133 hours at e030a's rate and 10 × ~12 min = 2 hours at e030b's rate. The 10 here is chosen so total gradient step count (160K) is still comparable to LeWM's (100K) — we're not under-training, we're just matching LeWM's effective step budget with our data.

3. **`lr = 4e-4` (not 5e-5)**. Linear scaling with batch size, standard technique for AdamW on Transformer-family models at the 128 → 1024 range. Lands within 40% of b002's converged LR on this data (5e-4). A sqrt-scaling alternative (1.4e-4) is queued as a follow-up sensitivity check — see the Lineage plan. Warmup (5% of steps, ~8K batches) protects the early-training instability that motivates sqrt as a safer default.

**None of these touch the architecture, loss, or paradigm.** Predictor depth, heads, SIGReg, AdaLN, encoder trunk — all identical. This is a hyperparameter rescale, not a rewrite.

## Probe methodology fix

Essential code change to `training/jepa_trainer.py` (and possibly a thin wrapper in `training/jepa_diagnostics.py`).

### What was broken in e030a

`diagnostic_batch = val_dataset.get_batch(np.arange(256))` was pulled once at init. This has **two** problems, not one:

**Problem 1 — in-batch holdout overfitting**: `linear_probe_regression` split the 256-sample batch 80/20 internally. A 192-dim linear probe (+ bias = 193 params) fit to 204 samples is underdetermined — it can memorize the train split and interpolate to the 52 val samples. Result: R² ≈ 1.000 at epoch 2 on the e030a run, regardless of whether the encoder learned anything.

**Problem 2 — single-game sampling**: `JEPAFrameDataset.valid_indices` is built in game order. Each val game contributes ~4497 valid starting frames. `np.arange(256)` pulls the first 256 indices, which all come from **one val game** — a specific matchup between specific characters. All probe/swap/straightness numbers were measuring the encoder's representation of that single game. The e030a run's `ditto=nan` confirms this: the first val game happens not to be a ditto, so the sharpest identity-collapse signal never fired.

### The fix: stride sampling + held-out eval batch

Pull **two** diagnostic batches, each stride-sampled across all val games so every game contributes proportionally:

```python
# In JEPATrainer.__init__
n_avail = len(val_dataset)
stride = max(1, n_avail // diagnostic_batch_size)
# Fit batch: even stride
fit_idx = np.arange(0, n_avail, stride)[:diagnostic_batch_size]
# Eval batch: offset by half a stride (disjoint from fit, still covers all games)
eval_offset = stride // 2
eval_idx = np.arange(eval_offset, n_avail, stride)[:diagnostic_batch_size]
self.probe_fit_batch = tuple(t.to(device) for t in val_dataset.get_batch(fit_idx))
self.probe_eval_batch = tuple(t.to(device) for t in val_dataset.get_batch(eval_idx))
```

Then `run_linear_probes` takes the two batches, fits on the embeddings of the fit batch, evaluates on the embeddings of the eval batch. No internal split, no memorization path. The swap test and temporal straightness still run on a single batch (they don't need a train/eval split).

### Why this helps

- **Cross-game coverage**: at `stride = 900K // 1024 ≈ 880`, each batch hits ~200 distinct games (one sample every ~880 frames spans the whole val set). Both batches see the full distribution of characters and matchups.
- **Genuinely held-out R²**: fit and eval indices are interleaved but disjoint, from the same distribution. A probe that memorizes the fit batch can't just interpolate to an adjacent val sample — the closest eval neighbor is `stride/2 ≈ 440` frames away, in the same game or a different one.
- **Ditto bucket gets populated**: at ~27% ditto rate across top-5 chars, we expect ~270 ditto samples in a 1024-sample batch instead of 0 or all-of-them.
- **Comparable across epochs**: same indices every epoch, same ground truth, so epoch-over-epoch deltas are meaningful.

### Scope

- `training/jepa_trainer.py::__init__`: ~15 lines for the stride sampling + two-batch pull
- `training/jepa_trainer.py::_diagnostic_eval`: pass both batches to the suite, handle the different method signature
- `training/jepa_diagnostics.py`: either modify `run_linear_probes` to accept separate fit/eval embeddings, or wrap it. ~10 lines.
- Smoke test via `scripts/run_jepa_diagnostics.py --smoke` — the CLI uses a single synthetic batch, so the CLI path can keep its current in-batch holdout (unchanged) and the trainer path gets the new held-out method. Two paths diverge, clearly documented.

Total: ~25 lines, one file edit + one minor signature change in another.

### Known issues with the fix (still imperfect)

Flagging these so future experiments don't rediscover them the hard way:

1. **Fit and eval batches come from the same population**. Both are strided samples of `val_dataset`, which is ~200 games split from the full 2K dataset. A probe that overfits to val-game-distribution-specific features could still get high numbers without generalizing to an unseen set of games. A stronger test would hold out a second "test" set of games that neither training nor the probe ever sees. Not done here — would require splitting val into val-for-loss and val-for-probe, reducing training signal on val_loss. Tradeoff deemed not worth it for e030b; flag as a follow-up if probe numbers look too-good.

2. **Stride sampling is deterministic but dataset-size-dependent**. If someone swaps the encoded file (different number of val games), the absolute indices shift, and probe numbers from e030b are no longer epoch-comparable with a future run on a different dataset. Cross-experiment probe comparisons should require the same data fingerprint.

3. **`linear_probe_classification` at 1024 samples still uses 200 SGD steps**. For a 400-class action classifier on 192-dim input with ~820 training samples (80% of 1024), 200 steps may not be enough to converge. If action probe accuracy looks suspicious (stuck at random, or asymmetric in ways that don't match the regression probes), the first fix is bumping `steps=200` to `steps=500` in `linear_probe_classification`. Not done in e030b to keep the rescale scope minimal.

4. **Diagnostic batch size still doesn't cover the full val set**. At B=1024 and ~900K val indices, each batch is 0.11% of val; fit+eval combined is 0.22%. For noisy metrics (action accuracy on 400 classes, where ~2 samples per class on average) the sample size still bites. Can't go much higher without making the probe eval expensive; this is the knee of the curve.

5. **Ditto detection assumes first-frame character ID is stable**. `JEPAFrameDataset` uses one frame's categoricals, and the swap test compares `int_frames[..., 2]` to `int_frames[..., ipp+2]` per sample. If two players ever swap ports mid-match (doesn't happen in our data but a theoretical edge case), the ditto buckets would be noisy. Not a current concern.

None of these are blockers for e030b. They're the things to look for in the closeout if anything seems off.

## Data

Unchanged from e030a: `encoded-e012-fd-top5.pt`, 1,988 FD top-5 games, ~1.9M frames, loads into RAM. Same b002 data contract (`state_flags`, `hitstun`, `ctrl_threshold_features`, `state_age_as_embed`, `multi_position`).

## Lineage plan (updated)

- **e030a** — first JEPA attempt, LeWM defaults. **Partial** (3 epochs, then killed). Proved pipeline + confirmed scaling mismatch + found probe methodology bug. Useful null result.
- **e030b** (this card) — same architecture, scale knobs fit to our data. Actual "does JEPA work" baseline. **Linear LR scaling (4e-4)**.
- **e030b-sqrt** *(queued)* — identical to e030b except `lr = 1.4e-4` (sqrt scaling from LeWM's 5e-5 by √8). Sensitivity check on the LR scaling rule decision. Run only if e030b shows unstable early training (NaN or loss oscillation in the first ~500 batches), or as a cheap ablation after e030b if the main result is borderline. Cost: ~$3, ~1.5 hours.
- **e030c** — v1-minimal encoding ablation on 2K (was e030b in the original plan). Drops `state_flags`, `ctrl_threshold_features`, `multi_position` — tests whether the b002 data contract helps or hurts JEPA. Runs after e030b establishes a working baseline number.
- **e030d** — data scaling on 7.7K with whatever 2K regime proved best.
- **e030e+** — LR/WD sweep at new scale, history_size lever, two-player embedding structure, multi-step prediction (after unblocking `num_preds==1` assert).

**Pre-registered `e030-identity-fix`** (unchanged): if e030b's swap test fires `swap.ditto_cosine_sim > 0.9` or per-player probe R² < 0.3 on the **held-out** probe (not the broken in-batch probe from e030a), run the per-player shared-weight sub-encoder + cross-attention fusion before anything else.

## Model

Unchanged from e030a:
- Encoder: MLP (288→512→512→2048→192 + BatchNorm)
- Predictor: 6-layer Transformer, 16 heads, dim_head=64, AdaLN-zero
- SIGReg: λ=0.1, 1024 projections, 17 knots
- Total: ~13.5M params

## Training

- AdamW, **lr=4e-4** (linear-scaled from LeWM's 5e-5 by the 8× batch jump), wd=1e-3
- **10 epochs, batch 1024**
- 5% linear warmup + cosine decay
- Gradient clip 1.0
- AMP (float16)
- **GPU: A100** — same as e030a. No reason to escalate. Expect higher VRAM (~3-5 GB est) but still far from saturated — this is the step after the obvious fix, not the final tuning run.

## Launch command

```bash
modal run --detach scripts/modal_train_jepa.py \
    --config experiments/e030b-jepa-rescale.yaml \
    --encoded-file /encoded-e012-fd-top5.pt \
    --gpu A100 \
    --run-name e030b-jepa-rescale
```

**Pre-launch status (updated post-review):**
1. ✅ e030a stopped cleanly after epoch 3, closeout Results section written.
2. ✅ Probe methodology fix landed in `training/jepa_diagnostics.py` (new `*_holdout` functions) and `training/jepa_trainer.py` (stride-sampled fit/eval batches). Smoke-tested via `python -m scripts.run_jepa_diagnostics --smoke` and via end-to-end trainer smoke on synthetic data.
3. ⏳ `/experiment-launch` pre-flight not yet run against this card — required before hitting Modal.

**First real-data check** is epoch 1 of e030b on Modal. See "Epoch 1 watchlist" below for the three specific things to confirm before trusting any of the new numbers.

### Epoch 1 watchlist (what to verify in the first epoch's wandb log)

Both the hyperparameter rescale and the probe fix land untested on real data in epoch 1. Three independent health checks to run against the wandb log as soon as epoch 1 completes — each one falsifies a specific assumption:

1. **`pred_loss` is stable or descending, no NaN, no oscillation.** Falsifies: linear LR scaling is wrong at 8×. Warmup over 5% of 160K steps = ~8K batches will hide early instability for most of epoch 1, so the critical window is ~step 8000 onward where the LR reaches its 4e-4 peak. If `pred_loss` starts oscillating or spikes there, **kill the run and relaunch as e030b-sqrt with lr=1.4e-4** (queued in the Lineage plan — no debate, no investigation). If it's stable through epoch 1 we can trust the LR choice for the rest of the run.

2. **`swap/ditto_cosine_sim` is a finite number, not NaN.** Falsifies: the stride sampling didn't actually hit dittos. At ~27% ditto rate on top-5 chars, a 1024-sample fit batch should have ~275 dittos — the bucket should be populated on the very first diagnostic eval. If it's still NaN, something's wrong with either the stride math (unlikely given verification) or the ditto detection path in `swap_test` (more likely — the `int_frames[..., 2]` vs `int_frames[..., ipp + 2]` comparison assumes the encoding's int layout holds). Debug before continuing — a NaN ditto bucket at epoch 1 means we're still blind on the sharpest identity-collapse signal.

3. **`probe/p0_x_r2` is not exactly 1.000 (to three decimals).** Falsifies: the probe holdout path is still broken somehow. At epoch 1 on real data with a partially-trained encoder, held-out position R² should land somewhere in the 0.1–0.6 range depending on how well the representation has formed. If it's 1.000 again, the fit and eval batches are not actually disjoint or the ridge regularizer is insufficient — either way, **stop and diagnose before trusting probe numbers for anything else in this run.** This is the specific bug that killed e030a; confirming it's dead is the single most important epoch-1 observation.

All three are visible in the first epoch's wandb log, no need to wait for full training. If any of them fails, the decision tree is clean:
- #1 fails → kill, relaunch e030b-sqrt
- #2 or #3 fails → stop, diagnose, do not trust any probe numbers

If all three pass, the rest of the run is basically "observe the curve and report at closeout."

## Evaluation

Primary signals unchanged from e030a. **One signal is now actually reliable**: the linear probes, because they're evaluated on held-out games rather than in-batch samples.

**Primary signals (per epoch):**
- `pred_loss`, `sigreg_loss`, `total_loss` — training dynamics (e030a showed these work)
- **Linear probe R² + game-units MAE** on **held-out val games** for position, percent, shield. Now actually held-out.
- **Linear probe accuracy for `action_state`** — same story, now on held-out games.
- **Identity diagnostics**: swap test (mean / ditto / non-ditto cosine similarity), per-player probes, relational probes.
- **Temporal straightness** — reliable from e030a (no methodology bug); expect it to trend up.

**Rollout coherence still deferred for e030b** — same reason as e030a. Decoder pattern is an e030d+ design decision.

## Success Criteria

All numbers on held-out val games (the real held-out probe, not in-batch) at the **final** epoch:

- **Promising**: per-player x/y R² > 0.7 (both P0 and P1) on **held-out** samples, per-player x/y MAE < ~20 game units, percent R² > 0.5, rel_x R² > 0.5, action probe accuracy > ~5%, swap test mean < 0.6 and ditto < 0.7. Worth continuing the lineage.
- **Competitive**: per-player x/y R² > 0.9 on held-out samples, MAE < ~8 game units, action accuracy ≳ 20%, swap test ditto < 0.5, relational probes > 0.8. Matches or exceeds what Mamba2 implicitly encodes — serious contender.
- **Not viable**: per-player held-out R² < 0.3 on position or percent after the curve clearly plateaus. Paradigm doesn't fit structured data at our scale — close the JEPA line.
- **Identity-collapsed**: held-out `swap.ditto_cosine_sim > 0.9` OR per-player probe R² asymmetric by > 0.3. Trigger `e030-identity-fix` before any other experiment.
- **Too early to tell**: curves still descending at epoch 10. Expected to be rare at this step count, but possible.

## Key Risks

Inherited from e030a, plus two specific to the rescale:

1. **Identity collapse** (unchanged from e030a — structural weakness vs Mamba2, pre-registered fix).
2. **50ms context.** history_size=3 is still unchanged. The first lever for a follow-up.
3. **Encoder capacity.** 2-layer MLP vs LeWM's 12-layer ViT, unchanged.
4. **Data contract inherited from b002.** Still directly tested by e030c.
5. **Two-player dynamics** as concat. Still an open question.
6. **New: linear LR scaling might be wrong.** 8× LR via linear scaling is standard but not universal — some models prefer √8 = 2.8× (LR = 1.4e-4). If the first epoch shows pred_loss exploding or oscillating, kill and drop to 1.4e-4 without relaunching the whole flow. We're within LeWM-reasonable territory either way.
7. **New: fewer total epochs means less SIGReg plateau time.** LeWM reports sigreg settles within ~5 epochs regardless of epoch budget; our 10-epoch run gives 5+ plateau epochs. Should be fine, but watch that sigreg_loss is genuinely flat by epoch 4-5 before trusting pred_loss trends.

## Budget

- A100 × ~1-2 hours × $2.10/hr ≈ **$2-4**
- Well under $30/day cap
- e030a already spent ~$10 on the partial run, so e030 series total is ~$12-14 — still fine

## Results

**Status: discarded, highly informative.** All 10 epochs completed cleanly. $4.54 total. No crashes, no NaN, no OOM, no Modal timeout. The run executed exactly as designed — the design was insufficient.

### The trajectory (all 10 epochs from `manifest.json`)

| Ep | pred_loss | sigreg | total | val_pred | swap/mean | ditto | p0_x R² | p1_x R² | rel_x R² | p0_act | p1_act | straight | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 0.080 | 2.07 | 0.288 | 0.048 | 0.035 | 0.191 | **0.904** | **0.891** | **0.895** | **0.753** | **0.700** | 0.512 | 829s |
| 2 | 0.018 | 1.60 | 0.178 | 0.018 | 0.030 | 0.210 | 0.901 | 0.898 | 0.772 | 0.595 | 0.575 | 0.607 | 825s |
| 3 | 0.011 | 1.47 | 0.159 | 0.014 | 0.037 | 0.219 | 0.868 | 0.868 | 0.785 | 0.489 | 0.460 | 0.663 | 751s |
| 4 | 0.008 | 1.14 | 0.122 | 0.010 | 0.043 | 0.206 | 0.841 | 0.830 | 0.781 | 0.293 | 0.244 | 0.709 | 753s |
| 5 | 0.006 | 0.79 | 0.085 | 0.007 | 0.042 | 0.203 | 0.734 | 0.681 | 0.639 | 0.217 | 0.174 | 0.726 | 758s |
| 6 | 0.005 | 0.74 | 0.078 | 0.005 | 0.042 | 0.197 | 0.480 | 0.516 | 0.524 | 0.172 | 0.150 | 0.740 | 753s |
| 7 | 0.004 | 0.71 | 0.075 | 0.004 | 0.044 | 0.195 | 0.418 | 0.361 | 0.354 | 0.163 | 0.132 | 0.756 | 763s |
| 8 | 0.004 | 0.70 | 0.074 | 0.004 | 0.045 | 0.194 | 0.349 | 0.350 | 0.321 | 0.152 | 0.140 | 0.770 | 786s |
| 9 | 0.003 | 0.70 | 0.073 | 0.003 | 0.044 | 0.192 | 0.345 | 0.294 | 0.286 | 0.152 | 0.132 | 0.776 | 771s |
| **10** | 0.003 | 0.69 | 0.073 | 0.003 | 0.044 | 0.192 | **0.305** | **0.246** | **0.257** | **0.151** | **0.133** | 0.777 | 788s |

**Observations**:

- **Loss curves are textbook healthy.** `pred_loss` dropped 27× (0.080 → 0.003). `sigreg_loss` dropped 3× (2.07 → 0.69). `val_pred_loss` tracks training loss with no overfit gap. `total_loss` monotonic decrease.
- **Temporal straightness rose monotonically** 0.51 → 0.78. LeWM's emergent diagnostic was firing strongly throughout.
- **Swap test stayed stable.** `swap/mean ≈ 0.04`, `swap/ditto ≈ 0.19-0.21`. Both well below the 0.9 collapse gate. Identity preservation was healthy the whole run.
- **No identity collapse was ever triggered.** The pre-registered `e030-identity-fix` is not needed.
- **But the linear probes degraded monotonically.** p0_x R² went 0.904 → 0.305. p1_x went 0.891 → 0.246. rel_x went 0.895 → 0.257. Action accuracy went 0.75/0.70 → 0.15/0.13. **Every single epoch had worse probe numbers than the previous epoch.**

The striking shape: loss metrics and the straightness diagnostic all improved smoothly, while the linear probe metrics — which measure "can you recover game state from the latent" — degraded smoothly in the opposite direction. Two internally consistent training signals pointing in opposite directions.

### Nonlinear probe diagnostic (to rule out probe methodology)

Before concluding the information was genuinely lost, I ran a nonlinear probe against the epoch-10 encoder on the canonical FD test game (`2fd092d61ac812828132334904bc5870`, Fox vs Marth, 2000-frame window, 50/50 temporal fit/eval split).

Script: `scripts/nonlinear_probe_diagnostic.py`. Probe: 2-layer MLP (192 → 64 → 1, ReLU, Adam lr=1e-3, 500 steps).

```
target             linear R²    nonlin R²    lin MAE   nonlin MAE
------------------------------------------------------------------------
p0_percent           -0.4148        0.1565      51.648      47.244  pct
p0_x                  0.4866        0.4247      30.499      28.004  px
p0_y                 -0.7705        0.0240      42.661      27.642  px
p0_shield               —              —        46.056       4.998  pct  ZEROVAR
p1_percent         -364.5406      -30.4055      19.765       5.838  pct
p1_x                  0.8232        0.7820      16.685      14.104  px
p1_y                  0.0731        0.2859      11.939       8.761  px
p1_shield         -9694.2607     -392.4787      45.985       9.186  pct

                  linear acc   nonlin acc  note
------------------------------------------------------------------------
p0_action             0.0680        0.0860  (random = 0.0025)
p1_action             0.1140        0.1120
```

**The nonlinear probe does not dramatically outperform the linear probe.** Where linear is decent, nonlinear is decent (p1_x: 0.82 → 0.78). Where linear is bad, nonlinear is barely better (p0_y: -0.77 → 0.02, a minor lift). Nowhere does the MLP find information the linear probe missed by more than ~0.3 R².

If the encoder were still carrying game-state information through nonlinear feature combinations, a 64-hidden-unit MLP with 500 SGD steps would find it — 1000 training samples with 192-dim input is plenty of signal for a small MLP to fit whatever structure is there. The fact that it doesn't means **the information really is gone from the encoder.**

### The failure mode: "prediction-shortcut collapse"

What happened: the encoder learned to minimize `||encoder(frame_{t+1}) - encoder(frame_t) - small_delta||²` by progressively dropping information about the fields that cause large frame-to-frame MSE spikes (percent on hits, action_state on transitions, shield on breaks). The encoder kept the smoothly-varying physics fields (position, velocity) because they're the ones that make the "small delta" prediction easy, and dropped everything that looked like discrete events.

SIGReg prevents **distribution-level collapse** (swap test stable, dittos ≠ non-dittos, ranks full) but has no mechanism to prevent **information-level compression** as long as the embedding distribution stays isotropic Gaussian. The encoder can drop whatever information it wants to as long as what's left is still approximately Gaussian-distributed. And because the loss actively rewards dropping information about discrete events, it does.

The field-specific pattern in the nonlinear probe confirms this diagnosis:

| Field | Layer | Epoch 10 R² (linear / nonlinear) | Interpretation |
|---|---|---|---|
| `p1_x` | continuous physics | 0.82 / 0.78 | preserved |
| `p0_x` | continuous physics | 0.49 / 0.42 | half-preserved |
| `p0_y` | continuous physics | -0.77 / 0.02 | mostly lost (asymmetric with x — direction-specific specialization) |
| `p1_y` | continuous physics | 0.07 / 0.29 | mostly lost |
| `percent` | discrete events | strongly negative | severely degraded |
| `shield` | discrete events | extremely negative | severely degraded |
| `action` | discrete transitions | 6-11% (was 75-70% at ep1) | mostly lost |

**X coordinates survived. Y coordinates mostly didn't. Discrete-event fields (percent, shield, action) are essentially gone.** This isn't random information loss — it's systematic specialization on continuous physics at the cost of discrete game rules.

### This is the same failure mode as three other observations from this session

1. **e030a's probe R² = 1.000 artifact at epoch 2** (which we discarded as a methodology bug): the e030a encoder hadn't yet specialized, and the in-batch holdout let a 193-param linear probe memorize the training split. **That encoder was actually healthy** (see retroactive viz finding on the FD test game) — we just couldn't tell because the probe was broken.
2. **Percent going backwards in the e030a reconstruction viz** (which we analyzed earlier today): the encoder at epoch 3 wasn't fully specialized, so percent decoded with R² ~0.9 on the FD game. But the reconstructed percent still drifts smoothly toward the mean in rollouts because the JEPA predictor has no concept of "discrete step functions."
3. **Death→respawn never working across any of our models**: the catastrophic version of the same story. Discrete events are rare, loss is continuous, gradient descent finds smooth-only solutions.

All four are symptoms of the same underlying issue. Full analysis in `docs/jepa-direction-notes/smooth-physics-vs-game-rules.md`.

### What e030b actually proved

- **The paradigm is not dead.** Epoch 1's probe numbers (R² 0.904 on position, 75% on action) were Competitive-zone results. JEPA on structured game state CAN learn a usable representation — we watched it happen, for about 800 seconds.
- **The rescale (batch 1024, lr 4e-4, 10 epochs) was correct** as a wall-clock fix. The run completed comfortably under budget. Nothing about the scale-knob choices is wrong.
- **The problem is not hyperparameters; it's loss structure.** Pure MSE in latent space + SIGReg isotropic-Gaussian prior has a joint optimum at "smooth latent trajectories with dropped discrete-event information." Training longer makes this worse, not better. Early stopping at epoch 1 would have given us the Competitive result we were chasing.
- **Checkpoint-saving strategy is broken.** `best.pt` is saved on lowest `val_total_loss`, which monotonically decreased. So `best.pt`, `latest.pt`, and `final.pt` are all epoch 10 — the worst-probe checkpoint. **We do not have the epoch-1 checkpoint on disk.** The Competitive checkpoint is gone. See `docs/jepa-scaling-notes.md` for the checkpoint-saving strategy fix.

## Decision

**Discarded, highly informative.** This run produced more useful research signal than any experiment we've run to date, specifically because the failure mode is new and reproducible and we now understand it. Listing explicitly what we learned:

1. **Prediction-shortcut collapse is a thing in our setup**, specifically with MSE + SIGReg + our dataset. It didn't fire in LeWM's environments because they're continuous control (Layer A only). It fires on Melee because Melee has both continuous physics (Layer A) and discrete game rules (Layer B).
2. **The rescale numbers are correct** for the question they asked (can we fit the training regime into reasonable wall clock). They're not the reason the run failed.
3. **Linear probe R² is a trustworthy signal**. The nonlinear probe confirms the linear numbers. Our diagnostic suite is working.
4. **Per-epoch monotonic degradation** is a signature pattern to watch for in future runs. Rising straightness + rising pred_loss health + falling probe R² = prediction-shortcut collapse in progress.
5. **Best.pt saved by val_loss is wrong for JEPA.** We need probe-R²-aware checkpointing, or (simpler) save every epoch so we can pick post-hoc. Logged as a follow-up in the scaling notes.
6. **Every training signal we had pointed at "healthy" except the probes.** Without the probes we'd have called this run a success. The probe infrastructure ScavieFae shipped in PR #22 is the reason we caught this at all — without it we'd have a Modal checkpoint we think is good, and we'd only notice it was broken when someone tried to visualize a reconstruction.

What this rules out:
- **NOT**: "JEPA is the wrong architecture for Melee"
- **NOT**: "Linear LR scaling at 8× was wrong"
- **NOT**: "The rescale hyperparameters were too aggressive"
- **YES**: "LeWM's loss structure, applied to a domain with discrete events, compresses out discrete-event information as training progresses"

What this enables:
- **A specific, testable hypothesis for e030c**: reducing SIGReg pressure (λ) should slow the specialization. If true, we can find a λ where Layer B info survives. If false, the MSE itself is the problem and we need structural changes.
- **A named failure mode** to watch for in future runs. "Prediction-shortcut collapse" is now a thing we can reference in run card Risks sections.
- **A research note** (`docs/jepa-direction-notes/smooth-physics-vs-game-rules.md`) that captures the framework tying together death→respawn, percent backwards, and the e030b trajectory. Reusable for any future experiment's context section.
- **A probe methodology confirmation**: linear probes are trustworthy. Nonlinear probes don't rescue what linear probes call "gone."

## Follow-ups queued

In priority order:

1. **e030c**: SIGReg λ sweep. Reduce λ from 0.1 to 0.01 (and maybe 0.03 as a middle point). Same model, same data, same rescale hyperparams. One change only, per Mattie's "move slowly" directive. Hypothesis: lower λ preserves Layer B info longer. This is the next experiment.

2. **Checkpoint-saving strategy fix**. Add `--save-every-epoch` to `scripts/modal_train_jepa.py` and `training/jepa_trainer.py`. When enabled, save `epoch_1.pt`, `epoch_2.pt`, ... alongside `best.pt`/`latest.pt`/`final.pt`. Cost: ~15 min of code + a few hundred MB per run on the Modal volume. Ships before e030c. See scaling notes update.

3. **Event-conditioned eval metric.** Split rollout eval into "context contained a hit / stock change / death" vs "no event," report `pos_mae`, `percent_mae`, `action_acc` separately for each split. Surfaces Layer B gap as a real number. Applies to Mamba2 as well. ~50 LOC in `scripts/eval_rollout.py`.

4. **e030b canonical viz artifact**. Run `scripts/visualize_jepa.py` against the `final.pt` checkpoint with the canonical Fox-Marth FD game. Expected: visually worse encoder reconstruction than e030a's 3-epoch checkpoint on the same game. If this prediction holds, it's direct visual confirmation of "longer training = worse representation in this regime" which is viscerally striking.

5. **Research note cross-reference**. Update `docs/jepa-direction.md` to link to `smooth-physics-vs-game-rules.md` as the canonical explanation for "why we expect certain fields to be hard."

## References

- LeWM paper: `research/sources/2603.19312-summary.md`
- **Research note**: `docs/jepa-direction-notes/smooth-physics-vs-game-rules.md` — the full framework tying e030b, the percent-backwards observation, and death→respawn into one disease
- **Scaling notes**: `docs/jepa-scaling-notes.md` — now includes a "prediction-shortcut collapse" section and checkpoint-saving strategy
- **e030a run card**: `docs/run-cards/e030a-jepa-baseline.md` — the baseline this rescale was designed to correct (now retroactively reinterpretable: e030a's 3-epoch encoder was actually less collapsed than e030b's 10-epoch encoder)
- **Diagnostic script**: `scripts/nonlinear_probe_diagnostic.py` — reusable tool for running linear-vs-nonlinear probe comparison on any JEPA checkpoint
- **Manifest**: `checkpoints/e030b-jepa-rescale/manifest.json` on Modal volume — full per-epoch history
- Code:
    - Models: `models/jepa/` (unchanged)
    - Data: `data/jepa_dataset.py` (unchanged)
    - Training: `training/jepa_trainer.py`, `training/jepa_diagnostics.py` (**probe methodology fix lands here**)
    - Modal entry: `scripts/modal_train_jepa.py` (unchanged)
    - Diagnostic CLI: `scripts/run_jepa_diagnostics.py` (unchanged)
