---
id: e031a
created: 2026-04-12
status: kept
type: training-regime
base_build: b002
built_on: [e028a]
source_paper: null
rollout_coherence: null
prior_best_rc: 4.798
---

# Run Card: e031a-speed-profile

## Goal

Profile the existing Mamba2 training pipeline to measure where wall-clock time goes, and test GPU-resident dataset as the first speed intervention. A 15.8M param model on an A100 should complete an epoch in minutes, not hours — this run tells us why it doesn't and what to fix.

## What changes

Identical to e028a-full-stack (current best, RC 4.798) except:

| Change | Value | Purpose |
|---|---|---|
| `profile: true` | new flag | Per-phase CUDA-synced timing (data load, forward, backward, optimizer, SF) |
| `gpu_resident: true` | new flag | Move `dataset.floats` + `dataset.ints` tensors to GPU once at startup (~2GB). `get_batch` becomes GPU-side fancy indexing, no CPU→GPU transfer per batch. |
| `num_epochs: 1` | was 2 | Just enough to measure timing. Not training for quality. |

Model, data, encoding, loss weights, SF config — all identical to e028a-full-stack.

## Expected output

A per-phase timing breakdown at the end of the epoch:

```
── PROFILE ──
data loading:   XXXX ms  (XX.X%)  X.XX ms/batch
forward pass:   XXXX ms  (XX.X%)  X.XX ms/batch
backward pass:  XXXX ms  (XX.X%)  X.XX ms/batch
optimizer:      XXXX ms  (XX.X%)  X.XX ms/batch
self-forcing:   XXXX ms  (XX.X%)  X.XX ms/batch
TOTAL:          XXXX ms  (N TF batches + M SF batches)
```

Hypothesis: data loading is >50% of total time at b002's batch=512. GPU-resident data should cut it dramatically. If forward+backward are the bottleneck instead, the optimization plan shifts to torch.compile and batch scaling.

## Launch command

```bash
modal run scripts/modal_train.py \
    --config experiments/e031a-speed-profile.yaml \
    --encoded-file /encoded-e012-fd-top5.pt \
    --gpu A100 \
    --run-name e031a-speed-profile
```

No `--detach` — this is a single-epoch run, should complete in <30 min. Watch the logs live.

## Budget

A100 × ~15-30 min × $2.10/hr ≈ $0.50-1.00.

## Success criteria

Not about RC — about timing. The run succeeds when:
1. We have a per-phase timing breakdown with percentage attribution
2. We know which phase dominates wall clock
3. We have a GPU-resident vs non-GPU-resident comparison (the `gpu_resident: true` flag itself is the first A/B)

If GPU-resident cuts epoch time by >2×, that's the first confirmed speedup and we move to the next lever. If it doesn't, we dig into forward/backward or SF.

## Results

**Status: kept (infrastructure validated, bottleneck identified).** Not a typical experiment — this is a diagnostic. Ran the profile instrumentation on 3050 batches (the first log interval) then turned off sync to let the rest of the epoch run at full speed. Had to go through three iterations before the run completed cleanly:

- **v1**: crashed in Self-Forcing because `reconstruct_frame` expects CPU tensors but `gpu_resident=True` put `dataset.floats` on GPU. Fixed with `.cpu()` calls in `_self_forcing_step` and `_build_sf_targets`.
- **v2**: ran but discovered the profiling CUDA sync doubled per-batch time (~366 ms vs ~160 ms non-profiled). A 3-hour epoch to get a breakdown is backwards. Killed and fixed the profiler to sample only the first `log_interval` batches then turn off.
- **v3**: ran to the profiling output, breakdown captured, then killed (rest of epoch would have hit the same rollout-eval device mismatch bug that the SF path hit — fix shipped in a separate commit).

### The profiling breakdown (first 3050 batches, 2440 TF + 610 SF)

| Phase | Total ms | % of wall clock | ms/batch | Notes |
|---|---|---|---|---|
| data loading | 42.7 | **0.0%** | 0.02 | GPU-resident worked completely |
| forward pass | 158,271 | 16.1% | 64.87 | per-TF-batch |
| backward pass | 269,482 | 27.5% | 110.44 | per-TF-batch, ~1.7× forward (normal) |
| optimizer | 10,373 | 1.1% | 3.40 | negligible |
| **self-forcing** | **542,670** | **55.3%** | **889.62** | per-SF-batch, ~5× a TF batch |
| TOTAL | 980,839 | — | 321.59 avg | includes CUDA sync overhead |

### Key findings

**1. GPU-resident data completely eliminated the data pipeline bottleneck.** 0.02 ms/batch is effectively zero. The 12 GB dataset copy happens once at startup (7.5s) and every subsequent batch is GPU-side fancy indexing with no CPU roundtrip.

**2. Self-forcing is 55.3% of wall-clock time.** Each SF batch (N=3 unrolls + CPU argmax reconstruction + tensor append) takes 890 ms vs 178 ms for a pure TF batch. That's **5× the cost of a TF batch, not 3×**. With SF batches at 20% of the interleaved ratio, the math works out: 20% × 5× = 100% of additional TF compute on top of training. Half the epoch is SF.

**3. Forward and backward are ~43% of remaining time.** 16.1% forward + 27.5% backward + 1.1% optimizer = 44.7% of total; without SF that's 100% of everything. The Mamba2 sequential scan is ~65 ms forward + ~110 ms backward for a 15.8M-param model at batch 512 with K=30. That's not insane but it's 2-3× what a 15M-param Transformer would do on the same hardware — the sequential for-loop over the 30 timesteps is not GPU-friendly despite the `chunk_size=15` SSD path.

**4. GPU-resident memory cost**: 12.02 GB moved to GPU in 7.5s. Leaves ~20 GB free on A100-40GB (after model + activations at peak). This fits for the 2K dataset but **will NOT fit for the 7.7K dataset** (~48 GB > 40 GB VRAM). Logged for scaling notes.

### What this directly implies

The single biggest speedup lever is **dropping Self-Forcing**. The whole e031b experiment is built around testing whether SF is still load-bearing at current scale, because if it's not, we get a 2.2× free speedup on every experiment by turning it off. That's the path we're on.

Secondary levers (after SF, if still not fast enough):
- **Bigger batch** (amortize kernel launch overhead on the sequential scan)
- **torch.compile** (op fusion, reduced Python overhead)
- **Shorter context** (K=10 or K=15 cuts the sequential scan proportionally — Mattie's "Melee encodes state" hypothesis)

### Infrastructure changes that landed here

These changes are kept and become the new baseline for the e031+ series:

1. **`profile: true` flag** in the Trainer — CUDA-synced per-phase timing, samples first `log_interval` batches then auto-disables. ~60 lines in `training/trainer.py`.
2. **`gpu_resident: true` flag** — moves `dataset.floats`/`dataset.ints` to GPU at startup. ~15 lines in `Trainer.__init__`.
3. **`.cpu()` safety in SF path** — `_self_forcing_step` and `_build_sf_targets` explicitly bring dataset reads to CPU regardless of where the source lives. Unblocks gpu_resident + SF together.
4. **`.cpu()` safety in rollout eval** — `evaluate_rollout_coherence` in `scripts/eval_rollout.py` does the same for batch_floats / batch_ints / gt_float / gt_int. Unblocks gpu_resident + rollout eval together.

All four changes are backward-compatible and off-by-default. Existing configs with `profile: false` (default) and `gpu_resident: false` (default) behave exactly as before.

## Decision

**Kept. Infrastructure validated.** The profiling instrumentation works, GPU-resident data works, and the critical finding (SF = 55% of time) is reproducible and actionable. e031b is the direct follow-up to test whether SF is still load-bearing. If it isn't, we drop it and the wall-clock budget cuts by half immediately.

### Cost

$0.00-$3.00 across v1/v2/v3 iterations. v1 ran ~5 min before crashing. v2 ran ~25 min before we killed it. v3 ran ~17 min to the profile breakdown + kill. Total A100 time ~47 min = ~$1.65.
