---
id: e031a
created: 2026-04-12
status: proposed
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
