---
id: e018a
created: 2026-03-09
status: proposed
type: training-regime
base_build: b001
built_on: [e017a]
source_paper: 2508.13009
rollout_coherence: null
prior_best_rc: null
---

# Run Card: e018a-self-forcing

## Goal

Implement true Self-Forcing: during training, periodically unroll the model autoregressively for K steps and backprop through the generated sequence. Unlike E015's scheduled sampling (which corrupts the last context frame), Self-Forcing runs full AR rollouts during training so the model learns to recover from its own compounding errors.

E015 replaced one context frame with a prediction. Self-Forcing goes further: the model generates a sequence of N frames autoregressively, and the loss is computed on the entire generated sequence against ground truth. The model sees exactly the kind of drift it produces at inference and learns to correct it.

**Key difference from E015 (true SS):**
- E015: corrupt 1 context frame → forward pass → loss on final prediction
- E018a: AR unroll N frames → loss on all N predictions vs ground truth

## What Changes

### Training loop (`training/trainer.py`)

Add a Self-Forcing training step, interleaved with standard teacher-forced steps:

1. Sample a starting context window from the batch (K frames of ground truth)
2. Unroll the model for N steps autoregressively (detach between steps for truncated BPTT, or full BPTT if memory allows)
3. At each AR step, use ground-truth controller inputs but the model's own predicted state
4. Compute loss at each AR step against ground truth
5. Backprop through the full unrolled sequence

### Schedule

- **Interleave ratio**: 1 Self-Forcing batch per P teacher-forced batches (start with P=4, so 20% SF)
- **Anneal**: Epoch 0 pure TF. Epoch 1 → 10% SF. Epoch 2+ → 20% SF.
- **Unroll length N**: Start with N=5. This is the key hyperparameter — longer unrolls train for longer-horizon stability but cost more memory and compute.

### What stays the same

- Model architecture (Mamba2, same heads, same encoding)
- Dataset, encoding, data loading
- Teacher-forced batches (identical to current training)
- Validation (teacher-forced, as baseline comparison)

## Target Metrics

| Metric | E017a baseline | Target | Kill threshold |
|--------|---------------|--------|---------------|
| val_change_acc | TBD (use E017a result) | No regression >2pp | <85% |
| val_pos_mae | TBD | No regression >10% | >0.85 |
| AR demo quality | Drift visible by frame ~50 | Stable to frame 100+ | Worse than baseline |

The real test is AR rollout quality, not TF metrics. TF metrics may regress slightly (the model spends some capacity on AR recovery). That's acceptable if AR rollouts improve.

## Cost Estimate

Self-Forcing batches are ~N× more expensive than TF batches (N sequential forward passes, can't parallelize). At 20% SF with N=5:
- Effective training cost: ~2× baseline (20% of batches are 5× more expensive)
- On H100 with E012 data (1.9K games): ~32 min baseline → ~65 min estimated
- **Est. cost: ~$4-5**

## Escape Hatches

- **OOM on unroll**: Reduce N from 5 to 3. Or use truncated BPTT (detach gradients every 2 steps).
- **TF metrics crater**: Reduce SF ratio from 20% to 10%. If still bad, the model may need more capacity to handle both TF and AR objectives.
- **No AR improvement**: Try longer unroll (N=10) or higher SF ratio (30%). If still nothing, the bottleneck may not be exposure bias.

## Open Questions

- **Truncated vs full BPTT**: Full BPTT through N steps gives better gradients but uses N× memory. Truncated BPTT (detach every M steps) trades gradient quality for memory. Start with full, fall back to truncated.
- **Loss weighting**: Should AR-step losses be weighted equally? Or ramp up with horizon (weight later steps more, since those are where drift matters most)?
- **Interaction with absolute_y (E017a)**: Absolute y already reduces one source of drift. Does Self-Forcing still help on top of it, or is the remaining drift mostly in other dimensions (action state, percent)?

## Prior Art

- Matrix-Game 2.0 (2508.13009): Self-Forcing distillation for video world models. 25 FPS real-time generation. Key insight: training on own outputs directly addresses exposure bias.
- E015 (true SS): Our previous attempt at feeding predictions back. Corrupted 1 context frame. Status: PENDING REVIEW — code not fully implemented in trainer.
- E012b (noise SS): Gaussian noise on context frames. Null result — random noise ≠ structured model errors.

## Implementation Notes

The core loop is roughly:

```python
# Self-Forcing step (runs every P batches)
context = batch[:, :K, :]  # ground truth context
gt_ctrl = batch_ctrl       # ground truth controller inputs for all frames

state = context
sf_losses = []
for step in range(N):
    ctrl = gt_ctrl[:, K + step, :]
    pred = model(state, ctrl)
    loss = compute_loss(pred, ground_truth[:, K + step, :])
    sf_losses.append(loss)
    # Reconstruct next state from predictions (same as AR demo generation)
    next_frame = reconstruct_frame(pred)
    state = torch.cat([state[:, 1:, :], next_frame.unsqueeze(1)], dim=1)

sf_loss = torch.stack(sf_losses).mean()
sf_loss.backward()
```

Key implementation details:
- `reconstruct_frame()` must match the AR demo generation logic exactly (deltas → absolutes, argmax for categoricals, sigmoid for binaries)
- Gradient flows through the reconstruction — this is what teaches the model about its own error patterns
- The `model()` call inside the loop must NOT be `@torch.no_grad()` — gradients must flow

## Launch Command

```bash
# TBD — depends on implementation
```
