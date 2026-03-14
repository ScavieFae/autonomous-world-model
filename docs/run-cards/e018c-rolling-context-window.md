---
id: e018c
created: 2026-03-09
status: proposed
type: architectural
base_build: b001
built_on: [e017a]
source_paper: 2505.20171
rollout_coherence: null
prior_best_rc: null
---

# Run Card: e018c-rolling-context-window

## Goal

Test whether a longer context window with rolling eviction improves AR rollout quality. Currently context_len=10 (K=10). The SSM paper (2505.20171) and Matrix-Game 2.0 (2508.13009) both show that longer temporal context helps — SSMs via recurrence, transformers via attention over more frames.

Our Mamba2 backbone already supports variable-length sequences. This experiment simply increases K and measures the effect on both TF metrics and AR rollout coherence.

## What Changes

- `context_len`: 10 → 30 (3× longer context)
- `chunk_size`: 10 → 30 (match context length for SSD)
- Batch size may need to decrease to fit in memory (4096 → 2048 or 1024)

Everything else stays the same.

## Why This Matters

Longer context gives the model more frames to establish patterns before predicting. During AR rollouts, it means the model can "see" more of its own recent predictions, potentially detecting and correcting drift patterns that are invisible in a 10-frame window.

The SSM paper's key insight: SSM recurrence carries state indefinitely at fixed cost per step, but you need enough context for the recurrence to build up useful state. K=10 at 60fps is 167ms — less than a single Melee reaction window. K=30 is 500ms — enough to see a full move sequence.

## Target Metrics

| Metric | E017a (K=10) | Target | Kill threshold |
|--------|-------------|--------|---------------|
| val_change_acc | TBD | No regression >2pp | <85% |
| val_pos_mae | TBD | Improvement | >0.85 |
| Rollout coherence | TBD (e018b) | Improvement | Worse than K=10 |

## Cost Estimate

Longer sequences = more compute per batch. With K=30 and half the batch size:
- ~2× wall time per epoch vs K=10
- On H100: ~32 min baseline → ~65 min
- **Est. cost: ~$4-5**

## Risks

- **Memory**: K=30 triples sequence length. May need to halve batch size. SSDs scale linearly with sequence length (not quadratic), so this should be manageable.
- **Chunk size**: Mamba2's SSD implementation uses chunk_size for its structured state computation. chunk_size=30 may interact differently than chunk_size=10. Need to verify the SSD implementation handles this.
- **Data loader**: Must verify the dataset can produce K=30 context windows (need games with 30+ consecutive frames, which is essentially all of them).

## Dependencies

- e018b (rollout coherence eval) should be done first, so we have a quantitative baseline for K=10.
