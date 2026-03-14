---
id: e018b
created: 2026-03-09
status: running
type: training-regime
base_build: b001
built_on: [e017a]
source_paper: null
rollout_coherence: null
prior_best_rc: null
---

# Run Card: e018b-rollout-coherence-eval

## Goal

Build the rollout coherence eval before running Self-Forcing experiments. Without a quantitative AR metric, we can't measure whether Self-Forcing actually helps — we'd be back to eyeballing demos.

This is infrastructure, not a training experiment. But it's a hard prerequisite for e018a and all future autoresearch.

## What to Build

`scripts/eval_rollout.py` — standalone script that takes a checkpoint and returns a divergence curve.

```
python scripts/eval_rollout.py --checkpoint checkpoints/e017a/best.pt --config experiments/e017a-absolute-y.yaml
```

### Algorithm

1. Load N starting windows from val set (N=300)
2. From each, autoregress K steps (K=20) feeding predictions back
3. At each horizon k, measure divergence from ground truth:
   - `pos_mae`: mean absolute error on (x, y) position
   - `vel_mae`: mean absolute error on velocities
   - `action_acc`: accuracy on action_state prediction
   - `percent_mae`: MAE on damage percent
4. Output: divergence curve (metrics at t+1, t+5, t+10, t+20)
5. **Summary metric**: mean `pos_mae` averaged over all K horizons. Single number.

### Constraints

- Must run in 30-60 seconds (fast enough for post-training eval)
- Deterministic: same checkpoint + same starting frames = same score
- Uses ground-truth controller inputs (testing physics prediction, not player intent)
- Works with any checkpoint that matches the config's architecture

## Why This Comes First

The autoresearch plan (`docs/autoresearch-plan.md`) identifies this as priority #1. Without it:
- E018a (Self-Forcing) has no quantitative success criterion
- Autoresearch loops can't compare experiments
- We keep relying on "does the demo look right" which doesn't scale

## Implementation Notes

Most of the AR loop already exists in `scripts/rollout.py` (the demo generator). The eval script reuses that logic but:
- Runs many short rollouts instead of one long demo
- Compares against ground truth at each step
- Returns numbers instead of JSON frames
- Must be `@torch.no_grad()` for speed (unlike Self-Forcing training)

## Deliverables

1. `scripts/eval_rollout.py` — the eval script
2. Baseline numbers for existing checkpoints (E012, E017a at minimum)
3. Update `docs/autoresearch-plan.md` with actual baseline numbers

## Launch Command

```bash
# Not a training run — just build and eval
python scripts/eval_rollout.py --checkpoint checkpoints/e017a/best.pt \
  --config experiments/e017a-absolute-y.yaml \
  --num-samples 300 --horizon 20
```
