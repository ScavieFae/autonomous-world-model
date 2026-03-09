# Batch Teacher-Forced Eval Analysis

**Date**: 2026-02-28
**Checkpoint**: `mamba2-v3-2k-test-v2` (Mamba-2 4.3M, K=10, v3 encoding, 2K games, 1 epoch)
**Data**: 30 games from `game-only-v3-2k`, seed=42
**Frames evaluated**: 644,486
**Tool**: `worldmodel/scripts/batch_eval.py`
**Viewer**: `worldmodel/batch_eval_viewer.html`

## Summary

| Metric | Value |
|--------|-------|
| Action accuracy | 96.7% |
| Change accuracy | 69.2% |
| On ground accuracy | 75.6% |
| Avg position error | 1.05 |

## Key Findings

### 1. on_ground has terrible recall (54.9%)

The model predicts "on ground" with high precision (86.7%) but misses nearly half of actual grounded frames (recall 54.9%, F1 67.2%). The bias is one-directional: the model defaults to "airborne" too often.

**Why this matters for autoregressive mode**: In teacher-forced mode, the model gets corrected every frame. But in autoregressive mode, a single false-negative on_ground prediction feeds back as context — the model sees "airborne" and predicts the next frame accordingly, compounding the error. This is likely the root cause of "characters stop taking visible actions" during autoregressive rollout.

### 2. facing has the same pattern (49.3% recall)

Facing precision is 99.4% but recall is only 49.3%. The model is reluctant to predict facing=true (facing right). This means in autoregressive mode, characters could get stuck facing one direction, which cascades into wrong attack predictions (BAIR vs FAIR depends entirely on facing).

### 3. invulnerable is always predicted false

0% recall, 100% accuracy. The flag is so rare (appears during certain specials, respawn invincibility) that the model learned to never predict it. This is correct behavior for overall loss minimization but means the model can't represent invulnerability states.

### 4. Action accuracy is dead flat across time

| Time bucket | Action acc | On ground acc | Pos error |
|-------------|-----------|---------------|-----------|
| 0-10% | 96.5% | 74.4% | 0.88 |
| 10-20% | 96.6% | 72.2% | 1.02 |
| 20-30% | 96.6% | 76.3% | 1.13 |
| 30-40% | 96.7% | 78.0% | 1.06 |
| 40-50% | 96.7% | 74.8% | 1.11 |
| 50-60% | 96.7% | 74.4% | 1.00 |
| 60-70% | 96.7% | 77.6% | 1.10 |
| 70-80% | 96.5% | 76.3% | 1.12 |
| 80-90% | 96.8% | 77.4% | 1.06 |
| 90-100% | 96.9% | 74.5% | 1.06 |

No temporal degradation in teacher-forced mode. The error rate is constant from start to finish. This confirms that drift is a compounding/autoregressive problem, not a base prediction problem.

### 5. Movement is the weakest category (89.6%)

| Category | Accuracy | Change acc | Count |
|----------|----------|------------|-------|
| aerial | 97.5% | 81.8% | 154,281 |
| damage | 97.3% | 24.2% | 88,514 |
| aerial_attack | 96.9% | 57.5% | 86,251 |
| movement | 89.6% | 57.2% | 65,945 |
| special | 98.0% | 81.5% | 60,504 |
| shield_dodge | 96.8% | 81.2% | 39,194 |
| ground_attack | 98.8% | 71.5% | 34,446 |
| grab | 98.0% | 79.8% | 24,525 |
| idle | 97.5% | 85.8% | 22,054 |
| edge | 96.3% | 60.6% | 12,958 |

### 6. Top error actions

| Action | Category | Accuracy | Change acc | Errors |
|--------|----------|----------|------------|--------|
| TURNING | movement | 59.9% | 28.2% | 3,095 |
| DASHING | movement | 95.5% | 75.3% | 1,625 |
| FALLING | aerial | 93.7% | 41.8% | 1,266 |
| LANDING | aerial | 94.0% | 65.4% | 1,019 |
| WALK_SLOW | movement | 69.5% | 31.2% | 960 |

TURNING (28.2% change_acc) and WALK_SLOW (31.2% change_acc) are the worst — instantaneous state transitions that the model consistently fails to predict.

### 7. Damage change_acc is 24.2%

The model can't predict *when* a player enters a damage state. This makes sense — getting hit depends on the opponent's attack connecting, which is a spatial+timing event the model can't anticipate from the victim's state alone.

## Implications

1. **Binary flag bias is the autoregressive bottleneck**: on_ground and facing have catastrophic recall. Fixing this (e.g., balanced sampling, loss reweighting, or separate binary heads) would likely have the biggest impact on autoregressive quality.

2. **Movement transitions are a physics gap**: TURNING, WALK_SLOW, and DASHING transitions require understanding Melee's ground movement state machine, which involves traction, initial dash length, and pivot mechanics. The state_flags (byte0.bit2 "actionable") should help here — the model needs to learn "you can turn during wait but not during dash startup."

3. **Damage is inherently hard**: Predicting when a hit connects requires spatial reasoning about hitboxes, which our model doesn't have explicit access to. This may be a fundamental ceiling without hitbox encoding.

4. **The model is good at steady-state, bad at transitions**: 96.7% overall accuracy is driven by the 90%+ of frames where the action doesn't change. The 69.2% change_acc on the ~10% of transition frames is where the model fails — and those are the frames that matter for gameplay.
