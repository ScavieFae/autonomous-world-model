# Empirical Findings

Observations from the experiment series. Stated as evidence with hit rates, not conclusions.

## Training Regime

| Finding | Evidence | Experiments |
|---------|----------|-------------|
| Self-Forcing improves rollout coherence | 3/3 SF experiments improved RC over TF-only | e018a (6.26), e018c (6.03), e020a (6.62 — worse than e018c but still SF) |
| Truncated BPTT saturates at N=3 | N=5 did not improve over N=3 (1/1) | e018b (6.45 vs e018a 6.26) |
| Horizon weighting harms RC | 1/1 regression | e018d (6.81 vs e018a 6.26) |
| TF metrics don't predict AR quality | e016 had 38% better val_loss than e012 but worse AR demos | e016 vs e012 |
| The lever is *how we train*, not *what we predict* | Pattern across multiple experiments | e012b, e015, e016, e018a |

## Encoding & Data

| Finding | Evidence | Experiments |
|---------|----------|-------------|
| Absolute y trades drift for jitter — structural improvement | 1/1 | e017a |
| Absolute x causes oscillation | 1/1 | e017c |
| Absolute velocities cause head decoupling | 1/1 | e017d |
| action_change_weight=5.0 is the strongest proven single improvement | +9.9pp change_acc | e009b |
| ctrl_threshold_features improve action prediction | +5.9pp | e008c |
| Character embedding changes are null at current data scale | 1/1 | e011 |
| Noise-based scheduled sampling is a null result | 1/1 | e012b |

## Architecture

| Finding | Evidence | Experiments |
|---------|----------|-------------|
| Rolling context window K=30 + SF is current best | RC=6.03 | e018c |
| Context K=50 marginal improvement over K=30 | RC=5.97 vs 6.03 (discarded — marginal gain) | e019a |
| Cascaded heads fixed damage drift but overfitted at 1.9K games | 1/1 | e014 |
| Velocity and dynamics heads were getting zero gradient (fixed) | Bug, not finding | pre-e018 |

## Current Best

- **Rollout Coherence: 6.03** (e018c — rolling context K=30 + Self-Forcing on b001)
- **Prior best: 6.26** (e018a — Self-Forcing alone on b001)

## Open Axes

- Batch size hasn't been swept since early experiments (pre-RC evaluation)
- d_state and d_model haven't been explored (always 64 and 384)
- Learning rate schedule hasn't been tested beyond cosine annealing
- Data scale: most RC-evaluated experiments ran on 2K sample subset
