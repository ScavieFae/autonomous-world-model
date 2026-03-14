# Goals

See `program.md` for the authoritative research direction document.

## Current Mode

**Autoresearch.** Run research cycles: hypothesize → Director review → execute on Modal → evaluate results. Each cycle tests one hypothesis against the E019 baseline (rollout coherence = 6.77).

## Active Baseline

- **Checkpoint:** `e019-baseline-1k/best.pt`
- **Rollout coherence:** 6.77 (mean pos_mae, K=20, N=300)
- **Config:** `experiments/e019-baseline.yaml` (b001 stable build, 1.9K data, bs=512, 1ep)
- **Base build:** b001

## Priority Order

1. Research directions from program.md (Self-Forcing is #1)
2. Hyperparameter exploration (batch size, learning rate, weight decay)
3. Data scaling (once 7.7K data loading is fixed)

## Constraints

- Budget: $30/day, $150/week
- GPU: A100 40GB only (H100 requires Mattie approval)
- Data: 1.9K encoded file for Scout/Confirm experiments
- One variable per experiment
