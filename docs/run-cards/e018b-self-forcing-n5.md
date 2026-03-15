---
id: e018b
created: 2026-03-15
status: discarded
type: training-regime
base_build: b001
built_on: [e018a]
source_paper: 2508.13009
rollout_coherence: 6.45
prior_best_rc: 6.26
---

# Run Card: e018b-self-forcing-n5

## Goal

Test whether longer Self-Forcing unroll (N=5 vs N=3) improves rollout coherence further. E018a established that N=3 achieves 7.5% RC improvement (6.77→6.26). This tests the horizon axis: does more drift exposure help?

## What Changes

One config change from E018a:

```yaml
self_forcing:
  unroll_length: 5      # was 3 in e018a
```

No code changes. The trainer already supports arbitrary unroll lengths.

## Results

**DISCARDED — regression.**

| Metric | E018a (N=3) | E018b (N=5) | Delta |
|--------|------------|------------|-------|
| **rollout_coherence** | **6.26** | **6.45** | **+3.0% (worse)** |
| change_acc | 61.6% | 57.4% | -4.2pp |
| pos_mae | 0.825 | 0.823 | ~flat |
| sf_loss | 0.383 | 0.744 | +94% |
| tf_loss | 0.164 | 0.234 | +43% |
| action_acc | — | 94.5% | — |

### Cost

- Runtime: 9017s (~150min) on A100 40GB
- Cost: ~$5.25 ($2.10/hr × 2.5hr)
- wandb: https://wandb.ai/shinewave/melee-worldmodel/runs/wdv9wynz

## Director Evaluation

**Verdict:** DISCARDED

**Confidence:** HIGH — regression is real and consistent across all metrics. SF loss nearly doubled, indicating the model couldn't learn from 5 steps of drift with truncated BPTT.

**Finding:** Longer Self-Forcing unroll (N=5 vs N=3) regressed rollout coherence by 3.0% (6.26→6.45) and nearly doubled SF loss (0.38→0.74). Truncated BPTT breaks gradient flow on timescales longer than ~150ms (N=3 at 60fps). The gradient signal degraded rather than improved with longer horizon.

**Implication:** Truncated BPTT saturates at N=3. To go longer, need either full BPTT (gradient through reconstruct_frame) or horizon-weighted loss (e018d) to focus on early AR steps where truncated BPTT still has signal. Test e018d on N=3 first.
