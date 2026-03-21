---
id: e020b
created: 2026-03-16
status: discarded
type: training-regime
base_build: b001
built_on: [e018c]
source_paper: 2508.13009
rollout_coherence: 6.289
prior_best_rc: 6.03
---

# Run Card: e020b-sf-ratio-30

## Goal

Test whether higher SF ratio (30% vs 20%) improves rollout coherence. More exposure to own errors = more learning signal for error recovery. Previously rejected by Director with 0 data points — running it to get actual data.

## What Changes

One config change from E018c: `self_forcing.ratio: 3` (30% SF, was `ratio: 4` = 20%).

## Data

1.9K FD top-5, 1 epoch, bs=512. K=30 context. Identical to E018c except SF ratio.

## Target Metrics

| Metric | E018c (20% SF) | Target | Kill threshold |
|--------|---------------|--------|---------------|
| **rollout_coherence** | **6.03** | **< 6.03** | ≥ 6.10 |
| change_acc | 62.3% | Expect regression (more SF) | < 55% |

## Cost Estimate

~$5 Scout (actual: ~$7.35, A100 3.5hr).

## Results

| Metric | E018c (25% SF) | E020b (33% SF) | E020a (10% SF) | Delta |
|--------|---------------|----------------|----------------|-------|
| **rollout_coherence** | **6.03** | **6.289** | 6.62 | **+4.3%** |
| change_acc | 62.3% | 60.2% | 63.1% | -2.1pp |
| pos_mae | 0.824 | 0.825 | 0.808 | flat |
| sf_loss | ~0.37 | 0.427 | 0.534 | +15% |
| tf_loss | — | 0.239 | — | — |

Runtime: 12748s (~3.5hr) on A100. Cost: ~$7.35.
wandb: https://wandb.ai/shinewave/melee-worldmodel/runs/8powgzv7

## Director Evaluation

**Verdict: DISCARDED**

Clear regression (+4.3%). SF ratio axis now has 3 data points showing symmetric degradation around 20%: 10% too little signal (RC 6.62), 20% optimal (RC 6.03), 33% too much (RC 6.289). Higher SF ratio overwhelms TF learning — SF loss jumped 15%.

**Finding:** SF ratio:4 (20%) is near-optimal. Axis closed with 3/3 data points. Don't test other ratios without strong new reasoning.
