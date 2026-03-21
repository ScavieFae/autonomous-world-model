# Research Brief: SF Ratio 30%

Run the SF ratio 30% experiment. E020b was previously cancelled without data — run it to get an actual measurement.

## Context

SF ratio 20% is the only proven value (E018a, RC 6.26). SF ratio 10% regressed (E020a, RC 6.62 — too little signal). 30% is the obvious next test point. More SF exposure = more error-correction training signal, but too much could destabilize teacher-forced learning.

Existing cancelled run card: `docs/run-cards/e020b-sf-ratio-30.md`. Un-cancel it and execute.

## What To Do

1. Un-cancel e020b (set status back to `running`)
2. Use `experiments/e018c-context-k30.yaml` as base config
3. Change `self_forcing.ratio: 3` (= 30% SF, was `ratio: 4` = 20%)
4. Train on 1.9K data, 1 epoch, bs=512
5. Compare RC to E018c baseline (6.03)

## Decision Rules

- RC < 6.03 → KEEP. 30% is better, opens question of 40%/50%.
- RC ≥ 6.03 but < 6.10 → Marginal. Note finding, likely discard.
- RC ≥ 6.10 → DISCARD. 30% is too much SF. 20% is near-optimal.
