---
id: e023a
created: 2026-03-19
status: running
type: architectural
base_build: b001
built_on: [e018c]
source_paper: null
rollout_coherence: null
prior_best_rc: 6.03
---

# Run Card: e023a-dmodel192

## Goal

Phase 1 Tier 1 architecture grid -- test whether d_model=384 is over-parameterized for the current 1.9K replay dataset. Halve d_model to 192 and measure whether the smaller model matches or improves rollout coherence via better bias-variance tradeoff at this data scale.

## Mechanism

SSM hidden state width capacity. At d_model=192, the Mamba2 trunk has 3.36x fewer parameters (1,294,684 vs 4,349,284). The narrower trunk forces more efficient feature extraction through the SSM layers. At 1.9K training games, the larger model may be over-parameterized -- a smaller model could generalize better by having less capacity to memorize.

Key dimensional changes:
- d_model: 384 -> 192
- d_inner (expand=2): 768 -> 384
- num_heads (d_inner/headdim): 12 -> 6
- d_state, n_layers, headdim: unchanged (64, 4, 64)

Note: headdim=64 divides cleanly into d_inner=384, yielding 6 heads. Valid configuration.

## Context

E018c achieves RC 6.03 at d_model=384 with 4,349,284 parameters. This is the current best. This experiment tests whether that width is necessary or whether 192 suffices at the current data scale.

## Param Counts (verified)

| Config | d_model | d_inner | num_heads | Total Params |
|--------|---------|---------|-----------|-------------|
| E018c (baseline) | 384 | 768 | 12 | 4,349,284 |
| **E023a (this)** | **192** | **384** | **6** | **1,294,684** |

Ratio: 3.36x fewer parameters.

## What Changes

One change from E018c: `d_model: 192` (was 384). All other hyperparameters identical.

## Target Metrics

- **Target:** RC <= 6.03 (match or beat E018c)
- **Kill threshold:** RC > 6.15 (clear regression -- discard)
- **Watch:** training loss convergence speed, action accuracy, SF loss stability

## Data

Same as E018c: 1.9K replays, stage_filter=32 (FD), character_filter=[1,2,7,18,22] (top 5).

## Model

Mamba2, d_model=192, d_state=64, n_layers=4, headdim=64, context_len=30, chunk_size=15, dropout=0.1.

## Training

Same as E018c: lr=0.0005, wd=1e-5, batch_size=512, 1 epoch, self-forcing ratio=4 unroll=3.

## Cost

Estimated ~$3-4 Scout GPU (smaller model trains faster due to 3.36x fewer backbone params).
