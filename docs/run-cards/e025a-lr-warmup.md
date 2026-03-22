---
id: e025a
created: 2026-03-21
status: proposed
type: training-regime
base_build: b001
built_on: [e023b]
source_paper: null
rollout_coherence: null
prior_best_rc: 5.775
---

# Run Card: e025a-lr-warmup

## Goal

Test whether linear LR warmup improves training stability and rollout coherence at d_model=768. With 15.8M parameters, the early learning phase may benefit from a gentler start before cosine decay. Warmup is standard practice for larger models but has never been tested in this pipeline.

## What Changes

One training parameter added: `warmup_pct: 0.05` (5% of total steps use linear warmup, then cosine decay for the remaining 95%). AMP enabled (validated safe in e023b-epoch2).

All other hyperparameters identical to E023b.

## Target Metrics

- **Keep:** RC < 5.775 (improvement over E023b)
- **Kill:** RC > 5.85 or training instability (loss spikes in first 5% of steps)

## Model

Identical to E023b: d_model=768, d_state=64, n_layers=4, headdim=64, 15,648,882 params.

## Training

- lr: 0.0005, weight_decay: 1e-5, batch_size: 512, 1 epoch
- **warmup_pct: 0.05** — linear warmup for first 5% of steps, then cosine decay
- AMP: enabled (float16 autocast + GradScaler)
- Self-Forcing: ratio=4 (20%), unroll_length=3
- context_len=30, chunk_size=15

## Cost

~$7-9 Scout tier (A100 40GB). AMP may reduce step time vs E023b.

## Confounds

- Only testing 5% warmup. If null, doesn't rule out longer warmup (10%, 20%).
- Interaction with Self-Forcing: SF steps happen from the start. Warmup affects LR during early SF batches too.
