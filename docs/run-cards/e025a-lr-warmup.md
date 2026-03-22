---
id: e025a
created: 2026-03-21
status: kept
type: training-regime
base_build: b001
built_on: [e023b]
source_paper: null
rollout_coherence: 5.146
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

~$5 (A100 40GB with AMP, ~3hr).

## Confounds

- Only testing 5% warmup. If null, doesn't rule out longer warmup (10%, 20%).
- Interaction with Self-Forcing: SF steps happen from the start. Warmup affects LR during early SF batches too.

## Results

| Metric | E023b (baseline) | E025a (warmup) | Delta |
|--------|-----------------|----------------|-------|
| **rollout_coherence** | **5.775** | **5.146** | **-10.9%** |
| change_acc | 66.0% | 65.6% | -0.4pp |
| pos_mae | 0.823 | 0.814 | -1.1% |
| loss | — | 0.426 | — |

wandb: https://wandb.ai/shinewave/melee-worldmodel/runs/obu0o7lf
Checkpoint: /data/checkpoints/e025a-lr-warmup/best.pt

## Director Evaluation

**Verdict: KEPT — New Best**

RC 5.146 is a 10.9% improvement over E023b (5.775) — the largest single-experiment gain since Self-Forcing (7.5% in e018a). This is now the largest single-experiment improvement in the project's history.

LR warmup stabilized early training at d_model=768. The 5% warmup period allowed the optimizer to find a better basin before cosine decay began. change_acc is essentially flat (-0.4pp), meaning the RC gain comes entirely from better position/dynamics prediction rather than action classification changes. pos_mae also improved slightly (0.823 to 0.814).

Cumulative RC improvement: 6.77 -> 6.26 -> 6.03 -> 5.775 -> 5.146 (-24.0% from baseline).

**Next:** warmup_pct is a new canonical hyperparameter. Consider testing 0.10 warmup for further gains.
