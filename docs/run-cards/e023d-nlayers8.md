---
id: e023d
created: 2026-03-20
status: running
type: architectural
base_build: b001
built_on: [e023b]
source_paper: null
rollout_coherence: null
prior_best_rc: 5.775
---

# Run Card: e023d-nlayers8

## Goal

Phase 1 Tier 1 depth test — does doubling depth at the wider width compound the capacity gain from E023b? E023b proved d_model=768 is monotonically better than 384 (RC 5.775 vs 6.03, -4.2%). This tests whether adding more layers at that width yields further gains, or whether 4 layers already saturate representation capacity at d_model=768.

## Context

| Config | d_model | n_layers | Params | RC | Notes |
|--------|---------|----------|--------|----|-------|
| E018c | 384 | 4 | 4,275,810 | 6.03 | Previous best |
| E023b | 768 | 4 | 15,648,882 | 5.775 | Current best |
| **E023d** | **768** | **8** | **30,314,386** | **?** | **This experiment** |

30.3M params at ~1.9K training games = ~15,900 params/game. This is a 1.94x increase over E023b and nearly doubles the overfitting risk. Dropout=0.1 and weight_decay=1e-5 are the only regularization.

## What Changes

One config change from E023b: `n_layers: 8` (was 4).

- d_model stays 768, d_inner = 1536, num_heads = 24 (all unchanged)
- headdim = 64, d_state = 64 (all unchanged)
- All other hyperparameters identical

## Target Metrics

- **Keep:** RC < 5.775 (improvement over E023b)
- **Kill:** RC > 5.90 or val loss divergence (overfitting)

## Model

| Param | E023b | E023d |
|-------|-------|-------|
| d_model | 768 | 768 |
| d_state | 64 | 64 |
| n_layers | 4 | **8** |
| headdim | 64 | 64 |
| d_inner | 1536 | 1536 |
| num_heads | 24 | 24 |
| Total params | 15,648,882 | **30,314,386** |

## Training

Identical to E023b:
- lr: 0.0005, weight_decay: 1e-5, batch_size: 512, 1 epoch
- Self-Forcing: ratio=4 (20%), unroll_length=3
- context_len=30, chunk_size=15

## Cost

~$12-15 (A100 required). 1.94x more params than E023b means ~2x longer step time. E023b took ~4.1hr on A100; expect ~7-8hr for this run.

## Confounds

- 15,900 params/game ratio is very high (nearly 2x E023b's 8,200). Primary risk is overfitting. Watch for val loss divergence vs train loss.
- Dropout 0.1 and WD 1e-5 are the only regularization. If this overfits, a follow-up with higher dropout (0.15-0.2) or WD would isolate depth capacity vs regularization.
- VRAM usage will be higher. A100 40GB should have headroom but monitor for OOM, especially during self-forcing unrolls.

## Results

*Pending.*
