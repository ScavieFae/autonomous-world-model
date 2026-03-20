---
id: e023c
created: 2026-03-20
status: running
type: architectural
base_build: b001
built_on: [e023b]
source_paper: null
rollout_coherence: null
prior_best_rc: 5.775
---

# Run Card: e023c-dmodel512

## Goal

Phase 2 — find the efficient frontier between d_model=384 (RC 6.026) and d_model=768 (RC 5.775). The width axis is monotonic so far (192 < 384 < 768). d_model=512 is the midpoint: does it capture most of 768's gain at ~46% the param cost?

This matters for deployment. Fewer params = smaller onchain weight accounts, faster upload, lower CU per frame. If 512 gets within ~1% of 768, it's the better production choice.

## Context

| Config | d_model | d_inner | num_heads | Params | RC | Notes |
|--------|---------|---------|-----------|--------|----|-------|
| E023a | 192 | 384 | 6 | ~1.3M | 6.065 | Underfitting |
| E018c | 384 | 768 | 12 | 4,275,810 | 6.026 | Former best |
| **E023c** | **512** | **1024** | **16** | **7,276,306** | **?** | **This experiment** |
| E023b | 768 | 1536 | 24 | 15,648,882 | 5.775 | Current best |

7.3M params at ~1.9K training games = ~3,800 params/game. Between E018c (2,250) and E023b (8,200). Moderate overfitting risk.

## What Changes

One config change from E023b: `d_model: 512` (was 768).

- d_inner = 2 * 512 = 1024
- num_heads = 1024 / 64 = 16 (headdim=64 unchanged)
- All other hyperparameters identical

## Target Metrics

- **Keep:** RC < 5.775 (improvement over E023b) or RC within ~2% of 5.775 with clear efficiency advantage
- **Kill:** RC > 6.03 (regression past E018c baseline)

## Model

| Param | E018c | E023c | E023b |
|-------|-------|-------|-------|
| d_model | 384 | **512** | 768 |
| d_state | 64 | 64 | 64 |
| n_layers | 4 | 4 | 4 |
| headdim | 64 | 64 | 64 |
| d_inner | 768 | 1024 | 1536 |
| num_heads | 12 | 16 | 24 |
| Total params | 4,275,810 | **7,276,306** | 15,648,882 |

## Training

Identical to E023b:
- lr: 0.0005, weight_decay: 1e-5, batch_size: 512, 1 epoch
- Self-Forcing: ratio=4 (20%), unroll_length=3
- context_len=30, chunk_size=15

## Cost

~$6 Scout tier (A100 40GB). 7.3M params is between E018c (4.3M, ~$4) and E023b (15.6M, ~$8.70). Expect ~2.5hr runtime.

## Confounds

- 3,800 params/game is moderate. Less overfitting risk than E023b (8,200) but watch val loss vs train loss.
- If RC lands between 384 and 768, the width-RC curve shape tells us whether returns are diminishing (concave) or accelerating (convex). This informs whether to test d_model=1024 next.

## Results

_Pending._
