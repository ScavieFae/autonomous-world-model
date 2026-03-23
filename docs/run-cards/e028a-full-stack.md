---
id: e028a
created: 2026-03-23
status: kept
type: training-regime
base_build: b002
built_on: [e026b, e026c]
source_paper: null
rollout_coherence: 4.798
prior_best_rc: 4.965
---

# Run Card: e028a-full-stack

## Goal

Validate that unimix (E026b) and SF curriculum (E026c) compound before committing to the 7.7K scaling run. This is a combination test — both techniques were kept independently, but they might interfere. If RC improves, this recipe becomes the basis for the scaling run.

## Context

| Experiment | Approach | RC | Notes |
|---|---|---|---|
| E026b | 1% unimix on categoricals | 5.120 | Kept — prevents overconfident collapse |
| E026c | SF curriculum N=1->2->3 (2 epochs) | 4.965 | Kept — largest change_acc gain ever |
| E027c | Loss reweight warm-start + standard epoch 2 | 4.939 | Kept — but h10 percent_mae 4.53 (e025b warm-start destroyed percent) |
| **E028a** | **Unimix + curriculum (full b002 stack)** | **4.798** | **KEPT — new best, -3.4%** |

## What Changes

From b002 baseline: combine both kept improvements that build on b002.

- `unimix_ratio: 0.01` (from E026b)
- `self_forcing.curriculum: [1, 2, 3]` with `num_epochs: 2` (from E026c)

No warm-start, no regime switching. Clean b002 + both kept techniques.

## Target Metrics

- **Keep:** RC < 4.965 (improvement over e026c — compounding confirmed)
- **Kill:** RC > 5.2 (interference between techniques)

## Model

Identical to b002: d_model=768, d_state=64, n_layers=4, headdim=64, 15,794,548 params.

## Training

- lr: 5e-4, weight_decay: 1e-5, batch_size: 512
- warmup_pct: 0.05, AMP: enabled
- Self-Forcing: ratio=4 (20%), curriculum=[1, 2, 3]
- context_len: 30
- 2 epochs, A100 40GB

## Cost

~$10 (A100 with AMP, 2 epochs).

## Confounds

- Two changes vs e026c (adds unimix) and vs e026b (adds curriculum + 2 epochs). But both were tested independently — this is a combination test, not a novel hypothesis.
- Comparison to e027c is complicated: e027c used e025b warm-start which destroyed percent tracking (h10 percent_mae 4.53 vs e026c's 1.70). e028a doesn't use warm-start.

## Results

**KEPT — new best. RC 4.798 (-3.4% vs e026c's 4.965). Unimix + curriculum compound.**

| Metric | E028a | E026c (prior best on full basket) | E027c (prior best RC-only) | Delta vs E026c |
|--------|-------|-----------------------------------|---------------------------|----------------|
| **rollout_coherence** | **4.798** | 4.965 | 4.939 | **-3.4%** |
| change_acc | 80.1% | 80.2% | 78.9% | -0.1pp |
| pos_mae | 0.618 | 0.642 | 0.650 | -3.7% |

- **wandb:** yqcjtzey
- **Cost:** ~$10 (A100 with AMP, 2 epochs)
- **Checkpoint:** e028a-full-stack/best.pt

### Per-Horizon Metrics

| Horizon | pos_mae | action_acc | percent_mae |
|---------|---------|------------|-------------|
| h1 | 0.522 | 99.2% | 0.214 |
| h5 | 2.386 | 91.0% | 1.122 |
| h10 | 4.560 | 78.7% | 1.849 |
| h15 | 6.876 | 65.3% | 2.771 |
| h20 | 8.878 | 53.0% | 3.636 |

### Percent Tracking Analysis (issue #21)

h10 percent_mae comparison across recent bests:

| Experiment | h10 percent_mae | Notes |
|---|---|---|
| E026c | 1.701 | Best percent tracking |
| **E028a** | **1.849** | Close to e026c — unimix + curriculum preserve percent |
| E027c | 4.527 | e025b warm-start destroyed percent tracking |

E028a maintains good percent tracking (h10: 1.849, +8.7% vs e026c's 1.701). This confirms that the e025b warm-start in e027c was the source of percent destruction, not unimix or curriculum. E028a is the true best on the full metric basket — best RC AND good percent tracking.

### Analysis

Unimix and SF curriculum are orthogonal improvements that compound cleanly. The 3.4% RC improvement vs e026c (which already had curriculum but no unimix) is larger than e026b's standalone 0.5% unimix contribution. The curriculum amplifies unimix's benefit: preventing overconfident categorical collapse matters more during multi-step AR unrolls (curriculum stages N=2, N=3) where error cascades through categorical predictions.

change_acc is flat (80.1% vs 80.2%) — the curriculum already captured most of the categorical benefit. The improvement is in position tracking: pos_mae 0.618 vs 0.642 (-3.7%), meaning unimix's regularization of categoricals translates to better continuous predictions downstream.

This is the recipe for the 7.7K scaling run: b002 + unimix + curriculum.

Cumulative RC improvement from E019 baseline: -29.1% (6.77 -> 4.798).
