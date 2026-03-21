# Experiment Index

*15 experiments — 3 kept, 2 running, 0 proposed, 9 discarded.*

*Generated 2026-03-20 20:04 UTC*

**Best rollout coherence:** 5.775 ([e023b](../run-cards/e023b-dmodel768.md))

## Running

| ID | Type | Base | RC | Built On | Paper |
|-----|------|------|----|----------|-------|
| [e018b](../run-cards/e018b-rollout-coherence-eval.md) | training-regime | b001 | — | e017a | — |
| [e019](../run-cards/e019-baseline.md) | training-regime | b001 | — | — | — |

## Kept

| ID | Type | Base | RC | Built On | Paper |
|-----|------|------|----|----------|-------|
| [e018a](../run-cards/e018a-self-forcing.md) | training-regime | b001 | 6.26 | e019 | [2508.13009](https://arxiv.org/abs/2508.13009) |
| [e018c](../run-cards/e018c-rolling-context-window.md) | architectural | b001 | 6.03 | e018a | [2505.20171](https://arxiv.org/abs/2505.20171) |
| [e023b](../run-cards/e023b-dmodel768.md) | architectural | b001 | 5.775 | e018c | — |

## Discarded

| ID | Type | Base | RC | Built On | Paper |
|-----|------|------|----|----------|-------|
| [e018b](../run-cards/e018b-self-forcing-n5.md) | training-regime | b001 | 6.45 | e018a | [2508.13009](https://arxiv.org/abs/2508.13009) |
| [e018d](../run-cards/e018d-horizon-weighted-loss.md) | training-regime | b001 | 6.81 | e018a | [2508.13009](https://arxiv.org/abs/2508.13009) |
| [e019a](../run-cards/e019a-context-k50.md) | architectural | b001 | 5.97 | e018c | [2505.20171](https://arxiv.org/abs/2505.20171) |
| [e020a](../run-cards/e020a-sf-ratio-10.md) | training-regime | b001 | 6.62 | e018c | [2508.13009](https://arxiv.org/abs/2508.13009) |
| [e020b](../run-cards/e020b-sf-ratio-30.md) | training-regime | b001 | 6.289 | e018c | [2508.13009](https://arxiv.org/abs/2508.13009) |
| [e021b](../run-cards/e021b-selective-bptt.md) | training-regime | b001 | 6.87 | e018c | — |
| [e022a](../run-cards/e022a-bs256.md) | hyperparameter | b001 | 6.026 | e018c | — |
| [e023a](../run-cards/e023a-dmodel192.md) | architectural | b001 | 6.065 | e018c | — |
| [e023c](../run-cards/e023c-dmodel512.md) | architectural | b001 | 6.203 | e023b | — |
