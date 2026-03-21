---
id: e024a
created: 2026-03-21
status: discarded
type: training-regime
base_build: b001
built_on: [e023b]
source_paper: null
rollout_coherence: 8.980
prior_best_rc: 5.775
---

# Run Card: e024a-full-bptt

## Goal

Test whether full backpropagation through Self-Forcing unroll steps improves rollout coherence. Currently SF truncates gradients between steps (per-step backward, `torch.no_grad()` on reconstruction). This prevents the model from learning multi-step error correction — e.g., "my position error at step 1 cascades into an action mispredict at step 3."

Full BPTT enables cross-step gradient flow via:
1. **Differentiable float reconstruction** — continuous deltas and sigmoid binary predictions keep gradients flowing through the float context.
2. **Soft categorical embeddings** — `softmax(logits) @ embed.weight` replaces `embed(argmax(logits))` for the reconstructed frame's action/jumps, making the categorical path differentiable.
3. **Single backward** — total loss accumulated across all N steps, one backward call. Gradients flow through the entire unroll chain.

## Context

| Experiment | Technique | RC | Notes |
|---|---|---|---|
| E018a | Truncated BPTT SF, N=3 | 6.26 | First SF, -7.5% |
| E018b | Truncated BPTT SF, N=5 | 6.45 | Regressed — longer unroll hurt with truncated BPTT |
| E021b | Selective BPTT (detach categoricals) | 6.87 | Regressed -13.9% — categoricals are structurally important |
| E023b | Truncated BPTT SF, N=3, d_model=768 | 5.775 | Current best |
| **E024a** | **Full BPTT SF, N=3, d_model=768** | **?** | **This experiment** |

E018b showed N=5 regresses with truncated BPTT. The hypothesis is that full BPTT unlocks longer effective horizons because gradient signal doesn't degrade between steps.

E021b showed detaching categoricals fails. This experiment is the opposite — it makes categoricals MORE connected via soft embeddings, not less.

## What Changes

One config change from E023b: `self_forcing.full_bptt: true`.

Code changes:
- `scripts/ar_utils.py`: New `reconstruct_frame_differentiable()` — functional float construction (no in-place ops), sigmoid binary, returns categorical logits separately
- `models/mamba2.py`: New `_soft_embed()` static method, `_encode_frames()` accepts `soft_cat_last` dict for differentiable last-frame encoding
- `training/trainer.py`: New `_self_forcing_step_full_bptt()` — accumulates total loss tensor, single backward, passes soft categoricals between steps

## Target Metrics

- **Keep:** RC < 5.775 (improvement over E023b)
- **Kill:** RC > 6.0 (regression) or CUDA OOM

## Model

Identical to E023b: d_model=768, d_state=64, n_layers=4, headdim=64. 15.8M params.

## Training

Identical to E023b except full BPTT:
- lr: 0.0005, weight_decay: 1e-5, batch_size: 512, 1 epoch
- Self-Forcing: ratio=4 (20%), unroll_length=3, **full_bptt=true**
- context_len=30, chunk_size=15

## Cost

~$8-10 (A100 40GB). Full BPTT keeps more computation graph in memory but batch_size unchanged. May be slightly slower due to larger backward pass.

## Confounds

- **VRAM:** Full BPTT retains N forward passes' computation graphs simultaneously. If OOM, fall back to gradient checkpointing or reduce batch_size.
- **Gradient scale:** Soft embeddings and differentiable reconstruction may produce different gradient magnitudes than truncated BPTT. Watch for gradient norm spikes via wandb.
- **Sigmoid vs threshold:** Binary predictions use sigmoid (0-1 continuous) during SF reconstruction but threshold at eval. Small representational mismatch.

## Risk

Medium. The code change is clean (no existing behavior modified, new method dispatched via config flag). Fallback is trivial: set `full_bptt: false`.

## Results

| Metric | E023b (truncated BPTT) | E024a (full BPTT) | Delta |
|---|---|---|---|
| **rollout_coherence** | **5.775** | **8.980** | **+55.5% (regressed)** |
| change_acc | 66.0% | 64.9% | -1.1pp |
| pos_mae | 0.823 | 1.123 | +36.5% |
| p0_action_acc | 95.8% | — | — |
| sf_loss | 0.618 | 0.443 | -28.3% (misleading) |

Runtime: 14745s (~4.1hr) on A100 40GB. Cost: ~$8.60.
wandb: https://wandb.ai/shinewave/melee-worldmodel/runs/udy9ail3

## Director Evaluation

**Verdict: DISCARDED — catastrophic regression.**

RC 8.980 is a 55.5% regression, the worst result in the experiment series. The lower sf_loss (0.443 vs 0.618) is misleading: the model optimized for the soft-embed reconstruction path (softmax @ embed.weight) during training, but eval uses hard argmax. This train/eval mismatch is the likely culprit.

The soft embedding approach creates a fundamentally different optimization landscape: the model learns to produce logit distributions that minimize loss through weighted-average embeddings, rather than learning to produce sharp, correct categorical predictions. At eval time, the hard argmax snaps to the single highest logit, which may not be the same as what the soft average was optimizing for.

**Key finding:** Differentiable categorical reconstruction via soft embeddings doesn't work as a drop-in replacement for truncated BPTT. The train/eval mismatch is fatal. Future attempts at full BPTT should either:
1. Use temperature annealing (start soft, anneal toward hard argmax during training)
2. Use straight-through estimator (hard argmax forward, gradient approximation backward) instead of soft embeddings
3. Only flow gradients through the continuous reconstruction path (but E021b showed this fails too)
