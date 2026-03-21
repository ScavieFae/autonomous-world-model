# Research Brief: Full BPTT Through Self-Forcing Steps

Investigate and implement full backpropagation through `reconstruct_frame()` in Self-Forcing steps, removing the gradient truncation that currently limits unroll effectiveness.

## Context

Currently, gradients are detached between SF steps (N=3 truncated BPTT). This limits what the model can learn about multi-step error recovery. E018b showed N=5 with truncated BPTT regressed — but full BPTT might unlock longer unrolls by letting the model learn to correct errors across multiple steps.

Read `program.md` section "Engineering directions → 1. Full backpropagation through Self-Forcing steps" for full context.

## Research Questions

1. **What is the current gradient truncation mechanism?** Find where in the training code gradients are detached between SF steps. Document the exact code path.
2. **What are the non-differentiable operations in `reconstruct_frame()`?** Identify every argmax or discrete operation that blocks gradient flow.
3. **Which relaxation approach is best for our case?** Evaluate: (a) Gumbel-softmax for categorical heads, (b) straight-through estimator, (c) gradient flow only through continuous heads (partial BPTT). Literature search for which works best in similar autoregressive settings.
4. **Memory implications?** Estimate VRAM requirements for full BPTT with N=3 and N=5 on A100 40GB. Will gradient checkpointing be needed?

## Deliverables

1. A detailed implementation plan for the most promising approach (written to a proposed run card)
2. Memory/compute estimates
3. Literature references if relevant papers exist on differentiable world model training

## Scope

This is a RESEARCH brief — investigate, plan, and propose. Do NOT implement code changes. The output should be one or more proposed run cards with `status: proposed`.
