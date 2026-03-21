# Research Brief: Architecture Grid (Phase 1)

Systematically explore the model's width, depth, SSM state size, and head granularity. The current architecture (d_model=384, n_layers=4, d_state=64, headdim=64, ~4.3M params) was inherited without ablation. We don't know if the model is the right size or shape.

## Context

Read `program.md` section "Architecture exploration (ref: issue #6)" for the full grid design, Phase 2 decision rules, and Phase 3 structural changes (Phase 3 requires Mattie approval — don't attempt).

## Experiment Plan

Run against E018c baseline (SF + K=30, RC 6.03). Change ONE dimension at a time. Priority order:

### Tier 1 (run first — highest signal)
1. **d_model=192** (~1.1M params) — Is the model over-parameterized for 1.9K data?
2. **d_model=768** (~17M params) — Is the trunk too narrow?
3. **n_layers=2** (~2.2M params) — Can it learn with half the depth?
4. **n_layers=8** (~8.5M params) — Does more depth help?

### Tier 2 (run after Tier 1 results)
5. **d_state=32** — Less SSM memory
6. **d_state=128** — More SSM memory
7. **headdim=32** (24 heads) — Many independent streams
8. **headdim=128** (6 heads) — Few rich streams

### Tier 3 (conditional on Tier 1-2 results)
9. **dropout=0.0** — No regularization
10. **dropout=0.3** — Heavy regularization
11. **Combined scale-up** (d_model=768, n_layers=8, ~34M) — Is the model just too small?

## Per-Experiment Protocol

- Base config: `experiments/e018c-context-k30.yaml`
- Change ONE variable
- Train 1.9K data, 1 epoch, bs=512
- ~$5/experiment for standard sizes, ~$10 for 768/8-layer (more compute)
- Compare RC to 6.03

## Decision Rules (from program.md)

Follow the Phase 2 decision rules in program.md exactly. Key: if ALL Tier 1 experiments are within ±2% of baseline, architecture isn't the bottleneck — stop and report.

## Budget Note

This is ~$50-60 for the full grid. Run Tier 1 first ($20), evaluate, then decide on Tier 2. Don't blow the entire budget on architecture if Tier 1 shows it's not the bottleneck.
