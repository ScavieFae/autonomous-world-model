# World Model Training Stages

How world model training pipelines are structured in the literature, and where we are.

## The General Pipeline

| Stage | What it learns | Data | Signal |
|-------|---------------|------|--------|
| **Pretrain** | How the world works — physics, dynamics, general patterns | Large-scale video/data | Self-supervised (predict next frame) |
| **Fine-tune** | How *this specific* world works | Domain-specific data | Supervised, narrower distribution |
| **RL / Reinforcement** | What "good" looks like — optimize beyond raw prediction | Interactive or labeled experience | Reward signal (task completion, constraint satisfaction, human feedback) |

Pretraining and fine-tuning teach the model to **predict**. RL teaches it to **prefer** — to be more accurate where it matters, or to satisfy constraints that raw prediction loss doesn't capture.

## Important Distinction: World Model vs. Policy

Most RL literature trains a **policy** (what to do). We train a **world model** (what happens). These are different:

- **Policy RL reward**: "Did you win? Did you complete the task?"
- **World model RL reward**: "Did you simulate a physically coherent state?"

We do NOT care whether the model predicts "good play." We care whether it predicts **possible play** — states that could actually exist in the game's physics. A model that accurately predicts a player getting combo'd is just as good as one that predicts a player landing a combo, as long as both are physically valid.

## Where We Are

```
Stage 1 — Pretrain: YES (current)
  Supervised next-frame prediction on Melee replay data.
  Self-Forcing adds autoregressive error exposure.
  fd-top5 dataset filtered by context (stage, characters), not by quality.

Stage 2 — Fine-tune: PARTIALLY
  Dataset is curated (top-5 characters, Final Destination).
  But filtering is by context, not by play quality or simulation difficulty.
  Could: weight by game quality, difficulty of sequences, known-hard transitions.

Stage 3 — RL-like: NOT YET
  No reward signal beyond reconstruction loss.
  No constraint checking on model output.
  This is the open frontier.
```

## What Stage 3 Could Look Like For Us

Since we're training a world model (not a policy), the "reward" is constraint satisfaction — did the model produce a state that could exist in Melee's physics?

### Detectable violations (automatic, no human needed)

- **Physics violations** — position deltas exceeding movement speed caps, characters below stage without falling/recovery action state
- **State machine violations** — illegal action state transitions (e.g., shield → aerial without jumpsquat)
- **Conservation violations** — stocks appearing/disappearing, percent decreasing without star KO
- **Continuity violations** — position teleportation, sudden velocity reversals without a hit

The game's rules are the reward model. These violations are programmatically checkable — Melee's state machine is well-documented.

### Possible approaches

1. **Constraint-weighted training** — roll out the model, check predictions against constraints, weight gradient higher on sequences that produce violations. No architecture change, just training regime.

2. **Curriculum by difficulty** — identify which replay sequences produce the worst rollout coherence, oversample them. Train harder on what's hard.

3. **RECAP-style loop (automated)** — deploy → roll out → detect violations → flag violation frames → retrain with higher weight on those frames → repeat. The "human corrector" is replaced by programmatic constraint checking.

4. **Auxiliary constraint loss** — add loss terms for specific physical constraints (e.g., "position delta must not exceed max speed"). Penalize violations directly rather than just through reconstruction loss.

### Prior art

- **RECAP / π₀.₆** (Physical Intelligence, arXiv 2511.14759) — deploy→fail→human corrects→retrain for robot VLAs. Our analog replaces human with programmatic constraint checker.
- **Self-Forcing** (Adobe/UT Austin, arXiv 2508.13009) — already addresses the train/inference gap. Stage 3 would build ON TOP of SF, not replace it.
- **DreamerV3** (DeepMind, 2023) — trains world model + policy jointly. We only need the world model half, but their reward shaping ideas may transfer.

### Open questions

- How much of rollout degradation is constraint violations vs. drift within valid states? If most errors are "wrong but physically possible," constraint-based reward won't help as much.
- Can we extract Melee's state transition table programmatically, or do we need to hand-code the constraints?
- Does weighting by violation severity outperform weighting by rollout coherence (which we already measure)?

## Deep Dives

- `.loop/knowledge/world-model-patterns.md` — training patterns from literature
- `research/sources/2508.13009-summary.md` — Self-Forcing paper (our current Stage 1+ technique)
- `docs/research-canon.md` — full paper list with paradigm context
