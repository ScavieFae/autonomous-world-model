# Autonomous World Model

A learned world model deployed onchain as an autonomous world. Trained on Melee replay data via the No Johns project, deployed on Solana using MagicBlock's ephemeral rollups and BOLT ECS.

## Architecture

```
nojohns / nojohns-training  →  trained weights (float32)
                                      ↓
autonomous-world-model       →  quantize → deploy → render
```

This repo is the right side of that arrow. Everything that takes trained weights and puts them onchain.

## Project Structure

```
autonomous-world-model/
├── viz/              # State visualizer — renders world model output
├── quantization/     # INT8 quantization + accuracy testing
├── solana/           # BOLT ECS scaffold + onchain inference programs
└── docs/             # Architecture decisions, specs
```

## Key Concepts

- **The model IS the world.** Learned rules become ground truth. Errors aren't bugs — they're the physics of a new world.
- **Arcade, not MMO.** Persistent weights/rules, not persistent world state. Sessions spin up in ephemeral rollups.
- **INT8 determinism for free.** Integer math is identical everywhere. Quantization solves both size and determinism.

## Related Projects

- **rnd-2026** — research docs (`llms/world-models.md`, `projects/autonomous-world-model/README.md`)
- **nojohns** — model code, arena, community platform
- **nojohns-training** — data pipeline, training runs, parsed replay data

## Model Output Format

The world model outputs one frame per timestep. Each frame contains per-player state matching the v2 encoding:

**Continuous (regression heads):** x, y, percent, shield_strength, speed_air_x, speed_y, speed_ground_x, speed_attack_x, speed_attack_y, state_age, hitlag, stocks

**Binary (classification):** facing, on_ground

**Categorical (classification heads):** action_state (400-class), jumps_left (8-class), character (33-class pass-through)

**Per-frame:** stage (33-class pass-through)

See `viz/visualizer.html` for the exact JSON shape consumed by the rendering layer.

## Onchain Target

- **Solana** via MagicBlock BOLT ECS + Ephemeral Rollups
- Weights stored on mainnet (permanent, forkable)
- Sessions in ephemeral rollups (10ms blocks, configurable CU, zero fees)
- 60fps achievable: 16.67ms frame budget - ~0.5-2ms inference - ~10ms block time
