# Autonomous World Model

A learned world model deployed onchain as an autonomous world on Solana.

Trained on competitive Melee replay data via [No Johns](https://github.com/your-org/nojohns). Deployed using [MagicBlock](https://www.magicblock.xyz/) ephemeral rollups and BOLT ECS.

## What's here

| Directory | What | Status |
|-----------|------|--------|
| `viz/` | State visualizer — renders model output as interactive stage view | Working (mock data) |
| `quantization/` | INT8 quantization pipeline + accuracy benchmarks | Planned |
| `solana/` | BOLT ECS scaffold + onchain inference programs | Planned |

## Quick start

```bash
# Open the visualizer
open viz/visualizer.html
```

The visualizer runs entirely in-browser with mock data. Three scenarios: neutral game, combo sequence, edgeguard. Keyboard: Space=play, arrows=step, +/-=speed.

## The concept

The world model IS the autonomous world. Learned rules become ground truth — the "errors" aren't bugs, they're the physics of a new world whose rules emerged from training data.

- **Arcade framing** — persistent weights/rules, not persistent world state
- **Forkable** — train on different data, fork the weights, the physics diverge
- **Composable** — agents, prediction markets, mods all call the same physics engine

## Deep research

Full research docs live in [rnd-2026](../rnd-2026/):
- [`llms/world-models.md`](../rnd-2026/llms/world-models.md) — world model fundamentals, taxonomy, MagicBlock analysis
- [`projects/autonomous-world-model/README.md`](../rnd-2026/projects/autonomous-world-model/README.md) — concept exploration, architecture comparison, character generation pipeline
