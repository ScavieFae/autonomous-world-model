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
├── site/             # "The Wire" — Next.js arena website (ScavieFae)
├── viz/              # State visualizer — renders world model output (Scav)
├── quantization/     # INT8 quantization + accuracy testing (Scav)
├── solana/           # Onchain code (ScavieFae)
│   ├── syscall/      # sol_matmul_i8 native syscall implementation
│   ├── programs/     # Solana programs (world-model, cu-benchmark, syscall-test)
│   ├── programs-ecs/ # BOLT ECS components + systems
│   ├── client/       # TypeScript SDK (@awm/client)
│   └── cli/          # Upload CLI tool
└── docs/             # Architecture decisions, specs, handoff (shared)
```

## Key Concepts

- **The model IS the world.** Learned rules become ground truth. Errors aren't bugs — they're the physics of a new world.
- **Arcade, not MMO.** Persistent weights/rules, not persistent world state. Sessions spin up in ephemeral rollups.
- **INT8 determinism for free.** Integer math is identical everywhere. Quantization solves both size and determinism.

## Two-Agent Development

This project is developed by two Claude Code agents. **Check which agent you are before editing files.**

| Agent | Role | Owns | Branch prefix |
|-------|------|------|---------------|
| **ScavieFae** | Website, smart contracts, onchain programs | `site/`, `solana/` | `scaviefae/` |
| **Scav** | Model training, quantization, inference implementation | `quantization/`, `viz/` | `scav/` |

### How to Know Which Agent You Are

- **ScavieFae**: Working on the website, Solana programs, BOLT ECS, syscall code, or deployment. On the deployment machine.
- **Scav**: Working on model training, quantization pipeline, or inference logic. On the training machine.

If unclear, ask Mattie.

### Directory Ownership

| Directory | Owner | Notes |
|-----------|-------|-------|
| `site/` | **ScavieFae** | Next.js website ("The Wire") |
| `solana/` | **ScavieFae** | All onchain code — programs, syscall, ECS, client SDK |
| `quantization/` | **Scav** | INT8 quantization pipeline |
| `viz/` | **Scav** | State visualizer, render modes |
| `docs/` | **Shared** | Both agents can edit |

**Do not edit files in another agent's directories.** If you need a change in their code, describe what you need in the handoff doc or a PR comment.

### Branches & Review Process

Two long-lived branches: **`main`** (public-facing) and **`dev`** (integration). Agents work on prefixed branches off `dev`, PRs merge into `dev`, curated releases go `dev` → `main`.

**Review is mandatory for:**
- Onchain programs before any deploy (Scav reviews ScavieFae's Solana code)
- Quantization pipeline changes that affect weight format (ScavieFae reviews Scav's changes)
- Any change to the shared interface between training output and onchain inference
- Large features — open a PR, request review from the other agent

**Handoff doc**: `docs/HANDOFF.md` is the coordination point. Use it for:
- Review requests (what changed, what to look at, what questions remain)
- Review responses (approvals, concerns, action items)
- Status updates on blockers or external dependencies (e.g., MagicBlock)

**Shared schemas**: The weight format (INT8 layout, shard structure, manifest schema) is the contract between quantization and onchain code. Changes require coordination via handoff doc.

### For ScavieFae

You own the website and all Solana code. Read `docs/HANDOFF.md` for current status and review items.

### For Scav

You own model training, quantization, and the visualizer. Read `docs/HANDOFF.md` for review requests from ScavieFae.

## Reference Docs

| Doc | What's in it |
|-----|-------------|
| [docs/HANDOFF.md](docs/HANDOFF.md) | Active handoff — review requests, responses, status |
| [docs/sol-matmul-i8-spec.md](docs/sol-matmul-i8-spec.md) | `sol_matmul_i8` syscall spec (shared with MagicBlock) |
| [docs/architecture-overview.md](docs/architecture-overview.md) | System architecture |
| [docs/cu-benchmark-findings.md](docs/cu-benchmark-findings.md) | CU measurements for INT8 ops |
| [docs/design-arena-mechanics.md](docs/design-arena-mechanics.md) | "The Wire" arena design |
| [docs/design-visual-ux.md](docs/design-visual-ux.md) | Visual/UX design for The Wire |

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
