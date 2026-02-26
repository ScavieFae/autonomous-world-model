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
├── models/           # PyTorch inference code — Mamba2, PolicyMLP (Scav)
├── crank/            # Offchain match runner — standalone + Solana (Scav)
├── viz/              # State visualizer — renders world model output (Scav)
├── quantization/     # INT8 quantization + accuracy testing (Scav)
├── solana/           # Onchain code (Codex)
│   ├── syscall/      # sol_matmul_i8 native syscall implementation
│   ├── programs/     # Solana programs (world-model, cu-benchmark, syscall-test)
│   ├── programs-ecs/ # BOLT ECS components + systems
│   ├── client/       # TypeScript SDK (@awm/client)
│   ├── cli/          # Upload CLI tool
│   └── tests/        # Integration tests (Mocha)
└── docs/             # Architecture decisions, specs, handoff (shared)
```

## Key Concepts

- **The model IS the world.** Learned rules become ground truth. Errors aren't bugs — they're the physics of a new world.
- **Arcade, not MMO.** Persistent weights/rules, not persistent world state. Sessions spin up in ephemeral rollups.
- **INT8 determinism for free.** Integer math is identical everywhere. Quantization solves both size and determinism.

## Three-Agent Development

This project is developed by three agents. **Check which agent you are before editing files.**

| Agent | Platform | Role | Owns |
|-------|----------|------|------|
| **ScavieFae** | Claude Code | Website, UX, design, overall experience | `site/` |
| **Scav** | Claude Code | Model training, quantization, inference, offchain crank | `models/`, `crank/`, `quantization/`, `viz/` |
| **Codex** | OpenAI Codex | Smart contracts, onchain programs, client SDK | `solana/` |

### How to Know Which Agent You Are (Claude agents only)

- **ScavieFae**: Working on the website, UX, design, product experience, or frontend integration.
- **Scav**: Working on model training, quantization pipeline, inference logic, or the offchain match runner.

If unclear, ask Mattie. If you're Codex, see `AGENTS.md`.

### Directory Ownership

| Directory | Owner | Notes |
|-----------|-------|-------|
| `site/` | **ScavieFae** | Next.js website ("The Wire") |
| `models/` | **Scav** | PyTorch inference code (Mamba2, PolicyMLP) |
| `crank/` | **Scav** | Offchain match runner (standalone + Solana bridge) |
| `quantization/` | **Scav** | INT8 quantization pipeline |
| `viz/` | **Scav** | State visualizer, render modes |
| `solana/` | **Codex** | All onchain code — programs, syscall, ECS, client SDK, tests |
| `docs/` | **Shared** | All agents can edit |

**Do not edit files in another agent's directories.** If you need a change in their code, describe what you need in `docs/HANDOFF.md`.

### Interface Contracts

These are the shared boundaries between agents. Changes require coordination via handoff doc.

**1. Scav ↔ Codex: Binary wire format**
- `PlayerState` = 32 bytes, field order matches Rust `AnchorSerialize`
- `crank/solana_bridge.py` and `solana/programs-ecs/components/session-state/` must agree byte-for-byte
- `models/encoding.py` (`EncodingConfig`) defines all normalization scales and vocab sizes
- Fixed-point: positions/velocities × 256, percent stored directly as u16

**2. ScavieFae ↔ Codex: TypeScript SDK surface**
- `solana/client/src/` exports functions and types that `site/` imports
- Codex owns the SDK implementation; ScavieFae consumes it as a dependency
- Function signatures and type shapes are the contract — changes require handoff

**3. ScavieFae ↔ Scav: JSON frame format**
- `{ meta, stage_geometry, frames[] }` — consumed by `viz/` and `site/` renderers
- Already stable. See `viz/visualizer.html` for the exact schema.

### Review Gates

| What changed | Who reviews | Why |
|--------------|-------------|-----|
| Onchain programs, syscall, ECS | Scav reviews math/format | Hardest to undo, must match model |
| Client SDK type changes | ScavieFae reviews | They're the consumer |
| Weight format / encoding changes | Codex reviews | Must match onchain structs |
| Binary wire format (PlayerState, etc.) | All three | Shared boundary |
| Site UX | No gate | Iterative, reversible |
| Model / crank code | No gate | Offchain, testable |

### Coordination

**Handoff doc**: `docs/HANDOFF.md` is the coordination point for all three agents:
- Review requests (what changed, what to look at, what questions remain)
- Review responses (approvals, concerns, action items)
- Status updates on blockers or external dependencies (e.g., MagicBlock)

### For ScavieFae

You own the website and overall product experience. You consume `@awm/client` (Codex's SDK) and `viz/` output format (Scav's). Read `docs/HANDOFF.md` for current status.

### For Scav

You own model training, quantization, inference, and the offchain crank. Your binary format in `crank/solana_bridge.py` must match Codex's Rust structs byte-for-byte. Read `docs/HANDOFF.md` for review requests.

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
