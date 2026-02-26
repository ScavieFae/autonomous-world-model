# Autonomous World Model

A learned world model deployed onchain as an autonomous world on Solana. The model IS the physics engine — trained on millions of frames of competitive Melee replay data, quantized to INT8, running deterministically in [MagicBlock](https://www.magicblock.xyz/) ephemeral rollups.

No game logic. No hardcoded physics. Just 15 million integers being multiplied together 60 times a second.

## How It Works

```
Melee replays (millions of frames)
        |
Mamba2 model trained on replay data (nojohns-training)
        |
INT8 quantization ──> weights stored on Solana mainnet (permanent, forkable)
        |
Ephemeral rollup session ──> 60fps onchain inference
        |
WebSocket ──> browser visualizer
```

**Weights on mainnet** are the permanent layer — the "cartridges" on the arcade shelf. Anyone can read them, fork them, retrain a variant, deploy alternate physics.

**Sessions in ephemeral rollups** are the gameplay layer — spin up on demand, 15ms blocks, zero fees, custom compute. Controller inputs go in, game state comes out. When the match ends, results settle back to mainnet permanently.

## Architecture

| Layer | What | Where |
|-------|------|-------|
| **Weights** | ~15MB INT8 (2 shards) + manifest with LUTs | Solana mainnet |
| **Sessions** | SessionState + HiddenState + InputBuffer | MagicBlock ephemeral rollup |
| **Inference** | Mamba2 forward pass (12 layers, 24 matmuls/frame) | Onchain BPF + `sol_matmul_i8` syscall |
| **Rendering** | Canvas 2D wireframe visualizer | Browser |

### Why Mamba2?

Fixed-size hidden state. Cost per frame is constant whether it's frame 1 or frame 10,000. No attention, no growing context window. The ideal shape for a computation that repeats 60 times a second.

### Why INT8?

**Size:** 15M params in fp32 = 60MB. In INT8 = 15MB.
**Determinism:** Integer math is identical on every machine. Float math isn't. When your physics engine runs onchain and multiple parties must agree on the result, exact reproducibility matters.

## Project Structure

```
autonomous-world-model/
├── viz/                  # Canvas 2D visualizer (working — renders frames from JSON or WebSocket)
├── quantization/         # FP32 → INT8 quantization pipeline + LUT generation (Python)
├── solana/
│   ├── programs/
│   │   └── cu-benchmark/ # CU measurement for INT8 ops on Solana VM
│   ├── programs-ecs/
│   │   ├── components/   # BOLT ECS: WeightShard, ModelManifest, SessionState,
│   │   │                 #           HiddenState, InputBuffer, FrameLog
│   │   └── systems/      # submit-input, run-inference, session-lifecycle
│   ├── client/           # TypeScript SDK (@awm/client)
│   └── cli/              # Weight upload CLI
└── docs/                 # Architecture specs, benchmarks, design docs
```

## Key Numbers

| Metric | Value |
|--------|-------|
| Model parameters | ~15M (INT8) |
| Architecture | Mamba2 (d_model=512, d_inner=1024, d_state=16, 12 layers) |
| MACs per frame | ~18.9M |
| CU per frame (BPF, packed) | ~300M (too high) |
| CU per frame (with `sol_matmul_i8` syscall) | ~8.7M |
| Target frame rate | 60fps |
| Weight storage | ~15MB across 2 onchain accounts |
| Hidden state | ~200KB (Mamba2 recurrent memory) |
| Onchain rent (recoverable) | ~104 SOL for 15M model |

## Documentation

| Doc | What |
|-----|------|
| [Architecture Overview](docs/architecture-overview.md) | Full system design — the experience, layer model, frame loop, rationale |
| [CU Benchmark Findings](docs/cu-benchmark-findings.md) | Measured costs for matmul, LUT activations, SSM scan on Solana VM |
| [Packed Matmul Explainer](docs/packed-matmul-explainer.md) | How we got 1.52x speedup with packed u32 loads + unsafe |
| [`sol_matmul_i8` Spec](docs/sol-matmul-i8-spec.md) | Technical spec for native INT8 matmul syscall in ephemeral rollups |
| [Path A: Agave Syscall](docs/path-a-agave-syscall.md) | Fork the validator, add the syscall yourself |
| [Path B: MagicBlock ER](docs/path-b-magicblock-er.md) | Run in MagicBlock ephemeral rollups with their infra |
| [Arena Mechanics](docs/design-arena-mechanics.md) | "The Wire" — agent arena, sponsorship economy, Tapestry social layer |
| [Visual & UX Design](docs/design-visual-ux.md) | Wireframe fighters, render modes, page layouts |
| [Site Map](docs/site-map.md) | Full frontend spec — routes, page states, component inventory, user flows |

## Quick Start

```bash
# View the visualizer (runs in-browser, no deps)
open viz/visualizer.html

# Or the juicy version with motion trails, screen shake, hit flash
open viz/juicy.html
```

The visualizer works with mock data out of the box. Three scenarios: neutral game, combo sequence, edgeguard. Keyboard: `Space` = play/pause, `arrows` = frame step, `+`/`-` = speed.

## Related Projects

| Project | What |
|---------|------|
| [nojohns](https://github.com/ScavieFae/nojohns) | Model code, arena, community platform |
| [nojohns-training](https://github.com/ScavieFae/nojohns-training) | Data pipeline, training runs, parsed replay data |

## The Concept

The model IS the autonomous world. Learned rules become ground truth. The "errors" aren't bugs — they're the physics of a new world whose rules emerged from training data.

- **Arcade, not MMO.** Persistent weights/rules, not persistent world state. Sessions spin up and tear down.
- **Forkable physics.** Train on different data, deploy alternate physics. Street Fighter replays produce Street Fighter physics. The arcade grows.
- **Composable.** Agents, prediction markets, mods — all calling the same onchain physics engine. Weights are just Solana accounts.
