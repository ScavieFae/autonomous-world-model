# AGENTS.md — Codex

You are the **onchain agent** for autonomous-world-model. You own all Solana code: programs, BOLT ECS components and systems, the custom syscall, the TypeScript client SDK, CLI tools, and integration tests.

## What You Own

```
solana/
├── programs/         # Anchor programs (world-model, cu-benchmark, syscall-test)
├── programs-ecs/     # BOLT ECS components (6) + systems (3)
├── syscall/          # sol_matmul_i8 native syscall for MagicBlock ER
├── client/           # TypeScript SDK (@awm/client) — session, state, input
├── cli/              # Weight upload CLI tool
└── tests/            # Integration tests (Mocha)
```

## What You Must Not Touch

| Directory | Owner | Why |
|-----------|-------|-----|
| `site/` | ScavieFae (Claude) | Website, UX, frontend |
| `models/` | Scav (Claude) | PyTorch model code |
| `crank/` | Scav (Claude) | Offchain match runner |
| `quantization/` | Scav (Claude) | INT8 quantization pipeline |
| `viz/` | Scav (Claude) | Standalone visualizer |

If you need a change in another agent's code, describe what you need in `docs/HANDOFF.md`.

`docs/` is shared — all agents can read and write.

## Project Context

A Mamba2 state space model trained on Super Smash Bros. Melee replays runs as an onchain physics engine. The product is **The Wire** — an AI agent arena where agents fight inside the learned world model, humans spectate and sponsor.

**Architecture overview**: `docs/architecture-overview.md`
**Full technical context**: `docs/CODEX-BRIEF.md`
**Current status and review items**: `docs/HANDOFF.md`

## Interface Contracts

These are the shared boundaries with other agents. Changes require coordination via `docs/HANDOFF.md`.

### Contract 1: Binary Wire Format (with Scav)

The Python offchain crank (`crank/solana_bridge.py`) reads and writes Solana accounts. The binary layout must match your Rust structs byte-for-byte.

**PlayerState = 32 bytes:**
```
Offset  Type   Field
0       i32    x                (fixed-point x256)
4       i32    y                (fixed-point x256)
8       u16    percent
10      u16    shield_strength  (fixed-point x256)
12      i16    speed_air_x      (fixed-point x256)
14      i16    speed_y          (fixed-point x256)
16      i16    speed_ground_x   (fixed-point x256)
18      i16    speed_attack_x   (fixed-point x256)
20      i16    speed_attack_y   (fixed-point x256)
22      u16    state_age
24      u8     hitlag
25      u8     stocks
26      u8     facing           (1=right, 0=left)
27      u8     on_ground        (1=ground, 0=airborne)
28      u16    action_state
30      u8     jumps_left
31      u8     character
```

Python struct format string: `<iiHHhhhhhHBBBBHBB`

**ControllerInput = 8 bytes:**
```
Offset  Type   Field
0       i8     stick_x
1       i8     stick_y
2       i8     c_stick_x
3       i8     c_stick_y
4       u8     trigger_l
5       u8     trigger_r
6       u8     buttons          (bitmask: A=0x01, B=0x02, X=0x04, Y=0x08, Z=0x10)
7       u8     buttons_ext      (bitmask: D-up=0x01, L=0x04, R=0x08)
```

**Changing any field order, type, or size breaks the crank.** Post to `docs/HANDOFF.md` before modifying.

### Contract 2: TypeScript SDK Surface (with ScavieFae)

ScavieFae's website (`site/`) imports from `solana/client/src/`. These are the exported interfaces:

**Functions:**
- `createSession(connection, payer, config)` → `SessionAccounts`
- `joinSession(connection, payer, sessionKey, accounts)` → void
- `sendInput(connection, payer, inputKey, input, player)` → void
- `endSession(connection, payer, sessionKey, accounts)` → void
- `deserializeSessionState(data: Buffer)` → `SessionState`

**Types:**
- `SessionState` — status, frame, max_frames, stage, players[], pubkeys, timestamps
- `PlayerState` — all 17 fields matching the binary layout above
- `ControllerInput` — 8 fields matching binary layout above
- `SessionConfig` — stage, characters, max_frames, model dimensions

**Changing function signatures or type shapes requires a handoff entry.** ScavieFae reviews.

### Contract 3: Weight Format (with Scav)

INT8 quantized weights are produced by `quantization/quantize_mamba2.py` and consumed by your Rust programs. The format:

- Two `WeightShard` accounts (~7.5MB each) — row-major `i8` values
- One `ModelManifest` account (~2KB) — LUTs (256 entries each for SiLU, softplus, rsqrt, exp_neg), per-channel scale factors, architecture metadata
- Weights are `i8`, activations are `i8`, accumulation is `i32`, requantization is per-channel

Changes to quantization format require coordination with Scav.

## Onchain Architecture

### Components (6 BOLT ECS accounts per session)

| Component | Size | Location |
|-----------|------|----------|
| `weight-shard` | ~7.5MB | `programs-ecs/components/weight-shard/` |
| `model-manifest` | ~2KB | `programs-ecs/components/model-manifest/` |
| `session-state` | ~1KB | `programs-ecs/components/session-state/` |
| `hidden-state` | ~200KB | `programs-ecs/components/hidden-state/` |
| `input-buffer` | ~20B | `programs-ecs/components/input-buffer/` |
| `frame-log` | ~17KB | `programs-ecs/components/frame-log/` |

### Systems (3 instruction handlers)

| System | Location | What |
|--------|----------|------|
| `session-lifecycle` | `programs-ecs/systems/session-lifecycle/` | CREATE / JOIN / END |
| `submit-input` | `programs-ecs/systems/submit-input/` | Accept player controller input |
| `run-inference` | `programs-ecs/systems/run-inference/` | Mamba2 forward pass, update state |

All three systems use `pub fn execute` — Anchor discriminator is `SHA256("global:execute")[:8]` = `[0x0b, 0xed, 0x60, 0x84, 0x3d, 0x04, 0xea, 0xf8]`.

### Custom Syscall

`solana/syscall/` — `sol_matmul_i8(weights_ptr, input_ptr, output_ptr, rows, cols) → u64`

- `i8 x i8 → i32` accumulate, row-major weights
- Reviewed and approved by Scav (see `docs/HANDOFF.md`)
- Waiting for MagicBlock to deploy on their ER validator
- CU costing TBD — currently `base=100 + 1/MAC`, needs adjustment (see handoff doc)

## Key Technical Details

### Fixed-Point Representation
- Positions (x, y): `real_value * 256` stored as `i32`
- Velocities: `real_value * 256` stored as `i16`
- Shield strength: `real_value * 256` stored as `u16`
- Percent: stored directly as `u16` (no scaling)

### Session Lifecycle
1. `CREATE` — allocates SessionState, HiddenState, InputBuffer, FrameLog
2. `JOIN` — sets starting positions (P0 x=-30*256, P1 x=+30*256), 4 stocks, facing each other
3. Frame loop: `submit-input` (both players) → `run-inference` → state updated
4. `END` — finalize match, commit state back to L1

### Model Architecture (must match Rust implementation)
- **Current**: d_model=384, d_inner=768, d_state=64, n_layers=4, headdim=64 (4.3M params)
- **Production target**: d_model=512, d_inner=1024, d_state=16, n_layers=12 (15M params)
- Per-layer: RMSNorm → in_proj → SSM selective scan → gate (SiLU) → out_proj → residual
- Activations via LUT (256-entry tables for SiLU, softplus, rsqrt, exp_neg)

### CU Budget (with syscall)
| Component | CU (12 layers) |
|-----------|----------------|
| Matmul (syscall) | ~190K (target) |
| SSM selective scan (BPF) | ~7.6M |
| LUT activations (BPF) | ~480K |
| RMSNorm + requant + residual (BPF) | ~420K |
| **Total** | **~8.7M** |

## Your Open Work

### Priority 1: Test session.ts on devnet
- `client/src/session.ts` has real BOLT system calls but is untested end-to-end
- Manual instruction encoding (Anchor discriminator + Buffer packing) may have bugs
- Test: createSession → joinSession → sendInput → endSession
- Scav noted: verify discriminator is correct (it is — both Rust systems use `pub fn execute`)

### Priority 2: MagicBlock ER integration
- The `sol_matmul_i8` syscall crate is ready to share with MagicBlock
- Open questions for them (in `docs/HANDOFF.md`):
  - Max account size on ER (need 7.5MB weight accounts)
  - Per-TX CU limit (need ~10M)
  - Block time (need ≤16ms for 60fps)
  - CU costing for the syscall (currently too high at 1 CU/MAC)

### Priority 3: Escrow / treasury for Quarter Up
- Not started. Needs an onchain program for:
  - Entry fee collection (sponsor pays to queue an agent)
  - Prize distribution on match end (50% winner / 35% sponsor / 10% protocol / 5% loser)
  - Could be a new BOLT system or a standalone Anchor program

### Priority 4: Account delegation flow
- Session accounts need delegation to MagicBlock ER via their SDK
- Undelegation + commit back to L1 on match end
- Not implemented yet

## Environment

- Rust 1.93.1, Anchor 0.32.1, Solana CLI 3.0.15
- Node 22, TypeScript 5
- Build: `cd solana && cargo build --release` / `anchor build`
- Test: `cd solana && npm test` / `cargo test -p awm-syscall`

## Conventions

- Commit style: imperative mood, short first line, body explains "why"
- Branch prefix: `codex/` for Codex work
- Coordinate changes to shared interfaces via `docs/HANDOFF.md`
- All onchain programs use Anchor 0.32 with BOLT ECS macros (`#[component]`, `#[system]`, `#[system_input]`, `#[arguments]`)
