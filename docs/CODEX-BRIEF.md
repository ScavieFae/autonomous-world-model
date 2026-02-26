# Codex Brief — Autonomous World Model

You are working on **autonomous-world-model**, a project that deploys a learned neural network as an onchain physics engine. A Mamba2 SSM (state space model) trained on Super Smash Bros. Melee replays runs inside Solana ephemeral rollups at 60fps. Two AI agents fight in a world where the "rules" are emergent from training data, not hand-coded game logic.

The product is **The Wire** — an agent arena at `world.nojohns.gg`. AI agents fight, humans spectate and sponsor ("quarter up"). Think arcade cabinet metaphor: weights on L1 are the cartridge, ephemeral rollup is the console, browser is the TV.

---

## Repository Map

```
autonomous-world-model/
├── models/           # PyTorch inference code (Mamba2, PolicyMLP)     ~980 LOC Python
├── crank/            # Offchain match runner (standalone + Solana)    ~1,260 LOC Python
├── quantization/     # FP32 → INT8 quantization pipeline             ~1,250 LOC Python
├── viz/              # Canvas 2D match visualizer (standalone HTML)   ~4,560 LOC JS/HTML
├── site/             # "The Wire" — Next.js arena website             ~6,000 LOC TypeScript
├── solana/
│   ├── programs/     # Anchor programs (world-model, cu-benchmark)   ~2,500 LOC Rust
│   ├── programs-ecs/ # BOLT ECS components (6) + systems (3)        ~1,600 LOC Rust
│   ├── syscall/      # sol_matmul_i8 native syscall for MagicBlock  ~400 LOC Rust
│   ├── client/       # TypeScript SDK (@awm/client)                  ~400 LOC TS
│   ├── cli/          # Weight upload CLI                              ~200 LOC TS
│   └── tests/        # Integration tests (Mocha)                     ~600 LOC TS
└── docs/             # Architecture, specs, design, handoff          ~4,800 LOC Markdown
```

Total: ~24K LOC across Rust, Python, TypeScript, JavaScript, Markdown.

---

## How It Works

### The Model

**FrameStackMamba2** — a recurrent state space model. Takes current game state + both players' controller inputs, predicts the next frame of game state. Constant cost per step (unlike transformers). Fixed-size hidden state (~200KB) carries forward match "memory."

Current training checkpoint: **4.3M params** (d_model=384, n_layers=4, d_state=64, headdim=64).
Production target: **15M params** (d_model=512, n_layers=12).

**Outputs per frame:**
- 12 continuous values per player (x, y, percent, shield, 5 velocities, state_age, hitlag, stocks)
- 2 binary per player (facing, on_ground)
- 3 categorical per player (action_state/400 classes, jumps_left/8, character/33)
- 1 categorical per frame (stage/33)

### The Frame Loop

```
Every 16ms:
  1. Read both players' controller inputs
  2. Read current hidden state (Mamba2 recurrent memory)
  3. Run one forward pass through all layers
  4. Write new hidden state + new game state
  5. Browsers subscribed via WebSocket see the new frame
```

### Two Execution Modes

**Mode A — Standalone** (working now): Python crank runs locally. Two AI agents (PolicyMLP or scripted) produce controller inputs. World model predicts frames. Output is a JSON file viewable in `viz/visualizer.html`.

**Mode B — Solana Crank** (skeleton): Same inference loop, but reads/writes Solana accounts via RPC. Session state lives in an ephemeral rollup. This is the production target.

### Onchain Architecture (BOLT ECS)

Six component accounts per session:

| Component | Size | What |
|-----------|------|------|
| WeightShard ×2 | ~7.5MB each | INT8 model weights (on mainnet, cloned to ER) |
| ModelManifest | ~2KB | LUTs, scales, architecture metadata |
| SessionState | ~1KB | Current frame: both players' positions, percents, stocks, actions |
| HiddenState | ~200KB | Mamba2 recurrent state (the world's "memory") |
| InputBuffer | ~20B | Both players' controller inputs for this frame |
| FrameLog | ~17KB | Ring buffer of last 256 frames |

Three systems (instruction handlers):

| System | What |
|--------|------|
| `session-lifecycle` | CREATE / JOIN / END — allocate accounts, set starting positions, finalize match |
| `submit-input` | Accept controller input from a player |
| `run-inference` | Execute one Mamba2 forward pass, update SessionState + HiddenState |

### The Custom Syscall

Standard Solana VM (BPF) charges ~25 CU per multiply-accumulate. One frame needs ~19M MACs → ~300M CU in BPF (impossible). A native `sol_matmul_i8` syscall runs the matmul at CPU speed, dropping to ~8.7M CU/frame (feasible at 60fps in an ephemeral rollup).

The syscall is implemented (`solana/syscall/`), reviewed, and waiting for MagicBlock to deploy it on their ER validator.

### INT8 Determinism

All weights and activations are INT8. Integer arithmetic is bitwise identical on every machine — no floating point nondeterminism. Quantization pipeline in `quantization/` converts FP32 training weights to INT8 + generates lookup tables for nonlinear activations (SiLU, softplus, rsqrt, exp_neg).

---

## Decisions Already Made

### Architecture
- **Mamba2 SSM**, not transformer. Constant per-step cost, fixed hidden state, ideal for 60fps onchain loop.
- **INT8 quantization**. Determinism + 4× size reduction. LUT activations instead of floating-point nonlinearities.
- **BOLT ECS** on Solana via MagicBlock ephemeral rollups. Not vanilla Anchor programs.
- **Offchain crank** for prototype. Python reads ER state, runs PyTorch inference, writes back. Onchain inference (BPF) is the goal but blocked on syscall deployment.
- **Weights stay offchain** for prototype. Mainnet upload (~$8K rent deposit at current SOL) deferred until syscall is live.

### Data Representation
- **Fixed-point ×256** for positions and velocities onchain (i32 for positions, i16 for velocities). Model uses normalized floats (×0.05 for xy, ×0.01 for percent, etc.).
- **PlayerState = 32 bytes** binary layout matching Rust `#[derive(AnchorSerialize)]` field order.
- **Controller input**: i8 sticks (-128..127), u8 triggers, u8 button bitmask. Model normalizes to [0,1] floats.
- **Hidden state**: raw f32 bytes for FP32 prototype. Will be raw i8 bytes for INT8.

### Product
- **"The Wire"** — agent arena, not a playable game. AI agents fight, humans spectate and sponsor.
- **Tapestry** (socialfi SDK) for all social features: agent profiles, follows, match results as content, leaderboard via custom properties. Namespace: `wire`.
- **ELO ranking** with tiers named after Melee stages (Final Destination, Battlefield, Dreamland, Flat Zone).
- **Quarter Up** = sponsor an agent for a match. Prize split: 50% winning agent / 35% sponsor / 10% protocol / 5% loser.
- **Wireframe visual style** as default render mode. Skeleton-mesh fighters with glowing joints, pulsing heart, CRT/diagnostic aesthetic.
- **Four render modes**: Wire (W), Character (C), Data (D), X-Ray (X).

### Codebase
- **Three-agent development**: ScavieFae (Claude) owns `site/`, Scav (Claude) owns `models/` + `crank/` + `quantization/` + `viz/`, Codex owns `solana/`. Coordination via `docs/HANDOFF.md`.
- **Codex ownership rules** are in `AGENTS.md` (root). Interface contracts, review gates, and priority work are defined there.
- **Model code copied from nojohns** (sibling repo) with adapted imports. This repo is inference-only, no training code.
- **Synthetic seed data**: 10 frames of starting positions (P0 x=-30, P1 x=+30, 4 stocks, facing each other) generated from character/stage selection. No external replay dependency.

### Tech Stack
- **Frontend**: Next.js 16, React 19, Zustand, Canvas 2D, fflate (for character animation ZIPs)
- **Onchain**: Anchor 0.32, BOLT ECS, Solana BPF, custom syscall
- **Inference**: PyTorch 2.0+, Python 3.14
- **Environment**: macOS, Node 22, Rust 1.93, Solana CLI 3.0.15

---

## Key Open Implementation Work

### Critical Path (make the demo work)

1. **Download model checkpoints from Modal**
   - World model: `mamba2-22k-ss-ep1.pt` → `checkpoints/world-model.pt`
   - Policy: `policy-p0-test/best.pt` → `checkpoints/policy.pt`
   - These exist on a Modal volume (`melee-training-data`), just need downloading.
   - Without these, standalone matches use random weights (garbage output).

2. **Run a real PolicyAgent match**
   - Load actual Phillip (PolicyMLP) checkpoints into `crank/agents.py:PolicyAgent`
   - Run `python -m crank.main --world-model ... --policy ... --output match.json`
   - Open in `viz/visualizer.html` and verify output looks like Melee, not noise.
   - The code is written and tested with scripted agents. Real policy is the next verification.

3. **Wire up The Wire frontend to real data**
   - `site/` has rendering, components, and pages scaffolded.
   - `site/src/providers/data.tsx` has a WebSocket listener but needs a real data source.
   - For demo: load a `match.json` from the crank as a static fixture → drive the renderer.
   - For production: WebSocket subscription to ER account changes.

4. **Tapestry integration**
   - Agent registration → `POST /v1/profiles/findOrCreate` with `wire` namespace
   - Match result posting → `POST /v1/contents/create` with custom properties
   - Leaderboard → `GET /v1/profiles/search` sorted by ELO custom property
   - SDK: `npm install socialfi`, API at `api.usetapestry.dev/v1/`
   - None of this is started. The design is in `docs/design-arena-mechanics.md`.

5. **Quarter Up flow**
   - Connect wallet (Phantom/Backpack/Solflare)
   - Pick an agent to sponsor
   - Pay entry fee → agent enters queue
   - Two sponsored agents → match starts
   - Prize distribution on match end
   - Requires: wallet adapter integration, an onchain escrow or treasury program, matchmaking logic.

### Important But Not Blocking Demo

6. **Test session.ts on devnet**
   - `solana/client/src/session.ts` was rewritten with real BOLT system calls.
   - Needs integration testing: createSession → joinSession → sendInput → endSession.
   - Manual instruction encoding (Anchor discriminator + Buffer packing) — may have bugs.
   - The BOLT ECS components and systems compile and have unit tests, but the TS SDK calling them end-to-end is untested.

7. **Mode B crank (Solana integration)**
   - `crank/main.py` has a skeleton `run_crank()` that polls ER accounts.
   - Needs: actual `solana-py` / `solders` dependency, tested RPC calls, hidden state read/write.
   - Blocked on: MagicBlock ER instance with `sol_matmul_i8` syscall enabled.
   - Workaround for demo: run Mode A standalone, feed output to frontend as static data.

8. **MagicBlock ER deployment**
   - The `sol_matmul_i8` syscall crate (`solana/syscall/`) is complete, reviewed, approved.
   - MagicBlock (Gabriele Picco) needs to integrate it into their validator and set up an ER instance.
   - Open questions: max account size (need 7.5MB), per-TX CU limit (need ~10M), block time (need ≤16ms), CU costing for the syscall.
   - This is an external dependency. The codebase is ready; we're waiting on them.

9. **Wireframe fighter rendering**
   - `site/src/engine/renderers/wire.ts` exists with basic skeleton-mesh.
   - Needs: action state → pose mapping (standing, dashing, jumping, attacking, damaged, KO), hit effects (flash, vertex burst), KO disintegration (lines scatter as particles).
   - Design spec is detailed in `docs/design-visual-ux.md`.

### Nice to Have

10. **Match event ticker** — auto-generate combat events (hits, KOs, combos) from frame data diffs. Component exists (`MatchTicker.tsx`) but needs the event detection logic.

11. **Heart rate monitor** — ECG-style line derived from `state_age` and action state transitions. Visual spec in design doc.

12. **Attract mode** — wireframe fighters shadowboxing on empty stage when no match is running. Text: `AUTONOMOUS WORLD MODEL — INSERT COIN`.

13. **Sound design** — CRT hum, synthetic hit sounds, crowd murmur during combos, coin-insert on quarter up. Entirely stretch.

14. **X-Ray render mode** — Wire mode + hitbox/hurtbox overlay. Frame data nerd mode.

15. **ELO calculations** — standard ELO with K-factor decay (K=32 new, K=24 established, K=16 veteran). Starting ELO 1200. Algorithm matches nojohns proper. Not implemented yet.

---

## File-Level Guide to Key Code

### Models (Python — inference only)

- `models/encoding.py` — **Read this first.** `EncodingConfig` defines every field index, scale factor, and vocab size. All conversion code depends on it.
- `models/mamba2.py` — `FrameStackMamba2`: embeds float+int frame data, runs through N Mamba2 layers (RMSNorm → in_proj → SSM scan → gate → out_proj → residual), predicts next frame via regression/classification heads.
- `models/policy_mlp.py` — `PolicyMLP`: observes game state, outputs controller values. `ANALOG_DIM=5` (sticks + trigger), `BUTTON_DIM=8`.
- `models/checkpoint.py` — `load_model_from_checkpoint()`: auto-detects MLP vs Mamba2 from weight keys, infers hyperparams from tensor shapes, handles old/new formats.

### Crank (Python — match execution)

- `crank/match_runner.py` — `run_match()`: the autoregressive loop. Seeds with K=10 frames, then iterates: agents produce controllers → model predicts next frame → decode continuous (deltas) + binary (logit threshold) + categorical (argmax) → clamp to valid ranges → check KO.
- `crank/agents.py` — `PolicyAgent.get_controller()`: runs PolicyMLP forward pass on observed state. `_swap_perspective()` mirrors P0/P1 so the policy always sees itself as P0.
- `crank/state_convert.py` — The critical glue. `session_to_tensors()` / `tensors_to_session()` for round-trip conversion. Key gotcha: uses `round()` not `int()` for float→fixed-point to avoid off-by-one.
- `crank/solana_bridge.py` — `deserialize_session_state()` / `serialize_player_state()`: struct.pack/unpack matching Rust `AnchorSerialize` layout byte-for-byte. `PLAYER_STATE_SIZE = 32`.

### Solana — ECS (Rust)

- `solana/programs-ecs/components/session-state/src/lib.rs` — `PlayerState` struct with 17 fields. Fixed-point representation. `NUM_CONTINUOUS = 12`. Status constants.
- `solana/programs-ecs/systems/session-lifecycle/src/lib.rs` — `ACTION_CREATE` / `ACTION_JOIN` / `ACTION_END`. JOIN sets starting positions (P0 x=-30, P1 x=+30).
- `solana/programs-ecs/systems/run-inference/src/lib.rs` — Calls into mamba2.rs for the BPF forward pass. Reads weights, hidden state, inputs. Writes new state.

### Solana — Client SDK (TypeScript)

- `solana/client/src/session.ts` — `createSession()`, `joinSession()`, `sendInput()`, `endSession()`. Manual Buffer encoding of Anchor instructions. Discriminator: `SHA256("global:execute")[:8]`.
- `solana/client/src/state.ts` — `SessionState` / `PlayerState` TypeScript types. `playerStateToViz()` converts to visualizer format.
- `solana/client/src/input.ts` — `ControllerInput` type, `InputManager` class, keyboard→controller mapping.

### Site (TypeScript/React)

- `site/src/providers/data.tsx` — `DataContextProvider`: WebSocket connection, frame buffer, match state management.
- `site/src/stores/arena.ts` — Zustand store for match state (players, scores, input).
- `site/src/engine/renderers/wire.ts` — Wireframe fighter renderer (skeleton-mesh, joint dots, body outline).
- `site/src/engine/playback.ts` — Playback state machine (play, pause, step, speed control).
- `site/src/components/arena/ArenaView.tsx` — Main match display layout.
- `site/src/components/modals/QuarterUpModal.tsx` — Sponsorship entry form.

### Visualizer (standalone HTML)

- `viz/visualizer-juicy.html` — Full-featured renderer with motion trails, particles, screen shake, parallax background, CRT overlay. Loads JSON or connects to WebSocket. This is the reference renderer that the site's Canvas engine is based on.

---

## Conventions

- **Fixed-point**: onchain values are `real_value × 256` for positions, velocities, shield. Percent is stored directly as u16 (no scaling).
- **Anchor discriminator**: `SHA256("global:execute")[:8]` = `[0x0b, 0xed, 0x60, 0x84, 0x3d, 0x04, 0xea, 0xf8]` for all BOLT system calls.
- **Player indexing**: P0 is always the first player in arrays/structs. PolicyMLP always expects to see itself as P0 — the `_swap_perspective()` method handles mirroring for P1.
- **Frame format** (JSON for viz): `{ meta: {...}, stage_geometry: {...}, frames: [{ players: [{x, y, percent, stocks, action_state, ...}, ...], stage: N }] }`
- **No `int()` for float→fixed-point conversion**. Always `round()`. The `int()` function truncates, causing off-by-one errors on round-trip.
- **Commit style**: imperative mood, short first line, body explains "why." Co-authored-by line for Claude.
- **Three-agent ownership**: Codex owns `solana/`, Scav owns `models/`+`crank/`+`quantization/`+`viz/`, ScavieFae owns `site/`. Rules in `AGENTS.md` (Codex) and `CLAUDE.md` (Claude agents). Coordinate via `docs/HANDOFF.md`.
