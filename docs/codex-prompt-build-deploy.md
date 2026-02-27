Read `AGENTS.md` and `docs/CODEX-BRIEF.md` for full project context.

**Deliverable: Build the ECS programs, generate real program IDs, wire up Anchor.toml, and get the lifecycle test passing on localnet.**

## Background

The previous Codex task audited the SDK wire format and wrote integration tests (`solana/tests/session.ts`). The wire format tests pass. The lifecycle test cannot run because:

1. No compiled `.so` files exist — nothing has been built
2. `Anchor.toml` only lists `cu_benchmark` under `[programs.localnet]`, not the ECS programs
3. All program IDs are placeholders (e.g., `SessLife11111111111111111111111111111111111`) — both in Rust `declare_id!()` and TypeScript constants
4. `programs-ecs/` is excluded from the Cargo workspace (`Cargo.toml` has `exclude = ["programs-ecs"]`)

## What I need

### 1. Build the ECS components and systems

The ECS crates live in `solana/programs-ecs/` and are currently excluded from the main workspace. Each is a standalone crate with its own `Cargo.toml`. You need to build them as BPF programs.

**Components** (6 crates in `programs-ecs/components/`):
- `session-state` — PlayerState (32 bytes), session status, players
- `hidden-state` — Mamba2 SSM hidden state buffer
- `input-buffer` — ControllerInput (8 bytes per player)
- `frame-log` — Ring buffer of compressed frames
- `model-manifest` — Model metadata
- `weight-shard` — Weight data shards

**Systems** (3 crates in `programs-ecs/systems/`):
- `session-lifecycle` — CREATE / JOIN / END actions
- `submit-input` — Write player controller input
- `run-inference` — Run world model inference step (has matmul, lut, mamba2 modules)

Build each with `cargo build-sbf` (or `anchor build` if you can wire them into Anchor). The session lifecycle test only needs the components and `session-lifecycle` + `submit-input` systems — `run-inference` is nice to have but not blocking.

### 2. Generate real program IDs

After building, each program gets a deploy keypair in `target/deploy/`. Use `solana-keygen pubkey target/deploy/<name>-keypair.json` to get the real base58 program ID.

Replace the placeholder IDs everywhere they appear:

**Rust files** (`declare_id!(...)` in each crate's `lib.rs`):
- `programs-ecs/systems/session-lifecycle/src/lib.rs`
- `programs-ecs/systems/submit-input/src/lib.rs`
- `programs-ecs/systems/run-inference/src/lib.rs`
- All 6 component crates in `programs-ecs/components/*/src/lib.rs`

**TypeScript** (`solana/client/src/session.ts`):
- `SESSION_LIFECYCLE_PROGRAM_ID`
- `SUBMIT_INPUT_PROGRAM_ID`
- `SESSION_STATE_PROGRAM_ID`
- `INPUT_BUFFER_PROGRAM_ID`
- `FRAME_LOG_PROGRAM_ID`
- `HIDDEN_STATE_PROGRAM_ID`

Note: Two placeholder IDs had invalid base58 (capital `I` in `SubInput` and `InpBuffer`). These were fixed to lowercase `i` (`Subinput`, `inpBuffer`). The Rust `declare_id!()` macros still have the original uppercase versions — they need to be replaced with real IDs anyway.

### 3. Wire up Anchor.toml

Add the ECS programs to `[programs.localnet]` so `anchor test` deploys them to the local validator. Currently it only has:

```toml
[programs.localnet]
cu_benchmark = "Ecs7bGY1zvN6Desmt1d98PcSicoYc6c93jvQXAosyePF"
```

Add entries for at minimum: `session-lifecycle`, `submit-input`, and the component programs that the lifecycle test allocates accounts for (`session-state`, `input-buffer`, `frame-log`, `hidden-state`).

You may also need to handle the workspace exclusion. `programs-ecs` is currently excluded from the Cargo workspace. Either:
- Add the ECS crates to the workspace `members` list, or
- Build them separately with `cargo build-sbf --manifest-path` and copy the `.so` files to `target/deploy/`

### 4. Run the lifecycle test

```bash
cd /Users/mattiefairchild/claude-projects/autonomous-world-model/solana
anchor test --skip-lint
```

The test file `solana/tests/session.ts` has two test suites:
- `"session sdk wire format audit"` — already passes, no localnet needed
- `"session sdk lifecycle (BOLT ECS localnet)"` — needs deployed programs

The lifecycle test will:
1. Check that system programs are deployed (`assertProgramIsExecutable`)
2. Airdrop SOL to two test keypairs
3. Create session → verify state
4. Join session → verify starting positions
5. Submit input for both players → verify input buffer
6. End session → verify status

### 5. Fix whatever breaks

The most likely failure point (from Scav's review): the lifecycle instructions were updated to NOT include the player pubkey as an account meta (it's only in `Args` data). If the BOLT runtime requires the player as a signer account for authority checks, you'll need to add it back as a non-component account: `{ pubkey: player, isSigner: true, isWritable: false }`.

## Key files

| File | Role |
|------|------|
| `solana/Cargo.toml` | Workspace — currently excludes `programs-ecs` |
| `solana/Anchor.toml` | Only lists `cu_benchmark` |
| `solana/programs-ecs/components/*/src/lib.rs` | Component crates with placeholder `declare_id!()` |
| `solana/programs-ecs/systems/*/src/lib.rs` | System crates with placeholder `declare_id!()` |
| `solana/client/src/session.ts` | TypeScript SDK with placeholder program ID constants |
| `solana/tests/session.ts` | Integration tests (wire format + lifecycle) |

## Success criteria

- `anchor test --skip-lint` passes both test suites
- All program IDs are real deploy addresses (no more placeholder strings)
- ECS programs are in `[programs.localnet]`
