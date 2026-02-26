Read `AGENTS.md` and `docs/CODEX-BRIEF.md` for full project context. Read `docs/HANDOFF.md` for Scav's review — there are specific action items for you.

Your first deliverable: **make the BOLT ECS session lifecycle work end-to-end on localnet.**

## What exists

- 6 BOLT ECS components in `solana/programs-ecs/components/` — compile, have Rust unit tests
- 3 BOLT ECS systems in `solana/programs-ecs/systems/` — compile, have Rust unit tests
- `solana/client/src/session.ts` — TypeScript SDK with `createSession`, `joinSession`, `sendInput`, `endSession`. **These were written by hand with manual Buffer encoding and are untested.** They likely have bugs.
- `solana/tests/session.ts` — existing test file, needs real end-to-end coverage

## What I need

1. **Audit `solana/client/src/session.ts`** against the Rust source in `solana/programs-ecs/systems/`. Verify:
   - Every Buffer offset in `encodeLifecycleArgs()` and `encodeInputArgs()` matches the Rust `Args` struct field order and sizes
   - Account metas (pubkeys, isSigner, isWritable) match the `#[system_input]` declarations in each Rust system
   - `deserializeSessionState()` and `deserializePlayerState()` read at correct offsets (PlayerState = 32 bytes, layout in `AGENTS.md`)
   - Scav flagged: stale "28 bytes" comment is already fixed, but double-check the actual deserialization math

2. **Write integration tests** in `solana/tests/session.ts` that exercise the full lifecycle:
   - Create session (allocate 4 accounts + call session_lifecycle with ACTION_CREATE)
   - Join session (call session_lifecycle with ACTION_JOIN, verify starting positions: P0 x=-7680, P1 x=+7680 in fixed-point)
   - Submit input (call submit-input for both players)
   - Read and deserialize SessionState, verify all fields
   - End session (call session_lifecycle with ACTION_END)
   - Run these against localnet via `anchor test`

3. **Fix whatever breaks.** The manual Buffer encoding is the most likely failure point. If the Anchor IDL approach (`program.methods.execute(...)`) is cleaner and works with BOLT systems, switch to that instead of raw Buffer packing.

## Key files to read

| File | What to check |
|------|---------------|
| `solana/programs-ecs/systems/session-lifecycle/src/lib.rs` | `Args` struct fields, `#[system_input]` account list, CREATE/JOIN/END logic |
| `solana/programs-ecs/systems/submit-input/src/lib.rs` | `Args` struct fields, account list |
| `solana/programs-ecs/components/session-state/src/lib.rs` | `PlayerState` struct (17 fields, 32 bytes), field order |
| `solana/programs-ecs/components/input-buffer/src/lib.rs` | `ControllerInput` struct (8 bytes) |
| `solana/client/src/session.ts` | The code under audit |
| `solana/client/src/state.ts` | TypeScript types that must match Rust |
| `solana/client/src/input.ts` | ControllerInput types |
| `solana/tests/session.ts` | Existing test patterns |
| `AGENTS.md` | Your ownership rules and interface contracts |
