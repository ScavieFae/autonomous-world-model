# Handoff — Autonomous World Model

Active coordination doc between Scav and ScavieFae. Newest entries at top.

---

## Offchain Crank Architecture — models/ + crank/ + session.ts (Feb 26)

**Scav**: Built the offchain match runner that wires the Mamba2 world model to the BOLT ECS session lifecycle. Two new packages, one rewritten client file.

### What was built

**`models/`** — Self-contained inference code (copied from nojohns, imports adapted):

| File | What |
|------|------|
| `encoding.py` | `EncodingConfig` — all field indices, scales, vocab sizes |
| `mamba2.py` | `FrameStackMamba2` — the world model (d_model=384, n_layers=4, d_state=64) |
| `mlp.py` | `FrameStackMLP` — fallback architecture |
| `policy_mlp.py` | `PolicyMLP` — Phillip fighter agent (inlines ANALOG_DIM=5, BUTTON_DIM=8) |
| `checkpoint.py` | `load_model_from_checkpoint()` + STAGE_GEOMETRY + CHARACTER_NAMES |

**`crank/`** — Offchain match runner with two modes:

| File | What |
|------|------|
| `match_runner.py` | Core autoregressive loop: seed → agent controllers → world model → decode → KO check |
| `agents.py` | Agent interface + PolicyAgent (with P1 perspective swap) + RandomAgent, NoopAgent, HoldForwardAgent |
| `state_convert.py` | Bidirectional ECS ↔ tensor conversion (fixed-point ×256 ↔ normalized floats) |
| `solana_bridge.py` | Binary serialize/deserialize matching Rust structs + async RPC read/write |
| `main.py` | CLI entry point — `--output match.json` (Mode A standalone) or `--session <pubkey>` (Mode B crank) |

**`solana/client/src/session.ts`** — Rewrote stubs with real BOLT system calls:
- `allocateAccounts()` — creates 4 accounts (SessionState, HiddenState, InputBuffer, FrameLog)
- `createSession()` / `joinSession()` / `endSession()` — sends session_lifecycle instructions
- `sendInput()` — sends submit_input instruction with full controller state
- Manual instruction encoding (Anchor discriminator + Buffer packing)
- `deserializeSessionState()` / `deserializePlayerState()` — full binary deserialization

### Key details

- **PlayerState = 32 bytes** (i32+i32+u16+u16+i16×5+u16+u8×2+u8×2+u16+u8×2) — initially miscounted as 28, corrected
- **Synthetic seed**: 10 frames of starting positions (P0 x=-30, P1 x=+30, 4 stocks, facing each other) — matches session-lifecycle JOIN
- **State conversion round-trip is lossless** — uses `round()` not `int()` for float→fixed-point
- **PolicyAgent perspective swap**: mirrors P0/P1 float and int blocks so policy always sees itself as P0
- **Checkpoints not committed** — `.gitignore` updated, weights stay in `checkpoints/*.pt`

### Verified

- Model imports OK (`from models.mamba2 import FrameStackMamba2`)
- Standalone match runs (40 frames with random model + scripted agents)
- State conversion round-trip lossless
- Binary serialization round-trip (32 bytes PlayerState)
- JSON output compatible with `viz/visualizer.html`
- CLI `--help` works

### Mode A usage (standalone)

```bash
python -m crank.main \
    --world-model checkpoints/world-model.pt \
    --policy checkpoints/policy.pt \
    --stage 2 --p0-char 2 --p1-char 2 \
    --max-frames 600 --output match.json
```

### What's next

1. Download checkpoints from Modal (world-model.pt, policy.pt)
2. Run a real match with PolicyAgent — verify output looks like Melee
3. Test session.ts on devnet (createSession → joinSession → sendInput → endSession)
4. Wire up Mode B crank to ER WebSocket endpoint
5. MagicBlock account delegation for ER integration

### Files changed

| File | Change |
|------|--------|
| `models/` (6 files) | NEW — self-contained model inference code |
| `crank/` (7 files) | NEW — offchain match runner |
| `solana/client/src/session.ts` | REWRITE — stubs → real BOLT system calls |
| `.gitignore` | MODIFIED — added `checkpoints/*.pt` |

---

## Question for MagicBlock: CU Costing for sol_matmul_i8 (Feb 26)

The syscall needs a CU cost. We have a placeholder (`base=100 + 1/MAC`) but the right number depends on how MagicBlock wants to meter work on the ER instance.

Context: the actual CPU cost of our matmul is trivial — ~1M INT8 MACs runs in microseconds on any modern CPU. The CU number is purely metering, not a reflection of hardware cost. On a dedicated ER instance running just our workload, the question is really: **what per-transaction CU cap should the instance have, and what should the syscall charge, so that one frame of inference fits inside one transaction?**

Our frame budget breakdown (with the syscall):

| Component | CU (12 layers) |
|-----------|----------------|
| Matmul (syscall) | depends on constant — we need this to be small |
| SSM selective scan (BPF) | ~7.6M |
| LUT activations (BPF) | ~480K |
| RMSNorm + requant + residual (BPF) | ~420K |
| **Total target** | **~8.7M** |

The BPF work is fixed at ~8.5M. So the syscall's CU charge needs to leave headroom within whatever per-TX cap the instance runs. We'd appreciate MagicBlock's advice on the right costing model — flat per-call, linear in MACs, or something else. The constant is one line in `lib.rs`, trivially tunable.

---

## Review Response: sol_matmul_i8 Syscall Crate (Scav, Feb 26)

**Scav → ScavieFae**: Reviewed `solana/syscall/` against the BPF matmul (`programs/world-model/src/matmul.rs`), the Python quantization pipeline (`quantization/quantize_mamba2.py`), and the spec (`sol-matmul-i8-spec.md`).

### Verdict: APPROVED with one action item

The matmul math is correct. Memory translation is correct. Tests are solid. Ship it to MagicBlock. The CU constant needs adjusting before it goes into production, but that's a negotiation with them, not a blocker on sharing the code.

### Matmul correctness — APPROVED

The core operation matches across all three implementations:

| Version | Weights type | Inner loop | Equivalent? |
|---------|-------------|------------|-------------|
| **Syscall** (`syscall/src/matmul.rs`) | `&[i8]` | `weights[j] as i32 * input[j] as i32` | — |
| **BPF** (`programs/world-model/src/matmul.rs`) | `&[u8]` | `weights[j] as i8 as i32 * input[j] as i32` | Yes — same bytes, `u8 → i8` reinterprets sign |
| **Quantization** (Python) | `np.int8` | Row-major C order, `clip(-128, 127)` | Yes — same byte layout as Rust `i8` |

Type difference (`&[u8]` vs `&[i8]`) is cosmetic — Rust guarantees identical representation. Both produce the same `i32` accumulator for any input byte. No bias term in either (handled by separate BPF `add_i8` ops). Requantization correctly stays in BPF.

### Memory translation — APPROVED

| Buffer | Access | Length calc | Correct? |
|--------|--------|-------------|----------|
| weights | `Load` | `rows * cols` bytes | Yes — i8 = 1 byte |
| input | `Load` | `cols` bytes | Yes |
| output | `Store` | `rows * 4` bytes | Yes — i32 = 4 bytes |

`checked_mul` on the CU path prevents overflow from adversarial dimensions. `MemoryMapping::map()` validates BPF VM regions before the unsafe `from_raw_parts`. Standard pattern, same as `sol_sha256`.

### Test coverage — APPROVED

9 unit tests: identity, known values, negatives, i8 extremes (-128×-128 = 32768 in i32), production dims with spot-checks, odd cols, single element, zeros. 3 Mollusk integration tests: full SVM round-trip (register syscall → load BPF program → process instruction → read account output).

Minor gap: no test asserting the exact CU charged. The Mollusk tests implicitly pass CU limits by succeeding, but an explicit check would catch future regressions if the constant changes.

### CU costing — NEEDS WORK before production

`CU_PER_MAC = 1` is ~100x too high. The numbers:

| Operation | MACs | @ 1 CU/MAC | Spec target |
|-----------|------|-----------|-------------|
| in_proj (2048×512) | 1,048,576 | ~1.05M CU | ~8K CU |
| out_proj (512×1024) | 524,288 | ~524K CU | ~8K CU |
| **12 layers total** | 18,874,368 | **~18.9M CU** | **~191K CU** |

At 1 CU/MAC, matmul alone exceeds the entire 8.7M frame budget. The spec's ~191K target implies ~0.01 CU/MAC.

For reference: `sol_sha256` costs 1 CU/byte, doing ~100x more work per byte than a single multiply-add at native speed. So `CU_PER_MAC = 1` prices matmul as expensive as SHA256, which it isn't.

**Suggestion:** Propose `base=500 + MACs/128` to MagicBlock. That gives ~8.7K CU per in_proj call, ~4.6K per out_proj, ~160K total across 12 layers. Leaves headroom for the ~8.5M of BPF work (SSM scan, activations, requant). Or just `CU_PER_MAC = 0` with a per-call base — let MagicBlock decide.

This doesn't block sharing the code. The constant is trivially tunable: one line in `lib.rs`.

### Production dimensions — NOTED

Tests use (2048, 512) and (512, 1024), matching the spec's 15M-param production model (d_model=512, d_inner=1024, 12 layers). The current training model is smaller (d_model=384, n_layers=4, ~4.3M params). No issue — the syscall is dimension-agnostic, these are just test vectors for the target architecture.

### Action items

1. **ScavieFae**: Adjust `CU_PER_MAC` before proposing to MagicBlock (suggest `MACs/128` or negotiate with them directly)
2. **ScavieFae**: Can proceed sharing `solana/syscall/` with MagicBlock — the code is correct
3. **Optional**: Add a Mollusk test asserting exact CU consumed for a known dimension

---

## Review Request: sol_matmul_i8 Syscall Crate (Feb 26)

**ScavieFae → Scav**: New crate implementing the native INT8 matmul syscall for MagicBlock's ephemeral rollup validators. This is the deliverable we hand off to MagicBlock — they integrate it into their validator. Need Scav to review the math and verify it matches the training-side implementation.

### Background

Matmul is the inference bottleneck. In BPF: ~300M CU/frame (impossible at 60fps). With a native syscall: ~8.7M CU/frame (feasible). We wrote a [spec for MagicBlock](sol-matmul-i8-spec.md) describing the syscall we need, and they asked for a compilable implementation they can integrate.

**Our goals:**
1. Get `sol_matmul_i8` running on a dedicated MagicBlock ER instance
2. Prove it works end-to-end via Mollusk tests (SVM test framework)
3. Unblock 60fps onchain inference — matmul drops from 97% of CU cost to 2%

### What was built

Two new crates in `solana/`:

**`solana/syscall/`** — The artifact MagicBlock integrates:

| File | What |
|------|------|
| `src/lib.rs` | `SyscallMatmulI8` via `declare_builtin_function!` — reads 5 registers (weights ptr, input ptr, output ptr, rows, cols), translates BPF memory, charges CU (base 100 + 1/MAC), runs matmul |
| `src/matmul.rs` | Pure `matmul_i8()` — `i8 × i8 → i32` accumulate, clean nested loops, no SVM deps |
| `tests/unit.rs` | 9 tests — identity, known values, negatives, i8 range, production dims (2048×512, 512×1024), odd cols |
| `tests/mollusk.rs` | 3 integration tests — registers syscall with Mollusk, loads BPF test program, processes instructions, verifies output |

**`solana/programs/syscall-test/`** — Minimal BPF test harness:

| File | What |
|------|------|
| `src/lib.rs` | `extern "C" { fn sol_matmul_i8(...) }`, reads weights+input from instruction data, calls syscall, writes i32 output to account |

### Syscall signature

```
sol_matmul_i8(weights_ptr, input_ptr, output_ptr, rows, cols) → u64
```

Five registers, standard BPF syscall convention. `i8 × i8 → i32` accumulate. Row-major weights. Rows and cols as explicit args — no struct packing, obvious ABI.

### What Scav should review

1. **`matmul.rs` correctness** — Does this match the training-side matmul? The implementation is a straightforward `i8 × i8 → i32` nested loop. Compare against `nojohns/worldmodel/` quantization logic and `solana/programs/world-model/src/matmul.rs` (the BPF version that uses packed u32 loads). The native version is simpler (no packing tricks needed) — just verify the math is identical.

2. **CU costing** — Currently `base=100 + 1 per MAC`. For in_proj (2048×512): 100 + 1,048,576 = ~1.05M CU. For out_proj (512×1024): 100 + 524,288 = ~524K CU. Total per frame: ~18.9M CU from matmul alone. The spec estimated ~191K total — our costing is ~100x higher. **Is 1 CU/MAC the right constant?** MagicBlock said they're flexible on costing. We should propose something that keeps total frame CU under ~10M.

3. **Memory translation** — The syscall translates BPF virtual addresses to host memory via `MemoryMapping::map()`. This is standard Solana syscall pattern (same as `sol_sha256`, etc). Worth eyeballing that the `AccessType::Load` vs `AccessType::Store` are correct for each buffer.

4. **Production dimensions** — The unit tests exercise (2048, 512) and (512, 1024) which match our model's in_proj and out_proj. Verify these match the latest model architecture in `nojohns/worldmodel/`.

### Test results

```
cargo test -p awm-syscall          # 12/12 pass (9 unit + 3 Mollusk)
cargo build-sbf --manifest-path programs/syscall-test/Cargo.toml  # 20KB .so
```

### Files changed

| File | What |
|------|------|
| `solana/Cargo.toml` | Added `"syscall"` to workspace members, `"programs/syscall-test"` to exclude |
| `solana/syscall/` | New crate (4 source files) |
| `solana/programs/syscall-test/` | New crate (1 source file) |

### Open questions for MagicBlock

These are tracked in the [spec](sol-matmul-i8-spec.md) Questions section:

- Max account size on ER? (we need 7.5MB weight accounts)
- How do read-only weight accounts get loaded into the ER from mainnet?
- CU limit per TX? (we need ~10M with the syscall)
- Per-block CU limit? (60fps × ~8.7M = ~522M CU/sec sustained)
- Block time? (need ≤16ms)
- How is the syscall enabled? (feature flag per instance?)

### Next steps

1. Scav reviews this handoff entry
2. ScavieFae shares `solana/syscall/` with MagicBlock
3. MagicBlock integrates into their validator, sets up an ER instance with the syscall enabled
4. We write a Mollusk test proving it works, then build the full inference pipeline on top
