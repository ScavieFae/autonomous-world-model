# Handoff — Autonomous World Model

Active coordination doc between Scav and ScavieFae. Newest entries at top.

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
