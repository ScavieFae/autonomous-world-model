# MagicBlock Integration — Syscall Handoff

Repo: https://github.com/ScavieFae/autonomous-world-model

---

## What to run

```bash
git clone https://github.com/ScavieFae/autonomous-world-model.git
cd autonomous-world-model/solana/syscall
cargo test
```

Three Mollusk integration tests + five unit tests. All pass.

---

## File map

| Path | What it is |
|------|-----------|
| `solana/syscall/src/lib.rs` | Syscall implementation — `declare_builtin_function!`, memory mapping, CU costing |
| `solana/syscall/src/matmul.rs` | Pure math — `i8 × i8 → i32` row-major matmul |
| `solana/syscall/tests/mollusk.rs` | Mollusk SVM integration tests (register + round-trip) |
| `solana/syscall/tests/unit.rs` | Unit tests for the math |
| `solana/programs/syscall-test/src/lib.rs` | Minimal SBF program that calls the syscall |
| `docs/sol-matmul-i8-spec.md` | Full technical spec — signature, dimensions, CU breakdown, questions |
| `docs/cu-benchmark-findings.md` | Empirical BPF CU measurements that motivated the syscall |

---

## Syscall signature

```rust
fn sol_matmul_i8(
    weights: *const i8,   // r1: row-major weight matrix
    input:   *const i8,   // r2: input vector
    output:  *mut i32,    // r3: caller-allocated output
    rows:    u64,         // r4
    cols:    u64,         // r5
) -> u64;
```

Standard 5-register convention. `i8 × i8 → i32` accumulate. Inner loop auto-vectorizes on ARM NEON.

---

## Registration

From `solana/syscall/tests/mollusk.rs`:

```rust
mollusk.program_cache.program_runtime_environment
    .register_function("sol_matmul_i8", SyscallMatmulI8::vm)
```

Register before loading the program. That's it.

---

## The numbers

4.5M parameter Mamba2 model, INT8 quantized, 60fps target.

| | BPF only | With syscall |
|---|---|---|
| CU per frame | ~90-100M | ~3M |
| 60fps feasible? | No | Yes |

Without the syscall, matmul is 97% of cost. With it, matmul drops to ~2% and the new bottleneck (SSM scan) is fine in BPF.

---

## CU costing

Current implementation: `100 base + 1 per MAC`. Happy to adjust — the actual CPU work is trivial (<1ms/frame native). Whatever costing works for you, as long as total matmul stays under ~500K CU/frame.

---

## What we need on the ER instance

| Requirement | Value |
|-------------|-------|
| Custom syscall | `sol_matmul_i8` (this crate) |
| CU limit per TX | ~5-10M |
| Block time | ≤16ms (your 5-10ms crank is perfect) |
| Weight account | ~4.5MB, read-only, cloned from mainnet |
| Hidden state account | ~200KB, read+write 60×/sec |
| Session state | 256B, write 60×/sec |

---

## Questions answered from the call

- Account size: 10MB max confirmed — our 4.5MB weights fit in one account
- Crank: 5-10ms built-in tick, well under our 16ms budget
- K compression: coming, will reduce rent from ~$8k to ~$80

## Still open

- How do read-only weight accounts clone from mainnet into the ER?
- Any write throughput concerns for 200KB mutated 60×/sec?
- Preferred CU costing model for the syscall?
