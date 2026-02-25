# Path A: Custom Agave Syscall (`sol_matmul_i8`)

## The Idea

Fork the Solana validator (Agave). Add a native syscall that performs INT8 matrix-vector multiplication at CPU speed instead of interpreted BPF. A 512×1024 matmul that costs ~13M CU in BPF drops to ~5K CU as a syscall. The 15M parameter model becomes trivially feasible in a single transaction.

## Why This Works

Solana's BPF VM is an interpreter (with JIT). Each BPF instruction costs 1 CU. Our matmul inner loop compiles to ~25 BPF instructions per multiply-accumulate. But the *actual CPU work* of an i8×i8 multiply is one ARM instruction. The 25:1 overhead is pure VM tax.

A syscall bypasses the VM entirely. The host CPU runs native Rust/ARM code. On Apple Silicon with NEON SIMD, you can do 16 i8 MACs per instruction. A 512×1024 matmul (524K MACs) takes ~33K native instructions — microseconds of wall time.

## What Exists Already

Agave has ~20 syscalls. The pattern is well-established:

| Syscall | What It Does | CU Cost Model |
|---------|-------------|---------------|
| `sol_sha256` | Hash memory region | 85 base + 1/byte |
| `sol_keccak256` | Hash memory region | 85 base + 1/byte |
| `sol_secp256k1_recover` | Signature recovery | 25,000 flat |
| `sol_curve25519_mul` | Elliptic curve multiply | 2,177 flat |
| `sol_poseidon` | ZK-friendly hash | 61n² + 542 |

The hash syscalls are the closest analog — they take a pointer to a memory region, process it, write output. Our matmul syscall does the same thing.

## The Implementation

### Files to touch (3-4 for a hackathon fork):

**1. `programs/bpf_loader/src/syscalls/mod.rs`** — the implementation

```rust
declare_builtin_function!(
    SyscallMatmulI8,
    fn rust(
        invoke_context: &mut InvokeContext,
        weights_addr: u64,    // r1: pointer to i8 weight matrix
        input_addr: u64,      // r2: pointer to i8 input vector
        output_addr: u64,     // r3: pointer to i32 output vector
        dims_addr: u64,       // r4: pointer to [rows: u32, cols: u32]
        _arg5: u64,
        memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Error> {
        // Read dimensions
        let dims = translate_slice::<u32>(
            memory_mapping, dims_addr, 2,
            invoke_context.get_check_aligned(),
        )?;
        let rows = dims[0] as usize;
        let cols = dims[1] as usize;

        // Charge CU: base + per-MAC cost
        let cost = 100 + (rows * cols) as u64 / 100; // ~1 CU per 100 MACs
        consume_compute_meter(invoke_context, cost)?;

        // Get memory slices
        let weights = translate_slice::<i8>(
            memory_mapping, weights_addr, (rows * cols) as u64,
            invoke_context.get_check_aligned(),
        )?;
        let input = translate_slice::<i8>(
            memory_mapping, input_addr, cols as u64,
            invoke_context.get_check_aligned(),
        )?;
        let output = translate_slice_mut::<i32>(
            memory_mapping, output_addr, rows as u64,
            invoke_context.get_check_aligned(),
        )?;

        // Native matmul — runs at full CPU speed
        for i in 0..rows {
            let mut acc: i32 = 0;
            let row = &weights[i * cols..(i + 1) * cols];
            for j in 0..cols {
                acc += row[j] as i32 * input[j] as i32;
            }
            output[i] = acc;
        }

        Ok(0)
    }
);
```

Then register it in `create_program_runtime_environment_v1()`:
```rust
result.register_function_hashed(*b"sol_matmul_i8", SyscallMatmulI8::vm)?;
```

**2. `compute-budget/src/compute_budget.rs`** — CU cost constants

```rust
pub matmul_i8_base_cost: u64 = 100,
pub matmul_i8_mac_cost_divisor: u64 = 100, // 1 CU per 100 MACs
```

**3. `sdk/program/src/syscalls/definitions.rs`** — SDK declaration

```rust
define_syscall!(fn sol_matmul_i8(
    weights: *const i8,
    input: *const i8,
    output: *mut i32,
    dims: *const u32,
    _unused: u64
) -> u64);
```

**4. Your BPF program** — calling the syscall

```rust
// In your Solana program:
extern "C" {
    fn sol_matmul_i8(
        weights: *const i8,
        input: *const i8,
        output: *mut i32,
        dims: *const u32,
        unused: u64,
    ) -> u64;
}

fn matmul(weights: &[i8], input: &[i8], output: &mut [i32], rows: u32, cols: u32) {
    let dims = [rows, cols];
    unsafe {
        sol_matmul_i8(
            weights.as_ptr(),
            input.as_ptr(),
            output.as_mut_ptr(),
            dims.as_ptr(),
            0,
        );
    }
}
```

### Estimated LOC: ~200-300 lines across 4 files.

## CU Impact

| Operation | BPF (current) | BPF (packed) | Syscall |
|-----------|-------------|-------------|---------|
| 512×1024 matmul | ~13.1M CU | ~8.4M CU | ~5,343 CU |
| 1024×512 matmul | ~13.1M CU | ~8.4M CU | ~5,343 CU |
| Per Mamba2 layer | ~40M CU | ~26M CU | ~15K CU |
| 12-layer model | ~480M CU | ~312M CU | **~180K CU** |

**180K CU for the full 15M parameter model.** Fits in a single mainnet transaction (1.4M limit) with room to spare.

## Hackathon Plan

### Day 0 (Setup, ~2 hours)
- Fork anza-xyz/agave at a stable tag (v2.1.x or v3.0.x)
- Build locally (`cargo build --release -p solana-test-validator`)
- Verify the unmodified validator works

### Day 1 (Syscall, ~4-6 hours)
- Add `SyscallMatmulI8` in the syscalls module
- Add cost constants
- Add SDK definition
- Build, run validator, test with a minimal BPF program that calls the syscall
- Benchmark: measure CU for various matrix sizes

### Day 2 (Model Integration, ~4-6 hours)
- Write the Mamba2 forward pass program using the syscall for all matmuls
- Upload quantized weights to the local validator
- Run full inference, measure total CU
- Demo: feed controller inputs, get frame state output

### Day 3 (Polish + Demo)
- Client visualization (reuse existing visualizer)
- Recording / slides
- Handle edge cases

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Agave build is complex (~500K LOC) | Medium | Only touch 4 files. Don't modify anything else. |
| Validator fork diverges from upstream | Low | Hackathon is a snapshot. No need to track upstream. |
| Syscall ABI breaks between versions | Low | Pin to one Agave version. |
| CU cost model is wrong | Low | Doesn't matter for a hackathon demo on your own validator. |
| ARM NEON not available on validator | Low | Even scalar native code is 100x faster than BPF. NEON is a bonus. |

## The Pitch

"We added a tensor math instruction to the Solana VM and ran a 15M parameter neural network in a single transaction. The model IS the world — no off-chain compute, no ZK proofs, no oracles. Pure onchain inference at 180K CU."
