# Packed INT8 Matmul: What We Did and Why It's 1.5x Faster

## The Problem

A matrix-vector multiply is the core operation in neural network inference. For each output element, you compute a dot product:

```
y[i] = W[i][0]*x[0] + W[i][1]*x[1] + ... + W[i][n]*x[n]
```

Our weights and inputs are INT8 (one byte each). The naive Rust inner loop:

```rust
for j in 0..cols {
    let w = weights[row_offset + j] as i8 as i32;
    let x = input[j] as i8 as i32;
    acc += w * x;
}
```

This compiles to roughly **25 BPF instructions per multiply-accumulate** on Solana's SBF VM:

```
                              BPF instructions
                              ────────────────
weights[row_offset + j]   →   add (compute index)
                              bounds check (compare + conditional jump)
                              ldxb (load 1 byte)
as i8 as i32              →   lsh + arsh (sign-extend byte to 32-bit)
input[j]                  →   bounds check (compare + conditional jump)
                              ldxb (load 1 byte)
as i8 as i32              →   lsh + arsh (sign-extend)
w * x                     →   mul32
acc += ...                →   add32
loop overhead             →   add (j++), compare, ja (jump)
                              ────────────────
                              ~20-25 instructions total
```

Each BPF instruction costs 1 CU. So ~25 CU per MAC.

## What "Packed" Does

Instead of loading one byte at a time, we load 4 bytes at once as a `u32`, then extract the individual bytes with bit shifts:

```rust
// Before: 4 separate byte loads (4 load instructions + 4 bounds checks)
let w0 = weights[base]     as i8 as i32;
let w1 = weights[base + 1] as i8 as i32;
let w2 = weights[base + 2] as i8 as i32;
let w3 = weights[base + 3] as i8 as i32;

// After: 1 u32 load + 3 shifts (no bounds checks, 1 load instruction)
let w4: u32 = ptr.read_unaligned();
let w0 = (w4 as u8)         as i8 as i32;   // low byte, free
let w1 = ((w4 >> 8) as u8)  as i8 as i32;   // 1 shift
let w2 = ((w4 >> 16) as u8) as i8 as i32;   // 1 shift
let w3 = ((w4 >> 24) as u8) as i8 as i32;   // 1 shift
```

For 4 MACs, the instruction count goes from:

```
Naive:   4 × (load + bounds_check + sign_extend + load + bounds_check + sign_extend + mul + add)
       = 4 × ~20 = ~80 instructions

Packed:  2 × (ptr_load) + 6 × (shift + truncate) + 8 × (sign_extend) + 4 × (mul + add) + loop
       = 2 + 6 + 8 + 8 + ~4 = ~28 instructions
```

So for 4 MACs: **~28 vs ~80 instructions. ~2.9x fewer.**

The measured improvement is 1.5x (not 2.9x) because there's fixed overhead per row (loop setup, output scaling) and the sign-extension is still per-element. But 1.5x is real and compounds across the full model.

## Combined With Unsafe

We also use `unsafe { get_unchecked() }` to eliminate bounds checks. Bounds checking alone accounted for ~20% of the cost (confirmed: 406K → 340K CU at 128×128). The packed variant is inherently unsafe (raw pointer reads), so we get both wins together.

Safety is maintained by checking dimensions once at the start:
```rust
require!(data.len() >= total_needed, BenchError::InsufficientData);
// After this point, all accesses are within bounds by construction.
unsafe { ... }
```

## Measured Results

128×128 matmul (16,384 MACs):

| Variant | CU | CU/MAC | Speedup |
|---------|-----|--------|---------|
| Naive (bounds-checked) | 406,010 | 24.8 | 1.0x |
| Unsafe only | 339,844 | 20.7 | 1.19x |
| Packed + unsafe | 266,380 | 16.3 | **1.52x** |

256×256 matmul (65,536 MACs):

| Variant | CU | CU/MAC | Speedup |
|---------|-----|--------|---------|
| Naive | >1,400,000 | — | exceeded |
| Unsafe only | 1,332,996 | 20.3 | — |
| Packed + unsafe | 1,038,604 | **15.8** | — |

Note: the naive 4x-unrolled variant was actually *slower* (507K vs 406K at 128×128). More register pressure without reducing loads doesn't help on a register-limited VM like SBF.

## What This Means for the Model

| Model size | MACs/step | CU @ 25/MAC | CU @ 16/MAC (packed) |
|------------|-----------|-------------|---------------------|
| 15M params (target) | ~19M | ~480M | ~300M |
| 4M params | ~3.2M | ~80M | ~51M |
| 1M params | ~600K | ~15M | ~9.6M |

Packed matmul alone doesn't make the 15M model fit in a single transaction. But it makes the 4M model much more tractable, and stacks with other optimizations (low-rank decomposition, reduced inference frequency, custom precompiles).

## Further Optimization Directions

What we haven't tried yet:

1. **u64 packed loads** — load 8 bytes at once. Diminishing returns (more shifts) but worth measuring.
2. **Inline SBF assembly** — hand-write the inner loop. Eliminate any remaining compiler overhead.
3. **Reordering accumulation** — accumulate into i64 to avoid overflow checks that the compiler may insert.
4. **Loop structure** — experiment with row-major vs column-major traversal for cache/memory access patterns in the BPF VM.
