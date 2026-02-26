# `sol_matmul_i8` — Technical Spec

> Native INT8 matrix-vector multiply for MagicBlock ephemeral rollup validators.

**Repo:** [autonomous-world-model](https://github.com/ScavieFae/autonomous-world-model) · **Benchmarks:** [CU findings](cu-benchmark-findings.md)

---

## TL;DR

We're running a 15M-parameter Mamba2 model (learned fighting game physics) inside an ephemeral rollup at 60fps. Matmul is the bottleneck — **~300M CU/frame in BPF, ~8.7M with a native syscall.** Everything else (SSM scan, activations, requantization) is fine in BPF.

| | BPF (packed+unsafe) | With syscall |
|---|---|---|
| CU per frame | ~300,000,000 | **~8,700,000** |
| Wall-clock per frame | ~600ms | **<1ms** |
| 60fps feasible? | No | Yes |

---

## The Syscall

Multiply a row-major INT8 weight matrix by an INT8 input vector, accumulate into INT32:

```
y[i] = sum(W[i][j] * x[j])  for j in 0..cols
```

### Signature

```rust
fn sol_matmul_i8(
    weights: *const i8,   // r1: row-major weight matrix
    input:   *const i8,   // r2: input vector
    output:  *mut i32,    // r3: output vector (caller-allocated)
    rows:    u64,         // r4
    cols:    u64,         // r5
) -> u64;
```

Standard 5-register syscall convention. The native implementation is just nested loops — on ARM with NEON the inner loop auto-vectorizes to 16 MACs/instruction.

### CU Costing

For reference, `sol_sha256` costs 85 base + 1/byte. A similar model here (base + linear in MACs) would work. We're not opinionated on the exact constants — happy to work with whatever costing you prefer.

---

## Matrix Dimensions

The model has **12 identical layers**, each making **2 matmul calls**:

| Operation | Weight Shape | Input Shape | Output Shape | MACs |
|-----------|-------------|-------------|-------------|------|
| **in_proj** | (2048, 512) | (512,) | (2048,) | 1,048,576 |
| **out_proj** | (512, 1024) | (1024,) | (512,) | 524,288 |

**Per frame: 24 calls, ~18.9M MACs, sustained at 60fps = 1,440 calls/sec.**

At native CPU speed this is trivial (~1.13 GOPS INT8, well under 1% of any modern CPU). The bottleneck is purely BPF VM overhead — each MAC compiles to ~16-25 BPF instructions due to bounds checks, sign extension, and loop overhead.

### Model Parameters

| Parameter | Value |
|-----------|-------|
| `d_model` | 512 |
| `d_inner` | 1024 |
| `d_state` | 16 |
| `num_layers` | 12 |
| Total params | ~15M |

---

## Data Types

All `i8 × i8 → i32`. No floating point anywhere in the inference path.

```
weights:  i8   (per-channel symmetric quantization)
inputs:   i8   (requantized between layers via fixed-point u16 scales)
outputs:  i32  (dot product accumulators)
```

Between matmul calls, the program requantizes `i32 → i8` in BPF (~15K CU, not a bottleneck).

---

## Per-Frame CU Breakdown (With Syscall)

```
For each of 12 layers:
  RMSNorm                    — BPF, ~15K CU (LUT-based)
  in_proj  (2048,512)×(512,) — SYSCALL
  SSM selective scan step    — BPF, ~631K CU (measured)
  SiLU gate                  — BPF, ~13K CU (LUT)
  out_proj (512,1024)×(1024,)— SYSCALL
  Requantize + residual      — BPF, ~20K CU
```

| Component | 12 Layers |
|-----------|-----------|
| Matmul (syscall) | ~191K |
| SSM selective scan | ~7.6M |
| LUT activations | ~480K |
| RMSNorm + requant + residual | ~420K |
| **Total** | **~8.7M CU** |

Matmul drops from 97% of cost to 2%. The new bottleneck is the SSM scan at 7.6M — but that's fine in BPF.

---

## Account Layout

| Account | Size | Access |
|---------|------|--------|
| `WeightShard[0]` | ~7.5 MB | Read-only, every frame |
| `WeightShard[1]` | ~7.5 MB | Read-only, every frame |
| `ModelManifest` | ~2 KB | Read once at session start |
| `HiddenState` | ~200 KB | Read+write every frame (Mamba2 recurrent state) |
| `SessionState` | 256 B | Write every frame (game state output) |
| `InputBuffer` | 32 B | Read every frame (controller inputs) |

Weights live on mainnet (permanent, forkable). Session accounts delegate to the ER.

---

## Questions

| Question | Context |
|----------|---------|
| **Max account size on ER?** | We need 7.5MB weight accounts. Same 10MB cap as mainnet? |
| **Weight cloning from mainnet?** | How do read-only weight accounts get loaded into the ER? Delegation? Clone on first access? Pre-warm? |
| **CU limit per TX?** | We need ~10M CU/TX with the syscall. |
| **Per-block CU limit?** | 60fps × ~8.7M = ~522M CU/sec sustained. |
| **Block time?** | Need ≤16ms. Your docs say 10ms minimum — perfect. |
| **Crank?** | We drive inference via a 16ms tick. Is this the built-in task scheduler? |
| **How is the syscall enabled?** | Feature flag per instance? Config? We'd want it on a dedicated instance. |
| **Write throughput?** | HiddenState (~200KB) mutated 60x/sec on a single account. Any concerns? |
