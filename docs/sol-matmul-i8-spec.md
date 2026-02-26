# `sol_matmul_i8` — Technical Spec for MagicBlock Ephemeral Rollups

> Enabling native INT8 matrix-vector multiplication inside ephemeral rollup validators to run a Mamba2 neural network at 60fps.

**Repo:** [autonomous-world-model](https://github.com/ScavieFae/autonomous-world-model)
**Context:** [Architecture Overview](architecture-overview.md) · [CU Benchmarks](cu-benchmark-findings.md) · [Path A: Agave Syscall](path-a-agave-syscall.md)

---

## TL;DR

We're running a 15M-parameter Mamba2 world model inside an ephemeral rollup. The model takes controller inputs and outputs game state at 60fps — it's a learned physics engine for a fighting game. Everything works except **matmul is 1,667x too expensive in BPF**. A native `sol_matmul_i8` syscall fixes this completely.

| | BPF (packed+unsafe) | With `sol_matmul_i8` |
|---|---|---|
| CU per frame (12-layer model) | ~300,000,000 | **~180,000** |
| Wall-clock per frame | ~600ms | **<1ms** |
| Fits in single TX? | No | Yes (mainnet limits) |
| 60fps feasible? | No | Yes |

---

## 1. What the Syscall Does

Multiply a row-major INT8 weight matrix by an INT8 input vector, accumulating into INT32 output:

```
y[i] = sum(W[i][j] * x[j])  for j in 0..cols

Where:
  W: [i8; rows × cols]   (row-major)
  x: [i8; cols]
  y: [i32; rows]          (accumulator, no overflow for cols ≤ 16384)
```

### Proposed Signature

```rust
fn sol_matmul_i8(
    weights: *const i8,   // r1: row-major weight matrix
    input:   *const i8,   // r2: input vector
    output:  *mut i32,    // r3: output vector (caller-allocated)
    rows:    u64,         // r4: number of rows
    cols:    u64,         // r5: number of columns
) -> u64;
```

This follows the existing Solana syscall convention (5 register args, return code). No pointers-to-pointers, no indirect addressing.

### CU Cost Model

```
cost = base_cost + (rows * cols) / mac_divisor

Suggested: base_cost = 100, mac_divisor = 100
→ 1 CU per 100 MACs
→ ~5,343 CU for the largest matmul (2048×512)
```

This is consistent with existing syscall cost models (e.g., `sol_sha256`: 85 base + 1/byte).

---

## 2. Matrix Dimensions (Every Call the Model Makes)

The model is Mamba2 with **12 identical layers**. Each layer makes exactly **2 matmul calls**:

| Operation | Weight Shape | Input Shape | Output Shape | MACs | Calls/Frame |
|-----------|-------------|-------------|-------------|------|-------------|
| **in_proj** | (2048, 512) | (512,) | (2048,) | 1,048,576 | 12 |
| **out_proj** | (512, 1024) | (1024,) | (512,) | 524,288 | 12 |

**Per frame total: 24 matmul calls, ~18.9M MACs**

The weight matrices are static (read-only during inference). The input vectors change every frame.

### Why These Dimensions

| Parameter | Value | What It Means |
|-----------|-------|---------------|
| `d_model` | 512 | Embedding dimension — the "width" of the model |
| `d_inner` | 1024 | SSM hidden dimension — 2× d_model by convention |
| `d_state` | 16 | SSM state dimension (used in selective scan, not matmul) |
| `num_layers` | 12 | Depth of the model |
| Total params | ~15M | Mostly in in_proj and out_proj weight matrices |

---

## 3. Data Types

Everything is **INT8 in, INT32 accumulate**. No floating point anywhere in the inference path.

```
weights:  i8   (quantized from fp32, per-channel symmetric, scale = abs_max / 127)
inputs:   i8   (requantized between layers via fixed-point scaling)
outputs:  i32  (accumulated dot products, requantized to i8 before next layer)
```

### Requantization (happens in BPF, between matmul calls)

```rust
output_i8[i] = clamp(round((output_i32[i] * scale) >> 16), -128, 127)
```

The `scale` is a per-layer `u16` stored in the model manifest. This is cheap (~15K CU for 2048 elements) and stays in BPF.

---

## 4. Throughput Requirement

| Metric | Value |
|--------|-------|
| Frames per second | 60 |
| Matmul calls per frame | 24 |
| **Matmul calls per second** | **1,440** |
| MACs per second | ~1.13 billion |
| Sustained for | Duration of a match (2-8 minutes) |

At native CPU speed this is trivial — ~1.13 GOPS INT8 is well under 1% of any modern CPU. The bottleneck is purely BPF VM overhead.

---

## 5. Account Layout

The syscall reads from / writes to BPF program memory, not accounts directly. But the program loads data from these accounts before calling the syscall:

### Read-Only (weights — loaded once per session)

| Account | Size | Access Pattern |
|---------|------|----------------|
| `WeightShard[0]` | ~7.5 MB | Read every frame, all 12 layers' in_proj weights |
| `WeightShard[1]` | ~7.5 MB | Read every frame, all 12 layers' out_proj weights |
| `ModelManifest` | ~2 KB | Read once at session start (architecture params + LUTs) |

**Weight storage detail:** Weights are packed contiguously. Layer 0 in_proj, layer 0 out_proj, layer 1 in_proj, ..., etc. The program computes offsets:

```rust
let in_proj_offset  = layer * (2048 * 512 + 512 * 1024);
let out_proj_offset = in_proj_offset + (2048 * 512);
```

### Read-Write (per frame)

| Account | Size | Access Pattern |
|---------|------|----------------|
| `HiddenState` | ~200 KB | Read + write every frame (Mamba2 recurrent state, d_inner × d_state × 12 layers) |
| `SessionState` | 256 B | Write every frame (game state output: positions, damage, actions) |
| `InputBuffer` | 32 B | Read every frame (both players' controller inputs) |

### Questions for MagicBlock

1. **Max account size on ER instances?** We need 7.5MB accounts. Standard Solana max is 10MB — are ER instances the same?
2. **Account cloning from mainnet?** Weight accounts live on mainnet. Do they get cloned to the ER on first access? Is there a pre-warm mechanism?
3. **Write throughput on a single account?** HiddenState (~200KB) gets mutated 60x/sec. Any concerns?

---

## 6. How It Fits in the Inference Loop

One frame of inference:

```
1. Read InputBuffer (32 bytes)                         — BPF
2. Encode inputs to model format                       — BPF (~5K CU)
3. For each of 12 layers:
   a. RMSNorm on input                                — BPF (~15K CU, LUT-based)
   b. in_proj matmul: (2048, 512) × (512,) → (2048,)  — SYSCALL (~10.6K CU)
   c. Split into gate (1024,) and SSM input (1024,)    — BPF (trivial)
   d. SSM selective scan step                          — BPF (~631K CU)
   e. SiLU gate activation                            — BPF (~13K CU, LUT)
   f. Element-wise gate × SSM output                   — BPF (~5K CU)
   g. out_proj matmul: (512, 1024) × (1024,) → (512,) — SYSCALL (~5.3K CU)
   h. Requantize i32 → i8                             — BPF (~15K CU)
   i. Residual connection                              — BPF (~5K CU)
4. Decode outputs to game state                        — BPF (~5K CU)
5. Write SessionState + HiddenState                    — BPF
```

### CU Budget With Syscall

| Component | Per Layer | 12 Layers | Notes |
|-----------|-----------|-----------|-------|
| Matmul (in_proj + out_proj) | ~15.9K | ~191K | **Via syscall** |
| SSM selective scan | ~631K | ~7.6M | BPF (scales linearly with d_inner) |
| LUT activations (SiLU, softplus, exp_neg, rsqrt) | ~40K | ~480K | BPF |
| RMSNorm + residual + requantize | ~35K | ~420K | BPF |
| **Per-layer total** | **~722K** | | |
| **Full model** | | **~8.7M** | |
| Encode/decode + overhead | | ~50K | |
| **Grand total per frame** | | **~8.7M CU** | |

Fits comfortably in a single TX even at mainnet's 1.4M CU limit? No — the SSM scan at 7.6M still exceeds it. But at **any ER CU limit above 10M**, the full 15M-param model runs in a single transaction per frame.

Without the syscall, matmul alone costs ~300M CU. **The syscall reduces total frame cost from ~308M to ~8.7M — a 35x reduction.**

---

## 7. Reference Implementation

We have a working BPF matmul kernel. The syscall would replace this hot path:

```rust
/// Current BPF implementation (~16 CU/MAC with packed+unsafe optimization)
pub fn matmul_i8_packed(
    weights: &[u8],
    input: &[u8],
    output: &mut [i32],
    rows: usize,
    cols: usize,
) {
    for i in 0..rows {
        let mut acc: i32 = 0;
        let row_base = i * cols;
        let mut j = 0;
        while j + 4 <= cols {
            unsafe {
                let w_ptr = weights.as_ptr().add(row_base + j) as *const u32;
                let x_ptr = input.as_ptr().add(j) as *const u32;
                let w4 = w_ptr.read_unaligned();
                let x4 = x_ptr.read_unaligned();
                // Extract bytes, sign-extend, multiply-accumulate
                acc += (w4 as u8 as i8 as i32) * (x4 as u8 as i8 as i32);
                acc += ((w4 >> 8) as u8 as i8 as i32) * ((x4 >> 8) as u8 as i8 as i32);
                acc += ((w4 >> 16) as u8 as i8 as i32) * ((x4 >> 16) as u8 as i8 as i32);
                acc += ((w4 >> 24) as u8 as i8 as i32) * ((x4 >> 24) as u8 as i8 as i32);
            }
            j += 4;
        }
        output[i] = acc;
    }
}
```

The native syscall version is simpler — just straight nested loops at CPU speed. On ARM with NEON, the inner loop vectorizes to 16 MACs/instruction automatically.

---

## 8. Instance Configuration Questions

For enabling this on an ER instance, we'd like to understand:

| Question | Context |
|----------|---------|
| How is the syscall enabled per-instance? | Feature flag at instance creation? Config file? Always-on for specific instance types? |
| What's the CU limit per TX on ER instances? | We need ~8.7M CU/TX with the syscall. Standard mainnet is 1.4M. |
| Is there a per-block CU limit? | At 60fps with one TX/frame, that's ~522M CU/sec sustained. |
| Block time configuration? | We need ≤16ms for 60fps. Your docs say 10ms minimum — perfect. |
| Crank integration? | We use a 16ms crank to drive inference. Is this the built-in task scheduler? |
| Account size limits? | 7.5MB weight accounts. Same as mainnet 10MB cap? |
| Weight account cloning? | Weights live on mainnet. How do they get loaded into the ER? Delegation? Direct clone? Pre-warm? |

---

## 9. What We Have Working

| Component | Status | Where |
|-----------|--------|-------|
| INT8 quantization pipeline | Done | `quantization/quantize_mamba2.py` |
| Activation LUT generation | Done | `quantization/generate_luts.py` |
| CU benchmarks (matmul, LUT, SSM) | Done | `solana/programs/cu-benchmark/` |
| Packed matmul kernel (1.52x opt) | Done | `docs/packed-matmul-explainer.md` |
| BOLT ECS components (6) | Done | `solana/programs-ecs/components/` |
| Inference system (BPF) | Done | `solana/programs-ecs/systems/run-inference/` |
| Session lifecycle system | Done | `solana/programs-ecs/systems/session-lifecycle/` |
| Frame visualizer (Canvas 2D) | Done | `viz/` |
| Syscall implementation (Agave fork) | Specced | `docs/path-a-agave-syscall.md` |
| ER integration (delegation + crank) | Specced | `docs/path-b-magicblock-er.md` |

We're ready to integrate as soon as the syscall is available on an ER instance.

---

## 10. The Big Picture

We're building an **autonomous world** — a learned physics engine deployed onchain as a public good. Trained on millions of frames of competitive Melee replay data, quantized to INT8, running deterministically in Solana ephemeral rollups.

The model IS the world. No game logic. No hardcoded physics. Just 15 million integers being multiplied together 60 times a second. The `sol_matmul_i8` syscall is the key that makes the target model size feasible at real-time frame rates.

```
Training data (Melee replays)
        ↓
Mamba2 model (fp32)
        ↓
INT8 quantization → weights on Solana mainnet (permanent, forkable)
        ↓
Ephemeral rollup session → sol_matmul_i8 → 60fps onchain inference
        ↓
WebSocket → Browser visualizer → you're watching an AI dream about Melee
```
