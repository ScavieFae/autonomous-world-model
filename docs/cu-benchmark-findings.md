# CU Benchmark Findings: INT8 Inference on Solana

**Date:** 2025-02-25
**Program:** cu-benchmark (anchor-lang 0.32.1, cargo build-sbf)
**Validator:** solana-test-validator 3.0.15 (local, default CU limits)

---

## Raw Results

### INT8 Matrix-Vector Multiply

| Dimensions | MACs | CU Consumed | CU/MAC | Notes |
|------------|------|-------------|--------|-------|
| 64×64 | 4,096 | 105,678 | 25.8 | Baseline |
| 128×128 | 16,384 | 406,010 | 24.8 | |
| 256×256 | 65,536 | >1,400,000 | — | Exceeded default limit |
| 512×512 | 262,144 | >1,400,000 | — | Exceeded default limit |

**Key number: ~25 CU per multiply-accumulate operation.**

The plan estimated ~3 CU/MAC. Actual cost is ~8x higher.

### INT8 Matmul Tiled (4x Unrolled)

| Dimensions | CU Consumed | Notes |
|------------|-------------|-------|
| 256×256 | >1,400,000 | Exceeded — unrolling didn't help enough |
| 512×512 | >1,400,000 | Exceeded |

The 4x unroll doesn't meaningfully reduce CU. The bottleneck is per-instruction cost, not loop overhead.

### LUT Activations

| Function | Elements | CU Consumed | CU/Lookup |
|----------|----------|-------------|-----------|
| SiLU | 256 | 4,791 | 18.7 |
| SiLU | 512 | 7,607 | 14.9 |
| SiLU | 1,024 | 13,243 | 12.9 |
| softplus | 256 | 4,792 | 18.7 |
| softplus | 512 | 7,608 | 14.9 |
| softplus | 1,024 | 13,244 | 12.9 |
| rsqrt | 256 | 4,793 | 18.7 |
| rsqrt | 512 | 7,609 | 14.9 |
| rsqrt | 1,024 | 13,245 | 12.9 |

LUT lookups are cheap. ~13-19 CU per lookup (amortizes with more elements). Activation functions are not the bottleneck.

### SSM Selective Scan Step

| d_inner | d_state | CU Consumed | Notes |
|---------|---------|-------------|-------|
| 256 | 16 | 159,248 | |
| 512 | 16 | 316,432 | Scales linearly |
| 1,024 | 16 | 630,808 | Full target size |

SSM step scales linearly with d_inner. At target size (1024×16), it's 631K CU — manageable.

### Full Mamba2 Layer (d_model=512, d_inner=1024, d_state=16)

| Benchmark | CU Consumed | Notes |
|-----------|-------------|-------|
| full_layer | >1,400,000 | Exceeded default 1.4M CU limit |

Even one layer exceeds the default CU limit.

---

## Revised Cost Model

Using measured ~25 CU/MAC:

| Operation | Dimensions | MACs | Estimated CU |
|-----------|-----------|------|-------------|
| in_proj | 512 → 2048 | 1,048,576 | ~26.2M |
| SSM step | 1024 × 16 | — | ~631K (measured) |
| Gate (SiLU) | 1024 | — | ~13K (measured) |
| out_proj | 1024 → 512 | 524,288 | ~13.1M |
| RMSNorm + residual | 512 | — | ~15K |
| **Per layer** | | | **~40M CU** |
| **12-layer model** | | | **~480M CU** |

The plan estimated 59M. Actual projected cost is **~480M CU** — roughly 8x higher.

---

## What Is a CU?

A Compute Unit is Solana's gas metering unit. Every BPF/SBF instruction executed by the VM costs CU. The mapping is roughly:

- **Each BPF instruction ≈ 1 CU** (this is the baseline cost model)
- But a single `i8 × i8` multiply-accumulate in our Rust code compiles to **multiple BPF instructions**: sign-extend i8→i32, multiply, add to accumulator, increment loop counter, bounds check, branch — easily 20-30 BPF instructions per MAC
- Memory access (array indexing) adds bounds-check instructions
- The SBF VM is a register machine with 64-bit registers but no SIMD

**This explains the ~25 CU/MAC:** it's not that CUs are expensive, it's that each high-level operation compiles to many BPF instructions. The inner loop of:
```rust
let w = weights[row_offset + j] as i8 as i32;
let x = input[j] as i8 as i32;
acc += w * x;
```
involves: 2 index calculations, 2 bounds checks, 2 loads, 2 sign extensions, 1 multiply, 1 add, loop counter increment, comparison, branch = ~15-25 instructions.

---

## How Far Can Ephemeral Rollups Go?

MagicBlock ephemeral rollups run a modified Solana validator. Key parameters:

- **CU limit is configurable** — this is the whole point of ephemeral rollups for compute-heavy applications
- Default Solana mainnet: 1.4M CU per transaction, 48M CU per block
- Ephemeral rollups: CU ceiling can be raised, but **how high is an open question**
- Block time: 10ms (vs mainnet's 400ms)

### Questions to resolve with MagicBlock team:
1. What is the practical CU ceiling per transaction in an ER?
2. Is there a CU ceiling per block?
3. At 480M CU per frame, can a single ER transaction handle this?
4. If not, can multiple ordered transactions within a single 10ms block?
5. What's the actual wall-clock time for executing 480M BPF instructions?

### Wall-clock estimate:
- Modern CPU executes BPF at roughly 100-500M instructions/sec (interpreted/JIT)
- 480M CU ≈ 480M BPF instructions
- At 200M inst/sec: **~2.4 seconds per frame** — way too slow for 60fps
- At 500M inst/sec (JIT): **~960ms per frame** — still too slow
- We need either: much smaller model, or fundamentally different execution

**This is a potential showstopper independent of CU limits.** Even if the ER allows unlimited CU, the wall-clock execution time for 480M BPF instructions may exceed the frame budget.

---

## Where Can We Find Optimizations?

### 1. Compiler-level: Reduce BPF instructions per MAC

The inner loop is naive Rust compiled to BPF. Possible wins:

- **Unsafe indexing** — skip bounds checks with `get_unchecked()`. Could save ~2-4 CU/MAC (the bounds checks are a significant fraction). ~20-30% speedup.
- **Packed operations** — load 4 bytes at once as a u32, extract individual bytes, multiply. Reduces load instructions. Requires careful alignment.
- **Inline assembly** — SBF supports inline asm. Hand-write the hot inner loop. Maximum control but maintainability cost.
- **Compiler flags** — ensure LTO, opt-level=3, no debug. Check what `cargo build-sbf` actually passes.

**Estimated ceiling: 2-3x improvement → ~10 CU/MAC → ~190M CU per model**

### 2. Algorithmic: Reduce total MACs

- **Structured sparsity** — if 50% of weights are zero, skip those multiplies. Requires sparse storage format. 2x reduction if achievable.
- **Weight clustering** — group similar weights, compute once per cluster. Effective for highly redundant weight matrices.
- **Low-rank factorization** — decompose W (m×n) into A (m×r) × B (r×n) where r << min(m,n). If rank-64 approximation works: 512×64 + 64×2048 = 163K MACs vs 1M MACs. ~6x reduction. **This is probably the biggest single win.**
- **Mixed precision** — use INT4 for less-sensitive layers. 4 values packed per byte, fewer loads. Risk: Mamba2's gating may degrade.

### 3. Architectural: Change the model shape

See model size analysis below.

### 4. Execution model: Parallelize across transactions

Even with optimizations, a single tx may be too much. Options:

- **Layer-parallel pipeline** — each tx does 1-2 layers, reads/writes HiddenState account between. Needs ordered execution within a block.
- **Matmul tiling across txs** — split a single large matmul into tiles, each in its own tx. Accumulate in an intermediate account. More complex but higher parallelism.
- **Precompute + lookup** — for repeated inference with same weights, precompute partial products. Probably not applicable (inputs change every frame).

---

## What Different Model Sizes Change

| Model | d_model | d_inner | d_state | layers | Params | MACs/step | Est. CU (@25) | Est. CU (@10 optimized) |
|-------|---------|---------|---------|--------|--------|-----------|---------------|------------------------|
| Target (15M) | 512 | 1024 | 16 | 12 | 15M | ~19M | ~480M | ~190M |
| Small (4M) | 256 | 512 | 16 | 8 | 4M | ~3.2M | ~80M | ~32M |
| Tiny (1M) | 128 | 256 | 16 | 6 | 1M | ~600K | ~15M | ~6M |
| Micro (250K) | 64 | 128 | 8 | 4 | 250K | ~100K | ~2.5M | ~1M |

### Analysis:

- **Micro (250K params):** Fits in a single mainnet transaction at 1.4M CU (with optimizations). But can it play Melee? Almost certainly not — too few parameters for complex dynamics.

- **Tiny (1M params):** Fits in a single ER transaction even unoptimized. With optimization, fits in ~4 mainnet txs. Could capture basic movement physics. Worth prototyping to see quality floor.

- **Small (4M params):** With optimization, ~32M CU. Feasible in a single ER transaction if CU ceiling is 50M+. This is the sweet spot to investigate — might capture enough dynamics while being computationally tractable.

- **Target (15M params):** Even optimized, 190M CU. Needs either very generous ER CU limits or multi-tx pipeline. Wall-clock time is the real concern.

### The quality question:
We don't yet know the minimum model size that produces "interesting" Melee dynamics. This is an empirical question that requires training runs at different scales. **Recommend training a 1M and 4M parameter model in parallel with the 15M target.**

---

## How Else Can We Change the Math?

### 1. Custom VM / Precompile

If MagicBlock's ER validator is modified Solana, could we add a **native INT8 matmul precompile**?

- A syscall that takes two account slices and does matmul natively (C/Rust, not BPF)
- Would reduce CU cost to ~1-2 CU/MAC (the actual CPU work is trivial)
- 15M param model at 1 CU/MAC: ~19M CU — easily fits in one tx
- **This is the nuclear option. Changes the economics completely.**
- Requires MagicBlock cooperation / custom validator fork

### 2. Hybrid: Offchain Compute + Onchain Verification

- Run inference offchain (any hardware)
- Submit result + proof onchain
- Verify proof is cheaper than running inference
- Options: optimistic (fraud proof), ZK (validity proof), TEE (attestation)
- **Breaks "the model IS the world" philosophy** but may be pragmatically necessary

### 3. WASM instead of BPF?

- Some L2s/rollups run WASM VMs
- WASM has SIMD proposals (128-bit vectors)
- i8x16 multiply: 16 MACs per instruction
- Could reduce cost by 8-16x
- **Not applicable to Solana/MagicBlock as-is**

### 4. Reduce Inference Frequency

- Not every frame needs full inference
- Could run model at 15fps (every 4th frame) and interpolate
- 4x reduction in compute budget
- Interpolation for continuous values (position, velocity) is straightforward
- Discrete values (action_state) need different handling — hold previous value

### 5. Progressive Model Loading

- Don't load all 12 layers at once
- Run first 6 layers in tx 1, store intermediate
- Run last 6 layers in tx 2, produce output
- Doubles latency but halves per-tx CU

---

## Where Can We Probe Deeper?

### Immediate experiments (prototype-ready):

1. **Unsafe inner loop** — rewrite bench_matmul with `unsafe { get_unchecked() }`, measure CU reduction. 30 minutes of work, could show 2-3x improvement.

2. **Packed byte loading** — load 4 or 8 bytes at once, extract and multiply. Test if the reduced load count helps on BPF.

3. **`cargo build-sbf` optimization flags** — check what optimization level is used. Try `RUSTFLAGS` overrides.

4. **Tiny model benchmark** — hardcode a 64×128×8×4 model directly in a Solana program (no account loading). Measure total CU for a complete forward pass. Answers: "can we run ANY model in a single tx?"

5. **Account data loading cost** — measure how much CU goes to just reading data vs. actual compute. If data loading dominates, zero-copy patterns matter more.

### Requires external answers:

6. **MagicBlock CU ceiling** — email/Discord: "What's the maximum CU per transaction in an ephemeral rollup? What's the wall-clock execution budget?"

7. **MagicBlock custom precompiles** — "Can we add a native matmul syscall to the ER validator?"

8. **Training at smaller scales** — train 1M and 4M param Mamba2 models on Melee replays. Evaluate quality. This determines whether the small models are worth pursuing onchain.

### Longer-term research:

9. **Low-rank decomposition** — apply SVD to trained weight matrices. Measure accuracy loss at rank 32, 64, 128. Could be the biggest win.

10. **Structured sparsity** — prune trained model to 50% sparsity. Measure accuracy. If quality holds, implement sparse matmul kernel.

11. **BPF JIT performance** — profile actual wall-clock time on the validator for large CU transactions. Is the relationship linear? Does the JIT help?

---

## What Can We Prototype Now?

### Prototype 1: Optimized Matmul Kernel (next step)
Add `bench_matmul_unsafe` instruction with bounds-check-free inner loop. Measure the gap. This tells us the optimization ceiling.

### Prototype 2: Tiny Model End-to-End
Hardcode a d_model=64, d_inner=128, d_state=8, 4-layer model. All weights inline (no account loading). Complete forward pass in one instruction. Measure total CU. This is the existence proof: "can any neural network run in a single Solana transaction?"

### Prototype 3: Multi-TX Pipeline
Implement a 2-transaction pipeline: tx1 runs layers 0-5, writes intermediate state to an account. tx2 reads intermediate, runs layers 6-11, writes output. Test with the local validator. This validates the multi-tx architecture.

### Prototype 4: CU Test on MagicBlock
Deploy the existing benchmark to a MagicBlock ephemeral rollup with raised CU limits. Run the 256×256 and 512×512 matmuls that exceeded on local. Get actual numbers.

---

## Bottom Line

| | Estimated (plan) | Measured | Gap |
|---|---|---|---|
| CU per MAC | ~3 | ~25 | 8x |
| Per layer | 4.9M | ~40M | 8x |
| 12-layer model | 59M | ~480M | 8x |
| Single-tx feasibility | Maybe (60M) | No (480M) | — |

**The plan's CU estimates were 8x too optimistic.** This doesn't kill the project — it redirects it.

### Update: Packed matmul optimization achieved 1.52x improvement

After implementing packed u32 loads + unsafe indexing, we got from ~25 to ~16 CU/MAC. See `docs/packed-matmul-explainer.md` for the full technical explanation.

Updated bottom line with packed optimization:

| | Estimated (plan) | Measured (naive) | Measured (packed) | Syscall (projected) |
|---|---|---|---|---|
| CU per MAC | ~3 | ~25 | ~16 | ~0.01 |
| Per layer | 4.9M | ~40M | ~26M | ~15K |
| 12-layer model | 59M | ~480M | ~300M | ~180K |

### Two paths forward:

**Path A: Custom Agave Syscall** (`docs/path-a-agave-syscall.md`)
Fork the Solana validator, add a native `sol_matmul_i8` syscall. 15M model drops to ~180K CU — fits in a single mainnet transaction. Maximum flex, maximum effort.

**Path B: MagicBlock Ephemeral Rollup** (`docs/path-b-magicblock-er.md`)
Run inference in a MagicBlock ER with their sponsor track. Smaller model (1-4M params) + packed optimization + configurable CU limits. More pragmatic, hackathon-aligned.

Both paths are viable. Path A is the better demo ("we modified the Solana VM"). Path B is the better product ("it runs on existing infrastructure").
