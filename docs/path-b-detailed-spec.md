# Path B Detailed Spec: MagicBlock Ephemeral Rollup

## Overview

Two-phase approach:
- **Phase 1 (hackathon):** Pure BPF inference in a MagicBlock ER, packed matmul optimization, smallest viable model. Standard stack, sponsor-friendly.
- **Phase 2 (proposal):** Custom `sol_matmul_i8` syscall in a self-hosted ER. 15M model in a single TX. Presented as "here's where this goes with native tensor support."

This doc specs Phase 1 in detail.

---

## Model Size Decision

We need to pick a model size that balances quality vs CU feasibility. Based on benchmarks:

| Model | Params | CU (packed, 16/MAC) | Fits in... | Quality guess |
|-------|--------|---------------------|------------|---------------|
| Micro | 250K | ~1.6M | Single mainnet TX | Probably garbage |
| Tiny | 1M | ~9.6M | Single ER TX easily | Basic movement? |
| Small | 4M | ~51M | Single ER TX if CU raised | Interesting dynamics? |
| Target | 15M | ~300M | Multi-TX or syscall only | Full Melee physics |

**Target: 4M (minimum) or 15M (full) with custom `sol_matmul_i8` syscall in the ER.**

The custom syscall changes the math completely. Native CPU execution drops cost to ~1 CU per 100 MACs. Both 4M and 15M models become trivially feasible in a single transaction:

| Model | Params | CU (pure BPF, packed) | CU (with syscall) | Verdict |
|-------|--------|-----------------------|--------------------|---------|
| Small | 4M | ~51M | ~32K | Trivial |
| Target | 15M | ~300M | ~190K | Trivial |

Without the syscall (pure BPF fallback), the 4M model at ~51M CU still works if the ER allows raised CU limits. The 15M model requires the syscall or multi-TX pipeline.

**The biggest unknowns are NOT infra — they're ML:**
1. Can a 4-15M Mamba2 learn Melee physics with sufficient fidelity?
2. Does INT8 quantization preserve that fidelity? Mamba2's Δ-dependent gating is more quantization-sensitive than transformer FFN layers.

These require training runs, not more Solana optimization.

### 4M Model Architecture

```
d_model  = 256
d_inner  = 512
d_state  = 16
n_layers = 8
```

Approximate parameter count:
- Per layer: in_proj (256×1024) + out_proj (512×256) + SSM params ≈ 524K
- 8 layers × 524K ≈ 4.2M
- Input/output embeddings ≈ ~100K
- **Total: ~4M parameters, ~4MB INT8**

CU breakdown per layer (pure BPF packed, ~16 CU/MAC):
- in_proj: 256 × 1024 = 262K MACs → ~4.2M CU
- SSM step: 512 × 16 = 8K ops → ~130K CU
- Gate: 512 LUT lookups → ~7K CU
- out_proj: 512 × 256 = 131K MACs → ~2.1M CU
- **Per layer: ~6.4M CU**
- **8 layers: ~51M CU**

With syscall: ~32K CU total. With pure BPF packed: ~51M CU (needs raised ER CU limit).

### 15M Model Architecture

```
d_model  = 512
d_inner  = 1024
d_state  = 16
n_layers = 12
```

- **Total: ~15M parameters, ~15MB INT8 (2 weight shards)**
- Pure BPF packed: ~300M CU (infeasible without syscall)
- With syscall: ~190K CU (trivial)

---

## Account Layout

### On Mainnet (permanent)

**4M model:**

| Account | Size | Contents | Rent deposit |
|---------|------|----------|-------------|
| `WeightAccount` | ~4MB | INT8 weights for all 8 layers | ~28 SOL |
| `ModelManifest` | ~512B | Architecture params, quant scales, LUT data | ~0.004 SOL |

**15M model:**

| Account | Size | Contents | Rent deposit |
|---------|------|----------|-------------|
| `WeightShard[0]` | ~7.5MB | INT8 weights, layers 0-5 | ~52 SOL |
| `WeightShard[1]` | ~7.5MB | INT8 weights, layers 6-11 | ~52 SOL |
| `ModelManifest` | ~512B | Architecture params, quant scales, LUT data | ~0.004 SOL |

Solana max account size is 10MB, so 15M model needs two shards.

### Delegated to ER (per session)

**4M model:**

| Account | Size | Contents | Rent deposit |
|---------|------|----------|-------------|
| `SessionState` | ~256B | Frame counter, player states, status | ~0.002 SOL |
| `HiddenState` | ~65KB | SSM recurrent state (8 layers × 512 × 16) | ~0.47 SOL |
| `InputBuffer` | ~32B | Controller inputs for both players | ~0.001 SOL |

**15M model:**

| Account | Size | Contents | Rent deposit |
|---------|------|----------|-------------|
| `SessionState` | ~256B | Frame counter, player states, status | ~0.002 SOL |
| `HiddenState` | ~196KB | SSM recurrent state (12 layers × 1024 × 16) | ~1.4 SOL |
| `InputBuffer` | ~32B | Controller inputs for both players | ~0.001 SOL |

All rent is recoverable on session close.

### Account Seeds / PDAs

```
WeightAccount:  [b"weights", model_id.to_le_bytes()]
ModelManifest:  [b"manifest", model_id.to_le_bytes()]
SessionState:   [b"session", session_id.to_le_bytes()]
HiddenState:    [b"hidden", session_id.to_le_bytes()]
InputBuffer:    [b"input", session_id.to_le_bytes()]
```

---

## Program Architecture

One program, not BOLT ECS. Simpler for hackathon, fewer dependencies, less overhead.

### Instructions

```rust
// Session lifecycle
pub fn create_session(ctx: Context<CreateSession>) -> Result<()>
pub fn join_session(ctx: Context<JoinSession>, player: u8) -> Result<()>
pub fn close_session(ctx: Context<CloseSession>) -> Result<()>

// Gameplay (called in ER)
pub fn submit_input(ctx: Context<SubmitInput>, input: ControllerInput) -> Result<()>
pub fn run_inference(ctx: Context<RunInference>) -> Result<()>

// Setup
pub fn upload_weights(ctx: Context<UploadWeights>, offset: u32, data: Vec<u8>) -> Result<()>
pub fn init_manifest(ctx: Context<InitManifest>, config: ModelConfig) -> Result<()>
```

### `run_inference` — the hot path

```rust
pub fn run_inference(ctx: Context<RunInference>) -> Result<()> {
    let weights = ctx.accounts.weights.try_borrow_data()?;
    let mut hidden = ctx.accounts.hidden_state.try_borrow_mut_data()?;
    let input_buf = ctx.accounts.input_buffer.try_borrow_data()?;
    let manifest = &ctx.accounts.manifest;

    // Encode current state + controller inputs → input vector
    let x = encode_input(&ctx.accounts.session_state, &input_buf);

    // Forward pass: 6 layers
    let mut activations = x;
    for layer in 0..manifest.n_layers {
        let layer_weights = get_layer_weights(&weights, manifest, layer);
        let layer_hidden = get_layer_hidden_mut(&mut hidden, manifest, layer);

        // RMSNorm
        let normed = rmsnorm_i8(&activations);

        // in_proj (packed matmul)
        let projected = matmul_packed_i8(&layer_weights.in_proj, &normed);

        // Split into SSM input + gate
        let (ssm_input, gate_input) = projected.split_at(manifest.d_inner);

        // SSM step (updates hidden state in-place)
        let ssm_output = ssm_step_i8(
            ssm_input,
            layer_hidden,
            &layer_weights.a,
            &layer_weights.b_proj,
            &layer_weights.c_proj,
            &layer_weights.dt_proj,
            &manifest.luts,
        );

        // Gate: SiLU(gate) * ssm_output
        let gated = gate_silu_i8(&gate_input, &ssm_output, &manifest.luts.silu);

        // out_proj (packed matmul)
        let output = matmul_packed_i8(&layer_weights.out_proj, &gated);

        // Residual add
        activations = residual_add_i8(&activations, &output);
    }

    // Decode output → game state
    decode_output(&activations, &mut ctx.accounts.session_state);

    // Increment frame counter
    ctx.accounts.session_state.frame += 1;

    Ok(())
}
```

---

## Inference Kernel Details

### Packed INT8 Matmul (proven: ~16 CU/MAC)

```rust
#[inline(always)]
unsafe fn matmul_packed_i8(
    weights: &[u8],  // row-major, rows × cols
    input: &[u8],    // length = cols
    output: &mut [i32], // length = rows
    rows: usize,
    cols: usize,
) {
    let chunks = cols / 4;
    for i in 0..rows {
        let mut acc: i32 = 0;
        let row_offset = i * cols;
        for j in 0..chunks {
            let w_ptr = weights.as_ptr().add(row_offset + j * 4) as *const u32;
            let x_ptr = input.as_ptr().add(j * 4) as *const u32;
            let w4 = w_ptr.read_unaligned();
            let x4 = x_ptr.read_unaligned();

            acc += (w4 as u8) as i8 as i32 * (x4 as u8) as i8 as i32
                 + ((w4 >> 8) as u8) as i8 as i32 * ((x4 >> 8) as u8) as i8 as i32
                 + ((w4 >> 16) as u8) as i8 as i32 * ((x4 >> 16) as u8) as i8 as i32
                 + ((w4 >> 24) as u8) as i8 as i32 * ((x4 >> 24) as u8) as i8 as i32;
        }
        *output.get_unchecked_mut(i) = acc;
    }
}
```

### SSM Step (measured: ~630K CU at 1024×16, scales linearly)

Direct port of the benchmark code with unsafe indexing. At d_inner=256, d_state=16: ~159K CU × scaling factor ≈ ~40K CU. Negligible compared to matmuls.

### LUT Activations (measured: ~13 CU/lookup)

Already cheap. No optimization needed.

### RMSNorm

```rust
#[inline(always)]
unsafe fn rmsnorm_i8(x: &[u8], scale: &[u8], output: &mut [u8], len: usize) {
    let mut sum_sq: i64 = 0;
    for i in 0..len {
        let v = *x.get_unchecked(i) as i8 as i64;
        sum_sq += v * v;
    }
    // Use rsqrt LUT for 1/sqrt(mean_sq)
    let mean_sq = (sum_sq / len as i64).clamp(0, 255) as usize;
    let rsqrt = RSQRT_LUT[mean_sq] as i32;

    for i in 0..len {
        let v = *x.get_unchecked(i) as i8 as i32;
        let s = *scale.get_unchecked(i) as i8 as i32;
        let normed = (v * rsqrt * s) >> 14; // fixed-point
        *output.get_unchecked_mut(i) = normed.clamp(-128, 127) as u8;
    }
}
```

---

## Client Architecture

### WebSocket Frame Subscription

```typescript
const erConnection = new Connection(ER_ENDPOINT, "confirmed");

// Subscribe to session state changes
erConnection.onAccountChange(sessionStatePubkey, (info) => {
  const frame = decodeSessionState(info.data);
  visualizer.renderFrame(frame);
});
```

### Controller Input

Map keyboard to the v2 encoding's controller format:
```typescript
interface ControllerInput {
  joystick_x: number;   // -128 to 127
  joystick_y: number;
  c_stick_x: number;
  c_stick_y: number;
  trigger_l: number;     // 0 to 255
  trigger_r: number;
  buttons: number;       // bitfield: A, B, X, Y, Z, Start, D-pad
}
```

Send as a transaction to the ER:
```typescript
async function sendInput(input: ControllerInput) {
  const ix = program.methods
    .submitInput(input)
    .accounts({ inputBuffer: inputBufferPubkey, player: wallet.publicKey })
    .instruction();

  const tx = new Transaction().add(ix);
  await sendAndConfirmTransaction(erConnection, tx, [wallet]);
}
```

### Existing Visualizer Integration

The existing `viz/visualizer-juicy.html` already renders frames from JSON. Add a live mode:

```javascript
// Live mode: subscribe to ER, convert account data to frame JSON
function connectLive(erEndpoint, sessionPubkey) {
  const ws = new WebSocket(erEndpoint.replace('http', 'ws'));
  ws.onmessage = (msg) => {
    const frame = accountDataToFrameJson(msg.data);
    renderFrame(frame);
  };
}
```

---

## Delegation Flow

### Session Creation (on mainnet)

```
Player 1 → create_session TX (mainnet)
  ├── Create SessionState PDA
  ├── Create HiddenState PDA
  ├── Create InputBuffer PDA
  ├── Initialize all to default values
  └── Delegate all three to ER validator
        └── CPI to Delegation Program (DELeGGvXpWV2fqJUhqcF5ZSYMS4JTLjteaAMARRSaeSh)
```

### Gameplay (on ER)

```
Every 16ms (crank-driven):
  ├── Player 1 → submit_input TX (ER)
  ├── Player 2 → submit_input TX (ER)
  └── Crank → run_inference TX (ER)
        ├── Read WeightAccount (cloned from mainnet, read-only)
        ├── Read InputBuffer
        ├── Read+Write HiddenState
        └── Write SessionState (new frame)
```

### Session End (back to mainnet)

```
Player 1 → close_session TX (ER)
  └── commit_and_undelegate_accounts
        ├── Final SessionState committed to mainnet
        ├── HiddenState committed (could be discarded)
        └── Accounts return to program ownership
```

---

## Weight Upload Strategy

For 1M params (~1MB INT8), upload in chunks:

```
TX 1: upload_weights(offset=0,      data=[0..900])     // 900 bytes
TX 2: upload_weights(offset=900,    data=[900..1800])
...
TX N: upload_weights(offset=999100, data=[999100..1000000])
```

~1112 transactions. At ~5 TXs/sec on devnet, takes ~4 minutes. Do this once.

Alternatively, use the existing `solana program deploy`-style chunked write, or write a simple CLI:

```bash
node cli/upload-weights.mjs \
  --weights quantization/output/weights_int8.bin \
  --manifest quantization/output/manifest.json \
  --cluster devnet
```

---

## Crank Configuration

MagicBlock's built-in crank system drives the inference loop:

```toml
[task-scheduler]
min-interval = "16ms"   # 60fps target
```

The crank calls `run_inference` automatically every tick. No external bot needed.

If cranks aren't available or reliable, fallback to client-driven cranking:
```typescript
// Client sends run_inference after both players submit inputs
setInterval(async () => {
  await program.methods.runInference()
    .accounts({ weights, hiddenState, inputBuffer, sessionState, manifest })
    .rpc();
}, 16);
```

---

## Testing Strategy

### Unit Tests (local validator, no ER)

1. Deploy program to `solana-test-validator`
2. Upload dummy weights (random data)
3. Create session, submit inputs, call run_inference
4. Verify: frame counter increments, output state changes, no panics
5. Measure CU for run_inference with 1M model dimensions

### Integration Tests (local ER)

1. Start `solana-test-validator` (base layer)
2. Start `ephemeral-validator` (ER)
3. Create session on base layer, delegate to ER
4. Submit inputs + run inference on ER
5. Close session, verify state committed back to base layer

### End-to-End Test

1. Deploy to devnet
2. Delegate to MagicBlock devnet ER
3. Two browser tabs connect, send inputs
4. Visualizer renders frames
5. Measure fps, latency, CU

---

## File Structure

```
solana/
├── programs/
│   └── world-model/
│       └── src/
│           ├── lib.rs              # Program entrypoint, instructions
│           ├── inference.rs        # Forward pass orchestration
│           ├── matmul.rs           # Packed INT8 matmul kernel
│           ├── ssm.rs              # Mamba2 selective scan step
│           ├── lut.rs              # LUT activation lookups
│           ├── state.rs            # Account structs, encode/decode
│           └── error.rs            # Custom errors
├── cli/
│   └── upload-weights.mjs         # Weight upload tool
├── client/
│   └── src/
│       ├── index.ts               # SDK entry
│       ├── session.ts             # Create/join/close session
│       ├── input.ts               # Controller input encoding
│       └── subscribe.ts           # Frame subscription
├── tests/
│   ├── bench-direct.mjs           # CU benchmarks (existing)
│   ├── unit-inference.mjs         # Unit tests for inference
│   └── integration-er.mjs         # ER integration tests
├── env.sh                         # Toolchain env
├── Anchor.toml
└── Cargo.toml
```

---

## Dependencies

```toml
[dependencies]
anchor-lang = "0.32.1"
ephemeral-rollups-sdk = { version = "0.2", features = ["anchor"] }
```

```json
{
  "dependencies": {
    "@coral-xyz/anchor": "^0.32.1",
    "@solana/web3.js": "^2.0",
    "@magicblock-labs/ephemeral-rollups-sdk": "latest"
  }
}
```

---

## Open Questions

1. **ER CU limits:** What's the actual per-TX CU ceiling on a MagicBlock ER? 10M? 50M? 200M? This determines whether we can use the 4M model.

2. **Crank reliability:** Does the MagicBlock crank guarantee consistent 16ms ticks? What happens if a tick is missed?

3. **Weight account cloning latency:** How long does it take for the ER to clone a ~1MB account from mainnet on first access? Does this add startup latency?

4. **Account size limits in ER:** Any limit on delegated account sizes? HiddenState at 25KB should be fine, but need to confirm.

5. **WebSocket subscription from ER:** Does the ER support standard Solana WebSocket subscriptions (onAccountChange)? The docs suggest yes but need to verify.

6. **Multiple concurrent sessions:** Can multiple sessions run in the same ER instance? Or one ER per session?

---

## Hackathon Timeline

### Pre-hackathon (2-3 days before)
- [ ] Get MagicBlock ER running locally
- [ ] Deploy cu-benchmark to ER, verify CU behavior
- [ ] Train or obtain a 1M parameter Mamba2 model on Melee replays
- [ ] Run quantization pipeline, produce weights_int8.bin
- [ ] Test weight upload flow on devnet

### Day 1: Foundation
- [ ] Implement world-model program (all instructions)
- [ ] Implement packed matmul kernel (port from cu-benchmark)
- [ ] Unit test: create session, submit input, run inference
- [ ] Deploy to local ER, test delegation flow

### Day 2: Integration
- [ ] Upload quantized weights to devnet
- [ ] Wire up crank-driven inference loop
- [ ] Client: input capture + frame subscription
- [ ] Connect visualizer in live mode
- [ ] Two-player test

### Day 3: Polish + Demo
- [ ] Measure and report fps, latency, CU
- [ ] Record demo video
- [ ] Write submission (include Path A syscall proposal as "future work")
- [ ] Handle edge cases (player disconnect, session timeout)

### Deliverables
1. Working onchain world model in MagicBlock ER
2. Two-player real-time gameplay
3. Visualizer rendering live frames
4. Benchmark data (CU per frame, fps, latency)
5. Technical writeup with syscall proposal (Path A as Phase 2)
6. Open-source repo
