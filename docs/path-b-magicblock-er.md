# Path B: MagicBlock Ephemeral Rollup (Sponsor Track)

## The Idea

Deploy the world model inference as a standard Solana BPF program, run it inside a MagicBlock ephemeral rollup with raised CU limits and fast block times, use their crank system to drive the inference loop at 60fps. No validator fork — pure BPF, but with the ER's configurable execution environment.

## Why MagicBlock

1. **Hackathon sponsor.** They run the Real-Time Hackathon (historically April-May, $10K prize + Colosseum accelerator fast-track). Their "Fully Onchain Game" track is exactly this project.
2. **Free transactions.** Public ER nodes charge 0 SOL per transaction. You only pay ~0.0003 SOL per session at undelegation.
3. **Configurable block time.** Default 400ms, configurable down to ~10ms. Our 16.67ms frame budget is achievable.
4. **Cranks.** Built-in tick-based scheduler with 10ms minimum interval. No external bot needed — the ER drives the inference loop.
5. **Account delegation.** Weights stay on mainnet (permanent, forkable). Game state delegates to the ER for fast execution. Clean separation.
6. **They want this.** Their roadmap includes "AI Oracles" — calling AI models from smart contracts. Onchain inference is aligned with their vision.

## Architecture

```
┌─────── SOLANA MAINNET/DEVNET ──────────────────────────┐
│                                                         │
│  WeightShard[0..1]   ModelManifest    Leaderboard       │
│  (permanent INT8)    (arch params)    (global state)    │
│                                                         │
│  1. Player creates session                              │
│  2. delegate(SessionState, HiddenState, InputBuffer)    │
│                                                         │
├─────── MAGICBLOCK EPHEMERAL ROLLUP ────────────────────┤
│                                                         │
│  Block time: 15ms                                       │
│  CU limit: TBD (need to push boundaries)                │
│  TX cost: 0 SOL                                         │
│                                                         │
│  WeightShards ← cloned read-only on first access        │
│                                                         │
│  Crank (every 16ms):                                    │
│    → read InputBuffer                                   │
│    → run_inference(weights, state, hidden)               │
│    → write SessionState, HiddenState                    │
│                                                         │
│  Players (via WebSocket):                               │
│    → submit_input TX (controller state)                 │
│    → subscribe to SessionState changes                  │
│                                                         │
│  3. Game ends → commit_and_undelegate                   │
│  4. Final state settles to mainnet                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## The CU Problem (and Solutions)

With our packed matmul optimization, we're at ~16 CU/MAC. The full model:

| Model | MACs/step | CU (packed) | Fits in... |
|-------|----------|-------------|------------|
| 15M (target) | ~19M | ~300M | Nothing standard |
| 4M (small) | ~3.2M | ~51M | Single ER TX if CU raised |
| 1M (tiny) | ~600K | ~9.6M | Single ER TX easily |
| 250K (micro) | ~100K | ~1.6M | Single mainnet TX |

### Solution 1: Smaller Model (most pragmatic)

Train a 1M or 4M parameter model. At 9.6M CU, the 1M model fits comfortably in a single ER transaction even at mainnet CU limits. The question is quality — can 1M params produce interesting Melee dynamics?

For a hackathon demo, even a micro model (250K) that produces *any* plausible movement is impressive. "It runs entirely onchain" matters more than "it plays tournament-level Melee."

### Solution 2: Multi-TX Pipeline

Split inference across multiple transactions per frame:

```
TX 1: layers 0-2   → write intermediate to IntermediateState account
TX 2: layers 3-5   → read/write IntermediateState
TX 3: layers 6-8   → read/write IntermediateState
TX 4: layers 9-11  → write final SessionState
```

At ~26M CU per layer (packed), 3 layers per TX = ~78M CU per TX. Still too much.

At 1 layer per TX: ~26M CU per TX × 12 TXs per frame. If ER can raise limits to 30M CU per TX, this works. But 12 sequential TXs at 15ms block time = 180ms per frame = ~5.5fps. Not 60fps.

**Multi-TX only works if the ER can batch ordered transactions in a single block.** This is the critical question for MagicBlock.

### Solution 3: Modified Validator CU Limits

The magicblock-validator is open source (BSL-1.1). You can build it from source and modify the CU limits in the SVM runtime. This is lighter than a full Agave fork — you're just changing a config constant in their existing codebase.

```toml
# config.toml — block time is already configurable
[ledger]
block-time = "15ms"

# CU limit would need a source code change in the SVM layer
# Look for max_compute_units or similar in magicblock-svm
```

For a hackathon: build magicblock-validator from source, raise the per-TX CU limit to 50M or 100M, run locally. The 4M model at 51M CU fits.

### Solution 4: Hybrid Crank Pipeline

Use the ER crank to drive a multi-TX pipeline within a single "logical frame":

```
Crank tick 0:  run_inference_chunk(layers=0..3)
Crank tick 1:  run_inference_chunk(layers=4..7)    (+10ms)
Crank tick 2:  run_inference_chunk(layers=8..11)   (+10ms)
Crank tick 3:  output_frame()                      (+10ms)
```

Total latency: 40ms per frame = 25fps. Playable. Each chunk stays under CU limits.

## Integration Details

### Rust SDK

```toml
[dependencies]
anchor-lang = "0.32.1"
ephemeral-rollups-sdk = { version = "latest", features = ["anchor"] }
```

### Account Delegation

```rust
use ephemeral_rollups_sdk::anchor::delegate;
use ephemeral_rollups_sdk::cpi::DelegateConfig;

#[delegate]
#[derive(Accounts)]
pub struct CreateSession<'info> {
    pub payer: Signer<'info>,
    #[account(mut, del)]  // marks for delegation
    pub session_state: Account<'info, SessionState>,
    #[account(mut, del)]
    pub hidden_state: Account<'info, HiddenState>,
    #[account(mut, del)]
    pub input_buffer: Account<'info, InputBuffer>,
}

pub fn create_session(ctx: Context<CreateSession>) -> Result<()> {
    // Initialize accounts...

    // Delegate to ER
    ctx.accounts.delegate_session_state(
        &ctx.accounts.payer,
        &[SESSION_SEED, &session_id.to_le_bytes()],
        DelegateConfig::default(),
    )?;
    // ... delegate other accounts

    Ok(())
}
```

### Crank Setup

The ER's built-in crank calls your inference instruction automatically:

```rust
// In your program — the crank calls this every tick
pub fn run_inference(ctx: Context<RunInference>) -> Result<()> {
    let weights = ctx.accounts.weights.try_borrow_data()?;
    let input = ctx.accounts.input_buffer.try_borrow_data()?;
    // ... run model forward pass
    // ... write output to session_state
    Ok(())
}
```

Crank configuration:
```toml
[task-scheduler]
min-interval = "16ms"  # ~60fps
```

### Local Development

```bash
# Terminal 1: Solana test validator (base layer)
solana-test-validator

# Terminal 2: MagicBlock ER validator
npm install -g @magicblock-labs/ephemeral-validator@latest
ephemeral-validator \
  --remotes "http://localhost:8899" \
  --remotes "ws://localhost:8900" \
  -l "127.0.0.1:7799"

# OR build from source for custom CU limits:
git clone https://github.com/magicblock-labs/magicblock-validator
cd magicblock-validator
# ... modify CU limits ...
cargo build --release
cargo run --release -- config.toml
```

### Client Connection

```typescript
// Player connects to the ER endpoint
const erConnection = new Connection("http://localhost:7799", "confirmed");

// Subscribe to session state changes
const subId = erConnection.onAccountChange(sessionStatePubkey, (accountInfo) => {
  const state = decodeSessionState(accountInfo.data);
  visualizer.renderFrame(state);
});

// Send controller input
async function sendInput(controllerState: ControllerInput) {
  const tx = new Transaction().add(
    submitInputInstruction(sessionId, controllerState)
  );
  await sendAndConfirmTransaction(erConnection, tx, [wallet]);
}
```

## Hackathon Plan

### Pre-hackathon (1-2 days before)
- Install magicblock-validator, get local ER running
- Deploy cu-benchmark program to ER, verify it works
- Determine CU limit story: can we raise it? Do we need smaller model?
- Start a conversation with MagicBlock team about CU limits / custom extensions

### Day 1 (Foundation, ~8 hours)
- Deploy inference program to local ER
- Implement account delegation (session_state, hidden_state, input_buffer)
- Test delegation → execute → undelegation lifecycle
- Set up crank for inference loop

### Day 2 (Model + Client, ~8 hours)
- Upload quantized weights (even tiny model) to devnet
- Wire up full inference loop via crank
- Client: WebSocket subscription + input sending
- Hook up the existing visualizer for frame rendering

### Day 3 (Demo + Polish, ~6 hours)
- Two-player demo (two browser tabs)
- Record video of onchain gameplay
- Measure actual fps, latency
- Submit

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| CU limits too low for any interesting model | High | Prepare micro/tiny model. "Any onchain inference" is still a first. |
| ER cranks can't hit 60fps | Medium | 25fps (40ms) is fine for a demo. Or manual cranking. |
| Weight account cloning is slow | Medium | Pre-warm: send a dummy TX that touches weights before game starts. |
| MagicBlock ER has bugs / instability | Medium | Test thoroughly pre-hackathon. Have local validator fallback. |
| BOLT ECS adds overhead | Low | Skip BOLT. Use raw Anchor + ephemeral-rollups-sdk. |

## The Pitch

"A learned world model running entirely onchain in a MagicBlock ephemeral rollup. Weights stored on Solana mainnet — permanent, forkable, composable. Sessions spin up on demand with zero gas fees and sub-20ms latency. The model IS the world. No off-chain compute. No oracles. No clients running inference. Just the blockchain, playing Melee."

## Why MagicBlock Picks This

1. **It's their "Fully Onchain Game" track** — literally.
2. **It pushes their platform to its limits** — CU, block time, cranks, account delegation. They want to see what's possible.
3. **"AI Oracles" is on their roadmap** — this is a more radical version of what they already want to build.
4. **It's a great story** — "the first neural network running inside a Solana ephemeral rollup."
5. **Forkable worlds** — anyone can copy the weights, deploy a variant, create alternate physics. This is the composability MagicBlock sells.
