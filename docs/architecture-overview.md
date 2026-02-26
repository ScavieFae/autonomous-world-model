# Autonomous World Model: Architecture Overview

## The Experience

You open a website. You see a list of worlds. Each world is a neural network that someone trained on fighting game replays — maybe 50,000 matches of Melee, maybe a million. You pick one. You click "Play."

A session spins up. You get a link. You send it to a friend. They open it, connect their wallet, pick a character. You both hit Ready.

Now you're playing. Keyboard inputs go in, a canvas renders two characters fighting at 60 frames per second. It looks like Melee. It moves like Melee. But nobody wrote the physics. Nobody wrote the hit detection, the knockback formulas, the edge-grab mechanics. A model *learned* all of that from watching humans play — and now it's running those learned rules, live, as you press buttons.

When the match ends, the result settles permanently on Solana. The replay is there forever. The model — the "physics engine" — is there forever too. Anyone can read it, copy it, retrain it, deploy a variant. Someone could train on Street Fighter replays and deploy "Street Fighter physics" to the same arcade. Someone could train on a dataset that's half Melee, half nonsense, and deploy a surrealist world with weird gravity. The arcade grows.

**The product: an onchain arcade where the games aren't coded — they're learned. And the physics are a public good that anyone can fork.**

---

## Layer 1: The Arcade's Shelf

The model weights live on Solana mainnet. This is the permanent layer — the shelf of cartridges at the arcade.

Each "cartridge" is:
- **WeightShards** — 4-15MB of INT8 numbers. This is the model. The physics of the world. Every rule the network learned about how characters move, how hits connect, how knockback works — all encoded as millions of tiny integers.
- **ModelManifest** — a small descriptor. "This model has 12 layers, 15 million parameters, was trained on 50K replays. Here are the activation lookup tables."

These live on L1 forever. Anyone can read them. Anyone can copy them and deploy a fork. The rent deposit (~28-104 SOL depending on model size) is recoverable if you retire the model. But the point is permanence — this is a public good. The physics of this world are open for anyone to inspect, study, or build on.

---

## Layer 2: The Arcade Cabinet

When you hit "Play," something ephemeral spins up: a MagicBlock ephemeral rollup. This is the console. It exists for one match and then it's done.

Three small accounts get created and "delegated" to this rollup:

- **SessionState** (256 bytes) — where the players are, their damage, their stocks, what action they're performing, which direction they're facing. This is what the visualizer reads to draw the frame.
- **HiddenState** (~65-196KB) — the model's *memory*. Mamba2 is a recurrent architecture — it doesn't look at a window of previous frames like a transformer would. Instead, it maintains a fixed-size hidden state that gets updated every single frame. That state carries forward everything the model "remembers" about this match — the rhythm of the fight, the spacing patterns, whether someone's been shielding too much. It's the world's subconscious.
- **InputBuffer** (32 bytes) — both players' controller states. Joystick position, buttons pressed, triggers.

The rollup runs at ~15ms block times with zero transaction fees. A crank — an automated heartbeat — fires every 16ms. That's 60 frames per second.

---

## The Frame Loop

Every 16 milliseconds:

1. Both players' inputs are sitting in the InputBuffer (sent via WebSocket transactions)
2. The crank fires `run_inference`
3. The program reads the inputs, the current hidden state, and the weight shards
4. It runs one step of the Mamba2 forward pass — RMSNorm, matrix multiply, selective scan, gating, output projection, for every layer
5. It writes a new HiddenState (the world's memory updated) and a new SessionState (new positions, new damage, new action states)
6. Both players' browsers are subscribed to SessionState via WebSocket — they see the new frame and the visualizer renders it

Input goes in, physics come out. The "physics engine" is 15 million numbers being multiplied together 60 times a second. No game logic. No `if (player.x > stage.edge)` checks. The model handles all of it because it saw tens of thousands of matches where characters fell off edges, and it learned what happens.

---

## Match Lifecycle

The match ends: KO, timeout, or someone quits. The final SessionState — who won, final stocks, final damage — gets committed back to Solana mainnet. The ephemeral rollup disappears. The HiddenState gets committed too (useful as replay data).

The match result lives on L1 forever. Leaderboards, replays, match history — all permanent, all readable by anyone.

Quarter in, play, done. State settles to L1. The cabinet is free for the next match.

---

## Why This Shape

**Why Mamba2 and not a transformer?** Mamba2 has a fixed-size hidden state. A transformer's attention grows with context length — the longer the match, the more memory it needs, the more computation per frame. Mamba2's cost is the same on frame 1 and frame 10,000. Fixed-size input, fixed-size output, fixed-size working memory. The ideal shape for an onchain computation that repeats 60 times per second.

**Why INT8?** Two reasons. Size: a 15M parameter model in float32 is 60MB, in INT8 it's 15MB. Determinism: integer math produces identical results on every machine. Float math doesn't (rounding modes, FMA fusion, etc.). When your physics engine runs onchain and multiple parties need to agree on the result, exact reproducibility matters. INT8 gives us both compression and determinism for free.

**Why a custom syscall in the rollup?** The Solana VM charges ~25 compute units per multiply-accumulate operation through the standard BPF interpreter. A single frame of the 15M model requires ~19 million multiplications — ~300 million compute units, way beyond any transaction limit. Our custom `sol_matmul_i8` syscall runs the matrix math at native CPU speed, dropping cost to ~190K compute units. The actual CPU work takes microseconds. The bottleneck was always the VM overhead, not the math.

**Why ephemeral rollups and not mainnet?** Three reasons: (1) ~15ms block times for 60fps — mainnet is 400ms; (2) zero-fee transactions — players send inputs 60 times per second; (3) configurable compute limits for the forward pass. The rollup is a single-operator runtime optimized for our use case, with the guarantee that final state settles back to L1 permanently.

---

## The Bigger Picture

What we're building is a platform for **learned worlds as public goods**.

The first world looks like Melee because that's what we have training data for. But the architecture is generic. The model takes `(current_state + player_inputs)` and outputs `(next_state)`. What "state" and "inputs" mean is determined by the training data. Train on driving games → driving physics. Train on RTS replays → strategy dynamics.

The onchain part matters because it makes the physics forkable. Don't like how a model handles edge-guards? Copy the weights, fine-tune on a different dataset, deploy "Melee Physics v2." The models compete for players. The best physics win. Evolution, but for game engines.

And the "constantly-evolving" part: weights are just accounts on L1. Anyone can deploy new models to the arcade. The library of worlds grows organically — not through a company shipping updates, but through a community training variants and deploying them. The arcade is a protocol, not a product.

---

## Economics (Not Decided — Notes Only)

Deploying a world costs rent (~28 SOL for 4M, ~104 SOL for 15M). This is a recoverable deposit — you lock capital to make a world exist.

Properties of the rent model:
- **Recoverable.** Close the accounts, get all SOL back. Deploying a world is locking capital, not spending it.
- **Removable.** Close the weight accounts and the world ceases to exist on-network. Useful if you need to take something down quickly. (Since weights are public, someone could have copied them — but you can truthfully say you removed it.)

Alternatives considered for weight storage:
- **Arweave / IPFS.** ~$1-2/MB one-time (Arweave) or free (IPFS). 100x cheaper. But the ER can't read from external storage directly — needs an oracle or load step. Breaks forkability (can't just copy a Solana account). Breaks self-containment.
- **Shadow Drive / Solana-native storage.** Cheaper than rent, Solana-adjacent. But separate system, separate availability, ER can't natively read from it.
- **Offchain + onchain hash.** Store anywhere, verify hash onchain. Cheapest. But creates an availability dependency — host goes down, world is unplayable. Forking requires hosting, not just copying.
- **Compressed onchain.** Store compressed, decompress at session start. Lower always-on cost, but adds startup latency.

Every alternative trades the "weights are just Solana accounts" property for cost savings. That property is what makes forkability trivial (copy an account), ER cloning automatic (delegation protocol), and the system self-contained. Current plan: pay the rent.

Revenue surfaces that are natural to the existing architecture (no new programs or accounts):
- **Session fee.** `create_session` charges a small fee (fraction of SOL) that flows to the model deployer. The "quarter in the arcade." One `transfer` CPI — trivial to add later.
- **Deployer yield.** If sessions generate fees, the deployer earns yield on their locked rent capital. The world pays for itself.

Conceptually interesting (future work, not building now):
- **World staking.** Anyone stakes SOL into a world's rent account, earns a share of session revenue proportional to stake. The more popular the world, the more sessions, the more revenue flows to stakers. Unpopular worlds lose stakers, get closed, rent returns. Natural selection for game engines.
- **Fork lineage.** Derivative worlds could reference their parent. Protocol-level royalties on forks. (High bloat risk — flag it and leave it.)

The analogy to validator economics is almost exact: stake capital to make infrastructure exist, earn yield from usage, capital migrates toward the best-performing systems. But this is token engineering territory — note it, don't build it, revisit when there's actual usage data.

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              WEBSITE                                     │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────────┐ │
│  │  Model Browser│  │ Session Lobby│  │         Game View              │ │
│  │              │  │              │  │                                │ │
│  │ • Browse     │  │ • Create     │  │  ┌──────────────────────────┐ │ │
│  │   models     │→ │   session    │→ │  │                          │ │ │
│  │ • See params │  │ • Join via   │  │  │    Visualizer Canvas     │ │ │
│  │ • Fork a     │  │   session ID │  │  │                          │ │ │
│  │   world      │  │ • Pick char  │  │  │  Renders frame state     │ │ │
│  │              │  │ • Wait for   │  │  │  from onchain data       │ │ │
│  │              │  │   opponent   │  │  │                          │ │ │
│  │              │  │              │  │  └──────────────────────────┘ │ │
│  └──────────────┘  └──────────────┘  │                                │ │
│                                       │  Keyboard → Controller Input   │ │
│                                       │  60fps subscribe to state      │ │
│                                       └────────────────────────────────┘ │
│                                                                          │
│  Wallet: Phantom / Backpack / Solflare                                   │
│  Connects to: L1 (session mgmt) + L2 (gameplay)                         │
│                                                                          │
└──────────────┬──────────────────────────────────┬────────────────────────┘
               │ Session create/close              │ Input TXs + state subscribe
               │ Weight reads                      │ (WebSocket)
               │ (HTTPS RPC)                       │
               ▼                                   ▼
┌──────────────────────────────┐  ┌────────────────────────────────────────┐
│        SOLANA L1              │  │     MAGICBLOCK EPHEMERAL ROLLUP (L2)   │
│        (mainnet)              │  │                                        │
│                               │  │  Custom validator w/ sol_matmul_i8     │
│  ┌─────────────────────────┐ │  │  Block time: ~15ms                     │
│  │     Weight Storage       │ │  │  TX cost: 0 SOL                       │
│  │                          │ │  │                                        │
│  │  WeightShard[0]  ~7.5MB │ │  │  ┌──────────────────────────────────┐ │
│  │  WeightShard[1]  ~7.5MB │─┼──┼─→│  Cloned weights (read-only)      │ │
│  │  ModelManifest   ~512B  │ │  │  │  Cached after first access        │ │
│  │                          │ │  │  └──────────────────────────────────┘ │
│  │  Permanent. Forkable.    │ │  │                                        │
│  │  Anyone can read/copy.   │ │  │  ┌──────────────────────────────────┐ │
│  └─────────────────────────┘ │  │  │  Session Accounts (delegated)     │ │
│                               │  │  │                                    │ │
│  ┌─────────────────────────┐ │  │  │  SessionState    256B   frame,    │ │
│  │   Delegation Program     │ │  │  │                        players,  │ │
│  │   (MagicBlock)           │ │  │  │                        positions │ │
│  │                          │ │  │  │  HiddenState    196KB  Mamba2    │ │
│  │  Locks accounts to ER    │◄┼──┼──│                        recurrent │ │
│  │  Commits state back      │ │  │  │                        memory    │ │
│  │                          │ │  │  │  InputBuffer    32B    both      │ │
│  └─────────────────────────┘ │  │  │                        players'  │ │
│                               │  │  │                        controls  │ │
│  ┌─────────────────────────┐ │  │  └──────────────────────────────────┘ │
│  │   Results / History      │ │  │                                        │
│  │                          │ │  │  ┌──────────────────────────────────┐ │
│  │  Match results committed │ │  │  │  Inference Loop (crank @ 16ms)   │ │
│  │  back on session close   │ │  │  │                                    │ │
│  │  Replay data on L1       │ │  │  │  1. Read InputBuffer              │ │
│  │  Leaderboards            │ │  │  │  2. Read HiddenState              │ │
│  │                          │ │  │  │  3. Read Weights                   │ │
│  └─────────────────────────┘ │  │  │  4. Forward pass (sol_matmul_i8)   │ │
│                               │  │  │  5. Write new HiddenState          │ │
│  World Model Program          │  │  │  6. Write new SessionState          │ │
│  (deployed to both L1 + L2)   │  │  │  7. frame++                        │ │
│                               │  │  │                                    │ │
└───────────────────────────────┘  │  └──────────────────────────────────┘ │
                                    │                                        │
                                    └────────────────────────────────────────┘
```

## UX Flow

```
 Player A                          System                          Player B
 ────────                          ──────                          ────────

 1. Connect wallet
    Browse models on website
    Pick "Melee Physics v1"
                          ───→  Read ModelManifest from L1
                          ←───  Show: 15M params, 12 layers,
                                trained on 50K replays

 2. "Create Session"
                          ───→  Create session accounts on L1
                          ───→  Delegate to ER
                          ←───  Session ID: "ABC123"

 3. Share session ID
    (link, QR, lobby)
                                                            ←───  Player B opens link
                                                                  Connects wallet

 4.                                                               "Join Session ABC123"
                          ───→  Join TX on ER
                          ←───  Both players connected
                                Pick characters

 5. "Ready"                                                       "Ready"
                          ───→  Session status: Active
                          ───→  Crank starts (16ms ticks)
                          ───→  Frame 0: initialize hidden state

 ┌─── GAMEPLAY LOOP (60fps) ──────────────────────────────────────────────┐
 │                                                                         │
 │  6. Keyboard input              7. Keyboard input                       │
 │     ↓                              ↓                                    │
 │     Encode to controller           Encode to controller                 │
 │     ↓                              ↓                                    │
 │     submit_input TX  ──→  ER  ←──  submit_input TX                     │
 │                            │                                            │
 │                     8. Crank fires run_inference                        │
 │                            │                                            │
 │                     ┌──────┴──────┐                                     │
 │                     │ Forward Pass │                                    │
 │                     │             │                                     │
 │                     │ inputs +    │                                     │
 │                     │ hidden →    │                                     │
 │                     │ 12 layers → │                                     │
 │                     │ new state + │                                     │
 │                     │ new hidden  │                                     │
 │                     └──────┬──────┘                                     │
 │                            │                                            │
 │                     9. SessionState updated                             │
 │                            │                                            │
 │     ←── WebSocket ────────┘────────── WebSocket ──→                    │
 │                                                                         │
 │  10. Visualizer renders       10. Visualizer renders                    │
 │      new frame                    new frame                             │
 │                                                                         │
 │  Repeat at 60fps...                                                     │
 │                                                                         │
 └─────────────────────────────────────────────────────────────────────────┘

 11. Game ends (KO, timeout, quit)
                          ───→  commit_and_undelegate
                          ───→  Final state → L1
                          ───→  Session accounts closeable
                          ←───  Match result on L1 forever

 12. View replay, share, rematch
```

## Data Flow Per Frame

```
                    ┌─────────┐
                    │ Player A│ Controller: joystick, buttons, triggers
                    │ Input   │ 12 bytes
                    └────┬────┘
                         │
                         ▼
              ┌──────────────────┐
              │   InputBuffer    │ 32 bytes on ER
              │   (both players) │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐     ┌──────────────────┐
              │  run_inference   │◄────│   WeightShards   │ ~15MB INT8
              │                  │     │   (read-only)    │ cloned from L1
              │  encode inputs   │     └──────────────────┘
              │       ↓          │
              │  for each layer: │     ┌──────────────────┐
              │    rmsnorm       │◄───►│   HiddenState    │ ~196KB
              │    in_proj       │     │   (read+write)   │ Mamba2 SSM
              │    ssm_step      │     │                  │ recurrent state
              │    gate          │     │  This IS the     │
              │    out_proj      │     │  world's memory  │
              │    residual      │     └──────────────────┘
              │       ↓          │
              │  decode outputs  │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  SessionState    │ 256 bytes on ER
              │                  │
              │  Per player:     │
              │   x, y           │ position
              │   percent        │ damage
              │   stocks         │ lives remaining
              │   action_state   │ what they're doing (400 classes)
              │   facing         │ left/right
              │   speed_*        │ velocity components
              │   shield         │ shield strength
              │   hitlag         │ freeze frames on hit
              │   on_ground      │ airborne?
              │                  │
              │  frame_number    │ monotonic counter
              │  stage           │ which stage
              └────────┬─────────┘
                       │
                       ▼ (WebSocket subscription)
              ┌──────────────────┐
              │   Visualizer     │ Browser canvas
              │                  │
              │  Character sprites at (x, y)
              │  Percent displays
              │  Stock icons
              │  Stage geometry
              │  Hit effects, dust, etc.
              └──────────────────┘
```

## What Lives Where

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  PERMANENT (L1 mainnet)                 EPHEMERAL (L2 per-session) │
│  ─────────────────────                 ──────────────────────────│
│                                                                  │
│  Model weights (INT8)                  Session state             │
│  Model manifest                        Hidden state (SSM memory) │
│  Program binary                        Input buffer              │
│  Match history/replays                 Frame counter             │
│  Leaderboards                                                    │
│                                                                  │
│  Survives forever.                     Lives for one match.      │
│  Anyone can read/fork.                 Created on session start.  │
│  Cost: ~104 SOL rent                  Delegated to ER.           │
│  (recoverable)                         Committed back on end.    │
│                                        Cost: ~1.4 SOL rent       │
│                                        (recoverable)             │
│                                                                  │
│  PUBLIC GOOD                           ARCADE CABINET             │
│  The physics of the world.             One play session.          │
│  Fork it. Retrain it.                  Quarter in, play, done.   │
│  Deploy alternate physics.             State settles to L1.      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Map

```
Repository: autonomous-world-model/

├── Training (external: nojohns-training)
│   └── Mamba2 model trained on Melee replays → FP32 weights
│
├── quantization/                    ← FP32 → INT8
│   ├── quantize_mamba2.py           Weights → weights_int8.bin
│   ├── generate_luts.py             Activation LUTs (SiLU, softplus, etc.)
│   └── benchmark_accuracy.py        FP32 vs INT8 fidelity comparison
│
├── solana/                          ← Onchain programs
│   ├── programs/
│   │   ├── cu-benchmark/            CU measurement (done, deployed)
│   │   └── world-model/             The actual inference program
│   │       ├── lib.rs               Instructions: create/join/close/input/infer
│   │       ├── inference.rs         Forward pass orchestration
│   │       ├── matmul.rs            Packed INT8 matmul kernel
│   │       ├── ssm.rs              Mamba2 selective scan
│   │       └── lut.rs              LUT activations
│   ├── cli/
│   │   └── upload-weights.mjs       Upload INT8 weights to L1
│   └── client/
│       ├── session.ts               Create/join/close session + delegation
│       ├── input.ts                 Controller encoding + TX sending
│       └── subscribe.ts             WebSocket frame subscription
│
├── viz/                             ← Rendering layer (exists)
│   └── visualizer-juicy.html        Canvas renderer, already works on JSON frames
│                                    Add: live WebSocket mode
│
├── website/                         ← Frontend (to build)
│   ├── Model browser                Read ModelManifest accounts
│   ├── Session lobby                Create/join/share sessions
│   └── Game view                    Visualizer + input capture + wallet
│
└── validator/                       ← Custom ER (Phase 2)
    └── Fork of magicblock-validator
        with sol_matmul_i8 syscall
```

## The "Console" Metaphor

```
Traditional game console:
  Cartridge (game ROM)  →  Console hardware  →  TV screen
  Fixed. Read-only.        Executes the game.   Displays output.

Autonomous world console:
  Weights (onchain L1)  →  ER validator (L2)  →  Browser visualizer
  Fixed. Forkable.         Executes inference.    Renders frames.
  The "ROM" is learned     The "hardware" is a    The "TV" is a
  physics, not coded        custom SVM runtime.    canvas element.
  physics.

Insert quarter:           = Create session + delegate to ER
Press start:              = Both players ready, crank starts
Play:                     = 60fps input→inference→render loop
Game over:                = Undelegate, state settles to L1
```
