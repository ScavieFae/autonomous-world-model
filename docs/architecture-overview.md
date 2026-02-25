# Autonomous World Model: Full Architecture

## The One-Liner

A learned world model deployed as an onchain game console. Load model, spin up session, connect players, play.

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
