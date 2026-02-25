# Custom Syscall Tradeoffs: What Breaks, What Doesn't, What People Will Say

## The Architecture

```
┌─── YOUR CUSTOM ER VALIDATOR ───────────────────┐
│                                                  │
│  Standard SVM + sol_matmul_i8 syscall            │
│  Single sequencer, no consensus peers            │
│  Executes inference at native CPU speed           │
│                                                  │
│  Every N frames: commit state to L1               │
│                                                  │
├─── DELEGATION PROTOCOL ─────────────────────────┤
│                                                  │
│  commit_accounts() → raw bytes to mainnet         │
│  Mainnet stores the data                          │
│  Mainnet never re-executes the computation        │
│                                                  │
├─── SOLANA MAINNET ──────────────────────────────┤
│                                                  │
│  Weights (permanent, forkable)                    │
│  Session results (committed from ER)              │
│  Leaderboards, replays, etc.                      │
│                                                  │
└──────────────────────────────────────────────────┘
```

## What Breaks and What Doesn't

| Property | Standard ER | Custom Syscall ER | Notes |
|----------|------------|-------------------|-------|
| Data availability | Yes | Yes | Inputs/outputs on L1 either way |
| Execution verifiability | Yes (any stock validator) | Yes (need our fork) | Weaker but not zero |
| Execution on L1 consensus | No | No | No ER has this |
| Fraud proofs (future) | Any validator can challenge | Only our fork can challenge | MagicBlock doesn't have fraud proofs yet anyway |
| Composability | Standard programs work | Standard programs work | The syscall is additive, doesn't break existing programs |
| State commitment to L1 | Works | Works | Delegation protocol doesn't re-execute |
| Portability | Any MagicBlock ER | Only our validator | Real limitation |
| Trust model | Trust ER operator | Trust ER operator | **Identical** — ERs are single-sequencer today |

**Key insight: the trust model is identical.** Both standard and custom ERs are trusted single-sequencer. The custom syscall doesn't degrade any guarantee that currently exists.

## The Three Levels of "Onchain"

1. **Data availability** — inputs and outputs live on L1. Anyone can read them. We have this regardless of approach.

2. **Execution verifiability** — anyone can re-execute the computation and check the result.
   - Standard ER: anyone with a stock MagicBlock validator can replay.
   - Custom syscall ER: anyone who builds our open-source fork can replay.
   - The gap is "download a binary" vs "git clone + cargo build." Real but not fatal.

3. **Execution on consensus** — L1 validators actually ran the computation. No ephemeral rollup has this. This is what mainnet Solana programs get, and what our CU benchmark showed is infeasible for large models (~480M CU).

## What an Experienced Consensus Engineer Would Say

### "You've reinvented an optimistic rollup with no fraud proofs."

The whole point of running inference in the SVM is verifiability. Anyone can replay and check. The moment you add a custom syscall, no standard validator can replay. You've broken verifiability — the thing that makes "onchain" mean something.

**Rebuttal:** Fraud proofs don't exist for MagicBlock ERs yet. We're not degrading a guarantee that exists. And verifiability is preserved for anyone who builds our (open source) fork. It's "anyone with git clone" vs "anyone with a stock validator" — weaker, but not zero.

### "You're confused about what 'onchain' means."

With a custom syscall, you've made your world model into a trusted oracle with extra steps. The ER operator could return any state they want — there's no way for mainnet to verify the computation was done correctly.

**Rebuttal:** This is true of ALL current ephemeral rollups. The ER operator is a trusted sequencer. This is the explicitly stated MagicBlock architecture today. Our syscall doesn't change the trust model.

### "The fraud proof story collapses."

MagicBlock's roadmap includes fraud proofs. With standard execution, any challenger can replay with a stock validator. With a custom syscall, the challenger needs your fork. "Anyone can challenge" becomes "anyone who builds our custom validator can challenge."

**Rebuttal:** Correct, and this is the real long-term concern. Mitigations:
- Publish the fork. It's open source. The build is reproducible.
- Work toward upstreaming: propose `sol_tensor` through the SIMD process.
- If MagicBlock implements fraud proofs, work with them to support plugin-aware verification.

### "You're optimizing the wrong thing."

The CU cost isn't a hardware problem, it's a metering problem. The actual CPU work for 19M INT8 MACs takes ~50 microseconds. The SBF VM charges ~25 CU per MAC because of instruction overhead. The right fix is:
- Push for a standard `sol_tensor` syscall (multi-year SIMD process)
- Use a VM with SIMD support (future SVM improvements)
- Accept that this needs ZK proofs

**Rebuttal:** We agree on the diagnosis — it's a metering problem, not a hardware problem. But the standard SIMD process takes years. ZK proof generation for a 15M parameter forward pass would dominate frame time (seconds, not milliseconds). For a product that needs to ship, a custom syscall in an app-specific rollup is the pragmatic answer. The "correct" answer is years away.

### "Just use a coprocessor."

Risc Zero, SP1, Jolt can prove arbitrary computation. Generate a ZK proof offchain, verify onchain for ~300K CU. You get verifiability AND performance AND mainnet compatibility.

**Rebuttal:** ZK-proving 19M MACs is not trivial. Current proof generation would take seconds to minutes per frame — incompatible with real-time gameplay. The prover hardware costs are significant. And you still need an offchain executor, which breaks "the model IS the world" just as much as a custom ER does. The coprocessor approach is legitimate but solves a different problem (batch/async inference, not real-time interactive).

### "Every L2 team has this idea. Here's why they don't ship it."

Custom opcodes are easy to add. The hard part is the ecosystem:
- Tooling doesn't support it (explorers, debuggers, indexers)
- Upgradability is your problem (bug in matmul = hard fork your own validator)
- Composability dies (other programs can't reason about your computation)
- You've created a single-operator chain with Solana account compatibility

**Rebuttal:** All fair points for a production system. But:
- Tooling: the syscall is invisible to external tools. They just see account state changes.
- Upgradability: we control the validator. We can upgrade it anytime (no consensus to coordinate).
- Composability: other programs CAN compose with our accounts. They read/write state normally. The syscall is an implementation detail of our inference program.
- "Single-operator chain with Solana compatibility" — yes. That's an app-specific rollup. Arbitrum Stylus, OP Stack custom precompiles, Starknet's Cairo builtins — the industry is going here.

## The Honest Framing

Don't say: "Trustless onchain neural network inference."

Do say: "An app-specific rollup with native tensor math that settles state to Solana. The model weights are permanent and forkable on L1. Sessions execute in a custom SVM runtime optimized for INT8 matrix operations. The trust model is identical to any current ephemeral rollup — single sequencer, with a path to fraud proofs as the ecosystem matures."

This is honest and still impressive. The innovation is:
1. A custom SVM runtime purpose-built for ML inference
2. Weights-as-public-goods on L1
3. Real-time interactive gameplay driven by a learned model
4. The "arcade" model — persistent rules, ephemeral sessions

## Comparison With Industry Approaches

| Approach | Verifiability | Latency | Complexity | Ships today? |
|----------|--------------|---------|------------|-------------|
| Pure BPF (no syscall) | Full (any validator) | Too slow (480M CU) | Low | No (CU limit) |
| Custom syscall ER | Fork-verifiable | Real-time | Medium | **Yes** |
| ZK coprocessor | Full (proof verification) | Seconds per frame | Very high | No (proof time) |
| Optimistic rollup + fraud proof | Eventual verification | Real-time | Very high | No (fraud proof infra) |
| Offchain + oracle | Trust oracle | Real-time | Low | Yes (but no "onchain" claim) |

The custom syscall ER is the only approach that ships real-time interactive inference today with any meaningful verifiability claim.

## Path to "Correct"

The custom syscall is a pragmatic shortcut. The long-term path:

1. **Today:** Custom syscall in self-hosted ER. Demonstrate feasibility.
2. **Near-term:** Propose `sol_tensor_ops` through MagicBlock's plugin system. If they adopt it, fraud proofs work with their standard validator.
3. **Medium-term:** SIMD proposal for `sol_tensor_ops` on mainnet Solana. If adopted, full L1 execution becomes possible for smaller models.
4. **Long-term:** SVM SIMD support (128-bit vector instructions in BPF). The metering problem goes away. Pure BPF inference becomes feasible.

Each step increases verifiability while maintaining the same user experience.
