# EigenCloud Analysis: Verified Offchain Compute for AWM

**Date:** 2026-03-02
**Status:** Research / evaluation

---

## Overview

EigenCloud (formerly EigenLayer) offers verifiable offchain compute backed by cryptoeconomic security (EIGEN token staking + slashing). Two primitives are relevant:

- **EigenCompute** — Run arbitrary Docker containers in Google Cloud TEEs. Python, Rust, Go, TypeScript supported. Verifiable execution via hardware attestation + slashing. Settles to Ethereum, Solana, Base, Arbitrum.
- **EigenAI** — Bit-exact deterministic LLM inference on GPUs. OpenAI-compatible API. Currently only 2 models (gpt-oss-120b-f16, qwen3-32b-128k-bf16). Built on llama.cpp. **Not applicable** — we need custom model support, not hosted LLMs.

**We would use EigenCompute** to containerize our crank (Python + PyTorch + Mamba2) and settle match results to Solana.

---

## Architecture: EigenCompute Path

```
┌──────────────────────────┐      ┌─────────────┐
│  EigenCompute (TEE)      │      │   Solana     │
│                          │      │              │
│  Docker: crank + model   │─────▶│  Settlement  │
│  - Mamba2 inference      │ post │  - FrameLog  │
│  - PolicyAgent           │ ───▶ │  - Results   │
│  - INT8 weights          │      │  - ELO       │
│  - Hidden state in RAM   │      │              │
└──────────────────────────┘      └─────────────┘
         │                              ▲
    TEE attestation              reads/verifies
         ▼                              │
   ┌──────────────┐              ┌──────────────┐
   │  EigenDA     │              │  The Wire    │
   │  (audit log) │              │  (site)      │
   └──────────────┘              └──────────────┘
```

### How it would work

1. Containerize existing crank (`crank/` + `models/` + checkpoints) as a Docker image
2. Deploy to EigenCompute — runs in a TEE with attestation
3. Container runs continuous match loop, streams VizFrames to site via WebSocket
4. Match results (winner, stocks, ELO delta) settle to Solana asynchronously
5. TEE attestation + EigenDA audit log prove the inference was executed correctly
6. Slashing-backed EIGEN stake enforces honesty

### What changes vs. MagicBlock

| Aspect | MagicBlock ER + syscall | EigenCompute |
|---|---|---|
| Where inference runs | Onchain (SBF VM + native syscall) | Offchain (Docker/TEE) |
| Verification model | Execution IS verification | TEE attestation + slashing |
| "Model IS the world" | Yes — chain computes reality | No — chain records/verifies results |
| Latency | ~10ms block time | Unknown; likely 100ms+ for settlement |
| 60fps rendering | From chain directly | Stream via WS, settle async |
| Custom syscall needed | Yes (MagicBlock cooperation) | No |
| Weight storage | Onchain accounts (7.5MB) | Container filesystem |
| Forkability | Anyone can fork weights + run parallel world | Weights not inherently public |

---

## Advantages

### Removes the biggest external dependency
No need for MagicBlock to add `sol_matmul_i8` to their validator. The crank already runs standalone — containerizing it is straightforward.

### Full model size, no CU constraints
15M params runs in ~0.5-2ms on native hardware. No BPF overhead, no CU limits, no compute budget concerns.

### INT8 determinism is free
Our INT8 inference is already bit-exact everywhere — integer math produces identical results on any hardware. TEE attestation proves it was run correctly. We don't need EigenAI's GPU determinism tricks (those solve floating-point non-determinism, which we don't have).

### Solana settlement
Match results, ELO updates, sponsorship transactions all settle to Solana. The onchain state is the source of truth for the competitive layer even if inference runs offchain.

### Same site integration
The site consumes VizFrames over WebSocket — doesn't care whether frames come from a local crank, a MagicBlock ER, or an EigenCompute container.

---

## Concerns

### 1. Settlement latency

No published round-trip numbers for EigenCompute → Solana. Their showcase app ("Humans vs. AI", ~70K WAU) is turn-based — no real-time constraint. For our use case:

- **Rendering**: Stream frames from container to site via WS — fast, independent of chain
- **Settlement**: Post match results to Solana — can be async (after match ends)
- **Per-frame settlement**: Probably not feasible or necessary. Settle match outcomes, not individual frames.

### 2. Statefulness

Mamba2 has ~200KB of recurrent hidden state that persists across frames within a match. The container must be long-running and stateful. EigenCompute seems designed for shorter-lived tasks. Need to verify that continuous inference loops are supported.

### 3. Documentation maturity

EigenCompute is in gated alpha. Developer docs consist of:
- CLI install command (`curl -fsSL https://tools.eigencloud.xyz | bash`)
- Language list (Python, Rust, Go, TS, Ruby)
- "Anything that runs in Docker"
- No SDK reference, no settlement API docs, no latency SLAs

"Onchain API coming later" for EigenAI suggests Solana integration may still be in development.

### 4. Philosophical shift

The project thesis is "the model IS the world" — the blockchain doesn't just record, it computes. EigenCompute changes this to "verified offchain compute with onchain settlement." Both are valid trust models, but different stories.

Alternative framing: **the onchain mechanisms verify the autonomousness** — the chain enforces that matches are computed by the declared model with no tampering, even if the computation happens offchain. The autonomy comes from the verification, not the execution location.

---

## Comparison: Monad vs. EigenCompute vs. MagicBlock

| | MagicBlock ER | EigenCompute | Monad |
|---|---|---|---|
| Inference execution | Onchain (BPF + syscall) | Offchain (TEE container) | Onchain (EVM bytecode) |
| 60fps feasible | Yes (with syscall) | Yes (compute) / No (per-frame settlement) | No (400ms blocks, gas limits) |
| Custom compute | sol_matmul_i8 syscall | Any Docker container | No custom precompiles |
| External dependency | MagicBlock adds syscall | EigenCompute alpha access | None |
| Weight storage | Solana accounts (10MB) | Container filesystem | 312 contracts × 24KB |
| Verification | Execution = truth | TEE + slashing | Execution = truth |
| Full model (15M) | Feasible with syscall | Trivial | Impossible (>block gas) |
| Settlement chain | Solana (native) | Solana (bridge) | Monad (native) |

---

## Hybrid Strategy

Use EigenCompute as the immediate execution layer, with MagicBlock as the long-term target:

```
Phase 1 (hackathon): EigenCompute container → WS stream → site
                     Match results settle to Solana
                     "Verified autonomous AI inference"

Phase 2 (post-hack): MagicBlock ER with sol_matmul_i8
                      True onchain inference
                      "The model IS the world"
```

Both phases use the same site, same WS protocol, same Solana settlement for results. The difference is where inference runs.

---

## Next Steps

1. **Get EigenCompute alpha access** — apply via developers.eigencloud.xyz
2. **Test containerization** — Dockerfile for crank + models + checkpoints, verify it runs
3. **Probe settlement API** — how does a container post results to Solana? What's the SDK?
4. **Measure latency** — end-to-end from container inference to Solana confirmation
5. **Verify statefulness** — can a container run a continuous loop, or is it request/response only?

---

## References

- [EigenCloud Developer Portal](https://developers.eigencloud.xyz/)
- [EigenCloud Technical Blog](https://blog.eigencloud.xyz/eigencloud-technical-blog/)
- [EigenAI Deterministic Inference](https://blog.eigencloud.xyz/deterministic-ai-inference-eigenai/)
- [EigenAI + EigenCompute Launch](https://blog.eigencloud.xyz/eigencloud-brings-verifiable-ai-to-mass-market-with-eigenai-and-eigencompute-launches/)
- [Monad Gas Pricing](https://docs.monad.xyz/developer-essentials/gas-pricing)
- [Monad Opcode Pricing](https://docs.monad.xyz/developer-essentials/opcode-pricing)
