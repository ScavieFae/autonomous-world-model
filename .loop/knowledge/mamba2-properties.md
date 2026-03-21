# Mamba2 Properties for Experiment Design

What matters about our architecture when designing experiments.

## Architecture

- **Mamba2 with SSD (Structured State Space Duality).** Not a transformer. No attention, no KV cache. State evolves through structured state space dynamics.
- **d_model=384, d_state=64, 4 layers, headdim=64, chunk_size=30.** ~18.5M params total. Tiny by LLM standards.
- **Constant memory in sequence length.** No quadratic attention cost. Memory scales linearly. This means context_len experiments are cheap to run — the model doesn't blow up at K=50 the way a transformer would.
- **SSD chunking.** Sequences are processed in chunks of `chunk_size`. Chunk boundaries are a potential information bottleneck — state must carry information across chunk boundaries through the SSM state alone. Experiments that change `context_len` should consider alignment with `chunk_size`.

## What's Different From Transformers

- **No attention over context.** The model can't "look back" at arbitrary positions. Past information must be compressed into the SSM state. This makes the model more sensitive to what information is encoded in the state representation.
- **Error propagation is different.** In transformers, attention can skip over corrupted positions. In Mamba2, errors in the state propagate forward through every subsequent timestep. This is why Self-Forcing matters more here — the model MUST learn to recover from its own errors because state corruption is permanent.
- **No position embeddings.** The model is position-aware through its recurrent dynamics, not explicit position encoding. Time is implicit.
- **Linear scaling = cheap AR rollouts.** Autoregressive generation is O(1) per step (just advance the state). This makes rollout coherence evaluation fast and makes the crank module viable for real-time inference.

## Training Implications

- **Batch size vs. gradient updates tradeoff.** Smaller batches = more gradient updates per epoch. With only ~18.5M params, the model may benefit from more updates rather than more stable gradients. bs=256 vs bs=512 is worth testing.
- **Self-Forcing is architecturally important.** The SSM state accumulates errors differently than transformer hidden states. SF forces the model to learn error correction in the state space, not just in the output space.
- **Context length affects what the SSM state must encode.** Longer context gives the state more information to compress. Shorter context means each frame carries more weight. The optimal context length depends on what temporal dependencies matter.
- **Weight decay on embeddings matters.** Embedding tables are a large fraction of total params (especially action_state with 400 classes). WD on embeddings has been a consistent positive signal.
- **The model is small enough to overfit.** 18.5M params on 82M frames of data — the model can memorize. Regularization (dropout, WD, data augmentation) is more important than for larger models.

## Open Questions

- Does chunk_size=30 (matching context_len=30) create an implicit single-chunk regime? What happens with context_len > chunk_size?
- How does SSM state capacity (d_state=64) limit what temporal patterns can be captured? Would d_state=128 help or just add parameters?
- The SSD algorithm's backward pass materializes full state trajectories — this is the main VRAM cost during training. Gradient checkpointing could reduce this but isn't implemented.

## Deep Dives

- `docs/MAMBA2-EXPLAINER.md` — full architecture explanation
- `models/mamba2.py` — implementation with SSD chunked algorithm
- `research/sources/` — Mamba papers and related SSM work
