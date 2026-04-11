# JEPA Adaptation Notes for Melee World Model

Working notes on how the JEPA/LeWM paradigm maps to our structured game state problem.

## The Core Paradigm Shift

**Current approach (Mamba2):** Structured game state → per-field prediction heads → supervised loss per field
- 16 prediction heads (continuous regression, binary classification, categorical classification)
- Each head predicts one aspect of the next frame
- Loss is a weighted sum of per-head losses
- The model must learn cross-field dependencies implicitly through the shared backbone

**JEPA approach:** Observations → learned latent embedding → predict next embedding → decode back
- Single embedding per frame (192-dim in LeWM)
- Prediction happens in latent space — the model chooses its own representation
- Anti-collapse regularizer prevents trivial solutions
- Cross-field dependencies are captured natively in the embedding

## What Changes for Melee (Game State, Not Pixels)

LeWM uses a ViT to encode pixels. We don't have pixels — we have structured game state (positions, velocities, action states, controller inputs). The encoder needs to be different.

### Encoder Options

**Option A: MLP Encoder**
- Take our existing frame encoding (116D per frame for 2 players)
- MLP: 116 → 512 → 256 → embed_dim
- Simple, matches our data format directly
- Risk: might not learn rich enough representations from flat features

**Option B: Per-Field Embedding + Fusion**
- Keep our current categorical embeddings (action_state → 32D, etc.)
- Concat with continuous features
- Transformer/attention layer to fuse cross-player and cross-field
- Richer but more complex

**Option C: Treat Frame as "Image" of Features**
- Reshape frame into a 2D grid (players × features)
- Small CNN or ViT-like patchification
- Interesting but probably overkill for 116D

**Recommendation: Start with Option A (MLP), graduate to Option B if needed.**

### Predictor Options

The predictor takes a sequence of embeddings + actions and predicts next embedding.

**Option A: Transformer with AdaLN (LeWM-style)**
- Most direct port of LeWM
- AdaLN conditions on controller inputs
- Causal masking, history window
- Proven to work in LWM

**Option B: Mamba-2 with AdaLN**
- Keep our SSM backbone, add AdaLN action conditioning
- Interesting hybrid — SSM dynamics + JEPA training objective
- Untested combination

**Option C: GRU (EB-JEPA style)**
- Simplest recurrent predictor
- Good baseline to start

**Recommendation: Start with Transformer (Option A) to stay close to proven LeWM recipe.**

### Decoder

LeWM doesn't decode — it plans entirely in latent space. But we need to produce interpretable game state for the visualizer and onchain deployment.

**Approach: Lightweight decoder heads**
- From embed_dim → per-field predictions
- Similar to our current prediction heads but operating on a single latent vector
- Train decoder alongside the JEPA objective (optional auxiliary loss, or train separately)

This is a departure from pure JEPA (which never decodes during training). Two options:
1. **Pure JEPA + post-hoc decoder:** Train the world model with JEPA loss only, then train a frozen-encoder decoder afterward. Cleaner separation, matches LeWM philosophy.
2. **JEPA + decoder auxiliary loss:** Add lightweight reconstruction loss during training. Slightly less pure, but ensures the latent space remains decodable. Risk: reconstruction loss might fight SIGReg.

**Recommendation: Start pure (option 1), add auxiliary decoder loss if latent space isn't decodable enough.**

### Action Conditioning

**LeWM:** Actions are continuous vectors → Conv1d + MLP → AdaLN modulation of predictor layers.

**Melee adaptation:**
- Controller inputs are 13D (analog sticks, buttons) — already continuous/binary
- Could use the same AdaLN pattern directly
- The Conv1d frameskip stacking isn't needed (we predict every frame, no frameskip)

**Julian Saks / action tokenization question:**
LeWM treats actions as continuous vectors that condition the predictor. The alternative is to define a discrete action vocabulary for Melee ("short hop", "wavedash", "grab") and embed those as tokens.

Arguments for discrete actions:
- Melee has a finite set of meaningful controller input combinations
- Action states already discretize what the character is doing
- Robot arm analogy: well-defined action primitives improve world model quality

Arguments against (for now):
- Raw controller inputs are already available and well-understood
- Defining the right action vocabulary is a research project in itself
- LeWM's continuous conditioning works — don't add complexity to the first experiment

**Recommendation: Keep continuous controller inputs for the first experiment. Action tokenization is an interesting follow-up.**

## Anti-Collapse for Game State

The collapse problem: if the encoder maps all game states to the same embedding, prediction loss goes to zero trivially.

With pixel observations, collapse is the central challenge. With structured game state, it may be less severe because:
- The input features are already semantically meaningful (not raw pixels)
- The encoder is simpler (MLP, not ViT)
- But it can still happen! Nothing prevents an MLP from collapsing.

**SIGReg should work here.** It's architecture-agnostic (paper ablates ViT and ResNet-18). An MLP encoder should be fine.

## Evaluation: How Do We Compare?

Our current metric is **rollout coherence** (mean position MAE over K=20 horizon). For JEPA:

1. **Rollout coherence (decoded):** Decode JEPA predictions back to game state, compute MAE. Direct comparison with Mamba2.
2. **Latent prediction error:** MSE in embedding space over rollout horizon. Internal metric for JEPA quality.
3. **Linear probe accuracy:** Can we linearly decode game state fields from the latent embedding? Measures representation quality.
4. **Temporal straightness:** Cosine similarity of consecutive latent velocity vectors. LeWM reports this as an emergent property.

## Open Questions

1. **Is latent-space prediction actually better for structured data?** LeWM's advantage comes from abstracting away pixel noise. Game state doesn't have "noise" in the same way — every field is semantically meaningful. The latent space might add indirection without benefit.

2. **How does the decoder quality affect onchain deployment?** If we need deterministic INT8 inference onchain, the decoder adds another quantization challenge.

3. **Multi-step rollout stability.** LeWM only predicts 1 step ahead during training (num_preds=1). Our self-forcing experiments showed that multi-step training helps. Can JEPA + self-forcing work together?

4. **Two-player dynamics.** LeWM handles single-agent environments. Melee has two interacting players. How do we structure the embedding to capture interactions?
