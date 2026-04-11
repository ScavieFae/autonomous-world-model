# How P0 and P1 Reach the JEPA Latent

A trace of the data flow from raw game state through the JEPA encoder, with focus on **where player identity lives at each step** and why it matters.

## Why this page exists

CLIP and similar contrastive models have a well-known "bag-of-words" failure mode: "put yellow cup on pink cup" and "put pink cup on yellow cup" pool to nearly the same vector because the attention-then-pool pipeline discards word order. The analogous question for our two-player world model is: **when JEPA encodes a frame, can it tell P0 from P1, or do their features get averaged into a single "game situation" vector that loses identity?**

Our architecture doesn't have CLIP's *exact* pathology — we concat, we don't pool — but it has a related one: the trunk MLP is free to learn swap-symmetric features even if the slots are dedicated. This page is the reference for understanding what the encoder actually does with P0/P1 features, so that the diagnostic instrumentation (`training/jepa_diagnostics.py`) and any future architectural changes are grounded in the real data flow.

## Input layout (what the dataset hands to the encoder)

A single frame is three tensors:

| Tensor | Shape | Contents |
|--------|-------|----------|
| `float_frames` | `(B, T, F)` where `F = 2 * float_per_player` | Continuous + binary + controller features for both players, concatenated |
| `int_frames` | `(B, T, I)` where `I = 2 * int_per_player + 1` | Categorical indices for both players, plus stage |
| `ctrl_inputs` | `(B, T, C)` | Controller features (extracted from float_frames, re-packed for predictor conditioning) |

### `float_frames` — the dedicated-slot layout

Everything is **concatenation by position**. For `float_per_player = fp`:

```
float_frames[..., 0      : fp    ]  ← P0's features
float_frames[..., fp     : 2*fp  ]  ← P1's features
```

Inside each player's `fp` slots, the order is fixed by `EncodingConfig`:

```
[percent, x, y, shield,                                     ← core_continuous (4)
 speed_air_x, speed_y, speed_ground_x, speed_attack_x, speed_attack_y,  ← velocity (5)
 hitlag, stocks, (state_age), (hitstun),                    ← dynamics (2-4)
 combo_count,                                                ← combat_continuous (1)
 (projectile features),                                      ← optional
 facing, invulnerable, on_ground, (state_flag bits×40),     ← binary (3 or 43)
 stick_x, stick_y, cstick_x, cstick_y, shoulder,            ← analog controller (5)
 a, b, x, y, z, l, r, start]                                ← button controller (8)
```

**Key: P0's `x` lives at index `1`. P1's `x` lives at index `fp + 1`.** They are never in the same slot. There is no averaging operation anywhere in the pipeline that would combine them before the trunk MLP sees them.

### `int_frames` — categoricals

For `int_per_player = ipp` (7 or 8 depending on `state_age_as_embed`):

```
int_frames[..., 0        : ipp    ]  ← P0's categoricals
int_frames[..., ipp      : 2*ipp  ]  ← P1's categoricals
int_frames[..., 2*ipp    : 2*ipp+1]  ← stage (shared)
```

Inside each player's block, the fixed order is:

```
[action, jumps, character, l_cancel, hurtbox, ground, last_attack, (state_age)]
```

So **P0's character ID is `int_frames[..., 2]`, P1's character ID is `int_frames[..., ipp + 2]`.** Dittos (same character on both ports) are detectable by comparing these two columns.

### `ctrl_inputs` — what the predictor sees for AdaLN conditioning

Built by `JEPAFrameDataset._extract_ctrl` from the float_frames. For `ctrl_threshold_features=true` and `press_events=false`, `lookahead=0`:

```
ctrl_inputs = cat([
    p0_ctrl,          # 13 dims  (analog + buttons)
    p1_ctrl,          # 13 dims
    p0_analog_thresh, # 5  dims  (|stick| > 0.3 binarization)
    p1_analog_thresh, # 5  dims
])                    # 36 dims total
```

Same story: positional slots, P0 and P1 never share a dim.

## Step-by-step through `GameStateEncoder.forward`

`models/jepa/encoder.py:75-115`. The encoder receives `(float_frames, int_frames)` and produces `(B, T, embed_dim)` latents. I'll trace one frame's worth.

```python
flat_float = float_frames.reshape(B*T, F)
flat_int   = int_frames.reshape(B*T, I)
```

So we're working on `(B*T, F)` and `(B*T, I)`. Every frame from every batch is now a row.

### 1. Categorical embedding lookups

```python
parts = [flat_float]

for offset in [0, ipp]:                     # ← P0 then P1
    parts.append(self.action_embed(flat_int[:, offset + 0]))       # action (32D)
    parts.append(self.jumps_embed(flat_int[:, offset + 1]))        # jumps (4D)
    parts.append(self.character_embed(flat_int[:, offset + 2]))    # character (8D)
    parts.append(self.l_cancel_embed(flat_int[:, offset + 3]))     # l_cancel (2D)
    parts.append(self.hurtbox_embed(flat_int[:, offset + 4]))      # hurtbox (2D)
    parts.append(self.ground_embed(flat_int[:, offset + 5]))       # ground (4D)
    parts.append(self.last_attack_embed(flat_int[:, offset + 6]))  # last_attack (8D)
    if self.cfg.state_age_as_embed:
        parts.append(self.state_age_embed(flat_int[:, offset + 7]))# state_age (8D)

parts.append(self.stage_embed(flat_int[:, -1]))                    # stage (4D)
```

Each `nn.Embedding.lookup` is an indexed fetch. `self.action_embed` is a single `nn.Embedding(400, 32)` whose weights are **shared between P0 and P1** — the model learns one action-embedding table and queries it twice, once for P0's action, once for P1's action. The two results land in **different `parts` entries**, which will become **different slot ranges** in the concatenated vector.

This is a deliberate weight-sharing choice: the meaning of action state 47 (say, "falling") is the same regardless of which player is in it. Shared-weight categorical embeddings give the right inductive bias without hurting identity — identity is preserved by *position* in the output concat, not by *embedding weights*.

### 2. The big concatenation

```python
x = torch.cat(parts, dim=-1)   # (B*T, total_input_dim)
```

This is the critical structural move. `parts` is ordered:

```
[flat_float,                              ← 2 * float_per_player dims (P0 floats || P1 floats)
 P0_action, P0_jumps, ..., P0_state_age,  ← P0 categorical embeddings
 P1_action, P1_jumps, ..., P1_state_age,  ← P1 categorical embeddings
 stage]                                    ← shared
```

After the cat, `x` has shape `(B*T, total_input_dim)` where `total_input_dim ≈ 278` for the b002 encoding. **Each element of this vector has a fixed, dedicated meaning.** P0's action-embedding dim 5 is always at the same index, and it is never the same index as P1's action-embedding dim 5.

This is the key difference from CLIP: **there is no pooling here.** CLIP would take a sequence of tokens `[yellow, cup, on, pink, cup]`, run them through a transformer, and then mean-pool (or CLS-token) the result to a single vector — that pooling is where word order gets lost. Our encoder has no equivalent step. Every input feature keeps its identity through the concat.

### 3. The trunk — where structural guarantees end

```python
x = self.trunk(x)   # Linear(278→512) → SiLU → Linear(512→512) → SiLU
```

**This is where the CLIP-adjacent failure mode can sneak in.** The first `nn.Linear` has a weight matrix of shape `(512, 278)`. Each output neuron is a learned linear combination of **all** input features, including both P0's and P1's features. A neuron whose weight happens to be large for P0's x-position and equally large for P1's x-position will produce a feature that's swap-symmetric: it doesn't matter whether you put player A on port 0 or port 1, this neuron fires the same.

The trunk can learn this, and in some regimes it **will** learn this, because:

- **Shortcut bias.** If the downstream loss can be minimized with features that are functions of "the game situation" rather than "which player is which," gradient descent will happily discover them. Half the parameters become redundant.
- **Dittos give no asymmetry signal.** In a Fox-vs-Fox match, the character embeddings for P0 and P1 are identical (same input → same embedding → same contribution to the trunk's weighted sum). The only P0/P1 distinguishing signal is positional/kinematic — x, y, velocity, action state. If those happen to also land on swap-symmetric trunk neurons, identity collapses for dittos specifically.
- **No loss term explicitly penalizes identity collapse.** SIGReg regularizes the *distribution* of encoder outputs to be Gaussian. MSE in latent space measures whether the predictor can match the encoder's output. Neither of these terms cares who is P0 and who is P1, as long as the encoder is self-consistent and the predictor can track it. This is structurally **weaker** than Mamba2's loss, which has per-player prediction heads that literally compute separate MSEs for P0's next position and P1's next position.

The trunk *can* learn to preserve identity — nothing forces it into collapse, and the downstream `pred_loss` genuinely requires knowing which player did what in order to predict the next frame. But it doesn't have to, and whether it does is an empirical question we need to measure.

### 4. The projector

```python
x = self.projector(x)   # Linear(512→2048) → SiLU → Linear(2048→192) → BatchNorm1d
```

Matches LeWM's projector pattern. `BatchNorm1d` at the end normalizes each of the 192 output features to zero-mean unit-variance across the batch. BN doesn't cause identity collapse directly (it operates per-dim), but it does mean the final latent lives on a well-scaled manifold that SIGReg can then pull toward isotropic Gaussian. The projector is 2 linear layers deep, so it preserves whatever identity-aware structure the trunk produces — it doesn't re-mix P0/P1.

### 5. Reshape back

```python
return x.reshape(B, T, self.embed_dim)
```

One 192-dim vector per (batch, time). All of P0 and P1's combined state is compressed into this single vector.

## How the predictor uses it — AdaLN action conditioning

`models/jepa/predictor.py`. The predictor takes `(context_embs, ctrl_inputs)` and outputs next-frame latent predictions.

- `context_embs`: `(B, H, D)` — the 192-dim latents for the `history_size=3` context frames.
- `ctrl_inputs`: `(B, H, 36)` — per-frame controller features (P0 and P1 concatenated, as above).

Inside:

```python
action_embs = self.action_encoder(ctrl_inputs)  # (B, H, 192)
x = context_embs + self.pos_embed[:, :H, :]
for block in self.blocks:
    x = block(x, action_embs, causal_mask)
```

The `action_encoder` is a 2-layer MLP that maps the 36-dim controller vector to a 192-dim action embedding. **Same positional-slot story as the state encoder:** P0's 13 controller dims and P1's 13 controller dims live in dedicated slots, the MLP can learn to preserve or collapse identity. The action embedding then modulates AdaLN shift/scale/gate at each transformer block — it's multiplicatively mixed into the latent, not appended as a separate token.

The predictor's self-attention operates on the sequence of frame latents (length 3). It doesn't re-mix P0/P1 at the attention level because each timestep is already a single combined-player latent. Attention lets the model aggregate *temporal* context, not player context.

## Where identity can fail

Given the above trace, there are three concrete places to watch:

1. **Encoder trunk MLP** learns swap-symmetric features. The 278→512 linear layer finds that averaging is a useful shortcut. This is the most likely failure mode. Detected by **swap test** on encoder outputs — swap P0↔P1 in the raw inputs, re-encode, measure cosine similarity. Expected low (distinct), observed high means collapse.

2. **Projector BN collapses informative variance.** Less likely but possible: BN's cross-batch normalization interacts badly with SIGReg's isotropy pressure and squashes identity dimensions. Detected by **per-player linear probes** — fit a linear model from latent → P0.x and another from latent → P1.x. Both should fit well. If one is noise, identity has collapsed.

3. **Action encoder MLP** collapses P0/P1 controller symmetry. The 36→192 action encoder could average P0 and P1 controllers. This would break the predictor's ability to condition on "who pressed what." Detected by **ctrl swap test** — swap P0/P1 controller inputs (without swapping the frame inputs), feed to predictor, verify predictions differ. Not yet implemented as a separate diagnostic — covered indirectly by the state-swap test.

## The Mamba2 comparison

Mamba2 has the same concat layout but a **different loss shape** that protects it: 16 prediction heads compute per-player MSEs and classification losses for each field separately. Every gradient step says "predict P0's x *separately from* P1's x." This is a strong, direct incentive for the encoder and backbone to keep identities distinct. The loss literally cannot be minimized without per-player representation.

JEPA has one latent MSE + SIGReg. Neither cares about per-player identity per se. The `pred_loss` term indirectly requires identity preservation (because you can't predict the next frame without knowing who is doing what), but only to the extent that the specific next-frame prediction is sensitive to the distinction. For large-scale motion prediction this is probably fine; for subtle relational events (who is in hitstun from whom) the signal is weak.

**This means JEPA is structurally more at risk of identity collapse than Mamba2, specifically because its loss doesn't explicitly bind identity.** That's why the diagnostic suite exists, and why we pre-register an architectural mitigation (per-player shared-weight sub-encoder + cross-attention fusion, canonical from AlphaZero) as the fix-of-first-resort if the swap test fires.

## Diagnostic hooks

All three failure modes have corresponding cheap diagnostics in `training/jepa_diagnostics.py`:

| Diagnostic | Detects | How |
|------------|---------|-----|
| `swap_test` | Encoder trunk collapse | Encode `(floats, ints)` and `swap_p0_p1(floats, ints)`, compute cosine similarity. Low = good. |
| `linear_probe_r2` applied per-player | Projector / downstream collapse | Fit linear regression from latent → P0.x, latent → P1.x separately. Both should be >0.8. |
| `relational_probe_r2` | Cross-player binding | Fit linear regression from latent → `(P0.x - P1.x)`. Should fit well if the model encodes relative position. |
| `temporal_straightness` | Trajectory quality (emergent from LeWM) | Cosine similarity of consecutive latent velocity vectors. Should increase during training. |

All are GPU-resident, closed-form, and run on a fixed held-out batch per epoch. No sklearn, no extra dependencies. See the diagnostics module and `scripts/run_jepa_diagnostics.py` for wiring.

## Related files

- `models/jepa/encoder.py` — the trace above
- `models/jepa/predictor.py` — AdaLN action conditioning
- `data/jepa_dataset.py` — `_extract_ctrl` packing
- `models/encoding.py` — `EncodingConfig` field layout and dimension properties
- `training/jepa_diagnostics.py` — the identity diagnostic suite
- `scripts/run_jepa_diagnostics.py` — CLI to run diagnostics on a trained checkpoint
- `research/sources/jepa-adaptation-notes.md` — Open Question #4 on two-player dynamics (now actively being instrumented)
