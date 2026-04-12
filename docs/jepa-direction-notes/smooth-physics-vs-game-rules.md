# Smooth Physics vs. Game Rules: JEPA's Structural Gap on Melee

**Date**: 2026-04-11
**Status**: research note, not a decision
**Derived from**: e030a viz analysis (Fox-Marth FD canonical test game), e030b full-training results, nonlinear probe diagnostic

## Tl;dr

Three independent observations from the JEPA lineage this session all turned out to be symptoms of **one underlying problem**: MSE loss in latent space plus a smoothness prior systematically produces encoders that are good at continuous physics and bad at discrete game-rule transitions. This note ties the observations together, explains why the mechanism is structural (not a training bug we can hyperparameter-tune out), and lists what we might try.

**Three observations, one disease:**

1. **We've never seen any world model correctly represent death → respawn.** Across Mamba2 and JEPA, the death→respawn cycle (character dies → teleports to center stage → resumes at 0%) is never learned. The model always predicts "character stays dead forever."

2. **Percent goes backwards in the e030a reconstructed viz.** Real percent is monotone non-decreasing (up on hits, resets to 0 on death). Reconstructed percent oscillates around the mean (encoder recon) or drifts smoothly toward the training-set mean (rollout recon). Never resets, never steps up discretely.

3. **e030b's linear probes degraded monotonically over training.** Epoch 1: p0_x R² = 0.904 (Competitive). Epoch 10: p0_x R² = 0.305 (Not viable). All loss curves looked textbook healthy throughout — pred_loss down, sigreg_loss down, straightness up. But the encoder was progressively dropping information about game state. Nonlinear probe confirmed this is real information loss, not a probe methodology artifact.

**The common cause**: Melee's state has two layers of structure — smooth physics (position, velocity) and discrete game rules (hits, shield break, death/respawn, action-state transitions). Our JEPA training regime explicitly optimizes for smooth prediction (MSE + SIGReg's isotropic Gaussian prior + rising temporal straightness emergence) and has no mechanism to reward representing the discrete layer. So the encoder progressively specializes on the smooth layer and drops the discrete layer.

This is a **structural** property of the loss, not a hyperparameter issue or a training-length issue. It would reproduce on other encoders and other architectures as long as the loss is MSE on continuous latents. You can't λ-sweep your way out of it — you can only mitigate.

## The three observations, in detail

### Observation 1: Death → respawn has never worked

**Setup**: We train world models that take a context of ~30 frames and predict the next frame's game state (Mamba2) or latent (JEPA). The training set contains games where characters die and respawn. Death ends a life (stocks decrement), then the character teleports to the respawn platform at the stage center (~(0, 40) on FD), and physics resumes at 0% damage.

**What we see**: No world model in our experiments has ever cleanly predicted this transition. In rollout visualizations, when the seed context includes a death, the model predicts the character stays dead forever (constant position, constant velocity, dead action state repeating). When the seed context doesn't include a death, the model never produces one — rollouts drift in position and percent but never reach a discrete "death event" that would reset.

**Why it's hard**:

1. **It's a discontinuity in a continuous loss landscape.** Between two ordinary frames, position changes by ≤10 pixels. Between the last dead-animation frame and the first respawn frame, position jumps 100–400 pixels. MSE's optimal strategy is "predict smooth continuation, eat the occasional big error on the teleport frame." The teleport is ~0.01% of frames; the gradient budget for learning it is negligible.

2. **The rare-event imbalance is brutal.** In a 5-minute Fox-Marth game, maybe 6–8 deaths happen. Each is 1 teleport frame + ~60 frames of death animation. Out of ~18K frames per game, the critical transition frame is 0.01% of training data. Not enough gradient to change the weights.

3. **Respawn is a game rule, not a dynamics rule.** Physics rules are local: given velocity and gravity, next position is p + v + g. Respawn is a state-machine rule: "if stocks decrement, teleport to (0, 40), set invincibility=true." There is no input feature that says "teleport now." The model must learn a symbolic conditional, which requires orders of magnitude more signal than learning a local continuous function.

4. **Context window too short to fit the full pattern.** At `context_len=30` (500ms at 60fps) for Mamba2, or `history_size=3` (50ms) for e030a/b JEPA, the model might not see both the death animation and the respawn frame in the same context window. For JEPA specifically at 50ms context, the death→respawn transition is guaranteed not to fit.

5. **For JEPA: latent trajectories want to be straight.** LeWM's temporal straightness diagnostic is literally a measure of latent smoothness over time. A perfectly straight latent trajectory cannot represent a teleport. The encoder would have to produce a discontinuous latent jump at death→respawn, and SIGReg + the smoothness prior actively push against this. We see this in e030b's data: straightness rises monotonically from 0.51 to 0.78 across training, which is "good" by LeWM standards but is also *directly forbidding* the kind of discontinuities that respawn requires.

### Observation 2: Percent going backwards in the viz

**Setup**: Running the JEPA visualizer against a verified FD test game (Fox-Marth, 600 frames, no deaths). Three tracks: Ground Truth, Encoder Recon (per-frame round-trip through encoder + linear probe decoder), Rollout Recon (autoregressive 60-step predictor rollout + same decoder).

**The actual numbers**:

| Track | p0 percent | p1 percent | Decreases / total |
|---|---|---|---|
| **Ground Truth** | 84 → 96 (step function, 1 bump) | 14 → 23 (step function, 1 bump) | **0** / 599 |
| **Encoder Recon** | 83 → 74 (noisy around mean) | 13 → 18 (noisy around mean) | 239 / 599, 190 / 599 |
| **Rollout Recon** | 83 → **24** (monotonic DOWN) | 13 → **0** (monotonic DOWN) | 61 / 62, 40 / 62 |

Ground truth is a step function: 99.7% of frames have the same percent as the previous frame, 0.3% are a +10 bump. No resets (nobody died in the 10-second window).

Encoder Recon oscillates around the mean: the probe fits to embedding dimensions that correlate incidentally with percent in the training window — probably position and velocity, which do vary — and those correlates oscillate even when percent doesn't. Linear R² for p0_percent on this viz is actually *negative*, meaning the probe is worse than predicting the mean. This is a probe methodology artifact (over-parameterized linear fit to a step-function target with 14 unique values) but the underlying encoder issue is that percent is represented weakly.

Rollout Recon drifts toward the training-set mean: starting at p0_percent = 83%, the rollout trends down ~1 pct per frame for a solid second straight, ending at 24%. This is regression-to-the-mean — MSE-trained predictors pull outputs toward the training distribution's center when extrapolating. Mean percent in a Melee game is probably around 30–50%. 83% is "unusual" (late-game, near-death), so the predictor drifts toward the typical region. For p1 starting at 13%, the drift happens to go the other way, ending at 0%.

**Why this matters**: Percent is a step function. The model should either predict no change (most frames) or a discrete +10 bump (rare frames). Nothing about "steady drift" is in the training distribution. The fact that we see steady drift at all means the model has learned "percent is a continuous value that changes slowly" — **the wrong abstraction for this field**. Percent is fundamentally a discrete counter, not a continuous fluid.

### Observation 3: e030b's probes degraded monotonically during training

**Setup**: e030b training run, 10 epochs on 15.6M training examples, batch 1024, lr 4e-4. The diagnostic suite runs a held-out probe every epoch: fit linear regression from encoder embeddings to game state targets (position, percent, action state) on one stride-sampled val batch, evaluate on a disjoint stride-sampled val batch.

**The trajectory**:

| Epoch | p0_x R² | p1_x R² | p0_action | p1_action | sigreg | straight |
|---|---|---|---|---|---|---|
| 1 | **0.904** | **0.891** | **0.753** | **0.700** | 2.07 | 0.51 |
| 5 | 0.734 | 0.681 | 0.217 | 0.174 | 0.79 | 0.73 |
| 10 | **0.305** | **0.246** | **0.151** | **0.133** | 0.69 | 0.78 |

Every epoch's probe numbers are worse than the previous epoch's. This is **monotonic degradation, not noise**. Meanwhile: loss curves are textbook healthy (pred_loss 0.080 → 0.003, val tracking train, no overfit, sigreg trending to plateau, straightness rising). Swap test is stable at ~0.035/0.19 the whole time (no identity collapse).

**Nonlinear probe test**: took e030b's epoch-10 encoder, fit a 2-layer MLP (192 → 64 → 1, ReLU) instead of a linear regression, and evaluated on held-out samples. Result: nonlinear R² is **roughly the same as linear R²** across all fields. Where linear is 0.82 (p1_x), nonlinear is 0.78. Where linear is -0.77 (p0_y), nonlinear is 0.02. The MLP never recovers information that the linear probe couldn't find. **The information really is gone**, not hiding in a nonlinear subspace.

**What this tells us**: the encoder at epoch 10 has progressively compressed out the fine details that were present at epoch 1. The representation space is smaller/coarser than it was, in a way that trades fidelity to specific game-state fields for better pred_loss on the smoothly-varying bulk of frames.

## Why these three observations are the same phenomenon

Melee's per-frame state has two structurally different layers:

**Layer A: Continuous physics**. Position, velocity, `state_age`. These change smoothly from frame to frame. Between any two consecutive frames, they change by ≤10 units in magnitude. They're predictable from local history via smooth functions.

**Layer B: Discrete game rules**. Percent (step function with discrete bumps), shield (step function with breaks), action_state (400-class categorical transition machine), stocks (integer counter), death→respawn (discrete state jump). These don't change smoothly. They change via events, and the events are rare and structured.

JEPA's training regime provides four signals, all of which favor Layer A:

1. **MSE in latent space** rewards smooth predictions. For Layer A fields, MSE-optimal is "continue the trend." For Layer B fields, MSE-optimal is "always predict the mean" — because getting a discrete event wrong costs a huge loss spike, and getting the smooth mean means you're wrong by a small constant on every frame. Gradient descent finds the local optimum.

2. **SIGReg** pushes latents toward isotropic Gaussian. A Gaussian is smooth, continuous, unimodal. It has no capacity to represent "discrete state machine in the encoder." SIGReg doesn't care what's in the encoder's output, only that the *distribution* looks Gaussian — which it will, even if every continuous dimension drops information about Layer B.

3. **Temporal straightness** — LeWM's emergent diagnostic — *is literally a measure of how smoothly latents move through time*. Rising straightness means the encoder is learning to produce trajectories that look like lines. Lines can't represent teleports. SIGReg + MSE + straightness form a joint optimum at "smooth latent flow," which is structurally incompatible with discrete Layer B events.

4. **No explicit Layer B reward**. JEPA's loss doesn't have a term that says "make sure we can decode action_state correctly." The only way Layer B information survives is if it happens to be useful for pred_loss — which it mostly isn't, because Layer B is rare and Layer A is constant.

**So the encoder specializes on Layer A.** Over training, it progressively compresses Layer B out of the representation because Layer B is loss-expensive and gradient-starved. What we see in e030b is the encoder trading Layer A fidelity (x, y, position) against Layer B fidelity (percent, action, shield) — and the trades go one way. At epoch 1, the encoder hadn't yet specialized and all fields had reasonable R². At epoch 10, fields that MSE trains against have survived (x, y, somewhat) and fields that MSE punishes (percent, action, shield) have withered.

This also explains the specific pattern in the nonlinear probe results: `p1_x R² = 0.82` but `p0_action_acc = 6%`. Position survived, action state didn't. Because position is Layer A and action state is Layer B.

The percent-going-backwards in the viz is the same mechanism acting on a slightly different time horizon. Over 600 frames, percent is supposed to be a step function. The encoder has learned to treat it as a continuous value and slowly drift. Drift is smooth. Step functions are not smooth. The reconstructed percent smoothly drifts because the encoder no longer has the concept of "percent jumps up by a discrete amount when hit."

Death→respawn is the extreme case where the Layer B event is catastrophic: position and action state both jump simultaneously. If the encoder can't represent small Layer B events (percent bumps), it has zero hope of representing the large ones (death respawn teleports).

## What this tells us about the e030b "degradation story"

At epoch 1, the encoder hadn't yet specialized. Layer B information was still present — action accuracy at 75%, percent linearly decodable at high R². The encoder was roughly a lossless projection of the input features into 192 dims.

Over epochs 2–10, the pred_loss gradient pushed the encoder to minimize `||encoder(frame_{t+1}) - encoder(frame_t) - delta||²`. Finding a representation where adjacent frames are close in latent space is easier if you drop information about things that *suddenly differ* between adjacent frames — which is Layer B. Specifically: if you drop percent from the representation, you no longer have a 10-point MSE spike on the hit frame, because the representation of the hit frame is almost identical to the non-hit frame. Low pred_loss, at the cost of decoder information.

At epoch 10, the encoder has reached a local optimum where Layer B information is mostly suppressed and Layer A information is mostly preserved. The predictor can accurately forecast `encoder(frame_{t+1})` because the encoder output is almost the same as `encoder(frame_t)`, plus a small smooth delta. pred_loss → 0. But the linear probe, which is asking "can you recover the original game state from this representation," says the answer is increasingly no.

**Epoch 1 was Competitive because the encoder hadn't specialized yet.** Every epoch after was the encoder doing its job according to the loss we asked it to minimize. The loss we asked it to minimize is the problem.

## What this is NOT

**Not an argument that JEPA is wrong for Melee.** JEPA's architectural choice (predict in latent space) is fine. The encoder and predictor shapes are fine. The problem is specifically in the loss structure: naive MSE in latent space + isotropic-Gaussian regularization has no mechanism to preserve discrete-event information, and Melee has a lot of discrete-event information.

**Not an argument to abandon the paradigm.** LeWM's environments (PushT, Reacher, Cube) are largely Layer-A-only — they're continuous control tasks where the dynamics are smooth. Melee has Layer B baked into the rules, which is a structurally different problem than the one LeWM was designed for. That doesn't mean LeWM's approach is bad; it means we need to extend it with a Layer-B-aware loss or architecture.

**Not a hyperparameter issue.** Reducing SIGReg λ would let the encoder be less Gaussian, which might preserve more raw information, but it doesn't directly reward representing Layer B. Training for fewer epochs would catch the encoder before it specializes, but that's fragile and requires knowing the right early-stopping epoch. These are mitigations, not fixes.

**Not specific to JEPA.** Mamba2 has the same death→respawn problem. Mamba2 avoids some of the symptoms because its per-field supervised heads force some Layer B representation (the 400-class action head has to get transitions right), but the rare-event imbalance still crushes the signal. We see percent bumps and action state transitions predicted poorly in Mamba2 rollouts too.

## What we might try

Ordered by cost, untested, all speculative:

### Cheap diagnostics (~$5 each, tests a specific hypothesis)

1. **Measure the Layer B gap explicitly.** Condition rollout eval on "hit landed during context window" vs "no hit." Compare `percent_mae` on the two splits. My prediction: the hit-present split is 5-10× worse than the hit-absent split. Same for `stocks` transitions if they exist in the eval window. Gives us a real number to improve against. Applies to both JEPA and Mamba2.

2. **SIGReg λ sweep.** Currently λ=0.1. Try 0.01, 0.03, 0.3. Hypothesis: lower λ gives the encoder more freedom to encode discrete structure. If the λ=0.01 run holds Competitive R² through 10 epochs, the SIGReg-smoothness-prior was the dominant specializer. If it degrades anyway, MSE is the dominant cause. This is e030c in the current lineage.

3. **Early stopping with intermediate checkpoints.** Save `epoch_N.pt` every epoch instead of overwriting `best.pt`. Pick the best-probe-R² checkpoint regardless of val_loss. Run the viz on each. Tells us "how early would we stop, and does early stopping actually work for deployment." Doesn't fix the structural issue but gives us a usable checkpoint for now.

### Moderate cost (~$15-30)

4. **Augment pred_loss with a Layer B reconstruction term.** Add a decoder head that predicts `(x, y, percent, action_state)` from the latent and train it jointly with MSE. The decoder's CE loss on action_state specifically forces the encoder to preserve action-state information. This is essentially "LeJEPA + auxiliary supervised decoder" — not pure JEPA anymore, but if it works, it gives us the JEPA latent without the Layer B blind spot. Flag this as a divergence from LeWM per our rule.

5. **Event-weighted loss.** Multiply pred_loss by a per-frame weight that's higher on frames where Layer B changed (percent went up, action state transitioned, stocks decremented). Upweight the rare events so they dominate the gradient budget. Risk: overfitting to the transition signature. Reward: the encoder has to represent the transitions to get low loss. ~50 LOC in `jepa_trainer.py`.

6. **Longer history window.** `history_size=3` can't fit a full death animation. Bump to 30 or 60 frames (matching Mamba2's context_len) and see if the encoder starts representing the longer-range patterns. Memory cost scales linearly; wall clock too. Possibly also: add frameskip like LeWM does, so `history_size=10` with `frameskip=5` covers 500ms.

### Structural (~e030e+ or a new base build)

7. **Hybrid latent with explicit event dimension.** Extend the encoder output from `(D_latent,)` to `(D_latent, D_event)`, where `D_latent` is the smooth continuous representation and `D_event` is a discrete-event code. Train the continuous part with MSE + SIGReg as today; train the discrete part with CE loss against action-state transitions. Decodes game state from both. This is the closest thing to "making JEPA's architecture actually handle Layer B."

8. **Hierarchical world model: physics + state machine.** Two models trained in parallel: a JEPA-style continuous predictor that handles physics, and a discrete Markov chain predictor that handles action-state / stocks / percent transitions. At inference, the hierarchical model chooses which predictor to consult for each field. Structurally matches how Melee actually works (physics engine + game logic) but it's a lot more code.

9. **Give up on JEPA for the Layer B fields, use Mamba2 for those.** Train JEPA for position/velocity/shield and Mamba2 for percent/action/death. Combine at the decoder. This is "use the right tool for each job" but it means shipping two models, which undermines the point of JEPA as a single coherent representation.

## Eval metric proposal

We should add a "Layer B recovery rate" metric to the JEPA eval suite. Something like:

- **Percent bump recall**: out of N frames where real percent increased, what fraction of predicted frames show a corresponding increase? Currently near zero.
- **Action transition accuracy**: conditioned on frames where `action_state_t != action_state_{t-1}`, what's the probe's accuracy? Currently near random.
- **Stock transition detection**: out of N frames where real stocks decreased, what fraction of predicted frames show the corresponding latent jump?

These metrics would surface the Layer B gap as a number we can watch evolve during training. Without them, all we have is "R² on continuous fields" which ignores Layer B entirely.

## Connections to the onchain deployment vision

If we ship a world model that can't handle Layer B events, **the first death in a match breaks the game**. The model predicts "character is still dead" and the rollout continues in that state, while the actual game engine has respawned the character at center stage. Any interactive use of the model downstream diverges from ground truth the moment a Layer B event fires.

The difference between a model that handles Layer B and one that doesn't is **the difference between a physics simulator and a game engine**. We've been building physics simulators. To ship a game engine, we need the Layer B story to work.

This isn't a JEPA problem specifically — Mamba2 has the same gap. But JEPA makes it more visible, because the gap shows up directly in the probe metrics, and we can measure the representation quality without needing to run a full rollout. That's a diagnostic advantage.

## Status

- **Research note**: this file. Not a decision.
- **Documented failure mode**: "prediction-shortcut collapse" / "Layer B blind spot" is now a named thing we can reference in future run cards.
- **Diagnostic to add**: `--split-by-event` flag in eval_rollout.py (queued, not blocking anything).
- **Next experiment direction**: e030c, following Mattie's "move slowly" guidance — pick ONE lever to move and see what happens, rather than stacking multiple changes. Current leaning: SIGReg λ sweep OR event-weighted loss. Not both in the same run.
