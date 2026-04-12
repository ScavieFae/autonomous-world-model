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

**Not *only* a hyperparameter issue.** This is worth unpacking carefully — see the next section for the full "levers vs fixes" breakdown. The short version: λ and early-stopping epoch are real levers that bias the Layer A / Layer B tradeoff, but they're knobs on the tradeoff, not ways to eliminate it.

**Not specific to JEPA.** Mamba2 has the same death→respawn problem, and the same percent-bump and stock-transition representation gaps. Mamba2 avoids some of the symptoms because its per-field supervised heads force some Layer B representation on categorical fields (the 400-class action head has to get transitions right), but the rare-event imbalance still crushes the signal on continuous fields. See the "JEPA vs Mamba2: loss or backbone" section below for the full breakdown of why this is mostly a loss-structure problem, not an architecture problem, and what that implies for porting fixes between the two lineages.

## Levers vs. fixes: what λ can and cannot do

This deserves its own section because the distinction is load-bearing for how we design e030c and beyond, and because "is this a hyperparameter issue" is the question every reader of this note will ask first.

### Why λ *is* a real lever (not zero signal)

Lower SIGReg λ gives the encoder more freedom to be non-Gaussian. That genuinely matters for Layer B, and the mechanism is concrete:

Step-function game state — percent, stocks, action_state transitions — produces **multimodal distributions** in the encoder output when the input distribution has discrete bumps. A frame where percent just jumped from 40 to 50 looks different from a frame where percent stayed at 40 across all the input features that covary with percent (the hit flash animation, the knockback velocity, the hitlag counter). If the encoder preserves that information, the latent distribution for frames-near-hit will be bimodal or multimodal — some samples sit in the "no hit" mode, some in the "hit just happened" mode, and the mixture is structured.

SIGReg explicitly pulls the encoder distribution toward an isotropic unimodal Gaussian. At high λ, any multimodal structure gets penalized because it's "non-Gaussian" — even though the multimodality is the information we want. At low λ, the encoder has slack to produce multimodal distributions, which is literally the shape Layer B information takes in embedding space.

So λ **is** a tuning knob on Layer B preservation. The λ sweep is not zero-signal work. A λ=0.01 run that holds Competitive probe R² through 10 epochs would tell us the SIGReg smoothness prior was the binding constraint on Layer B, and that a much lower-λ regime is the operationally-correct setting for our domain.

### Why λ is a knob on a tradeoff, not a fix

**Even at λ=0 you still have three other gradients pushing the encoder toward Layer A:**

1. **MSE in latent space itself.** MSE-optimal for step-function targets is "predict the mean" because the penalty for getting a step wrong scales with the step size squared, while the penalty for a small constant bias is linear in the bias. A smooth encoder that never predicts the step dodges the big squared penalty at the cost of a constant small miss on every frame. Low MSE, zero Layer B. This gradient is present regardless of SIGReg — turning off the regularizer doesn't turn off the MSE loss term that's rewarding smoothness.

2. **The rare-event gradient budget.** In our data, a Layer B transition (percent bump, action_state switch, death) occurs on roughly 0.01% to 3% of frames depending on the event type. The model sees ~97% of gradient updates that reward smooth continuation, ~3% that reward representing a transition. Gradient descent is a local-optimum search weighted by frequency. Even with a perfect loss that directly rewarded transition fidelity, you'd still need the frequency balance to be non-pathological, and it isn't. The rare-event imbalance bites independently of whether SIGReg is active.

3. **Temporal straightness as an emergent property.** LeWM's paper celebrates rising temporal straightness as a sign of successful training. It is literally a measure of how smoothly latent trajectories flow over time. A perfectly straight trajectory cannot contain a discontinuity, which means it cannot represent a Layer B event that by definition *is* a discontinuity. We observe straightness rising monotonically in e030b from 0.51 to 0.78 — the encoder is optimizing for line-like trajectories, and the prior literature interprets this as health. The "emergence" is the encoder actively learning to suppress the structure we need. Turning λ to zero does not remove this gradient — straightness is an emergent property of MSE + history-conditioning in a latent space, not something SIGReg directly induces.

**So the equilibrium point** — where the encoder's Layer A / Layer B tradeoff settles — is a function of λ, MSE weight, context length, and the data distribution's rare-event frequency. λ is one axis of that equilibrium. Moving λ shifts the equilibrium point along one axis but doesn't collapse the tradeoff to zero. You can bias the encoder toward Layer B, you can't eliminate the pressure pulling it toward Layer A.

### What counts as a fix vs. a lever

The distinction we care about:

- **Lever**: A setting whose adjustment shifts the A/B equilibrium point. The structural pressure against Layer B is still there; you're just finding a point on the curve with more Layer B than the default. Leverage is finite and bounded.
- **Fix**: A change that adds a gradient pushing toward Layer B, or removes a gradient pushing against it. Changes the shape of the loss landscape rather than finding a better point inside it.

**Leave at (levers, not fixes):**

- **SIGReg λ**. Reduces one of the smoothness priors. Doesn't touch MSE or straightness pressure.
- **Early stopping**. Catches the encoder before specialization completes. Fragile — requires knowing the right epoch, and every dataset or architecture change resets the schedule.
- **Decreased number of training epochs**. Same mechanism as early stopping, equally fragile.
- **Smaller SIGReg projection count**. Weaker distributional estimator → weaker regularization pressure. Same axis as λ, different knob.

**These are real fixes** (each one adds or removes a structural gradient):

- **Layer B reconstruction auxiliary head** (item 4 in "What we might try"). Adds an explicit loss term that rewards preserving action_state / percent / stocks information. The encoder now has a gradient telling it to keep Layer B, not just "try not to be too Gaussian." This is the closest thing to a direct fix in the short term.
- **Event-weighted MSE** (item 5). Upweights the gradient budget on rare-event frames. Doesn't just bias the tradeoff — literally rebalances the frequency imbalance that starves Layer B of gradient signal. Fix for the second structural issue from the list above.
- **Longer context window** (item 6). A 50ms context can't hold a full death animation. The encoder can't represent what it can't see. Extending `history_size` is structural because it changes what Layer B events are *expressible* given the input, not just how they're weighted.
- **Hybrid latent with explicit event dimension** (item 7). Splits the encoder output into a continuous part trained with MSE+SIGReg and a discrete part trained with CE. Removes the "SIGReg applies to everything" gradient from the discrete part. Structural.
- **Hierarchical world model** (item 8). Two models, one for each layer. Removes the "one representation must serve both layers" constraint entirely. Most structural.

**Item 9** ("use Mamba2 for the Layer B fields") is a hybrid approach that sidesteps the question rather than answering it. Listed as a fix for completeness but it's really "stop trying to fix JEPA for Layer B."

### What this means for e030c and the λ sweep

The λ sweep is still worth running. But we should run it with a **clear expectation of what it measures**, not with a hope that it fixes the problem.

**What e030c (λ sweep) actually measures:**

- **The λ→probe-R² curve on our data.** At λ ∈ {0.01, 0.03, 0.1, 0.3}, what's the Layer B information retention after 10 epochs? This tells us the structural floor of MSE+SIGReg on our data. If the curve is steep (λ=0.01 preserves most Layer B, λ=0.3 destroys all of it), the regularizer is the dominant specializer and we can find an operating point. If the curve is flat (all λ values produce similar degradation), the regularizer was not the binding constraint and we need a structural fix.
- **Whether the current λ=0.1 is approximately optimal.** Even "yes this λ is good" is useful to know — it closes the axis and focuses future work on the non-λ fixes.
- **A data point we can cite in the Layer B discussion going forward.** "We swept λ and got X → Y range on probe R²" is a specific, defensible empirical result. "We didn't try λ sweep" is an open question that will keep coming up in every future review.

**What e030c (λ sweep) does not and cannot measure:**

- Whether JEPA can handle Layer B events in principle. (No lever setting eliminates the other three gradients.)
- Whether death→respawn is representable. (It isn't, under any λ, because the other gradients still forbid the required discontinuity and context is too short.)
- Whether the paradigm is viable for onchain deployment. (Competitive-at-epoch-1 with aggressive early stopping is already answering that question today.)

**So the right framing for e030c** is "measure the λ axis so we know where the floor is" — a data-gathering run on an axis where we already know the mechanism. Not "test if λ fixes the problem." This framing matters because it tells us *in advance* what the success criteria should be:

- **Expected outcome if the mechanism is as described**: λ=0.01 yields some improvement (10–30% better probe R² at epoch 10 than λ=0.1) but still shows monotonic degradation across training. The axis has signal but doesn't fix the structural issue.
- **Surprising outcome if λ is the dominant constraint**: λ=0.01 holds probe R² flat or increases it across epochs. The regularizer was the binding constraint and we have an operating point that makes the paradigm usable as-is.
- **Surprising outcome if MSE is the dominant constraint**: λ=0.01 shows essentially the same degradation curve as λ=0.1. SIGReg wasn't binding and we need to move to structural fixes (item 4 or 5) for the next experiment.

All three outcomes are informative. All three update our model of the mechanism. The λ sweep is worth running not because it might fix anything but because each outcome narrows the space of what the next fix should look like.

### Checkpoint-saving is orthogonal to this discussion

Worth naming separately: **the checkpoint-saving strategy bug is its own issue, independent of the Layer A/B analysis**. e030b saved best.pt by val_total_loss, which was monotonically decreasing, so the Competitive checkpoint at epoch 1 was overwritten by the Not-viable checkpoint at epoch 10. This is a separate bug from the structural Layer B issue — it's a checkpoint-selection bug that happened to reveal the structural issue loudly.

Fixing checkpoint saving (select by probe R², or save per-epoch checkpoints and pick post-hoc) is a ~15 LOC change and a real lever — it doesn't fix Layer B, but it **makes the paradigm operationally usable today**. A JEPA model at epoch 1 of e030b was Competitive-zone. The only reason we don't have a deployable JEPA checkpoint right now is that we overwrote it. Fixing checkpoint saving means every future JEPA run produces a usable artifact whose quality is bounded by the early-epoch peak rather than the late-epoch floor.

This is a lever (doesn't fix the underlying issue) but a very cheap lever that unblocks deployment while the real fixes proceed.

## JEPA vs Mamba2: is this a loss problem or a backbone problem?

The natural reading of the observations so far is "JEPA is worse at Layer B because it doesn't carry explicit state the way SSMs do." That reading is wrong — or at least it's mis-localized in a way that matters for how we plan work. The dominant driver is the loss structure, not the backbone choice. This section unpacks why, because the conclusion changes what work we sign up for when we come back to JEPA.

### The core claim

**The difference in observed Layer B behavior between our JEPA experiments (loud failure) and our Mamba2 experiments (quiet partial failure) is almost entirely explained by loss structure, not backbone choice.** Mamba2 has per-field supervised heads — specifically the 400-class CE head on `action_state` — that create gradient pressure for Layer B representation regardless of backbone. JEPA has MSE on next-frame latent with no supervised targets, and that loss structure has nothing rewarding Layer B preservation.

If this claim is right, the three Layer-A gradients identified in "Levers vs. fixes" (MSE rewarding smooth predictions, SIGReg rewarding unimodal Gaussian distributions, rare-event gradient imbalance) are **properties of losses**, not backbones. SSMs don't neutralize them. Transformers don't amplify them. You get the same behavior from either backbone with the same loss.

### Two thought experiments

**Thought experiment 1: Mamba2 with JEPA loss.** Take our current Mamba2 code, keep the SSM backbone, keep the selective scan, keep the recurrent hidden state — but replace the 16 supervised per-field heads with "MSE on next-frame encoder latent + SIGReg(Z), no supervised targets." Predict what happens.

My prediction: same Layer B blind spot JEPA shows. Same percent oscillating. Same monotonic probe degradation across training. Same "death→respawn never learned." Because nothing in the new loss is rewarding Layer B representation, and the SSM backbone has no built-in preference for representing discrete state machines unless something in the loss asks it to.

**Thought experiment 2: Transformer predictor with Mamba2-style supervised heads.** Take a stateless Transformer backbone (no recurrence, attention-only state reconstruction) and train it with the same per-field supervised heads Mamba2 uses — 400-class CE on action_state, MSE on continuous fields, cross-entropy on jumps, etc. Predict what happens.

My prediction: handles Layer B about as well as Mamba2 does. Because the 400-class action_state cross-entropy head literally cannot be minimized without representing action transitions, regardless of whether the backbone has an explicit recurrent state or reconstructs state implicitly from attention over context.

Both predictions are untested. The first is cheaper to test (swap Mamba2's trainer to produce a latent-only loss, run one epoch) and would be the cleanest single data point on the loss-dominates-backbone claim.

### The backbone hedge: where SSMs might actually matter

There's one place the architectural difference might genuinely matter, and it's worth being honest about. SSMs are designed to carry state across time through recurrence: `h_t = A h_{t-1} + B x_t`. The A matrix is learned but the mathematical shape of the recurrence makes carrying information across time the "natural" thing for the network to do. For a small-cardinality discrete state machine — say, 5 Melee meta-states (grounded, aerial, hitstun, dead, respawning) — you could roughly write down the A/B/C matrices by hand that implement the transitions. A Transformer has to reconstruct state from attention context, which is a harder learning problem for an explicit FSM even though it's not impossible.

Whether learned SSMs discover this in practice is an empirical question we haven't tested. I'd guess there's a real but modest architectural advantage for SSMs on state transitions — maybe 10-30% improvement on the fraction of transitions they get right, not 10×. Second-order compared to the loss-structure effect.

**Evidence for second-order, not first-order**: we've never seen our Mamba2 runs produce a clean death→respawn either, despite having the architecturally "right" kind of backbone for it. If the backbone were the dominant factor, we'd expect Mamba2 to at least sometimes handle the transition. It doesn't. That's a data point against "SSM backbone solves Layer B" and for "loss structure dominates."

### What we actually observe in Mamba2 (empirical, not systematic)

We don't have systematic Layer B measurement on Mamba2 yet — the event-conditioned rollout eval is still a follow-up queued in "What we might try" item 1. So the Mamba2 observations here are anecdotal and from rollout visualizations, not run metrics. Framed honestly:

- **Stock count is deducted at wrong times.** Mamba2 has been observed predicting stock decrements when no death occurred (false positive) or missing decrements when deaths did occur (false negative). The stock field is a small integer counter with transitions that happen at the end of death animations — functionally a Layer B field. The supervised head on stocks (via the dynamics loss) is not sufficient to produce reliable transition prediction, because the rare-event frequency imbalance still bites.
- **We have never observed a clean respawn in any Mamba2 run.** The teleport from death animation to respawn platform, with the associated stocks decrement and position reset to (0, 40) on FD, is not something the Mamba2 rollouts produce. When the seed context includes a death, Mamba2 predicts the character stays dead. When the seed doesn't include a death, Mamba2 never produces one. Same symptom as JEPA.
- **Percent bumps are represented noisily in Mamba2.** Unlike `action_state` (which has a supervised CE head), percent has only a continuous MSE loss, no classification head. The smooth-continuation failure mode applies identically to Mamba2 on this field. We haven't measured it systematically, but the anecdotal observation from visualizations is that Mamba2's percent prediction also drifts smoothly toward stable values rather than producing discrete +10 bumps.

**This is exactly what the loss-dominates-backbone reframe predicts.** Mamba2's supervised heads partially mask the Layer B gap on categorical fields (action_state, jumps) because those fields have direct CE supervision. The same heads don't help with discrete-event continuous fields (percent, stock count) because those fields use MSE loss just like JEPA's latent does. The gap is there in both architectures; it's louder in JEPA because **nothing** masks it. In Mamba2 some fields get partial coverage and others don't.

**Epistemic caveat**: none of this is systematic measurement. We have anecdotal rollout visualizations, not metrics on event-conditioned eval splits. The "we've seen stocks at wrong times" observation is a rare occurrence that stood out when someone was watching a replay, not a measured rate over N transitions. A first useful step is landing the event-conditioned rollout eval (item 1 in the mitigation list) and putting real numbers on the Mamba2 Layer B gap. Until then, the comparison between "how bad is Mamba2 at Layer B" and "how bad is JEPA at Layer B" is grounded in loss-structure reasoning, not empirical measurement.

### Why we came back to Mamba2 (the clean reason)

The right framing is **not** "JEPA is structurally worse at state changes, so we're going back to the better architecture." That framing imports a backbone-dominant reading that the analysis above argues against. The right framing is:

1. **Reducing simultaneous unknowns.** Pursuing Layer B fixes on JEPA means debugging JEPA's training dynamics, tuning JEPA's hyperparameters, understanding its structural failure, *and* developing Layer B fixes — all at once. Going back to Mamba2 narrows the research question to just "developing Layer B fixes" on a baseline where training dynamics are well-understood and hyperparameters are stable.
2. **Faster iteration cycles.** The e031a profile surfaced that Mamba2 training has ~2.2× headroom (Self-Forcing overhead + data loader bottleneck), and infrastructure fixes are landing now. Every Layer B experiment after the SF ablation is cheaper to run.
3. **Existing instrumentation and success criteria.** Mamba2 has working per-field heads, established run cards, a known RC baseline (4.798), and a stable set of metrics. JEPA's probe methodology just got reworked in e030b and needs more runs to calibrate before it's a reliable go/no-go signal.
4. **Layer B work is architecture-independent.** The fixes we want to try (event-weighted loss, longer context, auxiliary decoder heads, event-conditioned eval) are not JEPA-specific. They're properties of how you train a world model on a state-machine domain. Doing the research concentrated on the baseline with tighter feedback loops is correct regardless of which backbone we eventually ship.

The load-bearing reason is #1. Reducing the number of simultaneous open variables is the right project management move when each variable costs wall clock and the outcomes are uncertain.

### The practical implication: work ports

Here's the corollary that makes this decision better than it looks. **If the loss-dominates-backbone reading is right, the Layer B fixes developed on Mamba2 will port directly to JEPA when we come back.** Specifically:

| Fix | Architecture dependency | Port cost |
|---|---|---|
| Event-conditioned rollout eval | Pure metric code, reads per-horizon results dict | Zero — same eval script runs on both |
| Event-weighted MSE | Property of how per-frame loss is summed; doesn't care about backbone or per-field vs latent | Low — same weighting scheme applies |
| Longer context window | Both architectures need to see the full transition to represent it | Medium — JEPA needs a bigger history buffer, Mamba2 needs a longer context_len |
| Auxiliary Layer B reconstruction head | This is literally "give JEPA the supervised heads Mamba2 already has" | Low — port the head definition directly |
| Hybrid latent with event dimension | Architecture-agnostic — applies wherever you have a continuous latent | Medium |
| Hierarchical physics + FSM | Applies to any world model with a loss gap on state machines | High regardless |

So the Mamba2 loop is **not** parallel wasted work from the JEPA perspective. It's doing the Layer B research once, on the better-instrumented baseline, with results that transfer. The only thing we lose by not running this work on JEPA right now is per-fix confirmation on the JEPA-loss variant — and that's a follow-up run, not duplicated research.

**Concrete prediction for when we come back to JEPA**: the first experiment should be "JEPA with an auxiliary supervised decoder head on action_state + percent + stocks, trained jointly with MSE+SIGReg." This is item 4 in the mitigation list, and it's essentially "port Mamba2's supervised-heads story to JEPA." If the loss-dominates-backbone reading is right, this should preserve Layer B information in JEPA comparably to how Mamba2 currently does. If it doesn't, we'd update toward "the backbone does matter for this domain" and pursue item 7 (hybrid latent) or item 8 (hierarchical model).

### Open questions for the research diary

1. **What's Mamba2's actual Layer B performance on the action_state transition frames?** Event-conditioned rollout eval answers this directly. First data point we should collect once the eval split lands.
2. **Does Mamba2 handle percent transitions better than a counterfactual "Mamba2 with only latent MSE loss"?** Would require training the counterfactual, which is ~1-2 days of work. Not urgent but would be the cleanest single test of the loss-dominates-backbone claim.
3. **Is there an SSM-specific advantage on small discrete state machines that we haven't measured?** Would require either a literature search for SSM-on-FSM results or a targeted experiment. Low priority — we have bigger fish to fry on the loss-structure side first.
4. **Does respawn work in either architecture at any context length?** Both Mamba2 (context=30) and JEPA (history_size=3) fail on this. At context_len=60 or 120, does Mamba2 start producing clean death→respawn transitions? If yes, context length was the binding constraint and the loss-structure reading needs revisiting.

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
