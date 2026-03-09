# E015 — Physical Consistency: What Actually Fixes Autoregressive Drift?

**Date**: 2026-03-02
**Context**: E014 cascaded heads reduced damage drift (1→0 impossible frames in demos) but regressed val change_acc (88.0-88.5% vs E012's 91.1%) due to overfitting on 1,988 games. Autoregressive demos still show y-position drift, on_ground inconsistency, and occasional catastrophic divergence. The question: what's the right next intervention?

## The Initial Framing (Wrong)

The starting intuition was to add penalty terms during training for physically impossible outputs:

1. Percentage going negative
2. Stock loss without blast zone crossing
3. Stock loss without respawn sequence
4. Y-position drift while grounded / on_ground=true while airborne

This maps to **soft constraint penalties** from the physics-informed neural networks (PINNs) literature — add `λ * violation_penalty` to the loss function.

## Why Soft Constraints Are the Wrong Tool Here

The impossible states we observe occur **during autoregressive rollout**, not during teacher-forced training. During training, the model sees ground truth context and almost never predicts negative percent or grounded-while-airborne. The penalty terms would fire infrequently and produce minimal gradient signal.

This is a fundamental mismatch: we'd be adding loss terms that penalize a failure mode the model barely exhibits in the regime where it's being trained. The PINNs literature itself flags this — soft constraint performance depends critically on the penalty weight, and insufficient violation frequency makes the signal noisy and hard to tune ([Raissi et al., 2019](https://epubs.siam.org/doi/10.1137/20M1318043)).

The real cause of the demo artifacts is **exposure bias**: the model trains on perfect ground-truth context (teacher forcing) but at inference consumes its own imperfect outputs. Small prediction errors compound because the model has never learned to handle imperfect inputs.

## Three Approaches, Grounded in Practice

### 1. Scheduled Sampling — Train on Own Predictions

**What it is**: During training, with some probability ε, replace ground-truth context with the model's own predictions. Anneal ε from 0 (pure teacher forcing) to some target (e.g., 0.3) over training. The model learns to predict correctly from imperfect inputs because it encounters them during training.

**Literature**: [Bengio et al., NeurIPS 2015](https://arxiv.org/abs/1506.03099). Standard technique for exposure bias in autoregressive models. Also: [Dynamic Scheduled Sampling with Imitation Learning](https://openreview.net/pdf?id=UmHG2bD7X3w) for more sophisticated scheduling.

**Why it fits**: Directly addresses the root cause. The model's impossible states emerge from compounding errors on its own outputs — scheduled sampling is exactly the intervention that makes it robust to that.

**Our data**: E012b tested "scheduled sampling" at rate=0.3 and lost 2.8pp change_acc (88.3% vs 91.1%). But what we implemented (`_corrupt_context` in trainer.py) is actually **random noise injection** — Gaussian noise at scale=0.1 added to the last 3 context frames' position/velocity values. This is generic noise robustness, not true scheduled sampling.

**The distinction matters**: True SS feeds the model's *own predictions* back as context. The corruption is structured — it has the model's actual biases (upward y-drift, damage creep, on_ground stickiness). Random noise at scale=0.1 doesn't simulate those patterns. The model learns "handle jitter" but not "handle my characteristic drift."

True SS requires an extra forward pass per corrupted sample (generate prediction, inject as context), making it ~2× more expensive per batch. But it's the intervention the literature actually describes. E015 should implement proper self-prediction SS, not just noise augmentation. The noise approach can stay as a complementary regularizer.

Additionally, E012b only completed 2 of its 3-epoch anneal ramp. Needs 3-4 epochs and possibly gentler annealing.

**Interaction with E014 cascade**: Scheduled sampling + cascaded heads should be strictly better than either alone. During SS, the model sees its own imperfect outputs as context. The cascade ensures that when it predicts an action, the physics heads condition on that action — so even if the action is wrong, the physics are *consistent with the predicted action*. Without cascade, SS helps the model handle noise but doesn't enforce cross-head consistency.

### 2. Post-hoc Clamping at Inference — Pragmatic Fix

**What it is**: During autoregressive rollout, apply hard constraints: `percent = max(0, percent)`, snap y to stage height when `on_ground=true`, enforce blast zone boundaries.

**Literature**: The [Model as a Game](https://arxiv.org/html/2503.21172v1) paper (2025) takes this further — they offload numerical and spatial consistency to separate modules (LogicNet for event prediction, external numerical record for scores, explicit map for spatial consistency) rather than expecting the generative model to handle it. Their insight: "decompose score at the character level" and integrate as external state.

**Why it fits**: Zero training cost. Fixes demos immediately. For our use case (autoregressive visualization, policy training environment), clamped outputs are strictly better than unclamped. ~10 lines in the rollout code.

**Limitation**: Doesn't teach the model anything. The clamped outputs don't flow back into training. But for the "two agents playing the world model" milestone, this gets us there.

### 3. Soft Constraint Penalties — Right Idea, Wrong Time

**What it is**: Add auxiliary loss terms for physical violations during training. E.g., reconstruct absolute percent from context + predicted delta, penalize if negative.

**Literature**: PINNs ([Raissi et al.](https://en.wikipedia.org/wiki/Physics-informed_neural_networks)) — soft constraints as loss penalties. Also [hard constraint approaches](https://epubs.siam.org/doi/10.1137/21M1397908) via differentiable projection, but these require knowing the constraint manifold geometry.

**When it would help**: *After* scheduled sampling is in place. Once the model trains on its own (sometimes wrong) outputs, constraint violations will actually occur during training, providing real gradient signal. At that point, auxiliary penalties for impossible states reinforce the physical rules the model struggles with.

**Not useful now**: With pure teacher forcing, the model rarely produces impossible states during training. The penalty terms would be near-zero almost everywhere.

## The Dreamer Comparison

DreamerV2/V3 handle consistency differently because they have a different architecture: they learn in a latent space, not in observation space. Their KL balancing between the encoder (what the world looks like) and the dynamics model (what the model predicts) keeps the two consistent. The reconstruction decoder implicitly enforces physical plausibility — latent states that produce implausible observations get weak gradients.

We don't have a latent space. Our model predicts directly in observation space (position deltas, action IDs, binary flags). The Dreamer approach isn't transferable, but the principle is: consistency comes from **structural coupling** between components, not from penalty terms. That's what cascaded heads (E014) does for us — it couples action prediction to physics prediction.

## Recommendation

**E015 should be scheduled sampling**, not impossible state penalties.

The implementation priority:
1. **Post-hoc clamping** in rollout code (immediate, ~10 lines)
2. **Scheduled sampling** as E015 proper (training change, addresses root cause)
3. **SS + cascade** combination run (E015 + E014 on larger dataset)
4. **Soft constraints as auxiliary losses** (only after SS is working, when violations occur during training)

### E015 Design Sketch

Same dataset as E012/E013/E014 (1,988 FD top-5 games) for apples-to-apples, then scale up.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| SS rate | 0.2 (anneal from 0 over epoch 1) | Gentler than E012b's 0.3 |
| Corruption | Replace 1 of K context frames with model's own prediction | Minimal disruption |
| Epochs | 3 | E012b's 2-epoch run didn't complete the anneal |
| Architecture | E012 baseline (no cascade) | Isolate SS effect first |
| Hyperparams | bs=4096, lr=0.0005 | Match E012 |

Follow-up: E015b = SS + cascade (E014 architecture + SS training).

### Post-hoc Clamping Sketch (for rollout code)

```python
# In autoregressive rollout, after model prediction:
# 1. Percent can't go negative
abs_percent = last_percent + pred_percent_delta
if abs_percent < 0:
    pred_percent_delta = -last_percent  # clamp to zero

# 2. Grounded actions lock y to stage height
GROUNDED_ACTIONS = {14, 15, 16, ...}  # STANDING, WALK_SLOW, etc.
if predicted_action in GROUNDED_ACTIONS:
    pred_y_delta = stage_ground_y - last_y  # snap to ground

# 3. on_ground consistency
if abs_y > stage_ground_y + threshold:
    on_ground = False
```

## References

- [Scheduled Sampling for Sequence Prediction (Bengio et al., NeurIPS 2015)](https://arxiv.org/abs/1506.03099)
- [Dynamic Scheduled Sampling with Imitation Learning (ICLR 2020)](https://openreview.net/pdf?id=UmHG2bD7X3w)
- [Model as a Game: Numerical and Spatial Consistency (2025)](https://arxiv.org/html/2503.21172v1)
- [Physics-Informed Neural Networks (Wikipedia)](https://en.wikipedia.org/wiki/Physics-informed_neural_networks)
- [PINNs with Hard Constraints (Lu et al., SIAM 2021)](https://epubs.siam.org/doi/10.1137/21M1397908)
- [Gradient Flow Pathologies in PINNs (Raissi et al., SIAM 2019)](https://epubs.siam.org/doi/10.1137/20M1318043)
- [DreamerV3: Mastering Diverse Domains (Hafner et al., Nature 2025)](https://www.nature.com/articles/s41586-025-08744-2)
- [Exposure Bias in Machine Learning](https://www.emergentmind.com/topics/exposure-bias)
