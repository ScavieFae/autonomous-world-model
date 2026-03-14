# Research Diary

Sequential record of experiments, findings, and decisions. Newest entries at top.
Run cards live in `docs/run-cards/`. Experiment configs in `experiments/`. This diary captures the *thinking* — what we tried, what we learned, what surprised us.

**Rule: append-only.** Add new entries at the top. Never overwrite or rewrite existing entries — they're the record of what we thought *at the time*. If a conclusion turns out wrong, add a new entry that corrects it and links back.

---

## 2026-03-14: Rollout Coherence Eval Built (e018b)

### What we built

The rollout coherence eval — `scripts/eval_rollout.py`. This is the quantitative AR metric we've been missing since E016 proved that TF metrics don't predict AR quality.

### Design decisions

- **Batched evaluation**: 20 batched forward passes of N=300, not 6000 sequential passes. Should run in 30-60s.
- **Game-unit metrics**: pos_mae, vel_mae, percent_mae all denormalized to game units. Comparing in normalized space would hide differences (xy_scale=0.05 vs percent_scale=0.01).
- **Deterministic sampling**: Fixed seed (default 42) + sorted indices. Same checkpoint + same data = same score. Always.
- **Config-driven reconstruction**: Factored the AR step out of `rollout.py` into `scripts/ar_utils.py`. Uses `EncodingConfig` properties for all index math — no more hardcoded `13:16` / `16:29` / `FPP=29`. This matters because E017a-style configs have `binary_dim=43` and `float_per_player=69`, not 3 and 29.

### The reconstruction factoring

`ar_utils.reconstruct_frame()` is now the single source of truth for: apply continuous deltas, threshold binaries, copy controller input, argmax categoricals. Three callers will share it:
1. `rollout.py` (demo generation) — uses it now
2. `eval_rollout.py` (evaluation) — uses it now
3. Self-Forcing training (e018a) — will need a differentiable variant, but the logic stays aligned

### What's still missing

- **Baseline numbers**: Need to run the eval against E012 and E017a checkpoints. These become the "prior best" that autoresearch experiments compare against.
- **`absolute_y` handling**: The flag is in E017a's YAML but not implemented in this repo's Python code (was likely in nojohns before migration). The eval uses the same reconstruction as rollout.py, so they're consistent — but both would need updating if we test an absolute_y model.

### Status

e018b: proposed → running. Eval script is built. Waiting on baseline numbers to close the card.

---

## 2026-03-02: E017c/d Results — Absolute Targets Prevent Drift but Cause Oscillation and Head Decoupling

### What we tried

Two more absolute-target experiments, extending E017a:

- **E017c (absolute x+y)**: Both spatial axes absolute, velocities still as deltas
- **E017d (absolute x+y+vel)**: Positions AND velocities all absolute — most aggressive delta elimination

### TF metrics (as expected, don't tell the story)

| Metric | E012 | E017a (abs y) | E017c (abs xy) | E017d (abs xy+vel) |
|--------|------|---------------|----------------|---------------------|
| change_acc | 91.1% | 90.7% | 90.4% | 90.5% |
| val_loss | 0.527 | 0.507 | 0.505 | 0.507 |

All look fine. Change_acc barely moved. val_loss slightly improved. The usual trap.

### AR demos — the real test (again)

Visually: both E017c and E017d were **worse** than E017a. Characters zoom off to either side immediately, eventually recover and yoyo back to center, then fly off again. The "recovery to center" isn't the model understanding physics — it's the absolute target acting as a spring, pulling predictions back toward zero.

Late-game numbers (last 200 frames) initially looked deceptively good:

| Run | Mean\|y\|p0 | Mean\|x\|p0 | On-ground% p0 |
|-----|------------|------------|---------------|
| E012 | 509.1 | 2635.7 | 12% |
| E017a (abs y) | 60.8 | 138.1 | 100% |
| E017c (abs xy) | 16.5 | 195.7 | 92% |
| E017d (abs xy+vel) | 17.1 | 6.7 | 100% |

E017d's mean |x| of 6.7 looked like a 400x improvement. But 100% on-ground was suspicious — Mattie flagged it immediately. Digging in:

### Head decoupling — the real finding

E017d: **165 out of 200 frames** had on_ground=true while the action was DAMAGE_FLY_LOW. That's physically impossible. The binary head collapsed to "always grounded" because absolute y≈0 made on_ground trivially predictable, while the action head independently predicted airborne actions. The heads stopped talking to each other.

| Run | Contradictions (on_ground + airborne action) | Unique actions | Top action |
|-----|----------------------------------------------|---------------|------------|
| E012 | 13 | 6 | DEAD_DOWN (82%) |
| E017a | 117 | 4 | JUMPING_ARIAL_BACKWARD (38%) |
| E017c | 121 | 13 | DEAD_RIGHT (33%) |
| E017d | 165 | 9 | DAMAGE_FLY_LOW (82%) |

More absolute targets → more contradictions → worse head coherence.

### The scorecard

| Variant | Drift? | Oscillation? | Heads agree? | Visual verdict |
|---------|--------|-------------|--------------|----------------|
| E012 (all deltas) | Fatal | No | Mostly | Drifts off, dies |
| E017a (abs y only) | Reduced | Jittery | Some contradictions | Best visual — ground-level, jittery |
| E017c (abs xy) | No | Flies off sides | 121 contradictions | Worse than E017a |
| E017d (abs xy+vel) | No | Ping-pong yoyo | 165 contradictions | Spatial mean deceiving, visually bad |

### Why absolute x was worse than absolute y

Y has low legitimate variance for grounded characters (≈0 on FD). Absolute y is a natural target — the answer is usually "zero." X has high legitimate variance — characters move all over the stage (-85 to +85). Making x absolute forces the model to predict the full position each frame, which it does poorly in AR, causing overcorrection and oscillation.

### Why absolute velocities made oscillation worse

Delta velocities act as a natural low-pass filter. Predicting "change by 0.3" limits how fast things shift frame-to-frame. Absolute velocities let the model jump to any velocity value each frame, removing the damping. Mattie's original intuition (absolute vel prevents "+1 +1 +1" runaway) was reasonable but the actual failure mode is different: instead of runaway accumulation, we get high-frequency oscillation.

### The conclusion: "change what we predict" is exhausted

We've now tried four variations of target representation:
- All deltas (E012): drift
- Absolute y (E017a): best so far, jittery
- Absolute xy (E017c): oscillates
- Absolute xy+vel (E017d): oscillates worse, heads decouple

The lever isn't *what* we predict. It's *how we train*. The model needs to see its own AR errors during training to learn to handle them. This points firmly at hybrid AR/TF training — run short AR rollouts inside the training loop so the model learns to recover from its own mistakes.

wandb: [E017c cc9yk53x](https://wandb.ai/shinewave/melee-worldmodel/runs/cc9yk53x), [E017d owwxoxpd](https://wandb.ai/shinewave/melee-worldmodel/runs/owwxoxpd)

---

## 2026-03-02: Cascade Architecture Analysis, Shorthop Diagnostic, and the Hybrid AR/TF Idea

### Why cascade fixed damage drift but not y-drift

Read through `_apply_heads()` in mamba2.py to answer this. The answer: **it's not architectural asymmetry — it's temporal.**

All physics heads (continuous, binary, velocity, dynamics) get the same conditioning: `h_joint = cat([h, p0_emb, p1_emb])`. The cascade embedding is a 16-dim vector for a single action ID. No head is treated differently. So why does the same mechanism fix damage drift and not y-drift?

- **Damage is a single-frame causal link.** Attack action → percent increases. The cascade embedding tells the continuous head "action is DOWN_B_GROUND" and the head learns "that means percent goes up by ~14." One action, one frame, done.
- **Y-trajectory is multi-frame.** KNEE_BEND (frames 1-4) → JUMPING_FORWARD (ballistic arc, frames 5-30) → aerial → LANDING. The cascade embedding says "action is KNEE_BEND" but doesn't encode "we're 3 frames into a 4-frame jumpsquat, y should be at 0.0 and about to launch." The trajectory tracking has to come from the SSM hidden state `h` — which works in TF (clean context) but corrupts in AR (each frame's y-error bleeds into the next hidden state).

The embedding knows *what* is happening. Damage only needs "what." Y needs "where in the trajectory," which only the hidden state can provide.

### Shorthops as a diagnostic

Mattie's insight: top players shorthop constantly. A Marth-Fox match on FD probably has 50-100+ shorthops in a 4-minute game. If the model can't properly sequence KNEE_BEND → JUMPING_FORWARD → aerial → LANDING with correct y-trajectory and on_ground transitions, every single shorthop accumulates error and the character floats.

**Proposed diagnostic:**
1. Extract shorthop sequences from existing demo data (find KNEE_BEND → airborne transitions)
2. Run short AR windows (10-30 frames) over just those sequences
3. Compare predicted y-trajectory vs ground truth — where does the error accumulate? The launch? The apex? The landing?
4. This would tell us whether the model understands shorthop *physics* but loses it in AR context, or whether it never properly learned the arc shape

If the model nails shorthops in AR, the y-drift problem is essentially solved. If it doesn't, we know exactly which frame of the sequence breaks down.

### Hybrid AR/TF training — a different idea from SS

Mattie proposed: run N frames in AR, snap back to teacher-forced ground truth, then do AR for the next N. This is fundamentally different from scheduled sampling:

- **SS corrupts individual frames in context.** "Frame t-3 is noisy, but t-2, t-1 are clean." The model never sees what happens when *multiple consecutive frames* are its own predictions.
- **Hybrid AR/TF runs actual multi-frame AR sequences** with periodic reality checks. It explicitly trains the model to recover from AR error accumulation — the exact failure mode we're seeing.

This is closer to what RL people call "truncated rollouts." Implementation sketch:
- During training, every M batches, run a 10-frame AR rollout from the current position in the context window
- Compute loss on each AR frame against the ground truth target
- Snap back to TF for the next window
- AR-frame losses get weighted (maybe ramping up over training)

The key insight: SS teaches the model to handle single corrupted frames. Hybrid AR/TF teaches it to handle the *temporal compounding* of its own errors. That's exactly the shorthop problem — one frame of y-error is fine, but 10 consecutive frames of y-error is a character floating off-stage.

This is hefty to implement. The current training loop processes one (context, target) pair at a time. Hybrid AR/TF requires running a multi-step autoregressive loop *inside the training loop*, accumulating gradients through the unrolled sequence. Essentially BPTT through the AR rollout. It's doable — the SS reconstruction machinery already exists — but it's a significant lift compared to a config flag. Filing this for after E017c/d results.

### Where else we use deltas — and where absolute makes sense

Audited all continuous targets:

| Target | Currently | Absolute candidate? | Reasoning |
|--------|-----------|---------------------|-----------|
| percent (idx 0) | Delta | No | Only changes on hit (~95% of frames it's 0.0). Absolute would predict "35.0" every frame instead of "0.0" — harder. |
| x (idx 1) | Delta | **Yes** | Same compounding problem as y. On FD, x is bounded ~(-85, 85). Absolute x means the model predicts "position 12.3" instead of accumulating "move right by 0.4, 0.3, ..." |
| y (idx 2) | Delta (absolute in E017a) | **Already done** | E017a proved absolute y structurally helps (jitter vs drift). |
| shield (idx 3) | Delta | No | Gradual decay/regen, mostly unchanged between frames. Delta works well. |
| velocities (5 per player) | Delta of velocity = acceleration | **Maybe** | Velocity resets on landing, hit, action change. The values are instantaneous. Predicting velocity directly is more natural than predicting change-in-velocity. |

### E017c and E017d — expanding absolute targets

Two new experiments to test:

- **E017c: absolute x+y positions** — Both spatial axes absolute, prevents compounding on both. Same config as E017a but with `absolute_positions: true` (supersedes `absolute_y`).
- **E017d: absolute x+y+velocities** — Positions absolute + velocities absolute. Predicts the full physics state directly instead of accumulating deltas. Most aggressive change — if velocity prediction regresses, we know the model needs the delta structure for velocities.

Both use E012's 1.9K encoded data, ~16 min/epoch, ~$2.60 each.

---

## 2026-03-02: E017 Results — Absolute Y and Physics Loss, Neither Solves AR Drift

### What we tried

Two experiments on E016's config using E012's 1.9K data for fast iteration:

- **E017a (absolute y)**: Replace Δy with absolute y prediction. The model directly predicts y-position each frame instead of accumulating deltas. For grounded characters the answer is just 0.
- **E017b (physics loss)**: Auxiliary loss penalizing |predicted_y| when ground truth says on_ground=true. Weight 0.5.

### TF metrics

| Metric | E012 (baseline) | E017a (absolute y) | E017b (physics loss) |
|--------|----------------|--------------------|--------------------|
| val_change_acc | 91.1% | 88.2% | 87.9% |
| val_pos_mae | 0.706 | 0.979 | 0.719 |
| val_loss | 0.527 | 0.507 | 0.521 |

E017a's pos_mae regression is a measurement artifact — absolute y has a wider target distribution than Δy, inflating MAE. The val_loss improvement (0.507) suggests it's actually learning well. E017b was essentially identical to E012 across the board.

### AR demos — the real test

| Metric | E012 | E017a | E017b |
|--------|------|-------|-------|
| Mean |y| grounded | 29.1 | 31.4 | 22.2 |
| Grounded |y|>5 violations | 61.5% | 54.3% | 84.3% |
| y range | [-148, 34] | [-182, 34] | [-17, 68] |
| p0 top-1 action % | 18.0% (FALLING) | 24.8% (NAIR) | 51.2% (DEAD_FALL) |

Numbers don't tell the full story. Visually, E017a was noticeably better — Mattie's assessment: "held it together a bit better, ground-levelish but jittery." The jitter makes physical sense: each frame independently predicts y without temporal smoothing, producing noise instead of smooth drift. E017b collapsed p0 into DEAD_FALL 51% of the time.

### Why E017b failed — predicted by our own analysis

The March 2 diary entry on E015 said: "Soft constraint penalties would become useful *after* SS is in place, when the model actually produces violations during training and the penalty terms have gradient signal."

E017b proved this exactly. Physics loss was 0.0034 out of 0.378 total — the model barely violates physics in teacher-forced mode. The penalty only matters during AR rollout, which training never sees. We built a fire alarm for a room that's never on fire during training.

### Why E017a is interesting despite the numbers

Absolute y changes the *shape* of the error. With Δy, errors are biased (small consistent drift that compounds → character falls through the floor over 200 frames). With absolute y, errors are noisy but roughly unbiased (centered near 0, jittery but ground-level). This is a real structural improvement — drift vs jitter are different failure modes, and jitter is more fixable (harness, smoothing, more data).

### The synthesis — what we've learned across E012-E017

Every fix that targets teacher-forced behavior doesn't help autoregressive quality. The model understands Melee physics when given perfect context (91-94% change_acc). The problem is entirely what happens when it consumes its own outputs.

| Fix | Targets | TF result | AR result |
|-----|---------|-----------|-----------|
| E014 cascade | Independent heads | Damage drift fixed | Y-drift unchanged, overfit |
| E015 true SS | Exposure bias | -2.8pp change_acc | Unclear AR benefit |
| E016 cascade+SS+4x | Everything | Best TF ever (94.3%) | Worse AR than E012 |
| E017a absolute y | Compounding Δy | pos_mae inflated | Visually better, jittery |
| E017b physics loss | Ground consistency | No signal (0.0034) | p0 collapsed to DEAD_FALL |

**TF metrics don't predict AR quality.** This is the single most important finding. E016 had 38% better val_loss than E012 and worse AR demos.

### What hasn't been fully explored

1. **E017a + harness**: Absolute y handles drift direction, harness clamps jitter. Zero retraining.
2. **Physics loss during SS**: The penalty has no signal in TF. But during true SS, the model IS seeing its own outputs — physics violations DO happen. `absolute_y + physics + ss_true` would give the penalty actual gradient signal.
3. **Hybrid physics integrator**: Predict velocity, derive position by integration, check against stage geometry. Position consistent by construction. Biggest change, most principled.
4. **Shorthop sequences as a diagnostic**: Top players shorthop constantly. If the model can properly sequence KNEE_BEND → JUMPING_FORWARD → aerial → LANDING with correct y-trajectory and on_ground transitions, that's a strong signal it understands the physics. Currently unclear if shorthop sequences specifically break down in AR.
5. **Hybrid AR/TF training**: Run N frames AR, snap back to teacher-forced ground truth, repeat. Exposes the model to its own errors (like SS) but with periodic "reality checks" that prevent total divergence. Different from SS — SS corrupts single frames, this corrupts multi-frame sequences.

wandb: [E017a xfarwruv](https://wandb.ai/shinewave/melee-worldmodel/runs/xfarwruv), [E017b 77vmem5u](https://wandb.ai/shinewave/melee-worldmodel/runs/77vmem5u)

---

## 2026-03-02: E013 Null Result, E015 True SS Launched

### E013 hitbox data — clean null, decision made

E013 came back: 91.15% change_acc vs E012's 91.07%. Every metric within noise. The 10 extra hitbox features (is_active, damage, angle, knockback growth, size — looked up per frame from a 33K-entry table keyed by character/action/state_age) added nothing the model didn't already know.

The one mildly interesting number: val_loss dropped 0.010 (0.527→0.517) while everything else stayed flat. The model learned to predict the hitbox columns themselves (they're in the continuous target), but that knowledge doesn't propagate to any metric we care about.

**Decision: drop hitbox_data permanently.** Not worth the encoding complexity, the pre-encode cost (hitbox table lookup per frame), or the 10 extra float columns. The model's implicit inference from (action, state_age) is sufficient. Crucially, this means we don't need to extrapolate the hitbox table across the full dataset — saves a chunk of data pipeline work.

Three recent nulls (E011, E013, E012b), but each failed for its own reason: E011 had insufficient data for rare characters, E013 offered redundant information, E012b used the wrong corruption method. Not a single pattern — three different lessons. Meanwhile E014 (cascaded heads) *did* move things structurally (fixed damage drift in demos), it just regressed on val due to overfitting at 1,988 games. The model isn't unreachable, we just need to pick the right lever.

### E015 true scheduled sampling — launched

Implemented and launched E015: true scheduled sampling. Instead of Gaussian noise (E012b's null result), this feeds the model's own predictions back as context during training. The corruption is structured — it matches the model's actual error patterns (y-drift, damage creep, action stickiness), not random jitter.

Key implementation: multi_position=true gives us predictions at every context position for free. Position K-2's prediction is "what the model thinks frame K-1 should be." We reconstruct that frame (deltas + absolutes + argmax + rules-based state_age), replace the last context frame, and train on the corrupted context. Two forward passes per corrupted sample — first (no_grad) to get predictions, second (with gradient) to train.

Config: rate=0.3, anneal over 2 epochs, 4 epochs total (2× E012 to give SS time to ramp and the model time to recover). Same data, same model, same everything else.

wandb: [1m4fr40y](https://wandb.ai/shinewave/melee-worldmodel/runs/1m4fr40y)

---

## 2026-03-02: E014 Results, E013 Launch, and the E015 Reframe

### E014 cascaded heads — promising structure, not enough data

Implemented and launched E014: action heads predict first (tier 1), their embeddings condition physics/binary/categorical heads (tier 2). Two runs — one with the plan's hyperparams (bs=1024, lr=0.001), one matching E012 (bs=4096, lr=0.0005).

| Run | val change_acc | train change_acc | pos_mae | val_loss |
|-----|---------------|-----------------|---------|----------|
| E014 bs1k | 88.5% | 95.3% | 0.61 | 0.58 |
| E014 bs4k | 88.0% | 93.9% | 0.59 | 0.56 |
| E012 baseline | 91.1% | — | 0.706 | 0.527 |

Both regressed on val change_acc. The train/val gap (95.3% vs 88.5%) signals overfitting — the cascade embedding gives the model extra capacity to memorize action transitions on 1,988 games. The architecture is sound, it just needs more data.

**Demo analysis**: Cascade fixed damage drift — 0 frames of "damage during STANDING" vs E012's 1. But E014 bs1k had catastrophic P1 divergence (y=-116K) and E014 bs4k drifted more than E012 overall. on_ground inconsistency got slightly worse. E012 still looks best visually.

### E013 hitbox data — launched

Pre-encoded and launched E013 (hitbox features as input). Pure A/B vs E012: same model, same hyperparams, same 1,988 FD top-5 games. Only difference: 5 continuous features per player per frame (is_active, damage/30, angle/361, kbg/200, size/10) looked up from a pre-computed hitbox table (33,739 entries).

Verification clean: 148 float columns (E012's 138 + 10 hitbox), 1,988 games, 15.7M train examples. Running on H100, ~1h estimated.

wandb: [5yw0k7vt](https://wandb.ai/shinewave/melee-worldmodel/runs/5yw0k7vt)

### E015 — the reframe: scheduled sampling, not impossible state penalties

Started with Mattie's intuition: penalize physically impossible outputs (negative percent, stock loss without blast zone, y-drift while grounded). Researched the field and concluded this is the wrong tool for the problem.

**The core issue**: impossible states happen during autoregressive rollout, not during teacher-forced training. With teacher forcing, the model sees ground truth context and rarely produces violations. Penalty terms would fire infrequently — minimal gradient signal. This is a well-documented limitation of soft constraints in PINNs: performance depends critically on violation frequency and penalty weight tuning.

**The real cause**: exposure bias. The model trains on perfect inputs but at inference consumes its own imperfect outputs. Errors compound because it's never seen imperfect context.

**Three approaches, ranked**:

1. **Scheduled sampling** (addresses root cause) — train on own predictions some % of the time. Important distinction: what we call "scheduled sampling" in `_corrupt_context` is actually **random noise injection** (Gaussian, scale=0.1). True SS feeds the model's *own predictions* back as context — structured corruption matching its actual failure modes, not random jitter. E012b's -2.8pp may partly reflect the weakness of noise augmentation vs true SS. E015 should implement proper self-prediction SS.

2. **Post-hoc clamping** (pragmatic, no training cost) — clamp outputs during rollout. The Model-as-a-Game paper (2025) does this at scale, offloading consistency to external modules. ~10 lines in rollout code. Good interim fix for demos.

3. **Soft constraint penalties** (right idea, wrong time) — would become useful *after* SS is in place, when the model actually produces violations during training and the penalty terms have gradient signal.

**Key insight**: SS + cascade (E015 + E014) should be strictly better than either alone. SS makes the model robust to imperfect inputs; cascade ensures cross-head consistency even when the predicted action is wrong. Neither alone solves both problems.

Full analysis: `research/notes/E015-consistency-approaches.md`

---

## 2026-03-01: The Independent Heads Problem — Position, Action, and Physics Don't Agree

### The observation

E012 hit 91.1% val change_acc and 94.2% teacher-forced batch eval — honest numbers, no focal offset inflation. But autoregressive demos revealed three manifestations of the same structural problem:

1. **Damage drift**: Percent increases without any attack action in context. The continuous head predicts damage going up because it learned "damage tends to increase over time" without conditioning on whether an attack actually connected. First noticed when the policy model (in autonomous-world-model) never pressed attack buttons but opponents still took damage.

2. **Y-position drift while grounded**: Model predicts STANDING (action 14) for 20+ consecutive frames while the position head drifts y from -1.7 to -3.8. In real Melee, STANDING locks y to the ground plane. The position head doesn't know the action head said STANDING — they both read from `h` independently.

3. **on_ground stays true while airborne**: Every frame shows `on_ground=true` even during jumps at y=15+. The binary head learned "on_ground is true ~70% of the time" and doesn't flip when the action head predicts JUMPING_FORWARD. Similarly, action sometimes says DASHING while y climbs upward — a physical impossibility.

### Root cause

The model has a shared Mamba-2 trunk producing hidden state `h`, with independent output heads for position (continuous), action (categorical), on_ground (binary), percent (continuous), etc. Each head reads from `h` and minimizes its own loss.

During teacher-forced training, the heads never need to coordinate because they each see perfect ground-truth context. Consistency comes "for free" from the data being physically consistent. The shared trunk learns statistical correlations, not hard physical constraints.

In autoregressive mode, small inconsistencies compound. The position head drifts y by 0.3/frame. The action head doesn't react to that drift because it also read from `h`, not from the drifted output. After 20 frames: character "standing" at y=15 with on_ground=true.

**The game engine doesn't work this way.** In Melee, action determines physics: STANDING → y is locked to ground, period. Action comes first, physics follow. Our model predicts all outputs in parallel from the same representation, violating this causal structure.

### Spectrum of fixes

From cheapest to most principled:

**1. Post-process constraints** (no training change)

If action is a grounded action, clamp y to nearest ground/platform height. If action is aerial, set on_ground=false. Encode Melee's physics rules as hard constraints on the model's output.

- Pro: Immediate, fixes demos now
- Con: Brittle, growing list of rules, doesn't teach the model anything

**2. Scheduled sampling** (E012b — already tested)

Corrupt context with noise during training. Forces model to predict correctly from imperfect inputs, simulating autoregressive drift.

- Pro: Already implemented, general-purpose robustness
- Con: E012b lost 2.8pp change_acc (88.3% vs 91.1%). May need 4+ epochs to recover. Doesn't address the structural cause — heads still independent, just more robust to noise.

**3. Consistency loss / cross-head regularization** (training change, no arch change)

Add loss terms penalizing inconsistency between heads:
- If action ∈ grounded_set but on_ground=false → penalty
- If action is STANDING but predicted y-velocity ≠ 0 → penalty
- If no attack in recent context but damage increases → penalty

- Pro: Teaches the model physics constraints through gradient signal
- Con: Requires enumerating the constraints manually. Grows with game knowledge.

**4. Cascaded heads** (architecture change)

Predict action first, then feed the predicted action INTO the position/binary/continuous heads as additional conditioning:

```
h = mamba_trunk(context)
action = action_head(h)
position = position_head(h, action_embedding(action))
on_ground = ground_head(h, action_embedding(action))
percent = damage_head(h, action_embedding(action))
```

The other heads literally *know* what action was predicted and can condition their physics on it. This mirrors the game engine's causal structure: action → physics.

- Pro: Architecturally principled. The model's information flow matches the game's causality. No manual constraint enumeration.
- Con: Adds sequential dependency (action head → other heads). During training with teacher forcing, use ground-truth action as conditioning (avoid exposure bias). At inference, use predicted action.

**5. Hybrid model with physics integrator** (biggest change)

Predict action + physics parameters (velocity, acceleration). Derive position by integrating velocity. Derive on_ground from position + stage geometry. The model only needs to predict what the game engine computes as intermediate values; a lightweight physics layer computes the rest.

- Pro: Position is physically consistent by construction
- Con: Significant rearchitecture. Need to know enough about Melee's physics engine to build the integrator.

### Which fix fits where we are

Options 1+2 get us to a better demo. Option 4 (cascaded heads) is the real fix — it's the minimum architectural change that addresses the root cause. Option 5 is the "right" long-term answer but is a larger project.

**Next step**: unpack cascaded heads architecture. Key design questions: how to embed the predicted action for conditioning, whether to use teacher forcing on the action input during training (probably yes — avoids compounding the action head's errors into the physics heads during learning), and how this interacts with multi-position prediction.

### Related observations

- **Hop #3 in the demo data** (P0 frames 165-182): The model *does* sequence KNEE_BEND → JUMPING_BACKWARD → NAIR correctly — the action head gets the transition right. But position and on_ground don't react appropriately. The heads produce correct marginal distributions but not a coherent joint state.

- **LANDING_SPECIAL for 25+ frames** (P1 frames 131-160): The model gets stuck in LANDING_SPECIAL long past its real duration (4 frames). This is a separate problem — the model doesn't enforce action duration constraints. state_age should help here but isn't being used as a hard constraint.

- **SHIELD_REFLECT while rising** (P1 frames 114-121): The model predicts SHIELD_REFLECT (a grounded action) while y rises from 3.6 to 9.5. Textbook independent heads — action says "shield" while position says "going up."

### E012b results (scheduled sampling)

| Metric | E012 (no SS) | E012b (SS 0.3) | Delta |
|--------|-------------|----------------|-------|
| val change_acc | 91.1% | 88.3% | -2.8pp |
| val action_acc | 97.7% | 97.1% | -0.6pp |
| val pos_mae | 0.706 | 0.703 | -0.003 |
| val_loss/total | 0.527 | 0.528 | +0.001 |

SS hurt val change_acc by ~3pp. The 2-epoch run only completes 2/3 of the 3-epoch anneal ramp. Might need 4+ epochs to recover. The real test is whether autoregressive demos drift less — that's the whole point of SS.

---

## 2026-03-01: E012 — Clean Slate, FD Top-5, Honest Numbers

### Context

After discovering the focal_offset inflation (see below), we stripped back to a clean configuration: focal_offset=0, top-5 characters only (Fox, Falco, Marth, Falcon, Sheik) on Final Destination only. This removes two sources of noise: rare-character physics confusion and platform geometry complexity.

### What E012 keeps vs excludes from E008-E011

| Keep | Source | Why |
|------|--------|-----|
| multi_position=true | E008c | 10× training signal, +4.2pp, architecturally clean |
| v3 encoding (state_flags, hitstun) | E009a | Direct actionability signals |
| action_change_weight=5.0 | E010b | +9.9pp, forces model to care about transitions |
| ctrl_threshold_features=true | E010c | +5.9pp, gives model Melee's stick deadzone boundaries |

| Exclude | Source | Why |
|---------|--------|-----|
| focal_offset | E008a/E009a | Inflated training metrics ~15pp, unclear inference benefit |
| ctrl_residual_to_action | E010a | +0.0pp — no measurable effect |
| projectiles | E009a data | Negligible impact (85.8% vs 85.2%), adds encoding complexity |
| character_conditioning | E011b | Null result at this data scale |
| character_embed_dim>8 | E011a | Null result |

### Dataset

- **Stage**: Final Destination only (stage 32) — flat stage, no platforms, pure neutral
- **Characters**: Fox (1), Falcon (2), Sheik (7), Marth (18), Falco (22) — 77.5% of all games
- **Games**: 1,988 (from 22K parsed-v3-24k dataset, 9.0% pass filter)
- **Frames**: 17,465,842 (~17.5M)
- **Encoded file**: `encoded-e012-fd-top5.pt` (12 GB)

### Results

**Training val**: 91.1% change_acc, 0.706 pos_mae, 0.527 val_loss

**Teacher-forced batch eval** (5 hand-picked FD games, diverse matchups):

| Game | Matchup | Frames | change_acc | pos_mae |
|------|---------|--------|-----------|---------|
| Fox vs Marth | spacie vs swordie | 5,997 | 93.6% | 0.72 |
| Marth vs Falco | swordie vs spacie | 7,157 | 92.9% | 0.66 |
| Marth ditto | mirror | 5,379 | 94.5% | 0.56 |
| Falcon vs Marth | grappler vs swordie | 3,718 | 95.1% | 0.58 |
| Sheik vs Falcon | mixup vs grappler | 5,289 | 95.3% | 0.59 |
| **Aggregate** | | **27,540** | **94.2%** | **0.63** |

Every game above 92%. The clean dataset (no rare characters, no platform complexity) removed noise and let the model learn cleaner physics.

**Error breakdown** (movement still #1 source, but massively improved):
- TURNING: 87.6% change_acc (up from 28.2% in E009a — biggest single improvement)
- TUMBLING: 41.9% (damage reactions, inherently stochastic)
- ON_HALO_DESCENT: 12.5% (respawn timing, partially random)
- Movement category overall: 180 change errors across 27K frames

### Autoregressive demo

Generated Fox vs Marth on FD, 3000 frames. Visually much better than previous demos. Wired SVG character animations from the web/ live viewer into the worldmodel viewer.html — characters now render as actual sprites instead of colored dots.

**Checkpoint**: `e012-clean-fd-top5/best.pt` (also copied to autonomous-world-model project)

---

## 2026-03-01: The Focal Offset Inflation — We've Been Measuring Wrong

### The question

Mattie: "I thought we saw 85% change_acc?"

E010d training val reported 85.8% change_acc. Batch eval (teacher-forced, 30 games, 644K frames) showed 70.7%. A 15pp gap. We spent a session chasing this down and found three compounding measurement issues.

### Finding 1: P0-only metric (minor — ~3pp)

The trainer's `action_change_acc` (`metrics.py:340-346`) only measured P0 action changes. Batch eval measured both P0 and P1. We added P1 tracking and found the gap is small: P0=70.2%, P1=67.4%, combined=68.8%. The model treats both players nearly equally. Not the main issue.

**Fixed**: trainer now reports `p0_action_change_acc`, `p1_action_change_acc`, and combined `action_change_acc`. Batch eval now reports per-player breakdown.

### Finding 2: Multi-position averaging (major — the real inflation)

With `multi_position=true`, the loss and metrics are computed over ALL K positions uniformly (reshape from `(B,K,...)` to `(B*K,...)`). With `focal_offset=3`, positions 7-9 see real future state in their causal past. These positions get artificially high change_acc because they have more information than positions 0-6.

The reported 85.8% is an average across all 10 positions. Positions 7-9 (with future context) inflate the average. The honest number at the focal position (6) — which is what inference actually uses — was never measured during training.

**This retroactively affects ALL experiments from E009a onward.** Every reported change_acc with `focal_offset>0 + multi_position=true` is inflated. Relative comparisons between experiments (E010d vs E010a, E011a vs baseline) are still valid — they all share the same inflation. But absolute values are overstated.

| Run | Reported change_acc | Honest (batch eval) | Inflation |
|-----|--------------------|--------------------|-----------|
| E008c (multi_pos, no focal) | 71.5% | ~71.5% (honest) | 0pp |
| E010d (multi_pos + focal_offset=3) | 85.8% | 70.7% | ~15pp |

E008c's 71.5% was honest because `focal_offset=0` — every position has the same information (only causal past). No position gets special treatment.

### Finding 3: Batch eval wasn't using focal offset at all

`generate_teacher_forced` fed the model context `[t-10:t]` — a standard window. But E010d was trained with `focal_offset=3`, meaning context `[t-7:t+3]`. The model was getting the wrong window shape at inference.

### The GGPO-rollback fix

Mattie's insight: treat it like rollback netcode. At inference, give the model the window shape it was trained on. Fill the 3 "future" slots with "assume unchanged" state (repeat last frame) plus real controller inputs. This is Slippi's "repeat last input" prediction applied to world model context.

We implemented this and tested:

| Mode | action_acc | change_acc (combined) | p0 | p1 |
|------|-----------|----------------------|-----|-----|
| No focal offset (wrong window) | 83.2% | 68.8% | 70.2% | 67.4% |
| Focal D=3, rollback fill | **90.7%** | **72.7%** | 74.6% | 70.7% |

+7.5pp action_acc, +3.9pp change_acc — free improvement from using the window shape the model expects. The remaining gap from 85.8% is the multi-position inflation (positions 7-9 seeing real future during training val).

### Why better state in future slots doesn't help

We tested real future controllers vs repeated controllers in the future slots — identical results. This is because the causal SSM at position 6 **cannot see positions 7-9** (causality). The only thing from position 7 that affects position 6's prediction is the controller columns, extracted by the multi-position ctrl conditioning code. The state at positions 7-9 is invisible to the focal prediction.

This also means "simulate the future slots better" is a dead end for improving position 6's output. The 72.7% is the real ceiling for this model at this inference configuration.

### The honest comparison we haven't done

We've never compared multi_position+focal_offset=3 vs multi_position+focal_offset=0 on the same v3 data, measuring at the inference-relevant position. E008c (multi_pos, no focal, pre-v3) got 71.5%. E010d (multi_pos + focal, v3) gets 72.7% at the focal position. The E010b/c improvements (transition weighting + threshold features) are real, but focal_offset's contribution to inference accuracy might be near zero — it primarily inflated the training metric.

### Autoregressive demos

Generated Fox vs Marth on FD (stage 32) autoregressive demos with and without focal offset for visual comparison:
- `demo-e010d-ar-fd-nofocal.json` — baseline
- `demo-e010d-ar-fd-focal3.json` — with rollback fill

Focal-3 version is more conservative (fewer P1 action transitions: 27 vs 58 changes in 600 frames). Whether that's stability or sluggishness requires visual inspection.

### Implications

1. **All future experiments should report per-position change_acc** (at minimum: focal position vs average). The combined number is misleading with focal_offset>0.
2. **Focal offset may not help inference.** It shapes training gradients but the benefit might not survive to the inference-relevant position. Need an A/B test: same model config with focal_offset=0 vs focal_offset=3, measured at the last position (no focal) and focal position respectively.
3. **The rollback context fill is the correct inference pattern** for focal_offset models. `--focal-offset 3` flag added to both `batch_eval.py` and `generate_demo.py`.
4. **Rare characters are a bigger factor than architecture tweaks.** Samus vs Yoshi game dragged change_acc to 65.6% while Fox/Falco games hit 75%+. E011's null result (character embedding changes don't help) confirms: the problem is insufficient training data for rare characters, not embedding capacity.

---

## 2026-02-28: E011 — Character Importance (Null Result)

**Run card**: `docs/run-cards/e011-character-importance.md`

Motivated by the character distribution analysis showing extreme skew (Fox 26.9%, top 3 = 61.7%, Fox:Mewtwo = 410:1), we tested whether stronger character signals would help the model learn character-specific physics.

Three experiments, all branching from E010d (85.8% change_acc):
- **E011a**: Bigger character embedding (8→32 dim)
- **E011b**: Character conditioning (additive projection into hidden state, like ctrl conditioning)
- **E011c**: Combined a+b

**Result: complete null.** All three landed at 85.9% change_acc — within noise of the 85.8% baseline. No improvement on any metric.

**What this tells us:**
1. The 8-dim character embedding is already sufficient. With 77.5% of games being Fox/Falco/Marth, there aren't enough rare-character examples for bigger embeddings to learn from.
2. Character conditioning (injecting identity into the hidden state) adds nothing. The model is already getting character identity through the concatenated input embedding — it knows which character it's simulating.
3. The remaining change_acc headroom is NOT about character awareness. The model isn't confused about *which character* is playing; it's confused about *what that character will do next*.

**Implications for next experiments**: Character-side interventions are a dead end at this data scale. The bottleneck is elsewhere — likely data quantity (more epochs, more games) or model capacity (bigger d_model). The pos_mae regression from E010d (0.65→0.67-0.68) is also worth investigating — the transition weighting may be stealing gradient from position prediction.

**Ops note**: Config validator had a false positive on `character_embed_dim: (32, 8)` because it was comparing model-side embedding dims against saved encoding config. Fixed by adding all `*_embed_dim` fields and model-side experiment flags to `_TRAINING_ONLY_FIELDS` exclusion list.

---

## 2026-02-28: E010 — Movement Suite, the Data Incident, and Character Distribution

### E010 results

Movement was the weakest action category at 55.6% change_acc (TURNING 28.2%, WALK_SLOW 31.2%). E010 attacked the ctrl→action pathway from three angles:

| Experiment | What | change_acc | vs E009a (+70.9%) |
|-----------|------|-----------|-------------------|
| E010a | Ctrl residual to action heads | 70.9% | +0.0pp — no effect |
| E010b-v1 | Transition-weighted loss (5×) | 80.8%* | +9.9pp |
| E010c-v1 | Deadzone threshold features | 76.8%* | +5.9pp |
| **E010d** | **B+C combined (no proj)** | **85.8%** | **+14.9pp** |
| E010e | B+C combined (with proj) | 85.2%* | +14.3pp |

*v1 results ran on wrong data (see below). E010d is the clean result on the same data as E009a.

E010a (ctrl residual) did nothing — the SSM isn't diluting the ctrl signal as much as hypothesized. The additive conditioning is already sufficient. The real problems were: (1) the loss doesn't push hard enough on transitions (E010b), and (2) the model has to learn Melee's stick thresholds from continuous values (E010c).

B+C is super-additive. Threshold features give the model sharp decision boundaries. Transition weighting forces it to focus on the frames where those boundaries matter. Together: 85.8%, beating either alone (80.8%, 76.8%).

Projectiles had negligible impact: 85.8% vs 85.2%. Different game sets muddy this slightly, but the signal is clear — projectile encoding isn't the lever for movement prediction.

### The data incident

We launched E010a/b/c on `encoded-game-v3-2k.pt` (projectiles=true) instead of `encoded-v3-ranked-50k-part-0.pt` (projectiles=false, same data as E009a). The config validator caught the mismatch: `projectiles: (False, True)`. Instead of investigating why the file had the wrong encoding, Scav flipped the config flags to match the data. Wrong reflex.

Three mismatches went unlogged: different encoded file, different encoding, different game count (17.5M vs 21.4M train examples). The v1 results are directionally interesting but not comparable to the E009a baseline.

**What we fixed:**
1. Deleted `encoded-game-v3-2k.pt` from Modal volume (local copy retained)
2. Enhanced `modal_train.py` wandb.init to log `encoded_file`, `float_columns`, `saved_encoding_config`, `train_examples`, `val_examples`, `model_params`
3. Created `/launch-experiment` skill: reads run card → launches with --detach → polls wandb for actual params → compares against card → logs verification to card
4. Key rule baked into skill: **never change a config flag to match an encoded file — the validator is telling you you're using the wrong file**

The deeper issue: we have 10+ encoded files on Modal with encoding configs sealed inside .pt blobs. No manifest, no way to know what's in a file without loading it. `encoded-game-v3-2k.pt` and `encoded-v3-ranked-50k-part-0.pt` sound interchangeable but differ in encoding flags and game source. Mattie pointed out that `state_flags` is the critical feature here — losing it silently would mean misattributing a regression to the experiment rather than the data.

### Character distribution

Analyzed the 22K dataset. The distribution is extremely top-heavy:

- **Top 3** (Fox 26.9%, Falco 19.4%, Marth 15.3%) = **61.7%** of all appearances
- **Top 5** (+ Falcon 8.6%, Sheik 7.3%) = **77.5%**
- **Top 8** (+ Peach 5.0%, Puff 4.2%, ICs 2.6%) = **89.3%**
- **Bottom 13** characters share 3.4%. Fox:Mewtwo ratio is 410:1.

The ranked 50K is likely even more skewed (high-rank players disproportionately play spacies).

Currently character is an 8-dim embedding — one feature among dozens. The model sees it but doesn't know it's load-bearing for physics prediction. Characters with <1% representation (Ganon and below) have effectively undertrained embeddings. The model probably defaults to "average character" physics for them.

This motivates E011: making character identity more prominent in the model. Bigger embeddings, character conditioning (like ctrl conditioning — project into hidden state at every layer), and possibly filtering to top-8 characters to remove noise from the long tail.

Mattie's hunch: character importance might also help with the ctrl override problem. If the model knows "this is Fox, Fox has these dash thresholds" rather than learning blurred average thresholds, it can more precisely map ctrl→action.

---

## 2026-02-28: E009 — v3 Encoding + State Flags, and the Ctrl Override Problem

### Context

E008c (multi-position) hit 71.5% change_acc — best clean result. But all E008 experiments ran on `encoded-2k.pt` which predates v3 features: `state_flags=false, hitstun=false, projectiles=false`. The model was learning physics without the bits that directly encode actionability.

E009 graduates E008's best ideas (multi-position + focal context) to v3 encoding with state_flags and hitstun enabled. Projectiles remain off.

### The actionability insight

The model struggles with "can't make a move while in dash startup." Our state representation has `action + state_age` which implicitly encode this — but the model has to learn the full IASA table (~400 action states × variable interruptibility frames) from data. That's a lot to ask of 4.3M parameters.

State flags give it directly:
- **byte0.bit2** (84.9% freq): "has control / actionable" — OFF during transitions like walk start, jump startup, dash startup
- **byte3.bit5** (33.9% freq): "can cancel / IASA window" — ON when an action can be interrupted

These are the two most important bits for the world model. They directly answer "will this controller input take effect?"

### The ctrl override problem

Mattie raised a sharp question: can the model decide a player's input is "stupid" and predict what it thinks they *should* do instead?

Yes. This is a real risk. The model learned from thousands of games where 95% of Foxes FAIR in a given context. When a player NAIRs, the contextual prior (K frames of "every Fox FAIRs here") might overwhelm the controller conditioning signal. Our ctrl is additive (`h + ctrl_proj(ctrl)`) — 26 floats projected and added to a 384-dim hidden state. The model can learn to predict the modal action from context alone and treat ctrl as noise.

This is "posterior collapse" in conditional generative models. The conditioning signal gets ignored when the learned prior is strong enough. It would manifest as: the model predicts the most common action for a given game state regardless of what buttons are actually pressed. change_acc would look decent (the common action IS usually correct), but unusual plays would be "corrected" by the model.

**Connection to GGPO**: In Melee's game engine, physics are mechanical — press A during wait = jab, always, no opinions. Our model has opinions. It learned "what usually happens" not "what the rules say." The state_flags help because they encode the rules directly (actionable/not actionable), reducing the space where the model's opinions can override the input.

**Potential fixes** (not yet tested):
- Stronger ctrl conditioning: cross-attention or FiLM (feature-wise linear modulation) instead of additive
- Direct ctrl → action shortcut: let the action head see raw ctrl, bypassing the SSM's contextual prior
- Data augmentation with unusual inputs: force the model to handle uncommon ctrl patterns
- Measure the problem first: compare action accuracy on "common" vs "uncommon" ctrl→action pairs

### Runs launched

| ID | K | focal | multi_pos | state_flags | hitstun | binary weight | Data |
|----|---|-------|-----------|-------------|---------|---------------|------|
| **E009a** | 10 | 3 | true | true | true | 1.0 | ranked-v3, 2K games |
| **E009b** | 30 | 5 | true | true | true | 1.0 | ranked-v3, 2K games |

Early observation: K=10 and K=30 are tracking closely on loss. Hypothesis: the "compressed history" features (last_attack_landed, action+state_age, hitstun_remaining, state_flags) carry enough information that raw context length doesn't add much. The extra 20 frames of position/velocity history in K=30 are mostly redundant when the model already knows what attack hit, how long hitstun lasts, and whether the character can act.

wandb: E009a `wqkltuzj`, E009b `c3d4rxln`.

### Future experiments: state_flags optimization

Logged for later — wait for E009 results first.

1. **Split binary head**: Separate "core binary" (facing, invulnerable, on_ground) from "state_flags" with their own loss weights. The 7 high-value bits are currently mixed with 12 dead bits and ~20 low-information bits.

2. **Curate state_flags**: Encode only the 7 high-value bits instead of all 40. Smaller head, every bit meaningful, no noise.

3. **Combat phase categorical**: Derive a single feature — 0=neutral, 1=attacking, 2=in_hitstun, 3=in_knockback, 4=recovering, 5=dead. One embedding that *is* the relationship between last_attack, hitstun, and state_flags. Manual feature engineering, but the model gets the structure for free.

4. **Trust the model** (baseline): The SSM should learn feature correlations from co-occurrence. Multi-position training (10× signal density) helps.

### Future experiments: ctrl conditioning strength

The ctrl override problem (above) won't bite us until policy play — autoregressive eval uses real replay ctrl, which aligns with the model's learned prior. But worth solving before two-agent play. Logged for later.

1. **Ctrl residual to action head** (easiest, ~10 lines in mamba2.py). Give the action prediction head direct access to raw ctrl features: `action = action_head(cat(h, ctrl))` instead of `action = action_head(h)`. Can't be ignored — it's a separate input channel bypassing the SSM's contextual prior.

2. **FiLM conditioning** (more powerful). Replace additive ctrl (`h + ctrl_proj(ctrl)`) with multiplicative modulation: `h = gamma(ctrl) * h + beta(ctrl)`. Multiplicative conditioning is much harder for the model to collapse away. Used in StyleGAN for this exact reason.

3. **Ctrl-change loss weighting** (no architecture change). Weight the loss higher on the ~5% of frames where ctrl actually changes. Forces the model to pay attention to ctrl precisely at transitions.

4. **Concatenation** instead of addition. Concat ctrl to the hidden state before prediction heads: `heads(cat(h, ctrl_proj(ctrl)))`. Wider input to heads, ctrl preserved as a distinct signal.

These only matter for policy play. For autoregressive accuracy (current goal), the model sees real ctrl from replays which matches its training distribution.

### Ops fix

`max_games` in YAML config wasn't being applied to pre-encoded .pt data — only to raw parsing. Added slicing in `modal_train.py` to cap games from pre-encoded payloads. Without this, loading `encoded-v3-ranked-50k-part-0.pt` (12K games) would train on all 12K despite `max_games: 2000`.

---

## 2026-02-28: E008 Focal Context — Five Experiments, One Cheater

### The question

Characters stop taking visible actions during autoregressive rollout. We hypothesized two possible causes: (1) the model can't predict action transitions well (a learning problem), or (2) small errors in binary flags like `on_ground` compound and push the model into a "frozen" state (a drift problem). Batch eval confirmed teacher-forced error is flat across time — it's not drift in teacher-forced mode, so the model genuinely struggles with transitions.

**Core idea**: give the model future context during training. If it can see where the game is going (even a few frames ahead), it should learn better representations of *what's about to change*.

### The experiments

All use Mamba-2 4.3M, 2K games, 2 epochs, H100.

| ID | What it does | Prediction target | Result |
|----|-------------|-------------------|--------|
| **E008a** | Tap hidden state *before* the target in context | Frame t (from position 6, with t,t+1,t+2 visible after) | **93.0% change_acc** |
| **E008b** | Positional conditioning — scalar tells model which frame to predict | Frame t (from last position, with future visible) | 99.4% — **cheating** |
| **E008c** | GPT-style multi-position — predict at every position | Every frame (no focal offset needed) | **71.5% change_acc** |
| **E008d** | Bidirectional Mamba — backward pass sees future | Frame t (from combined fwd+bwd hidden state) | ~99% — **cheating** |
| **E008e** | Future controller conditioning — see ctrl at T+1,T+2 | Frame t (standard position, no future state) | 64.7% change_acc |

**Baseline for comparison**: 67.3% change_acc (Mamba-2 4.3M, K=10, 2K/2ep).

### The cheaters (E008b, E008d)

E008b and E008d both hit 99%+ because the SSM's forward pass processes the target frame *before* the output position. The model literally reads the answer from its own input.

- **E008b**: positional scalar tells the model "predict frame at position 6." But the SSM hidden state at position 9 (where prediction happens) has already processed frames 7-9 — including the target. The model just copies.
- **E008d**: bidirectional pass explicitly gives the backward hidden state access to future frames. Same leakage, different mechanism.

Useful as diagnostics — they prove the SSM *can* propagate information across positions. But useless as training strategies.

**Key test**: E008b in teacher-forced mode WITHOUT focal context (standard 10-frame past window) scored 17.8% change_acc — *worse* than E008a's 25.9% autoregressive. The cheater model learned to copy, not to predict physics. When you take away its cheat sheet, it's worse than a model that never had one.

### E008a: the real result

93% change_acc is a massive jump from 67.3% baseline. Is it legitimate?

**Why it's not cheating**: Mamba-2's causal scan guarantees the hidden state at position 6 only depends on positions 0-6. Frames t, t+1, t+2 sit at positions 7-9 and are blocked by the causal mask. This holds in SSD mode — the chunked computation is mathematically equivalent to the sequential scan, with a lower-triangular mask on Q*K^T.

**Why it's so much better**: The model was *trained* with future context present. Even though position 6 can't see positions 7-9, the gradients from position 6's loss flow back through weights that also process positions 7-9 during the forward pass. The training dynamics produce richer representations — the model learns features that coexist with future context, which happen to be more informative for prediction.

**The honest caveat**: 93% is teacher-forced. The model expects real frames at positions 7-9 (the future context it was trained with). In autoregressive mode, those slots don't exist — the model only has past context. Autoregressive E008a was "better but still a mess." The training environment doesn't match inference.

**Sanity check we haven't done yet**: take the E008a checkpoint, give it 10 frames of pure past context (no future), evaluate from position 9 (like baseline). If it still beats 67.3%, the learned representations are genuinely better regardless of future context. If it drops back to baseline, the 93% was an artifact of train/eval mismatch.

### E008e: the rollback approximation

Mattie's idea: instead of future *state* (which you don't have at inference), give the model future *controller inputs* (which you could have via speculative lookahead or opponent modeling). Predict frame t, but condition on ctrl at T, T+1, T+2.

Result: 64.7% change_acc at 2 epochs. Roughly baseline. The extra ctrl signal didn't clearly help — the model may need more epochs to learn to use it, or the future button presses alone (without future state) aren't informative enough for 2-frame lookahead at 60fps.

### E008c: the winner so far

**71.5% change_acc** — beats baseline by 4.2pp. Also best pos_mae (0.61) and best val_loss (0.289) of the entire series.

GPT-style predict-at-every-position. At each position i, predict frame i+1 from the causal hidden state. Gives K× more training signal per sample. No future context needed — this is architecturally clean.

Why it works: 10× more loss terms per sample acts as massive regularization. The model can't overfit to predicting only the final frame — it has to learn physics that generalizes across all positions in the context window. Every position is a slightly different prediction problem (different amount of history), which forces robust representations.

Hit a bug on first Modal launch: `_val_epoch` called `action_tracker.update()` with the original (B,K,...) shaped predictions after `compute_loss` had only reshaped them internally. The `.pop("_multi_position")` mutated the caller's dict but created a new reshaped dict internally — the caller still had (B,K,...) tensors. Fixed by using `.get()` and adding dim-check reshaping in `_val_epoch`.

**Full metrics**: action_acc=97.1%, pos_mae=0.61, val_loss=0.289. wandb: `hcujei2k`.

### Decisions made

1. **Focal offset is training-only**: `focal_offset`, `multi_position`, and `bidirectional` are excluded from the config validator's comparison against pre-encoded data. They change model behavior but not tensor dimensions.
2. **Config validator fix**: compare only explicitly-set fields in saved config, not resolved defaults. Prevents false mismatches when defaults change (the root cause of the session's first 3 failed Modal launches).
3. **Tensor dimension sanity check**: added after data load, catches encoding config mismatches immediately.
4. **Branch per experiment**: E008a-E008e each on their own branch. Ops fixes cherry-picked (or manually applied when conflicts arise).

### Next: E008f/g — combining focal context with multi-position

E008c's multi-position and E008a's focal context address different weaknesses. Multi-position forces the model to predict well at *every* position (regularization). Focal context gives richer gradient signal from future frames (representation quality). The combination should fix E008a's train/eval mismatch — positions 0 through K-D-1 are past-only, forcing the model to also learn without future context.

| ID | K | focal_offset | multi_position | Estimate |
|----|---|-------------|----------------|----------|
| **E008f** | 10 | 3 | true | ~$2.30, ~35 min (control) |
| **E008g** | 30 | 5 | true | ~$7.78, ~2 hours (scale) |

E008f isolates the interaction effect at baseline K. E008g tests whether 3× context window + 3× training signals (529M/epoch vs 177M) justifies the 3× compute. K=30 with chunk_size=15, batch_size=2048.

### Open questions

- Does E008a's 93% survive the sanity check (eval without future context)?
- Does E008f beat E008c (does focal context help when multi-position is already present)?
- Does K=30 (E008g) justify 3× compute over K=10 (E008f)?
- Is E008e worth more epochs, or is future ctrl fundamentally less useful than future state?

---

## 2026-02-27: Batch Eval — The on_ground Crisis

*See full analysis: `research/notes/BATCH-EVAL-ANALYSIS.md`*

Teacher-forced eval across 30 games revealed:
- **on_ground recall: 52%** — model predicts "airborne" when characters are grounded half the time
- **facing recall: 51%** — same one-directional bias
- Error rate flat across game time — not a drift problem, a prediction problem
- BAIR/FAIR (back-air / forward-air) confusion is downstream of facing failure
- Action transitions are the hard part: 67.3% change_acc vs 99%+ when action stays the same

This motivated the E008 focal context series above.

---

## 2026-02-25: Mamba-2 Beats MLP

| Model | Params | Data | Epochs | change_acc | pos_mae |
|-------|--------|------|--------|-----------|---------|
| MLP (best) | 2.1M | 22K/4ep | 4 | 77.5% | — |
| Mamba-2 | 4.3M | 2K/2ep | 2 | 67.3% | 0.65 |
| MLP | 2.1M | 2K/2ep | 2 | 64.5% | — |

At equal data and epochs, Mamba-2 beats MLP by 2.8pp on change_acc. Architecture validated. The MLP ceiling at 77.5% was the architecture bottleneck, not data.

Key finding: SSD mode requires `chunk_size` to evenly divide `context_len`. K=10 needs chunk_size=10 or 5. K=60 needs chunk_size=30 or 15.

---

## 2026-02-24: Hackathon Win

Moltiverse Hackathon, $200K prize pool. See `docs/RETROSPECTIVE.md`.

---

*Earlier entries predate this diary format. See `research/README.md` for the full experiment index and `docs/ROADMAP.md` for the sprint timeline.*
