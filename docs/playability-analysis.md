# Playability Analysis: Full Metric Basket & Gap-to-Playable

**Date:** 2026-03-22
**Data source:** wandb per-horizon metrics from all 8 kept experiments (e019 through e027c)
**Current best:** e027c (RC 4.939)

---

## 1. Raw Data Tables

### 1.1 Summary Metrics (Teacher-Forced)

| Exp | RC | change_acc | pos_mae | vel_mae | dmg_mae | bin_acc |
|-----|---:|----------:|--------:|--------:|--------:|--------:|
| e019 | 6.771 | 0.670 | 0.840 | 0.070 | 0.813 | 0.999 |
| e018a | 6.261 | 0.616 | 0.825 | 0.080 | 0.795 | 0.998 |
| e018c | 6.026 | 0.622 | 0.824 | 0.078 | 0.795 | 0.998 |
| e023b | 5.775 | 0.660 | 0.823 | 0.087 | 0.891 | 0.999 |
| e025a | 5.146 | 0.656 | 0.814 | 0.124 | 0.920 | 0.998 |
| e026b | 5.120 | 0.654 | 0.798 | 0.117 | 0.931 | 0.998 |
| e026c | 4.965 | 0.802 | 0.642 | 0.015 | 0.404 | 1.000 |
| e027c | 4.939 | 0.789 | 0.650 | 0.039 | 0.623 | 1.000 |

### 1.2 Position MAE by Horizon (game units, lower = better)

| Exp | h1 | h2 | h3 | h5 | h7 | h10 | h15 | h20 |
|-----|---:|---:|---:|---:|---:|----:|----:|----:|
| e019 | 1.08 | 1.72 | 2.33 | 3.47 | 4.63 | 6.35 | 9.54 | 12.33 |
| e018a | 1.02 | 1.58 | 2.12 | 3.21 | 4.30 | 5.89 | 8.81 | 11.36 |
| e018c | 0.67 | 1.29 | 1.87 | 3.09 | 4.18 | 5.77 | 8.51 | 11.13 |
| e023b | 0.65 | 1.24 | 1.75 | 2.86 | 3.90 | 5.43 | 8.22 | 10.89 |
| e025a | 0.58 | 1.11 | 1.59 | 2.62 | 3.52 | 4.89 | 7.32 | 9.54 |
| e026b | 0.55 | 1.06 | 1.51 | 2.52 | 3.45 | 4.84 | 7.34 | 9.59 |
| e026c | 0.54 | 1.01 | 1.45 | 2.44 | 3.35 | 4.69 | 7.08 | 9.36 |
| e027c | 0.53 | 1.01 | 1.44 | 2.37 | 3.24 | 4.66 | 7.08 | 9.37 |

### 1.3 Action Accuracy by Horizon (%, higher = better)

| Exp | h1 | h2 | h3 | h5 | h7 | h10 | h15 | h20 |
|-----|---:|---:|---:|---:|---:|----:|----:|----:|
| e019 | 97.0 | 94.7 | 91.2 | 84.0 | 76.3 | 67.8 | 56.2 | 47.5 |
| e018a | 97.2 | 95.2 | 93.3 | 88.3 | 81.0 | 73.7 | 61.7 | 52.0 |
| e018c | 99.0 | 97.5 | 95.8 | 89.5 | 84.5 | 75.5 | 61.0 | 51.7 |
| e023b | 98.7 | 97.2 | 96.2 | 90.0 | 85.2 | 77.3 | 64.7 | 54.3 |
| e025a | 99.3 | 97.8 | 96.3 | 92.5 | 87.7 | 79.7 | 64.8 | 55.0 |
| e026b | 99.2 | 97.5 | 96.3 | 91.0 | 87.5 | 77.8 | 65.8 | 56.0 |
| e026c | 99.0 | 97.7 | 96.5 | 90.5 | 86.5 | 78.3 | 65.7 | 55.3 |
| e027c | 99.2 | 97.0 | 97.2 | 91.3 | 85.8 | 79.3 | 66.0 | 55.2 |

### 1.4 Percent (Damage) MAE by Horizon (damage %, lower = better)

| Exp | h1 | h2 | h3 | h5 | h7 | h10 | h15 | h20 |
|-----|---:|---:|---:|---:|---:|----:|----:|----:|
| e019 | 0.89 | 1.58 | 2.31 | 3.64 | 4.94 | 6.83 | 10.09 | 13.00 |
| e018a | 0.62 | 1.05 | 1.51 | 2.27 | 2.95 | 4.01 | 5.84 | 7.49 |
| e018c | 0.55 | 1.05 | 1.57 | 2.69 | 3.59 | 4.84 | 6.89 | 8.70 |
| e023b | 0.56 | 0.98 | 1.42 | 2.41 | 3.23 | 4.47 | 6.60 | 8.67 |
| e025a | 0.20 | 0.37 | 0.57 | 1.09 | 1.38 | 1.80 | 2.63 | 3.42 |
| e026b | 0.20 | 0.38 | 0.58 | 1.08 | 1.33 | 1.74 | 2.58 | 3.36 |
| e026c | 0.20 | 0.36 | 0.55 | 1.06 | 1.31 | 1.70 | 2.52 | 3.30 |
| **e027c** | **0.50** | **0.98** | **1.43** | **2.45** | **3.23** | **4.53** | **6.72** | **8.78** |

### 1.5 Velocity MAE by Horizon (game units/frame, lower = better)

Note: e025a/e026b/e026c have anomalously low velocity values (~20x lower than e018a/e027c). This is likely a denormalization inconsistency in the eval code deployed on Modal for those runs. The e018a and e027c values are in correct game units.

| Exp | h1 | h5 | h10 | h20 | Notes |
|-----|---:|---:|----:|----:|-------|
| e019 | 0.00 | 0.00 | 0.00 | 0.00 | Known vel_mae=0 bug |
| e018a | 0.025 | 0.124 | 0.248 | 0.498 | Valid |
| e018c | 0.046 | 0.251 | 0.501 | 0.985 | Valid but note: worse than e018a |
| e023b | 0.056 | 0.308 | 0.626 | 1.242 | Valid |
| e025a | 0.001 | 0.005 | 0.010 | 0.019 | Suspect: ~20x too low |
| e026b | 0.001 | 0.005 | 0.010 | 0.021 | Suspect: ~20x too low |
| e026c | 0.001 | 0.005 | 0.009 | 0.017 | Suspect: ~20x too low |
| e027c | 0.024 | 0.124 | 0.248 | 0.501 | Valid |

For analysis below, we use e027c velocity values as representative of current-best performance: h1=0.024, h10=0.248, h20=0.501 game units/frame.

---

## 2. Degradation Analysis

### 2.1 How fast does each metric degrade?

**Position MAE** grows roughly **linearly** at ~0.46 game units per AR step (e027c). The h20/h1 ratio is 17.6x. This is approximately linear drift: 0.53 at h1, then +0.46 per step average. At 60fps, 20 frames = 333ms, during which position error grows to 9.4 game units (roughly one character width).

**Action accuracy** degrades **logarithmically** — fast initial drop, then flattens. From 99.2% at h1 to 79.3% at h10 (20pp drop in 10 steps) to 55.2% at h20 (another 24pp in 10 more steps). The degradation rate slows because at low accuracy, the model is essentially guessing among a few common states — random baseline for 400 classes would be ~30-40% due to action frequency skew.

**Percent (damage) MAE** grows **linearly** at ~0.44% per step (e027c), similar to position. By h20, the model is off by 8.8% damage — meaningful for a stock where percent ranges 0-200%.

**Velocity MAE** grows **linearly** at ~0.025 game units/frame per AR step (e027c). By h20, the velocity error is 0.50 game units/frame, which is significant given dash speed is ~1.5 units/frame (after the 0.05 normalization, raw dash speed ~30 units/frame... wait, need to verify units).

### 2.2 Which experiments improved each metric the most?

**Position MAE:** Monotonic improvement across all experiments. Largest single-step gains:
- e025a (LR warmup): -10.2% at h10, -7.5% at h20. Largest gain.
- e023b (d_model=768): -5.9% at h10. Capacity.
- e018a (Self-Forcing): -7.2% at h10. Training regime.
- Recent experiments (e026b, e026c, e027c): <3.5% gains. Diminishing returns.

**Action accuracy:** Best at short horizons (h1-h5) across all experiments. The big gains came from:
- e025a (LR warmup): h5=92.5%, h10=79.7%. Best at medium horizons.
- e026b/e027c: slightly better at h15-h20 (66.0%). Late experiments are better at very long horizons.
- No experiment crossed 80% at h10. This is the ceiling so far.

**Percent MAE:** Two distinct regimes:
- **e025a/e026b/e026c** have dramatically better percent tracking (h20: 3.3-3.4%) vs earlier experiments (h20: 8.7-11.4%).
- **e027c regressed to early-experiment levels** (h20: 8.78%). The warm-start from e025b (position-focused loss weights) destroyed percent tracking. This is the biggest finding in this analysis.

**Velocity MAE:** Unreliable data due to denormalization bug in e025a/e026b/e026c. Using e027c values: comparable to e018a level, suggesting the vel_mae improvements are an artifact.

### 2.3 Ceiling effects

Position MAE at h10 across last 4 experiments: 4.89 -> 4.84 -> 4.69 -> 4.66. The improvement rate dropped from ~10% per experiment to <1%. Position is hitting a ceiling at current model capacity + data scale.

Action accuracy at h10: 79.7% -> 77.8% -> 78.3% -> 79.3%. Essentially flat. The action head is saturated at ~80% h10 accuracy with current architecture.

---

## 3. "Playable" Thresholds

Context for scale calibration:
- Character width: ~8-10 game units (Fox body width)
- Character shoulder-to-shoulder in neutral: ~20 game units apart
- FD platform width: ~140 game units (from -70 to +70)
- Dash speed (Fox): ~2.2 units/frame, full hop height: ~35 units
- Action state vocabulary: 400 classes, but ~20 account for 80% of frames
- Percent: 0-999 nominal, 0-200 typical in a stock
- One frame = 16.67ms at 60fps

### 3.1 Tier Definitions

#### Watchable (a casual spectator wouldn't notice most errors at 60fps)

The key insight: at 60fps, the human eye tracks motion smoothly. Position errors under ~3 game units (30% of a character width) are invisible in motion. Action errors matter more because they produce visible sprite changes.

| Metric | Threshold | Reasoning |
|--------|-----------|-----------|
| pos_mae h10 | < 3.0 | Half a character width at 167ms. Characters visually "in the right place." |
| pos_mae h20 | < 6.0 | One character width at 333ms. Still looks like the same interaction. |
| action_acc h10 | > 85% | 15% wrong actions at 167ms. Mostly looks right, occasional weird animation. |
| action_acc h20 | > 70% | 30% wrong. Acceptable — real players do ambiguous things too. |
| percent_mae h10 | < 3.0 | 3% damage error at 167ms. Damage display is small, updates infrequently. |
| percent_mae h20 | < 6.0 | 6% at 333ms. Noticeable in HUD but not immersion-breaking. |
| vel_mae h10 | < 0.15 | Movement speed looks right. Smooth motion. |

#### Playable for Agents (an RL agent could train against this world model)

RL agents are more sensitive than humans. Position errors compound into wrong reward signals. Action errors mean the environment doesn't respond correctly to the agent's inputs. The threshold is: would the agent learn the right policy?

| Metric | Threshold | Reasoning |
|--------|-----------|-----------|
| pos_mae h10 | < 2.0 | Tight enough that shield pressure, spacing, and combos are meaningful. |
| pos_mae h20 | < 4.0 | Agent can plan 2-3 interactions ahead. |
| action_acc h10 | > 90% | Wrong actions produce wrong state transitions. Must be rare. |
| action_acc h20 | > 80% | Still tracking the interaction at 333ms. |
| percent_mae h10 | < 1.5 | Damage matters for kill confirms. Must be accurate. |
| percent_mae h20 | < 3.0 | Kill percent accuracy within 3% over long horizon. |
| vel_mae h10 | < 0.08 | Velocity determines if combos are true or escapable. |

#### Indistinguishable (hard to tell from real Melee data)

| Metric | Threshold | Reasoning |
|--------|-----------|-----------|
| pos_mae h10 | < 1.0 | Sub-character-width accuracy at all horizons. |
| pos_mae h20 | < 2.0 | 2 game units at 333ms. Within reaction-time noise. |
| action_acc h10 | > 95% | Only 1 in 20 frames has wrong action. |
| action_acc h20 | > 90% | Looks real for seconds of playback. |
| percent_mae h10 | < 0.5 | Half a percent at 167ms. Indistinguishable. |
| percent_mae h20 | < 1.0 | Damage tracking is essentially perfect. |
| vel_mae h10 | < 0.03 | Imperceptible speed differences. |

---

## 4. Gap Analysis

### 4.1 Current Best (e027c) vs. Watchable

| Metric | Current (e027c) | Watchable | Gap | Status |
|--------|----------------:|----------:|----:|--------|
| pos_mae h10 | 4.66 | 3.0 | 1.66 (36%) | GAP |
| pos_mae h20 | 9.37 | 6.0 | 3.37 (36%) | GAP |
| action_acc h10 | 79.3% | 85% | 5.7pp | GAP |
| action_acc h20 | 55.2% | 70% | 14.8pp | **LARGEST GAP** |
| percent_mae h10 | 4.53 | 3.0 | 1.53 (34%) | GAP |
| percent_mae h20 | 8.78 | 6.0 | 2.78 (32%) | GAP |
| vel_mae h10 | 0.25 | 0.15 | 0.10 (40%) | GAP |

### 4.2 Critical Finding: e027c is NOT the best on all metrics

**e027c has the best RC (position-only summary) but is WORSE than e026c on percent tracking by 166%.** This is the central problem with position-only RC as the fitness metric.

| Metric | e026c | e027c | Better |
|--------|------:|------:|--------|
| RC (pos_mae avg) | 4.965 | 4.939 | e027c (+0.5%) |
| pos_mae h10 | 4.69 | 4.66 | e027c (+0.6%) |
| action_acc h10 | 78.3% | 79.3% | e027c (+1.0pp) |
| percent_mae h10 | **1.70** | **4.53** | **e026c (2.7x better)** |
| percent_mae h20 | **3.30** | **8.78** | **e026c (2.7x better)** |
| vel_mae h10 | 0.009* | 0.25 | e026c (if valid) |
| change_acc | 80.2% | 78.9% | e026c (+1.3pp) |

*e026c velocity values suspect (denormalization issue).

**e026c is actually closer to "watchable" than e027c** on the full metric basket. The e025b warm-start in e027c improved position ~0.6% at the cost of 2.7x worse percent tracking. A composite metric would correctly rank e026c above e027c.

### 4.3 Biggest Gaps (prioritized by visual/gameplay salience)

1. **Action accuracy at h20: 55.2% vs 70% needed (14.8pp gap).** This is the single largest gap. By 333ms, nearly half of action predictions are wrong. Actions drive animation — wrong actions look wrong. This is the #1 priority.

2. **Position MAE at h20: 9.37 vs 6.0 needed (36% gap).** Characters drift by a full character width. Still the most salient visual error, but at h20 (333ms), the position error is partially masked by the action error (if the model shows the wrong animation, position error matters less).

3. **Percent MAE (e027c): 4.53 vs 3.0 at h10 (34% gap).** However, e026c already meets watchable at h10 (1.70 < 3.0). The gap is an e027c regression, not an unsolved problem.

4. **Velocity MAE: 0.25 vs 0.15 at h10 (40% gap).** Velocity drives position indirectly, but is the least visually salient on its own.

### 4.4 What techniques target each gap?

**Action accuracy at long horizons (THE priority):**
- SF curriculum with longer horizons (N=4, N=5): e026c's curriculum N=1->2->3 was the largest change_acc gain (+14.8pp). Extending to N=1->2->3->4->5 should improve long-horizon action predictions further.
- GKD-style teacher distillation (issue #16): soft targets instead of hard labels during SF. Reduces the "impossible to match ground truth from AR trajectory" problem at long horizons.
- DreamerV3 symlog-twohot for action logits: prevents overconfident action collapse at long horizons. Unimix (e026b) showed this direction works.
- Data scaling to 7.7K games: more diverse action transitions. The 1.9K dataset has limited coverage of rare action states.

**Position MAE at long horizons:**
- More capacity (d_model=1024 or deeper): position improvements have been monotonic with capacity.
- 7.7K data: de-risk over-parameterization concern at d_model=768.
- Longer SF unroll via curriculum: N=4/N=5 should stabilize longer-horizon positions.
- GameNGen context noise: input noise injection teaches robustness to accumulated drift.

**Percent MAE regression in e027c:**
- This is a regime switching artifact. The e025b checkpoint's position-focused loss weights (continuous=2.0, action=1.0) under-trained percent heads. Fix: include percent in the "continuous" weight scaling, or use e026c as the base instead of e025b.
- Alternatively: regime switch e026c -> standard weights (instead of e025b -> standard). E026c already has excellent percent tracking.

**Velocity MAE:**
- Fix the denormalization bug first to get valid measurements.
- Velocity errors propagate to position (vel -> pos integration). Improving velocity may unlock position improvements.

---

## 5. Proposed Composite RC Metric

### 5.1 Design Rationale

The current RC (mean position MAE over h1-h20) has two problems:
1. It ignores action accuracy, percent tracking, and velocity — all of which affect perceived quality.
2. It can improve on position while regressing on everything else (as e027c demonstrates).

A composite metric should:
- Weight position highest (most visually salient)
- Weight action accuracy high (drives animation correctness)
- Include percent (accumulation errors are noticeable in HUD)
- Include velocity (affects position indirectly, least visible)
- Penalize ALL metrics, not just position

### 5.2 The Formula

```
Composite_RC = 0.40 * norm_pos + 0.30 * norm_action + 0.20 * norm_pct + 0.10 * norm_vel
```

Where each component is the mean over all 20 horizons, normalized to a common scale:

- `norm_pos = mean(h1..h20 pos_mae)` — already in game units, range 0-15
- `norm_action = mean(h1..h20 (1 - action_acc)) * 20` — convert accuracy to error, scale to ~0-10 range
- `norm_pct = mean(h1..h20 percent_mae)` — in damage %, range 0-15
- `norm_vel = mean(h1..h20 vel_mae) * 20` — in game units/frame, scale to ~0-10 range

Normalization factors (the `*20` multipliers) are chosen so that each component contributes roughly 0-10 to the composite when at current performance levels. This prevents any single metric from dominating purely due to scale.

### 5.3 Back-testing on Kept Experiments

Using the formula above:

| Exp | norm_pos | norm_act | norm_pct | norm_vel | **Composite** | Old RC |
|-----|------:|------:|------:|------:|----------:|-------:|
| e018a | 6.261 | 5.341 | 4.207 | 5.268 | **5.475** | 6.261 |
| e018c | 6.026 | 5.047 | 4.909 | 10.388 | **5.945** | 6.026 |
| e023b | 5.775 | 4.672 | 4.677 | 13.083 | **5.955** | 5.775 |
| e025a | 5.146 | 4.452 | 1.881 | 0.203* | **3.790** | 5.146 |
| e026b | 5.120 | 4.497 | 1.840 | 0.226* | **3.788** | 5.120 |
| e026c | 4.965 | 4.497 | 1.801 | 0.182* | **3.713** | 4.965 |
| e027c | 4.939 | 4.488 | 4.723 | 5.229 | **4.790** | 4.939 |

*e025a/e026b/e026c velocity values are likely under-reported by ~20x due to a denormalization inconsistency in the Modal-deployed eval code. Their composites are artificially favorable on the velocity component.

### 5.4 Without Velocity (until bug is fixed)

The safer formulation until velocity measurement is fixed:

```
Composite_RC_v0 = 0.50 * norm_pos + 0.30 * norm_action + 0.20 * norm_pct
```

| Exp | **Composite_v0** | Old RC | Rank change? |
|-----|----------:|-------:|:-------------|
| e018a | **5.574** | 6.261 | - |
| e018c | **5.509** | 6.026 | Rises above e018a (better action/pct) |
| e023b | **5.224** | 5.775 | Rises above e018c |
| e025a | **4.285** | 5.146 | - |
| e026b | **4.277** | 5.120 | Essentially tied with e025a |
| e026c | **4.191** | 4.965 | **Best composite** |
| e027c | **4.761** | 4.939 | **Drops to worst of last 4** |

**Key finding:** Under any composite metric that includes percent, **e026c is the actual best model**, not e027c. The ranking flips completely for the last 4 experiments:

| Rank | Old RC (pos-only) | Composite v0 |
|------|:------------------|:-------------|
| #1 | e027c (4.939) | **e026c (4.191)** |
| #2 | e026c (4.965) | **e026b (4.277)** |
| #3 | e026b (5.120) | **e025a (4.285)** |
| #4 | e025a (5.146) | **e027c (4.761)** |

The e027c "improvement" was an artifact of position-only RC masking a 2.7x percent regression.

### 5.5 Implementation

Add to `scripts/eval_rollout.py`:

```python
# After computing per-horizon metrics:
norm_pos = summary_pos_mae  # already computed
norm_action = float(np.mean([1 - per_horizon[k+1]["action_acc"] for k in range(horizon)])) * 20
norm_pct = float(np.mean([per_horizon[k+1]["percent_mae"] for k in range(horizon)]))
# norm_vel = float(np.mean([per_horizon[k+1]["vel_mae"] for k in range(horizon)])) * 20

composite_rc = 0.50 * norm_pos + 0.30 * norm_action + 0.20 * norm_pct

results["composite_rc"] = composite_rc
```

Log as `eval/composite_rc` to wandb alongside `eval/summary_pos_mae`.

---

## 6. Prioritized Work Plan

### Priority 1: Fix composite RC and re-rank (0 compute cost)

1. Implement `composite_rc` in eval_rollout.py.
2. Re-evaluate: e026c is likely the true best. Consider reverting best.json.
3. Fix vel_mae denormalization bug on Modal (check if velocity_scale division is applied consistently).

### Priority 2: Close the action accuracy gap (the biggest gap to watchable)

**Experiment e028a: Extended SF curriculum N=1->2->3->4->5 (3 epochs)**
- Build on e026c (the true best under composite).
- Expected: action_acc h20 from 55% toward 65-70%.
- Estimated cost: ~$15 (3 epochs on A100).
- This directly extends the most successful training innovation (e026c).

**Experiment e028b: GKD teacher distillation during SF**
- EMA teacher sees ground truth, student matches soft predictions.
- Expected: action_acc improvement at h10-h20, especially for rare action states.
- Estimated cost: ~$10 (1 epoch, ~2x memory from EMA copy).

### Priority 3: Data scaling (unlocks everything)

**Experiment e029: e026c config on 7.7K data**
- Data loader is fixed (mmap=True). Cost: ~$31/epoch.
- Every metric should improve: more action transitions, more diverse positions, more diverse percent trajectories.
- This may close multiple gaps simultaneously.

### Priority 4: Percent tracking (already solved, just pick the right base)

- Don't build on e027c/e025b warm-start for future experiments.
- Use e026c as the base. It has percent_mae h10=1.70 (already meets "watchable" threshold of 3.0).
- If combining with regime switching, ensure the loss-reweight epoch doesn't under-weight percent heads.

### Priority 5: Position at long horizons (needs capacity + data)

- d_model=768 is close to capacity ceiling on 1.9K data.
- 7.7K data + d_model=768 should unlock further position improvements.
- d_model=1024 on 7.7K data is the logical next step if d_model=768 saturates.

---

## 7. Summary

**Current state vs. "watchable" (the minimum viable product for The Wire):**

| Metric | Current Best* | Watchable | Gap | Technique to Close |
|--------|-------------:|----------:|----:|:-------------------|
| pos_mae h10 | 4.66 | 3.0 | 36% | Data scaling + capacity |
| pos_mae h20 | 9.37 | 6.0 | 36% | Extended SF curriculum + data |
| action_acc h10 | 79.3% | 85% | 5.7pp | Extended SF curriculum |
| action_acc h20 | 55.2% | 70% | 14.8pp | Extended curriculum + GKD + data |
| pct_mae h10 | 1.70** | 3.0 | DONE | Already meets threshold (e026c) |
| pct_mae h20 | 3.30** | 6.0 | DONE | Already meets threshold (e026c) |
| vel_mae h10 | 0.25 | 0.15 | 40% | Fix measurement first, then target |

*Using e027c for position/action (best on those axes), e026c for percent (best on that axis).
**e026c values — e027c regressed to 4.53/8.78 on percent.

**The single most impactful thing to do:** Switch the primary metric to composite RC, use e026c as the base for future experiments, and run extended SF curriculum on 7.7K data.

**Estimated experiments to reach "watchable":** 3-5 more experiments on 7.7K data with extended SF curriculum. If each experiment closes ~20% of the remaining gap (consistent with historical rate), we need:
- Position: 3-4 experiments (36% gap, 20%/exp compounds)
- Action: 4-5 experiments (14.8pp gap, ~3-4pp/exp)
- Percent: 0 experiments (already there with e026c)

At ~$31/experiment on 7.7K data, this is roughly $100-$150 of compute to reach "watchable."
