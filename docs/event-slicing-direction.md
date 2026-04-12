# Event Slicing as a Data Curriculum for Rare State-Machine Events

**Status**: direction note, not a decision
**Last updated**: 2026-04-12
**Related**: [jepa-direction-notes/smooth-physics-vs-game-rules.md](jepa-direction-notes/smooth-physics-vs-game-rules.md) — the diagnosis this note partially addresses

## Tl;dr

A data-level approach to the Layer B (state-machine) gradient starvation problem. Instead of architecture changes or new loss terms, restructure the training distribution so rare events (deaths, stock changes, respawns) appear more frequently per unit of training compute. Extract short windows around Layer B events from the larger (7.7K-game) dataset and inject them into the main (2K-game) training data as additional mini-episodes. The existing sampler handles boundary enforcement automatically; no new architecture or new sampling logic is needed.

This is directly inspired by how frontier robotics world models are trained: episodically, on many short recordings of specific tasks, rather than on continuous multi-hour sessions.

Applies to both Mamba2 and JEPA — the data restructuring is architecture-independent.

**This is a partial fix**, targeting one specific Layer-A gradient (rare-event frequency imbalance) from the smooth-physics diagnosis. It doesn't address MSE smoothness or SIGReg pressure. Stackable with other fixes. Whether it's the binding constraint or just one of several is empirically unknown — the experiment answers that question.

## Context: how frontier robotics labs train world models

The default way to train a world model in the Melee domain — sample continuous 30-frame windows from full matches — is meaningfully different from how robotics labs actually train their world models. The robotics pattern is:

- **Episodic, task-specific data.** Not "record 10 hours of robot operation," but "record 10,000 attempts at picking up a red square." Each episode is short (50-300 frames), focused on one task, with clean start/end boundaries.
- **High-quality labeling and setup.** 2-3 calibrated cameras, task descriptions, success labels per episode.
- **Training on episode batches.** The model sees thousands of short focused episodes instead of hundreds of long multi-task streams.

The reason this works isn't aesthetic — it's the rare-event gradient budget problem under a different name. "Robot sitting idle" is the Layer A of robotics: frequent, smoothly varying, gradient-rich. "Successful grasp on novel object" is the Layer B: rare, structured, gradient-starved in the natural distribution. Episodic training concentrates gradient on the thing you want the model to learn by making every episode be about that thing.

Examples across the frontier-robotics world model lineage: RT-2, NVIDIA GR00T, Google OpenVLA, BridgeData, RoboNet, Open X-Embodiment. None of them train on "a full day of robot operation." They all train on labeled, short, task-specific episodes.

## The Melee application: sub-episodes around rare events

Our situation is closer to a "factory day" than a "pick up a red square" — a Melee match contains many micro-tasks interleaved, ~99% smooth combat with ~1% state-machine transitions scattered throughout. We can't treat each full match as an episode; we'd still have the frequency imbalance.

But we can **extract sub-episodes centered on specific rare events**. The concrete idea:

1. Scan the 7.7K-game dataset for every death event (detectable from stock decrements in the categorical int field).
2. Extract a short window of frames around each death (exact window size TBD — see below).
3. Add those windows to the 2K-game training dataset as additional "games" in the existing flat-tensor layout.
4. The existing sampler draws training examples from the combined distribution, naturally concentrating gradient on death-containing frames without any new sampling logic.

**Scale estimate**: the 7.7K dataset has roughly 7 deaths per match × ~5,700 additional matches (beyond the 2K already in training) ≈ **~40,000 additional death events available for slicing**. That's a significant concentration of Layer B signal that isn't currently being used in training.

### Why short windows from a larger dataset, rather than training on 7.7K directly

The 7.7K dataset exists but has scaling issues: the e029a retroactive closeout showed 7.7K was within noise of 2K on the legacy RC metric, and the e031a profile confirmed data-on-GPU won't fit at 7.7K scale on A100-40GB. **We don't need to train on the full 7.7K — we only need its rare events.** Extracting just the death windows gives us the rare-event signal without the wall-clock and memory cost of training on the full dataset.

This reframes "should we use the 7.7K dataset?" from a yes/no question to a "what parts of it do we use?" question. The interesting answer is: use the parts that are scarce in the main training distribution and are known to be failure modes for our current models.

## Why this works mechanically

The existing sampler already handles arbitrary concatenated datasets via `game_offsets` boundaries. A walk-through of why adding slices doesn't break anything, because "silent bugs in data loading" was a real concern raised in the discussion:

### The invariant

The 2K dataset is *already* all concatenated in memory. `dataset.floats` and `dataset.ints` are single flat tensors with ~1.9M frames from 1988 matches stacked end-to-end. The only signal marking "match boundaries exist" is the `game_offsets` array: `[0, game_1_end, game_2_end, ...]`.

When the sampler builds valid starting indices, it walks `game_offsets` and enforces that `(t - context_len) >= game_start` and `t < game_end - lookahead`. Every training example's context window is fully inside one game, by construction. The sampler refuses to generate indices that straddle game boundaries.

### Why short slices work the same way

Adding death slices is just extending `game_offsets`. A 120-frame slice gets appended to the flat tensors, and a new boundary entry is added. The sampler enumerates valid starts within the slice exactly the same way it does for full matches — `start + context_len` to `end - lookahead`. Every training example drawn from a slice is a 30-frame window fully inside that slice.

**The model cannot see two slices as continuous** because:

1. Mamba2's SSM hidden state is re-initialized for every training example. There's no state carryover between forward passes.
2. The sampler refuses to generate any window that crosses a boundary.
3. The model has no concept of a "game" at all — it only sees 30-frame windows. Whether the window is from a full match or a short slice is invisible to the model, and there's no information channel that would let it be visible.

Storage layout is orthogonal to semantic continuity. This is the same reason the 2K dataset can concatenate 1988 matches into one tensor without the model thinking they're all one match.

### One-line sanity check

A runtime assertion at dataset construction time catches any violations of the invariant:

```python
# Every valid starting index must have its context window fully within one game
assert all(
    t - context_len >= game_offsets[game_containing(t)]
    and t + lookahead < game_offsets[game_containing(t) + 1]
    for t in valid_indices
)
```

Cheap insurance that makes the invariant explicit in code rather than relying on convention. Worth landing regardless of whether event slicing happens — it catches any future regression in the sampler that would silently leak data across boundaries.

## Window size and event structure

A death on FD isn't one frame — it's a specific multi-frame sequence with distinct phases:

1. **Pre-death knockback**: character trajectory heads off-stage after a hit, ~20-60 frames depending on knockback velocity
2. **Blast zone cross**: character position exits the stage bounds at X ≈ ±246 (side blast) or Y ≈ -140 (bottom blast)
3. **Death animation**: ~10-30 frames of silhouette or falling-star animation
4. **Stock decrement + despawn**: stock counter updates, character becomes invisible
5. **Teleport**: character appears on respawn platform at ~(0, 168)
6. **Invincibility**: ~120 frames of post-respawn invulnerability
7. **Resume**: character is actionable again, game continues

The full sequence is ~150-200 frames (~2.5-3.3 seconds at 60fps). Different window choices give the model different training signals:

| Window | What the model sees | What it learns |
|---|---|---|
| 30 before / 30 after teleport (60 total) | Late death animation + teleport + early invincibility | The teleport mechanic itself (observe, not predict) |
| 90 before / 30 after teleport (120 total) | Knockback + blast zone + death animation + teleport + early invincibility | Predict the teleport from pre-death context |
| 60 before / 90 after teleport (150 total) | Blast zone cross + teleport + full invincibility window | Full transition sequence including resume |

The first option is narrow and teaches just the teleport. The second and third teach the model to *predict* the death from its precursors, which is the harder but more valuable skill.

**Starting point for the first experiment**: 90 before + 30 after the teleport = 120-frame slices, biased toward pre-event context. Rationale:
- Centers the slice on the most dramatic moment (teleport)
- Gives enough pre-event frames for the model to learn to predict the death rather than just observe it
- Small enough post-event tail that we're not burning compute on "player is invincible, continues being invincible"
- Tunable based on results of the first run

At `context_len=30`, a 120-frame slice gives 90 valid starting positions (frames 30 through 119). Of those, roughly 30-60 are "predict a frame during the death sequence" and the rest are "predict something smoothly within the slice context." Each slice contributes maybe 20-40 training examples where the target is a Layer B transition frame.

**Rough scale math**: 40K slices × ~30 transition-targeted examples per slice = ~1.2M "learn the transition" training examples from the sliced data. Compared to the 2K training set's ~14K deaths × ~1 transition-targeted example per death (because deaths are scattered across long contexts) ≈ ~14K, that's roughly an 85× increase in direct transition gradient signal.

These numbers are back-of-envelope — the actual ratio depends on window width, number of slices retained after filtering, and exact distribution of death events within slices. But the order of magnitude is the point: this takes a rare-event concentration from the "effectively zero gradient" regime to the "meaningful fraction of training" regime.

## Spatial landmark learning (a separate benefit)

One insight from the discussion: **deaths happen at specific X/Y positions along FD's blast zones**, and the model currently has no explicit representation of this geometry. The stage is encoded as a single 33-class categorical (`stage = 32` for FD), but the spatial extent — where the walls are, where the blast zones are, where the respawn platform is — is implicit information the model has to extract from observing many frames.

More deaths at varied X positions give the model more evidence for the "blast zone is at X≈±246" pattern. With ~14K existing deaths in the 2K dataset, the model *could* learn this regularity from the base data. Whether it actually does depends on whether the 0.05% per-frame frequency gives enough gradient signal to shape the spatial representation. Adding ~40K more deaths at diverse positions gives the pattern ~4× more examples.

**This is a separate learning problem from rare-event frequency.** Rare-event frequency is about the *temporal* mechanism (transitions get more gradient budget for chain learning). Spatial landmark learning is about the *positional* mechanism (the model needs to learn "X=246 is special" as a stable feature of the stage geometry). Both benefit from more death examples, but they're different pressures acting on different parts of the model's representation.

**Implication**: if spatial landmark learning is the binding constraint, window width matters less than total count and positional variety. A narrow window (30+30) from 40K deaths could be genuinely enough to teach blast zone positions even if it's too narrow to teach the full teleport sequence. A wider window teaches both. The first experiment should probably try the wider window; if results are promising, a followup can ablate "does narrow work too?" to isolate which mechanism is doing the work.

## Mix ratio and the real risk

The actual thing to be careful about isn't boundary enforcement — the sampler handles that automatically. It's the mix ratio. Concretely:

- 2K dataset contributes ~20M training examples (1988 games × ~10K valid starts each)
- Adding 40K slices × ~90 valid starts each (at 120-frame width) = ~3.6M training examples from slices
- Slices would be **~15% of total training examples**
- Every slice example has a death event somewhere in its context or target window

Compared to the baseline where ~0.05% of training examples contain a death event, this is a **~300× concentration** on death-relevant training data.

### The distribution shift concern

If we over-oversample, the model develops a prior that deaths happen much more often than they actually do. At inference time on regular (non-slice) matches, this prior could cause **false-positive death predictions** — the model predicting a stock decrement or a teleport when the ground truth shows no death. That's the classic rare-event oversampling failure mode, and it's the main thing to monitor.

**Detection**: on a held-out set of regular matches, count how often the model predicts stock decrements or position teleports that don't happen in ground truth. If the baseline is 0.1% and the post-slicing rate spikes to 5%, we oversampled.

**Mitigation knobs** (in order of preference, least complexity first):
1. **Narrower slices.** Fewer valid starts per death event, same count of death events. Directly reduces mix ratio without new sampling logic.
2. **Fewer slices.** Use a sampled subset of the 40K, not all of them. Same pattern.
3. **Lower mix ratio via explicit sampler weighting.** Adds complexity to the training loop. Avoid if the first two knobs are sufficient.

The first two knobs fall out of how many slices we add to `game_offsets`, not how we sample from them, so no new sampling logic is needed for most of the tuning range.

### The sweet spot is empirically determined

There's no theoretical answer for "the right mix ratio" — it depends on:
- The model's capacity to represent multimodal distributions
- How binding the frequency imbalance actually is vs the other Layer-A gradients
- How much the spatial landmark learning benefit contributes independently

The right approach is to try a conservative mix ratio first (~10-15% — close to what the default "add all slices" config produces) and measure. If the event-conditioned rollout eval shows meaningful improvement without false-positive explosion, consider pushing higher. If it shows nothing, either the mix ratio is too low or frequency wasn't the binding constraint and we need to update toward other fixes.

## What's blocking

**Event-conditioned rollout eval.** We cannot measure success from event slicing with the current rollout eval, because the current eval aggregates position MAE across all frames — event slicing will barely move that number even if it completely fixes death→respawn, because transitions are still <1% of eval frames even in the sliced distribution. Any improvement signal gets washed out by the 99% smooth-combat frames.

We need a metric that specifically scores performance on death/transition frames:

```
eval_rollout.py --split-by-event=death
  → pos_mae_death_present: X   (MAE on frames where a death occurs in context)
  → pos_mae_death_absent: Y    (MAE on frames where no death occurs)
  → death_transition_recall: Z/N   (of N ground-truth deaths, how many did the model predict within tolerance?)
  → death_false_positive_rate: W   (of non-death contexts, how many did the model hallucinate a death?)
```

This is ~50 LOC in `scripts/eval_rollout.py`. **It's blocking for event slicing** because without it we can't tell if the experiment worked. It's also useful independently — it would surface the Layer B gap as a real number on any current or past run, not just event-slicing experiments. Running it against the current Mamba2 baseline (e028a-full-stack) would establish a floor we can compare future runs against.

**Recommendation**: land the event-conditioned rollout eval before committing to event slicing. Run it against the current Mamba2 baseline to establish a floor. Then decide whether the measured Layer B gap warrants the event slicing effort, and whether to push on it or on one of the other mitigations from `smooth-physics-vs-game-rules.md`.

## Implementation outline

Not a detailed plan — just enough to estimate scope. Actual plan should come after the event-conditioned eval lands and tells us what we're solving.

Rough phases, each additive and reversible:

1. **Event detection script** (~50 LOC, runs once, cached)
   - Scan `dataset.ints[:, stocks_col]` across all games, find frames where stocks decrement
   - Filter for "clean deaths" (stocks decrement followed by invincibility within ~120 frames) to avoid disconnects, self-destructs, or early endings with atypical patterns
   - Output: list of `(game_index, event_frame_index)` tuples, cached to disk

2. **Slice extraction** (~40 LOC)
   - For each detected death, pull frames from the raw dataset with configurable `(pre_window, post_window)` around the event
   - Append to a new tensor with its own offset array
   - Optionally cache the slices as their own `.pt` file for reproducibility and reuse across experiments

3. **Merging slices into the main training dataset** (~20 LOC)
   - Concatenate slice tensors onto `dataset.floats` and `dataset.ints`
   - Extend `game_offsets` with the slice boundaries
   - No changes to the sampler or any other training code

4. **Sanity assertion** (~5 LOC, runs once at dataset construction)
   - The one-line check from the "Why this works mechanically" section
   - Makes the boundary invariant explicit in code

5. **Run the experiment** with the merged dataset
   - Same trainer, same sampler, same everything else
   - Compare Layer B metrics (from the new event-conditioned eval) against baseline
   - Watch false-positive rates

**Approximate total scope**: ~115 LOC of additive, testable code changes. Plus the event-conditioned rollout eval (~50 LOC) which is prerequisite but useful on its own.

## What we don't know

Honest list of things this direction leaves uncertain:

1. **Is rare-event frequency the binding constraint?** The smooth-physics analysis identifies three Layer-A gradients (MSE smoothness, SIGReg unimodality, frequency imbalance). Event slicing attacks only the third. If MSE or SIGReg is the binding constraint, slicing alone won't fix death→respawn.

2. **Is the spatial landmark problem binding?** The "learn blast zone positions" story is plausible but we have no direct evidence the current model fails at spatial landmark learning specifically. If the model already represents FD geometry well enough for prediction purposes, the benefit of spatial variety is smaller than the raw numbers suggest.

3. **Does the model generalize from sliced deaths to novel deaths?** Oversampling teaches the model the pattern in the sliced data. Generalization to unseen matchups or blast zone positions is the standard ML concern and not specific to slicing — but it's worth watching.

4. **What's the right window width?** 30+30, 90+30, 60+90 — all plausible, none measured. The first experiment should probably fix a window width and report results; a followup can sweep this axis if the first is inconclusive.

5. **Does the effect compound with other fixes?** Event slicing + event-weighted loss + longer context + supervised heads on stocks could stack multiplicatively (each attacks a different part of the mechanism) or be redundant (once one of them breaks the logjam, the others add nothing). Unknown.

6. **Is 40K slices enough, or do we need to resample them?** Each slice is seen once per training epoch at the default sampler. If the effect requires seeing each transition many times to shape the representation, we might need to cycle the slices multiple times per epoch via explicit oversampling — but that's a knob for later, not the first run.

7. **Is current-Mamba2's Layer B performance actually as bad as we think?** We have anecdotal observations (stocks deducted wrong, no clean respawns seen in rollouts) but no systematic measurement. The event-conditioned eval answers this directly. Until it does, we're guessing at how much room there is for improvement.

## Connection to the smooth-physics framework

Event slicing is a **partial fix** in the levers-vs-fixes framework from `docs/jepa-direction-notes/smooth-physics-vs-game-rules.md`. Specifically:

- It **does** address the rare-event gradient budget (one of three Layer-A gradients identified in the note)
- It **does not** address MSE smoothness rewarding interpolation
- It **does not** address SIGReg pushing toward unimodal Gaussian (only relevant for JEPA, and slicing is architecture-neutral anyway)
- It **does not** address temporal straightness as an emergent property

Because it only attacks one of the three gradients, it's insufficient on its own if the other two are also binding. The value of the experiment is partly the outcome (did event metrics improve?) and partly the diagnostic (which Layer-A gradient was the binding constraint?):

- **If event slicing alone produces Competitive Layer B metrics**: frequency was dominant, and we have a fix. The other two gradients were following, not leading.
- **If event slicing produces modest improvement**: it's one of multiple binding constraints, and we combine with other fixes (event-weighted loss, supervised heads, longer context).
- **If event slicing produces nothing**: frequency wasn't the issue, and we update toward MSE smoothness or SIGReg being the binding constraint. Move to event-weighted loss or structural fixes.

All three outcomes update our model of the mechanism. The experiment is worth running not because it might fix everything but because each outcome narrows the space of what the next fix should look like.

## What this is NOT

**Not architecture-specific.** Applies to both Mamba2 and JEPA. The data restructuring is orthogonal to backbone choice and loss structure. Same slices, same sampler, same expected effect on gradient concentration.

**Not a replacement for other fixes.** Stackable with event-weighted loss, auxiliary supervised heads, longer context windows, hybrid latent architectures. The mitigation list in `smooth-physics-vs-game-rules.md` is additive, not mutually exclusive. This is one knob, and it might need to be pulled alongside others.

**Not a new architecture or a new loss term.** Pure data restructuring. The model code and training loop are untouched except for the dataset construction step — slices go into the flat tensor, game_offsets gets more entries, everything downstream is unchanged.

**Not a proposal to replace the 2K dataset with 7.7K.** It's the opposite — we're extracting just the parts of 7.7K that are scarce in 2K, keeping the main training distribution focused on what already works. The 7.7K-in-full approach has known wall-clock and memory issues from the e031a profile; this sidesteps them.

**Not an RL approach.** The discussion that led to this note also touched on whether state-machine behaviors would be better taught through reinforcement learning after imitation learning. Reasonable question, non-trivial scope, set aside for a separate conversation. Event slicing is a pure supervised-learning intervention that happens before any training-dynamics or loss-shape changes.

**Not yet a decision.** This is a direction note. The event-conditioned rollout eval is the blocking prerequisite — we can't measure whether this experiment would work, or whether it's needed in the first place, without it.

## Status

- **Research note**: this file
- **Prerequisite**: event-conditioned rollout eval in `scripts/eval_rollout.py` (~50 LOC, useful independently)
- **First experiment candidate**: `e032a-event-slicing` or similar on the Mamba2 lineage, after the e031 speed series closes out and the event-conditioned eval lands
- **Key open question for the next discussion**: *what does Mamba2's current death-transition recall actually look like?* — the number that tells us how much room there is for improvement. If it's already high, slicing is solving a problem we don't have. If it's near zero, we have a clear target and event slicing is a plausible cheap intervention to try first.
- **Priority**: medium. Not blocking any current work. Not the highest-priority direction either — the e031 infrastructure speedups and the event-conditioned eval both come first, both unblock multiple downstream experiments beyond this one.
