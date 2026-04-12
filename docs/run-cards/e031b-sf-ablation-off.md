---
id: e031b
created: 2026-04-12
status: kept
type: training-regime
base_build: b002
built_on: [e028a, e031a]
source_paper: null
rollout_coherence: 4.9643
rollout_coherence_k5: 1.4874
prior_best_rc: 4.798
prior_best_rc_k5: 1.446
---

# Run Card: e031b-sf-ablation-off

## Goal

Ablation: how much quality does Self-Forcing buy at the current scale (b002 + d_model=768 + context_len=30)? SF adds ~55% of wall-clock time per the e031a profile. We need to know whether that cost is load-bearing before we commit to it for the rest of the e031 speed push, or whether it's a hold-over from a smaller regime that newer scale has made redundant.

This is the first ablation run since e028a became b002's canonical best. Its real job is to establish a **reference point on the speed/quality frontier** — a cheap, known-quality "SF off" data point that every downstream e031 experiment can compare against.

## What Changes

Identical to e028a-full-stack except:

- `self_forcing.enabled: false` (was on, ratio=4, curriculum=[1,2,3])
- `profile: true`, `gpu_resident: true` — inherited from e031a

Everything else identical: Mamba2, d_model=768, n_layers=4, context_len=30, unimix_ratio=0.01, lr=5e-4, batch_size=512, 2 epochs, A100 + AMP.

## What actually happened

Run completed 2026-04-12. 2 epochs, 6567 s per epoch (~109 min), ~3.6 hours wall clock total including val + rollout eval + checkpoint writes. Clean completion — no crashes, no restarts.

**Profiling (e031b captured its own profile, confirming e031a's):**

```
data loading:      66.6 ms  (  0.0%)   0.02 ms/batch
forward pass:  198640.9 ms  ( 36.1%)  65.13 ms/batch
backward pass: 340260.0 ms  ( 61.8%)  111.56 ms/batch
optimizer:      11462.0 ms  (  2.1%)   3.76 ms/batch
TOTAL:         550429.4 ms  (3050 TF batches + 0 SF batches)
avg ms/batch:  180.47
```

With SF removed the per-batch cost is ~180 ms (vs ~405 ms for e031a's SF-on profile at ~56% SF overhead). Data loading is 0% — GPU-resident data path is working exactly as designed. Wall clock is ~40% lower than a matched e028a-style SF run would be.

## Results

Formal rollout eval via `modal_train.py::eval_checkpoint`, 300 samples from the 2K val split, seed=42, K=20 horizon, L4 GPU, $0.60. Same code path that produced the retroactive e028a and e029a numbers, so comparisons are apples-to-apples.

### Per-horizon (K=1 through K=20)

| h | pos_mae | vel_mae | action_acc | pct_mae |
|---|---------|---------|------------|---------|
| 1 | 0.53 | 0.00 | 0.992 | 0.22 |
| 5 | 2.46 | 0.01 | 0.887 | 1.10 |
| 10 | 4.71 | 0.01 | 0.737 | 1.81 |
| 20 | 9.25 | 0.02 | 0.527 | 3.46 |

### Summary suite

Computed from the per_horizon dict (means over t=1..K), same convention as e028a and e029a retroactive closeouts.

| Metric | K=5 | K=10 | K=20 |
|---|---|---|---|
| pos_mae | **1.4874** | 2.6486 | 4.9643 |
| action_acc | 0.9437 | 0.8762 | 0.7469 |
| percent_mae | 0.6058 | 1.0642 | 1.9032 |
| vel_mae | 0.0032 ⚠ | 0.0056 ⚠ | 0.0099 ⚠ |

⚠ vel_mae numbers are suspiciously low (same range as e028a's retroactive eval, two orders of magnitude below e029a's 0.12–1.27). Flagged as a known denormalization/config mismatch on the 2K eval path — not load-bearing for the go/no-go, but not comparable cross-run until fixed.

### Matched comparison to e028a-full-stack

Both evals use the same code path, 300 samples, seed=42, 2K val split, K=20 horizon, same eval JSON format. e028a ran the full b002 stack (SF on, ratio=4, curriculum=[1,2,3]) for 2 epochs; e031b differs only in `self_forcing.enabled: false`.

| Metric | K=5 e031b / e028a | K=10 e031b / e028a | K=20 e031b / e028a |
|---|---|---|---|
| pos_mae | 1.487 / 1.446 (**+2.9%**) | 2.649 / 2.565 (**+3.3%**) | 4.964 / 4.798 (**+3.5%**) |
| action_acc | 0.944 / 0.962 (**−1.8pp**) | 0.876 / 0.904 (**−2.8pp**) | 0.747 / 0.775 (**−2.8pp**) |
| percent_mae | 0.606 / 0.614 (**−1.3%**) | 1.064 / 1.086 (**−2.0%**) | 1.903 / 1.966 (**−3.2%**) |

### What SF was actually buying — and what it was costing

**Position tracking:** SF is the winner by a consistent ~3% across horizons. Real but small. This is the clearest case for keeping SF.

**Action accuracy:** SF wins by ~2–3pp across horizons. Smaller than the single h=10 number I initially keyed on (that number was −4.0pp in the wandb snapshot but the K-averaged summary is milder). Still a real win for SF, just smaller than it first looked.

**Percent tracking:** **SF was actively hurting percent tracking.** e031b (no SF) is **better** at every horizon — 1.3% at K=5, 2.0% at K=10, 3.2% at K=20. This is the single most surprising finding of the ablation. SF's consistency pressure on the unrolled head apparently interferes with the percent-tracking head's learning dynamics, probably because percent is a step-function signal and the SF consistency loss smears predictions across time steps in a way that suits continuous fields but penalizes sharp transitions.

**h=1 is identical** — 0.53 vs 0.52 pos_mae, 99.2% action_acc on both. SF's effect (positive or negative) only manifests once the model has to unroll.

### The Layer A / Layer B framing, revised

This matches the pattern we've now seen twice — on the JEPA side (`docs/jepa-direction-notes/smooth-physics-vs-game-rules.md`) and on the e029a 7.7K data scaling comparison — that continuous physics (Layer A) and discrete game-rule fields (Layer B) respond to training-regime changes in opposite directions. Here it shows up a third time, on a fully held-constant architecture, with only the SF flag flipped:

- **Layer A (pos_mae):** SF helps, consistently small margin
- **Layer B discrete (action_acc):** SF helps, modest margin at mid-horizons
- **Layer B step-function (percent_mae):** SF HURTS, consistently

The step-function vs smooth distinction within Layer B matters. Action state changes are smooth-ish — they stick for many frames, then transition. Percent is a pure step function with long runs of zero derivative followed by instantaneous jumps. SF's consistency loss wants temporal smoothness; that's good for pos/action but bad for percent.

If anything this ablation strengthens the framing from the JEPA notes: there is no single "good" training regime across field types. Different supervision shapes for different field shapes.

### Reconciling with the in-loop wandb numbers

The trainer's in-loop rollout eval at end-of-training reported `K=20 4.9643 | K=5 1.4874 | K=10 2.6486` — matching the formal eval exactly (as it should, same code path). The wandb per-epoch eval reported `h10_pos_mae 4.71`, `h10_action_acc 0.737`, `h10_percent_mae 1.81` — these are the per-horizon h=10 row, not K-summaries, and they also match.

**Closeout-writing mistakes I made along the way, captured so future-me doesn't repeat them:**

1. **First framing: "SF is dead weight"** — based on cost hand-waving, no data. Wrong in direction (SF helps pos_mae and action_acc, consistently).
2. **Second framing: "SF earns its keep"** — based on the `h10_action_acc 0.737` vs e028a's 0.787 gap alone, extrapolated as "−5pp everywhere." Wrong in magnitude. The K-averaged action_acc gap is actually −1.8 to −2.8pp across horizons, not −5pp.
3. **Third framing: "only 3% regression"** — based on the trainer's K-summary pos_mae output. Right on pos_mae but missed that action_acc regressed more at mid-horizons AND that percent_mae actually *improved* without SF.
4. **Final framing (this card): the Layer A / Layer B / step-function-vs-smooth regime tradeoff** — the one that actually holds when you look at the full matrix.

The through-line failure was celebrating or regretting on a single scalar. The new `/experiment-complete` skill now requires leading with the K=5/K=10/K=20 multi-metric suite for exactly this reason.

## Decision

**Kept as ablation reference.** Not a new best (+2.9% K=5 pos_mae vs e028a), but the single most informative result since e028a was promoted. It establishes a quantified trade-off instead of a hand-wave: **SF buys ~3% on position and ~2–3pp on action at the cost of ~2–3% on percent tracking and ~40% of wall clock.** Every downstream speed-push experiment now has a concrete reference for the quality floor.

### What this enables

1. **Two explicit points on the speed/quality frontier.** e028a (SF on, ~5.5 hours, the pos/action optimum) and e031b (SF off, ~3.6 hours, the percent optimum). Future screening runs pick their trade-off with numbers instead of intuition.
2. **SF is demoted from "canonical" to "regime-specific."** Its role depends on which field family you care about. For pos/action optimization: keep it. For percent optimization or general iteration velocity: drop it.
3. **The e031 push's primary frame locks in as iteration velocity, not cost parity.** Goal: **sub-2-hour screening experiments** so we can run 3–4× more ablations per day. Any recipe over that budget is disqualified at config-selection, not at result interpretation. e031b at 3.6 hours is already borderline; e031c targets ~1.5 hours.

### What this rules out

- **"SF is dead weight"** — it consistently helps position and action, just modestly.
- **"SF is load-bearing"** — 3% pos_mae is not a rout; it's a trade-off.
- **"Layer A always wins together with Layer B"** — percent is a step-function Layer B field and it responds to SF *opposite* to how action (smooth-transition Layer B) does. The split-within-Layer-B is real.
- **"One scalar is enough"** — see the reconciliation section above for the three wrong conclusions I almost shipped off single-metric views.

## Speed push framing (forward-looking)

This closeout also fixes a confused time-budget expectation I was carrying. The e031 series is about **iteration velocity**, not cost parity: the goal is sub-2-hour screening experiments so we can run 3–4× more ablations per day. Any recipe that takes >2 hours is disqualified at the config-selection step — a "clean but unusable" answer is just an unusable answer.

e031b at 3.6 hours is already borderline. e031c (batch=2048, 2 ep, no SF) targets ~1.5 hours as the new screening baseline candidate.

## Cost

- Training: ~$6 A100 (6567 s × 2 epochs ≈ 3.6 hours)
- Formal eval: $0.60 L4
- Total: **~$7**

## References

- Config: `experiments/e031b-sf-ablation-off.yaml`
- Checkpoint: `/data/checkpoints/e031b-sf-ablation-off/best.pt` on the Modal volume
- Eval JSON: `/data/checkpoints/e031b-sf-ablation-off/eval_rollout.json`
- wandb: `shinewave/melee-worldmodel/runs/cpgfxnjm`
- Launch commit: `85d252b` (fix) / launched from that branch
- Companion profile: `docs/run-cards/e031a-speed-profile.md`
- Follow-up: `experiments/e031c-batch2048.yaml`
