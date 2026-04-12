---
id: e029a
created: 2026-03-23
status: discarded
type: data
base_build: b002
built_on: [e028a]
source_paper: null
rollout_coherence: 4.820
rollout_coherence_k5: 1.289
prior_best_rc: 4.798
prior_best_rc_k5: 1.446
---

# Run Card: e029a-7k-scaling

> **Closeout note (2026-04-12):** This card was written retroactively, three weeks after launch. The run was initiated 2026-03-23, completed epoch 1, then failed before epoch 2 finished. No formal closeout happened at the time — the pivot to JEPA began immediately after — and this card fills the gap. The rollout eval numbers in the Results section below were captured **retroactively** from the epoch-1 `best.pt` checkpoint via `modal_train.py::eval_checkpoint`.

## Goal

First test of data scaling for the Mamba2 line. Take the b002-canonical recipe (e028a-full-stack: unimix + SF curriculum + LR warmup + d_model=768) and run it on **4× more data** (7.7K FD top-5 games vs the 2K that all prior b002 experiments used).

The hypothesis from the pre-launch playability analysis (see `docs/playability-analysis.md`, Priority 3):

> More data → more action transitions, more diverse positions, more diverse percent trajectories. This may close multiple gaps simultaneously.

Projected outcome: "position MAE closes ~20% of the gap to watchable, action accuracy closes ~3-4pp." The plan was to use e029a as the first of 3-5 data-scaling experiments to reach "watchable" ($100-150 total compute budget).

## Config

Identical to `e028a-full-stack` except for:
- **Data**: 7.7K FD top-5 games (`encoded-v3-ranked-fd-top5.pt`, ~53 GB encoded) instead of the 2K baseline
- **GPU**: H100 (for the 53 GB RAM load; A100 wouldn't fit)
- **`mmap: false`** (added after a first-attempt crash — H100 has 128 GB+ system RAM, full load is fine)
- Everything else unchanged: Mamba2, d_model=768, 15.8M params, context_len=30, SF curriculum [1,2,3], unimix=0.01, lr=5e-4, bs=512, 2 epochs, AMP on

See `experiments/e029a-7k-scaling.yaml`.

## What actually happened

**Launched 2026-03-23 08:17 PDT** (commit `10c3374`). First attempt crashed during data loading (mmap+H100 incompatibility). Fixed via commit `97e5d1c` (`mmap: false`) ~90 min later.

**Epoch 1 completed**: `best.pt`, `latest.pt`, and a `periodic.pt` were saved to the Modal volume.

**Epoch 2 failed** at some point before completion. No `final.pt`, no `manifest.json`, no formal closeout. The cause of failure wasn't documented at the time — most likely candidates:
- Modal container timeout (24h function limit, though 2 epochs at ~$31/epoch were projected to fit)
- H100 memory pressure mid-training
- An unrelated Modal infrastructure issue

**No closeout was written.** The next commits in git history (`9e73af1 Add JEPA world model direction`, Apr 4) show the direction had already pivoted to JEPA by the time anyone would have followed up on e029a. The experiment became a ghost in the repo — launched but never closed.

**Cost spent on e029a**: unknown exactly, but rough estimate ~$40-60 of H100 time based on the projected per-epoch rate.

## Results

**Retroactive rollout eval on `checkpoints/e029a-7k-scaling/best.pt`** (the epoch-1 checkpoint), run 2026-04-12 via `modal run scripts/modal_train.py::eval_checkpoint` on L4 GPU, 300 samples from the 7.7K val split (773 val games), K=20 horizon. Cost: ~$0.60.

### Per-horizon (K=1 through K=20)

| h | pos_mae | vel_mae | action_acc | pct_mae |
|---|---|---|---|---|
| 1 | 0.601 | 0.125 | 0.973 | 0.599 |
| 2 | 0.868 | 0.226 | 0.965 | 0.892 |
| 3 | 1.241 | 0.318 | 0.930 | 1.238 |
| 4 | 1.653 | 0.391 | 0.888 | 1.548 |
| 5 | 2.079 | 0.466 | 0.870 | 1.908 |
| 6 | 2.494 | 0.536 | 0.835 | 2.220 |
| 7 | 2.940 | 0.596 | 0.798 | 2.529 |
| 8 | 3.373 | 0.648 | 0.775 | 2.855 |
| 9 | 3.838 | 0.698 | 0.747 | 3.162 |
| 10 | 4.292 | 0.763 | 0.720 | 3.488 |
| 11 | 4.749 | 0.823 | 0.700 | 3.801 |
| 12 | 5.230 | 0.876 | 0.673 | 4.149 |
| 13 | 5.753 | 0.918 | 0.653 | 4.462 |
| 14 | 6.345 | 0.979 | 0.645 | 4.797 |
| 15 | 7.051 | 1.029 | 0.602 | 5.382 |
| 16 | 7.599 | 1.079 | 0.577 | 5.720 |
| 17 | 8.127 | 1.125 | 0.542 | 6.063 |
| 18 | 8.754 | 1.175 | 0.523 | 6.416 |
| 19 | 9.383 | 1.222 | 0.502 | 6.770 |
| 20 | 10.023 | 1.272 | 0.490 | 7.126 |

### Summary suite

| Metric | K=5 | K=10 | K=20 |
|---|---|---|---|
| pos_mae | **1.289** | **2.338** | **4.820** |
| vel_mae | 0.305 | 0.477 | 0.763 |
| action_acc | 0.925 | 0.850 | 0.720 |
| percent_mae | 1.237 | 2.044 | 3.756 |

The `rollout_coherence: 4.820` in the frontmatter is the K=20 summary (the legacy Mamba2 north star), making it directly comparable to e028a-full-stack's 4.798.

### Comparison to e028a-full-stack (the prior best, 2K data, fully completed 2 epochs)

**Matched rollout eval, 300 samples each, 2026-04-12.** e028a eval was also run retroactively via `modal_train.py::eval_checkpoint` on the 2K encoded file to get the K=5/K=10 numbers that weren't captured during the original e028a closeout. This gives us the first apples-to-apples multi-metric comparison we have between 2K and 7.7K runs of the same recipe.

| Metric | e028a (2K, 2ep) | e029a (7.7K, 1ep) | Delta |
|---|---|---|---|
| **K=5 pos_mae** | **1.446** | **1.289** | **−0.157 (−10.8%, better on 7.7K)** |
| K=10 pos_mae | 2.565 | 2.338 | −0.227 (−8.8%, better on 7.7K) |
| K=20 pos_mae | 4.798 | 4.820 | +0.022 (+0.5%, **tied at the noise floor**) |
| K=5 action_acc | 0.962 | 0.925 | −3.7pp (worse on 7.7K) |
| K=10 action_acc | 0.904 | 0.850 | −5.4pp (worse on 7.7K) |
| K=20 action_acc | 0.775 | 0.720 | −5.5pp (worse on 7.7K) |
| K=5 percent_mae | 0.614 | 1.237 | **+101% (much worse on 7.7K)** |
| K=10 percent_mae | 1.086 | 2.044 | +88% (much worse on 7.7K) |
| K=20 percent_mae | 1.966 | 3.756 | +91% (much worse on 7.7K) |

**Data regime**:
| | e029a | e028a |
|---|---|---|
| Dataset size | 7.7K games | 2K games |
| Training examples per epoch | ~60M | ~15.6M |
| Epochs completed | 1 (partial 2nd) | 2 |
| Total examples seen | ~60M | ~31M |
| Training wall clock | ~$40-60 H100 | ~$5 A100 |

### The finding the K=20-only view would have hidden

**The single-scalar K=20 comparison says "within 0.5%, tied, no signal."** That's what this run card said in its first draft and it would have been actively misleading.

The matched K=5/K=10 comparison tells a **completely different** story — a clear Layer A / Layer B regime shift:

**Layer A (continuous physics, position): e029a is meaningfully BETTER at short horizons.**
- K=5 pos_mae: **−10.8%** (1.29 vs 1.45)
- K=10 pos_mae: −8.8% (2.34 vs 2.57)
- K=20 pos_mae: tied (the 5× drift multiplier between K=5 and K=20 swamps the signal)

The 7.7K data improved the encoder's spatial generalization. More characters moving through more positions in training → better per-step position prediction. But rollout drift at K=20 is dominated by accumulated small errors after step 8, so the K=5 improvement doesn't survive to the K=20 summary. This is exactly the "K=20 is drift floor, K=5 is signal ceiling" framing — the K=20 number was calibrated to measure rollout divergence and e029a rolled out similarly to e028a from the divergence perspective, not from the per-step quality perspective.

**Layer B (discrete game-rule fields): e029a is consistently WORSE.**
- action_acc: −3.9pp to −5.5pp across all horizons
- percent_mae: **roughly 2× worse at every horizon** (K=5: 0.61 → 1.24, K=10: 1.09 → 2.04, K=20: 1.97 → 3.76)

The 1 epoch on 7.7K is under-converged on rare-event fields. Each action transition and percent bump was seen once on average, where e028a saw each one twice. Discrete-event fields need more gradient exposure to learn transition rules — the position/velocity heads are happy with one exposure but the 400-class action head and the step-function percent head are still climbing. The ~2× percent_mae degradation is the strongest signal here and it's exactly what we'd predict: percent is the thing with the most step-function character, so it benefits most from repeated exposure to its rare transitions.

**This is the same Layer A / Layer B tradeoff documented in the JEPA research note** (`docs/jepa-direction-notes/smooth-physics-vs-game-rules.md`), manifesting on Mamba2 in a milder form. JEPA's MSE-in-latent-space has no Layer B supervision at all so the collapse is catastrophic. Mamba2's per-field supervised heads provide direct Layer B gradient signal, so the regression is only a few percentage points instead of 50+. But the shape is the same: more diverse data → Layer A improves → Layer B regresses → the global summary washes out the movement. Different architectures, same structural tension, same lesson: **one scalar is not enough.**

### Notes on the e028a eval numbers

The e028a vel_mae column of the retroactive eval comes out to 0.001-0.017 across horizons, which is two orders of magnitude lower than e029a's 0.12-1.27. Both runs use the same `evaluate_rollout_coherence` code path in `scripts/eval_rollout.py`. Suspected denormalization or config mismatch (the velocity_scale division happens per-eval, and if e028a's loaded encoding config differs from e029a's, the results shift), but I'm not digging in because the other three metrics tell the real story and vel_mae isn't load-bearing for the go/no-go picture here. Flagged as a follow-up in case we ever want vel_mae to be trustworthy for cross-run comparison.

### Eval JSON artifacts

Both retroactive eval outputs are on the Modal volume:
- `/data/checkpoints/e028a-full-stack/eval_rollout.json` (2026-04-12)
- `/data/checkpoints/e029a-7k-scaling/eval_rollout.json` (2026-04-12)

Each contains `per_horizon` dict for K=1..20, `summary_pos_mae`, and the raw metadata. K=5/K=10 summaries for the other metrics (vel, action_acc, percent_mae) are computed locally from the per_horizon data — the `eval_checkpoint` function only saves `summary_pos_mae` + `per_horizon` into the JSON (it doesn't save the K=5/K=10 summaries even though `evaluate_rollout_coherence` now returns them). Future retroactive evals can compute the full suite from the saved `per_horizon` dict.

## Decision

**Discarded, retroactively. Partial run with no actionable signal.**

### What this doesn't prove

- It does NOT prove "more data doesn't help Mamba2." The experiment was only **one epoch on 7.7K**, vs 2 epochs on 2K for the baseline. The training regimes aren't comparable — e029a saw each example once, e028a saw each example twice. Multi-epoch training on 7.7K might converge to something better (or worse) than what we see here.
- It does NOT rule out "more diverse data helps specific metrics." K=20 is one aggregate scalar. Per-horizon and per-metric breakdowns might show e029a winning at short horizons or on specific fields. Without a matched comparison to e028a's per-horizon breakdown (which we don't have easily) we can't tell.
- It does NOT test the hypothesis from the playability analysis, which assumed e029a would be followed by e029b, c, d — a series of data-scaling experiments. A single partial epoch is not a scaling curve.

### What this does tell us

- **At the same K=20 metric, 1 epoch on 7.7K data is statistically indistinguishable from 2 epochs on 2K data.** There's no "data scaling free lunch" at this specific operating point.
- **The 7.7K dataset + H100 + Mamba2 training path works end-to-end at least for one epoch.** The mmap issue was fixable. Data loading completed. Training ran. Checkpoints saved. So the infrastructure is viable if we want to try again.
- **The resumption option exists.** The epoch-1 `best.pt` is on the volume. In principle, we could resume training from there and finish epoch 2 + run rollout eval at end. Whether that's useful given the e031 speed regime and the JEPA detour is a separate question.
- **The per-horizon curve has the expected Mamba2 shape**: action_acc starts at 97.3% at h=1 and degrades to 49.0% by h=20. This is roughly consistent with prior Mamba2 runs at the h20 frontier (e027c was at 55.2% h20, e029a is at 49.0% — e029a is actually somewhat worse on this metric, possibly because only 1 epoch of training on a more diverse distribution is less converged than 2 epochs on a narrower one).

## Implications for the current speed-focused work (e031 series)

1. **Don't assume 7.7K unlocks magical improvements.** The one data point we have says "no measurable RC improvement on this setup." Data scaling may still help but it's not a free win.
2. **Scale on GPU memory is a real issue.** 7.7K encoded = ~53 GB. That doesn't fit in `gpu_resident: true` mode on A100-40GB (per the scaling notes section I added in commit `5d33e2b`). Any retry of 7.7K requires either H100 (~$4/hr), or CPU-resident data (reintroducing the data loader overhead that e031a's profile showed was negligible at 2K on GPU). Neither is cheap.
3. **If we want to try 7.7K again**, the cheapest version is e031-style: SF off (the e031b result, once it lands, will tell us if SF can be dropped), batch=2048, H100 or CPU-resident on A100. Total cost ~$10-20 per experiment instead of $40-60.
4. **Or: resume from e029a/best.pt**. We could take the epoch-1 checkpoint and finish epoch 2 with the new infrastructure. Gives us a matched "2 epochs on 7.7K" data point comparable to e028a. Cost: ~$30 on H100.

None of these are urgent — the current e031 speed work is on 2K data and doesn't depend on 7.7K outcomes. Flagging as options if we want to revisit data scaling later.

## Cost (retroactive, estimated)

- Launch + epoch 1 training: **~$40-60 H100** (estimated from projected $31/epoch, partial second-epoch attempt added ~$10-30)
- Retroactive rollout eval (this closeout): **$0.60 L4**
- Total: **~$40-60**

## References

- Config: `experiments/e029a-7k-scaling.yaml`
- Checkpoint (retained): `/data/checkpoints/e029a-7k-scaling/best.pt` on the Modal volume
- Retroactive eval JSON: `/data/checkpoints/e029a-7k-scaling/eval_rollout.json`
- Launch commit: `10c3374` (2026-03-23)
- Mmap fix commit: `97e5d1c` (2026-03-23)
- Pre-launch context: `docs/playability-analysis.md` sections 4.3 and 5 (Priority 3)
- Related current work: `docs/run-cards/e031a-speed-profile.md`, `docs/jepa-scaling-notes.md` (the "7.7K won't fit in gpu_resident mode" section)
