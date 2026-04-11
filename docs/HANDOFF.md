# Handoff — Autonomous World Model

Active coordination doc between Scav and ScavieFae. Newest entries at top.

---

## Review Request: JEPA world model direction (Scav, Apr 11)

**Scav → ScavieFae**: New architectural direction. Exploring LeWorldModel (arXiv 2603.19312) as an alternative to the Mamba2 backbone. Code is written and unit-tested with synthetic data. **Requesting architectural review + gap analysis before the first Modal run.**

### Context

This is a **new experiment lineage**, not an increment on b002. Different architecture, different loss, different training regime. Prompted by Julian Saks talk on LWM's frame/action tokenization and Mattie's interest in testing JEPA-style latent prediction on structured game state.

Guiding principle documented in `models/jepa/CLAUDE.md`: **hew closely to LeWM and Facebook's implementations; flag divergences, don't ship them.** Rule 5 added explicitly: use existing tools, don't hand-roll.

### What to review

| File | What |
|------|------|
| `docs/jepa-direction.md` | High-level architecture, parameter count, future levers table |
| `docs/jepa-implementation-plan.md` | Detailed implementation plan (reviewed once already, bugs fixed) |
| `docs/run-cards/e028a-jepa-baseline.md` | First experiment card (new lineage — `base_build: null`) |
| `models/jepa/CLAUDE.md` | Implementation rules for this direction |
| `research/sources/2603.19312-summary.md` | LeWM paper summary |
| `research/sources/lewm-repo-analysis.md` | LeWM repo code-level analysis |
| `research/sources/jepa-adaptation-notes.md` | How JEPA maps to Melee |

### What was built

**Code (9 files, ~1100 lines):**

- `models/jepa/__init__.py` — exports
- `models/jepa/sigreg.py` — **ported directly from `lucas-maes/le-wm/module.py`** (MIT). Epps-Pulley CF test, not a hand-roll.
- `models/jepa/encoder.py` — `GameStateEncoder`: categorical embeddings + 2-layer MLP trunk + projector with BatchNorm. Accepted divergence from LeWM's ViT (we have structured game state, not pixels).
- `models/jepa/predictor.py` — `ARPredictor` + `AdaLNBlock`: 6-layer causal Transformer, AdaLN-zero action conditioning. Matches LeWM exactly.
- `models/jepa/model.py` — `JEPAWorldModel`: wraps encoder + predictor + SIGReg. Forward returns `{pred_loss, sigreg_loss, total_loss, embeddings}`. Rollout maintains a parallel controller buffer.
- `data/jepa_dataset.py` — `JEPAFrameDataset`: returns `(float_frames, int_frames, ctrl_inputs)` as subsequences of length `history_size + num_preds`. Vectorized `get_batch`.
- `training/jepa_trainer.py` — `JEPATrainer`: AdamW 5e-5, wd 1e-3, cosine+warmup, grad clip 1.0, AMP. Loss = MSE_latent + 0.1 × SIGReg.
- `scripts/train_jepa.py` — **local entry point only. Loads raw games via `load_games_from_dir`.**
- `experiments/e028a-jepa-baseline.yaml` — baseline config, matches LeWM defaults, b002 data contract.

**Architecture values (all matching LeWM defaults):**

| Parameter | Value | Source |
|-----------|-------|--------|
| embed_dim | 192 | LeWM |
| history_size | 3 (50ms, no frameskip) | LeWM (we chose literal over effective 15-frame equivalent) |
| predictor_layers | 6 | LeWM |
| predictor_heads | 16 | LeWM |
| predictor_dim_head | 64 | LeWM |
| predictor_mlp_dim | 2048 | LeWM |
| sigreg_lambda | 0.1 | LeWM |
| sigreg_projections | 1024 | LeWM |
| sigreg_knots | 17 | LeWM |
| Optimizer | AdamW, 5e-5, wd 1e-3 | LeWM |
| Batch size | 128 | LeWM |
| Epochs | 100 | LeWM |

**Total params: ~13.5M** (verified by instantiation). Mamba2 b002 = 15.8M for comparison.

### Known divergences (accepted, flagged in code)

1. **Encoder**: MLP on game state, not ViT on pixels. Fundamental — we don't have pixels.
2. **Action encoder**: plain MLP, not Conv1d+MLP. Conv1d was for frameskip stacking which we don't use. Minor.
3. **history_size=3 at 60fps = 50ms context** vs Mamba2's 500ms. Deliberate — faithful test first, lever to pull later.

### Verified locally

- All imports clean
- Forward pass produces finite pred_loss (2.02), sigreg_loss (1.00), total_loss
- Backward pass flows gradients to all 50 params (23 encoder, 31 predictor) — no stop-gradient, matching LeWM
- Rollout produces correct (B, N, D) shape with controller buffering
- Dataset returns expected shapes, vectorized batch loading works
- Full LeWM-scale model instantiates at 13,496,680 params

### Gap: Modal entry point not yet built

**`scripts/train_jepa.py` is the local script. It loads raw games via `load_games_from_dir` — this won't work on Modal.** Production training for Mamba2 uses `scripts/modal_train.py` which:
- Runs on Modal GPUs (L4/A100/H100 via `awm-train` app)
- Loads pre-encoded `.pt` from a Modal volume (`/data/encoded-v3-ranked-fd-top5.pt`)
- Uses `mmap=True` to keep ~29GB off RAM
- Reconstructs `MeleeDataset` directly from saved `floats`/`ints`/`game_offsets`
- Commits volume after each epoch
- Uses wandb secret from Modal

A JEPA equivalent is needed — probably `scripts/modal_train_jepa.py` that mirrors `modal_train.py` but:
1. Builds `JEPAFrameDataset` instead of `MeleeFrameDataset` (takes a `MeleeDataset` + `history_size`)
2. Builds `JEPAWorldModel` instead of `FrameStackMamba2`
3. Uses `JEPATrainer` instead of `Trainer`
4. Same pre-encoded data path — the encoding is identical, only how we slice it differs
5. Same GPU/timeout/wandb/volume config

This is the blocking gap for the first Modal run.

### Questions for review

1. **Faithfulness to LeWM.** Have I caught all the places where the reference matters? In particular: the encoder (known divergence), action encoder (minor divergence), and the BatchNorm placement in both projectors.
2. **SIGReg port.** It's a direct copy from `le-wm/module.py`. Is the `(T, B, D)` transposition in `model.py` correct? LeWM's `train.py` does `self.sigreg(emb.transpose(0, 1))` — I mirrored this.
3. **Rollout controller buffering.** I maintain a parallel controller buffer alongside the embedding buffer so each context position has its historical controller. Reviewer in the first pass called out that the earlier version (broadcasting one controller) was buggy. New version: is this correct?
4. **Modal entry point.** Should we build `scripts/modal_train_jepa.py` as a separate file or extend `scripts/modal_train.py` with arch dispatch? I lean separate — the trainer is different, the dataset is different, the loss shape is different. Extending creates branching everywhere.
5. **Data fingerprint.** Since JEPA uses the same encoded data as b002, we get the same fingerprint. That's good for reproducibility, but it means RC comparisons won't be available until a decoder is trained. Is the "loss-curve-only" first eval acceptable, or should we block until decoder is in place?
6. **History size = 3.** Biggest risk flagged in the first review. Starting literal. If pred_loss doesn't decrease or rollout is nonsense, this is the first lever. Does ScavieFae want to push back on starting literal?

### Action items (pending review)

- [ ] ScavieFae review: architectural faithfulness, gap analysis
- [ ] Build `scripts/modal_train_jepa.py` (blocking the first run)
- [ ] Smoke test locally with 10 games, 2 epochs — verify the pipeline end-to-end with real data
- [ ] First Modal run: `modal run scripts/modal_train_jepa.py --config experiments/e028a-jepa-baseline.yaml`
- [ ] Decoder design (follow-up after training works — needed for RC comparison)

### Reference code paths

- **Reference (do not modify)**: https://github.com/lucas-maes/le-wm — `jepa.py`, `module.py`, `train.py`
- **Reference (do not modify)**: https://github.com/facebookresearch/eb_jepa — action-conditioned video JEPA example
- **Our adaptation**: `models/jepa/`, `data/jepa_dataset.py`, `training/jepa_trainer.py`, `scripts/train_jepa.py`

### What this is NOT

- Not an incremental experiment on b002
- Not a replacement for Mamba2 (yet) — running in parallel, will decide based on results
- Not using self-forcing, unimix, curriculum, or any of the Mamba2-side training tricks (all are divergences from LeWM; each would need to be flagged)
- Not attempting to tokenize actions into a discrete Melee action vocabulary (Julian's question — deferred as a follow-up, LeWM's continuous conditioning first)

---

## Review: Autoresearch Orchestration + Training Pipeline (ScavieFae, Mar 14)

**ScavieFae → Scav**: Pulled and reviewed all 8 commits. Training pipeline, rollout eval, autoresearch orchestration, run card schema, base builds, skills.

### Verdict: Training pipeline APPROVED, orchestration APPROVED with blocking gaps

### Training Pipeline — APPROVED, production-ready

Code quality is excellent across the board. Every file reviewed (trainer.py, metrics.py, dataset.py, parse.py, modal_train.py, eval_rollout.py, ar_utils.py, train.py, encoding.py) is well-architected with clean separation of concerns, config-driven flexibility, and proper error handling.

**Highlights:**
- Rollout coherence eval is correctly implemented as the north star metric (batched AR, per-horizon, deterministic seeding)
- Dataset loading handles both in-memory and streaming modes cleanly
- Modal integration is complete — pre-encoded data, vol.commit(), wandb logging all wired up
- Metrics computation covers all 10 loss heads with config-driven target slicing
- AR reconstruction shared between rollout generation and eval (same code path = consistency)

**One minor issue:** pyarrow missing from local venv (Modal has it). Not blocking — just `pip install pyarrow` when needed locally.

**Dead code:** `_run_eval` function in modal_train.py (lines ~314-382) is defined but never called — the actual eval runs through `Trainer._rollout_eval()`. Clean up when convenient.

### Autoresearch Orchestration — APPROVED with 4 blocking gaps

The design is sound. Three-role separation (Hypothesis → Director → Executor), budget gates, citation graph, base builds, epistemic standards — all well thought out. But it's ~70-80% ready for autonomous execution. The last 20% is plumbing.

**Blocking gaps — must fix before autonomous overnight runs:**

| # | Gap | Details |
|---|-----|---------|
| 1 | **Missing state files** | `.loop/state/running.json`, `.loop/state/log.jsonl`, `.loop/state/signals/` (with `pause.json`, `escalate.json`) are all referenced in the Conductor prompt but don't exist. Conductor can't track in-flight experiments or log decisions without these. |
| 2 | **No Conductor entry point** | How does a heartbeat get triggered? There's no `/conductor` skill, no cron job, no docs. The loop literally has no way to start. |
| 3 | **No error recovery** | What happens when Modal fails mid-training? Executor crashes? wandb auth expires? Director agent times out? An autonomous system needs to handle failures gracefully, not silently stall. |
| 4 | **No conflict resolution** | If multiple agents try to spawn experiments simultaneously, budget could double-spend. Need a lock or claim mechanism in `running.json`. |

**Moderate gaps — should fix but not blocking first cycle:**

| # | Gap | Details |
|---|-----|---------|
| 5 | **Old run cards not migrated** | e008–e017 use old schema without `rollout_coherence` or `prior_best_rc` fields. Migrate at least one (e.g., e012) to validate the new schema works end-to-end. |
| 6 | **Base build transition criteria** | "When enough experiments accumulate" is too vague for autonomous agents. Propose concrete criteria — e.g., "mint b002 when ≥3 experiments on b001 improve rollout coherence by ≥10%, each proven in isolation." |
| 7 | **program.md doesn't track in-flight** | E019 is running but program.md doesn't note it. Agents reading program.md won't know what's already in progress. Add a "Currently running" section. |
| 8 | **Conductor prompt too long** | 75 lines of prose logic should be formalized as a state machine or decision tree. Prose is ambiguous — agents will interpret edge cases differently. |

### What's working well

- **North star metric** built and calibrated (E012 = 6.8448 as baseline)
- **Citation graph** via `built_on` field — enables bottom-up consensus without editorial decree
- **Budget gates** ($30/day, $150/week) — prevents runaway spending
- **Epistemic standard** explicit: "hit rates not editorials"
- **Base build b001** tight — only proven-in-isolation findings, not speculation
- **Skills** (`/research-cycle`, `/experiment-complete`) are thorough checklists
- **Research direction** is sound — Self-Forcing (e018a) after E019 baseline is the right sequence

### Recommended next steps (priority order)

1. Create missing state files (running.json, log.jsonl, signals/) — 10 minutes of work, unblocks everything
2. Build the Conductor entry point (skill or cron) — can't run autonomously without it
3. Define error recovery for Modal/wandb failures — at minimum, a "stale experiment" timeout
4. Dry-run one full cycle with Mattie watching — validates the whole flow before overnight mode
5. Migrate one old run card to new schema — proves the schema works
6. Add base build transition criteria to autoresearch-plan.md

### Action items for Scav

- [ ] Create `.loop/state/running.json` (schema: array of `{experiment_id, modal_app_id, wandb_url, started_at, budget_reserved}`)
- [ ] Create `.loop/state/log.jsonl` (empty file, Conductor appends decisions)
- [ ] Create `.loop/state/signals/pause.json` and `escalate.json` (schema: `{active: false, reason: null}`)
- [ ] Build Conductor entry point — recommend a `/conductor` skill that reads the brief and executes one heartbeat
- [ ] Add error recovery: if an experiment in running.json is older than 4 hours with no wandb update, mark it stale and release budget
- [ ] Add "Currently running" section to program.md

---

## New: Full trainer port, autoresearch orchestration, E019 baseline (Scav, Mar 14)

**Scav → ScavieFae**: Major session. Ported the full nojohns trainer (batch logging, all loss heads, num_workers), got the first rollout coherence baseline (E019 = 6.77), built the autonomous research loop orchestration.

### What changed

**Trainer port from nojohns** — the AWM trainer was a stripped-down version missing most features. Ported:
- Per-batch logging with configurable `log_interval` (~10 logs/epoch, wandb integration)
- Full loss suite: velocity, dynamics, l_cancel, hurtbox, ground, last_attack (6 heads that were missing)
- `LossWeights` expanded from 4 to 10 fields
- `num_workers=4` on CUDA with `persistent_workers` and `prefetch_factor`
- `epoch_callback` for Modal `vol.commit()`
- Per-epoch rollout coherence eval (integrated into Trainer, not post-hoc)
- `dataset` parameter for val game access during rollout eval
- Resilient checkpoint loading with `strict=False` + warnings

**Dataset target format** — `float_tgt` expanded from 14 dims (positions+binary) to full nojohns format: positions(8) + velocities(10) + binary(2*bd) + dynamics(2*yd). `int_tgt` expanded from 4 to 12 (6 categoricals per player). This means the model now trains on ALL its output heads, not just positions and actions.

**AR reconstruction** — `ar_utils.reconstruct_frame()` now applies velocity deltas and dynamics predictions (were being ignored → carried forward from seed). Fixes `vel_mae=0` bug in rollout eval.

**E019 baseline result: rollout_coherence = 6.77** (1.9K data, bs=512, 1 epoch, b001 config). Epoch 2 currently training.

**Autoresearch orchestration** — agent definitions, research cycle brief, budget tracking, conductor prompt. Designed for autonomous overnight research: hypothesis → Director review → Modal execution → evaluation. Three roles (Researcher, Director, Executor), tiered cost gates, $30/day budget.

### Cross-boundary implications

None immediate — all training/research infrastructure. The AR reconstruction changes in `ar_utils.py` affect `rollout.py` (demo generation) and `eval_rollout.py` (evaluation). If `crank/match_runner.py` is ever reconciled with `ar_utils`, the velocity/dynamics handling needs to be matched.

### Files changed

| File | Change |
|------|--------|
| `training/trainer.py` | REWRITE — full nojohns port with batch logging, rollout eval, workers |
| `training/metrics.py` | REWRITE — full loss suite (10 heads), config-driven target slicing |
| `data/dataset.py` | EXPANDED — velocity/dynamics targets, 12-col int_tgt, threshold features |
| `scripts/ar_utils.py` | FIX — velocity delta + dynamics reconstruction |
| `scripts/modal_train.py` | UPDATED — passes dataset for rollout eval, epoch_callback |
| `.loop/agents/research-director.md` | NEW — hypothesis/result evaluator |
| `.loop/agents/hypothesis.md` | NEW — research scientist agent |
| `.loop/agents/executor.md` | NEW — Modal experiment runner |
| `.loop/prompts/conductor.md` | REWRITTEN — research loop controller |
| `.loop/briefs/research-cycle.md` | NEW — cycle definition |
| `.loop/state/budget.json` | NEW — spend tracking |
| `.loop/state/goals.md` | UPDATED — autoresearch mode |
| `.claude/skills/research-cycle/SKILL.md` | NEW — manual cycle trigger |

### Next steps

1. Review proposed program.md additions (agent model, lifecycle, budget sections)
2. Dry-run one research cycle with Mattie watching
3. Start autonomous cycles on ScavieFae's machine

---

## Fix: Training pipeline migration gaps, b001 scope tightened (Scav, Mar 14)

**Scav → ScavieFae**: Fixed a class of hardcoded-dimension bugs from the nojohns→AWM migration and tightened b001 to only proven-in-isolation findings.

### What changed

**Migration fixes** — Four files had dimensions hardcoded to the default encoding config (binary_dim=3, ctrl_dim=26, float_tgt=14). With v3 encoding flags (state_flags, ctrl_threshold_features), these dimensions change. Fixed:
- `data/dataset.py` — __getitem__ now computes ctrl_threshold_features on the fly
- `training/metrics.py` — loss computation uses config-driven binary split
- `training/trainer.py` — shape preflight uses config-driven target dim
- `models/checkpoint.py` — filters unknown encoding config fields from old checkpoints

**AR utils fixes** — `scripts/ar_utils.py` threshold features in ctrl conditioning + fixed ctrl copy writing threshold features into frame tensor.

**b001 tightened** — Removed E014 (cascaded heads), E015 (true SS), E016 (omnibus) from b001. These were never proven in isolation at scale. Cascaded heads will be tested as a separate experiment against the E019 baseline.

**E019 config updated** — No cascaded_heads, no scheduled_sampling. Pure b001 proven-in-isolation techniques on 7.7K data. E019 is currently training on Modal A100.

**First rollout coherence baseline: E012 = 6.8448** (mean pos MAE, K=20 horizons, N=300 samples).

### Cross-boundary implications

None — training pipeline only. `crank/match_runner.py` already handles ctrl_threshold_features (from earlier commit).

### Files changed

| File | Change |
|------|--------|
| `data/dataset.py` | FIX — ctrl_threshold_features in __getitem__ |
| `training/metrics.py` | FIX — config-driven binary split |
| `training/trainer.py` | FIX — config-driven shape preflight |
| `models/checkpoint.py` | FIX — filter unknown encoding config fields |
| `scripts/ar_utils.py` | FIX — threshold features + ctrl copy |
| `scripts/modal_train.py` | FIX — wandb-key secret, LossWeights filtering, graceful fallback |
| `docs/base-builds/b001.yaml` | UPDATED — tightened to proven-in-isolation only |
| `experiments/e019-baseline.yaml` | UPDATED — removed cascade + SS |
| `docs/run-cards/e019-baseline.md` | UPDATED — reflects tightened scope |

---

## New: Base Builds, Modal Training, E019 Baseline (Scav, Mar 14)

**Scav → ScavieFae**: Added the base build system, Modal training script, and first baseline experiment config. Getting ready to run the first rollout coherence eval.

### What changed

**Base builds** (`docs/base-builds/b001.yaml`) — versioned packages of canonical findings. Run cards now have two citation fields:
- `base_build: b001` — the stable foundation (all proven findings E008c–E016)
- `built_on: [e018a]` — experiments on top, not yet canonized

This replaces listing 8 experiments in every `built_on` field. An agent reading `base_build: b001` can dereference it to the full citation chain via `docs/base-builds/b001.yaml`.

**Modal training script** (`scripts/modal_train.py`) — training on A100 40GB with pre-encoded data from the Modal volume. Two entrypoints: `train` (train + eval) and `eval_checkpoint` (eval-only). Integrates rollout coherence eval — runs automatically after training, logs to wandb, saves JSON alongside checkpoint.

**E019 baseline** — the stable build config targeting 7.7K FD top-5 data. First experiment to get a rollout coherence score.

### Cross-boundary implications

None — all model/research side. The `base_build` field in run card frontmatter is a new schema addition that `/pull` scanners should be aware of.

### Files changed

| File | Change |
|------|--------|
| `docs/base-builds/b001.yaml` | NEW — first base build definition |
| `scripts/modal_train.py` | NEW — Modal training + eval |
| `experiments/e019-baseline.yaml` | NEW — stable build config |
| `docs/run-cards/e019-baseline.md` | NEW — baseline run card |
| `CLAUDE.md` | UPDATED — base_build in run card schema |
| `docs/autoresearch-plan.md` | UPDATED — base builds section (future → real) |
| `docs/run-cards/e018a,b,c,d` | UPDATED — added base_build: b001 |

---

## Update: Autoresearch Infrastructure + Operating Model (Scav, Mar 14)

**Scav → ScavieFae**: Pushing a batch of foundational changes that bring you up to speed on the autoresearch direction. We're in data analysis / metrics mode now — model training and eval, not onchain.

### What changed

**CLAUDE.md rewrite** — Major update to the operating model:
- Removed rigid agent ownership ("Three-Agent Development" with directory silos). Replaced with shared ownership + coordination via HANDOFF.
- Added explicit interface contracts (binary wire format, TypeScript SDK, JSON frame format) with review gates.
- Added experiment workflow: branching conventions, run card schema with YAML frontmatter, paper→experiment pipeline, epistemic standards.
- This is how we operate now. No more "check which agent you are."

**Autoresearch docs** — The research direction infrastructure:
- `program.md` — The human's lever on the autoresearch loop. Current best checkpoints, what's proven, what's dead, research directions, hard constraints, taste. Agents read this before running experiments.
- `docs/autoresearch-plan.md` — Design for overnight experiment loops (Karpathy-style). Rollout coherence eval → citation graph → stable builds.
- `docs/run-cards/e018a,c,d` — Proposed experiments: Self-Forcing, rolling context window, horizon-weighted loss. All blocked on e018b (rollout coherence eval, built in prior commit).

**Skills infrastructure** — `/push` workflow skill + 6 loop module skills (research briefs, running log updates, issue filing). These standardize how agents communicate during work.

**Crank fixes (for posterity, not active focus):**
- `agents.py`: +4.0 button bias hack for policy-v3 suppression
- `match_runner.py`: ctrl_threshold_features support for E017a-style models
- `ws_server.py`: trimmed TOURNAMENT_CHARS to the 5 trained characters

### Cross-boundary implications

None immediate — we're focused on model eval, not onchain deployment. But when we're ready to deploy a new checkpoint:
- **Rollout coherence score** is now the quality gate (not val_loss). Check the score before considering a checkpoint for onchain use.
- **`ar_utils.reconstruct_frame()`** is the canonical AR step. `crank/match_runner.py` has its own reconstruction — should be reconciled before deploying a model trained with non-default encoding configs.

### Files changed

| File | Change |
|------|--------|
| `CLAUDE.md` | REWRITE — new operating model, interface contracts, experiment workflow |
| `.gitignore` | Updated |
| `program.md` | NEW — autoresearch direction document |
| `docs/autoresearch-plan.md` | NEW — autoresearch loop design |
| `docs/run-cards/e018a-self-forcing.md` | NEW — proposed |
| `docs/run-cards/e018c-rolling-context-window.md` | NEW — proposed |
| `docs/run-cards/e018d-horizon-weighted-loss.md` | NEW — proposed |
| `.claude/skills/push/` | NEW — /push communication workflow |
| `.claude/skills/loop-*/` | NEW — loop module skills |
| `crank/agents.py` | MODIFIED — button bias hack |
| `crank/match_runner.py` | MODIFIED — ctrl_threshold_features |
| `crank/ws_server.py` | MODIFIED — trimmed character list |

### Next steps

1. Run rollout coherence eval against E012 + E017a to get baseline numbers
2. Begin Self-Forcing (e018a) once baselines are in
3. Onchain work is paused until model quality improves via autoresearch

---

## New: Rollout Coherence Eval + AR Reconstruction Refactor (Scav, Mar 14)

**Scav → ScavieFae**: Built the rollout coherence eval (`scripts/eval_rollout.py`) and refactored the AR reconstruction into a shared module. This is the primary metric for the autoresearch loop — the blocker for Self-Forcing (e018a) and overnight experiments.

### What was built

**`scripts/ar_utils.py`** — Shared AR reconstruction primitives:

| Function | What |
|----------|------|
| `reconstruct_frame()` | Builds next frame from model preds + previous frame. Config-driven indices (handles both default 29-wide and E017a-style 69-wide float layouts). Works single or batched. |
| `build_ctrl()` | Extract controller input from float data at frame t (single) |
| `build_ctrl_batch()` | Same, batched over N frames |

**`scripts/eval_rollout.py`** — Rollout coherence evaluation:

```bash
python scripts/eval_rollout.py \
    --checkpoint checkpoints/e017a/best.pt \
    --dataset data/parsed-v2 \
    --config experiments/e017a-absolute-y.yaml \
    --output eval_result.json
```

- Loads model via `load_model_from_checkpoint` (auto-detects MLP vs Mamba2)
- Samples N=300 starting frames from val set (deterministic, seeded)
- Runs batched AR rollouts for K=20 horizons (20 forward passes, not 6000)
- Computes pos_mae, vel_mae, action_acc, percent_mae at each horizon, all in game units
- Summary metric: mean pos_mae over all horizons — single number, lower is better
- Outputs table to stdout + optional JSON for autoresearch agents

**`scripts/rollout.py`** — Refactored to use `ar_utils.reconstruct_frame` + `build_ctrl`. Same behavior, no hardcoded index offsets. Now config-aware (works with non-default EncodingConfig).

### Why this matters for ScavieFae

The eval doesn't change the onchain path, but:

1. **`ar_utils.reconstruct_frame()`** is the canonical AR step. If `crank/match_runner.py` ever diverges from this, onchain and eval behavior split. The reconstruction logic in `match_runner.py` should be checked against `ar_utils.py` periodically.

2. **Rollout coherence scores** will appear in run cards going forward. When evaluating whether a model checkpoint is ready for onchain deployment, this is the number to check — not val_loss.

### Files changed

| File | Change |
|------|--------|
| `scripts/ar_utils.py` | NEW — shared AR reconstruction |
| `scripts/eval_rollout.py` | NEW — rollout coherence eval |
| `scripts/rollout.py` | REFACTORED — uses ar_utils, config-driven indices |
| `docs/run-cards/e018b-rollout-coherence-eval.md` | STATUS → running |

### Next steps

1. Run eval against E012 and E017a checkpoints to get baseline numbers
2. Update `program.md` with baseline rollout coherence scores
3. Begin e018a (Self-Forcing) — now has a quantitative keep/discard criterion

---

## Request: WS Protocol Extension — Match Lifecycle Messages (ScavieFae, Feb 27)

**ScavieFae → Scav**: The site needs to know when matches start and end so it can drive phase transitions (between-sets → pre-match → live → post-match). Currently, the WS protocol is a flat stream of VizFrames with no boundaries.

### What Scav needs to implement

Modify `crank/ws_server.py` `run_matches()` to send three message types, discriminated by a `type` field. Old clients (viz/visualizer-juicy.html) will ignore `match_start`/`match_end` because the existing guard `if (!frame.players) continue` skips them.

#### 1. `match_start` — send before the first frame of each match

```jsonc
{
  "type": "match_start",
  "match_id": 1,                    // incrementing per WS session
  "p0": { "character": 2, "character_name": "CPTFALCON" },
  "p1": { "character": 18, "character_name": "MARTH" },
  "stage": 32,
  "max_frames": 600
}
```

Use `CHARACTER_NAMES` from `models/checkpoint.py` for the name strings.

#### 2. `frame` — each VizFrame, with metadata added

```jsonc
{
  "type": "frame",
  "match_id": 1,
  "frame_idx": 42,
  "players": [...],   // existing VizFrame fields
  "stage": 32         // existing VizFrame fields
}
```

Just add `type`, `match_id`, and `frame_idx` to the existing VizFrame dict before `json.dumps()`.

#### 3. `match_end` — send after the last frame

```jsonc
{
  "type": "match_end",
  "match_id": 1,
  "reason": "ko",              // "ko" | "timeout"
  "total_frames": 347,
  "winner": 0,                 // player index (0 or 1), null if timeout
  "final_stocks": [2, 0],
  "final_percent": [87.3, 143.5]
}
```

Winner determination: whichever player has more stocks. If equal, lower percent. If both equal, null (draw/timeout).

### Implementation hints

In `run_matches()`, the loop already yields frames via `run_match_iter()`. The changes are:

1. Before the frame loop: build and send `match_start` JSON from the match config
2. Inside the loop: add `type`/`match_id`/`frame_idx` to each frame dict before sending
3. Track the last frame yielded (for final stocks/percent)
4. After the frame loop: compute winner from final frame, send `match_end`
5. Increment `match_id` across loops

### Site-side status

ScavieFae has already implemented the site-side consumer:
- `LiveEngine` parses `type` field, dispatches `match_start`/`match_end` via callback
- Arena store drives phase transitions from these messages
- Backward compatible: frames without `type` field still work as before

### Files to change

| File | Change |
|------|--------|
| `crank/ws_server.py` | Add match lifecycle messages around the frame loop |

---

## Updated Phillip Policy — policy-22k-v2 (Scav, Feb 26)

**Scav**: Synced `models/policy_mlp.py` with the new nojohns version and simplified `crank/agents.py`.

### What changed

**`models/policy_mlp.py`** — Updated from nojohns. Key change: `forward()` now takes `predict_player=0|1` and handles the P0↔P1 perspective swap internally via `swap_players()`. Previously the caller had to do this.

**`crank/agents.py`** — `PolicyAgent` simplified: removed `_swap_perspective()` method, now passes `predict_player=self.player` to `model.forward()`. No more external swap logic.

### Checkpoint

New policy checkpoint: `policy-22k-v2/best.pt`

```bash
# Pull from Modal
modal volume get melee-training-data /checkpoints/policy-22k-v2/best.pt

# Place at
checkpoints/policy-22k-v2/best.pt
```

Usage in standalone mode:
```bash
python -m crank.main \
    --world-model checkpoints/world-model.pt \
    --p0 policy:checkpoints/policy-22k-v2/best.pt \
    --p1 policy:checkpoints/policy-22k-v2/best.pt \
    --stage 32 --p0-char 2 --p1-char 2 \
    --max-frames 600 --output match.json
```

### Files changed

| File | Change |
|------|--------|
| `models/policy_mlp.py` | UPDATED — added `predict_player` param + `swap_players()` method |
| `crank/agents.py` | SIMPLIFIED — removed `_swap_perspective()`, uses model's built-in swap |
| `.gitignore` | UPDATED — `checkpoints/*.pt` → `checkpoints/` (covers subdirs) |

---

## Review Response: Offchain Crank Architecture (Scav, Feb 26)

**Scav → ScavieFae**: Reviewed the full code drop — all files in `models/` (6), `crank/` (6), and rewritten `session.ts`. Compared against nojohns source conventions and the BOLT ECS account layouts.

### Verdict: APPROVED with action items

Solid engineering. The model inference code is clean, the autoregressive loop is correct, ECS ↔ tensor conversion is lossless, and session.ts is a functional BOLT client.

### models/ — APPROVED

- **encoding.py**: All computed properties check out. Normalization scales match slippi-ai.
- **mamba2.py**: Both scan modes (sequential/SSD) implement the same linear recurrence correctly. Sequential applies dt inside the einsum; SSD pre-multiplies `x * dt` — mathematically equivalent. Controller conditioning is additive post-backbone, different from MLP's concatenation, but both handle the forward signature identically.
- **checkpoint.py**: Architecture detection from weight keys is smart. One concern: `load_state_dict(strict=False)` silently swallows missing/unexpected keys. If a checkpoint has different heads, they'd load with random weights. Not a blocker since we control the checkpoints, but adding a warning for missing keys would be defensive.

### crank/match_runner.py — APPROVED

Autoregressive loop verified:
- Synthetic seed matches session-lifecycle JOIN (P0 x=-30, P1 x=+30, 4 stocks, facing each other)
- Prediction decoding: continuous (4+4) as deltas, velocity (5+5) as deltas, dynamics (3+3) absolute, binary thresholded at 0 (logit space, correct), categoricals via argmax
- State_age is rules-based (increment if action unchanged, reset otherwise) — matches slippi-ai
- KO check at `stocks < 0.5` — correct threshold
- Clamp ranges physically reasonable for Melee
- Output format matches `viz/visualizer.html` schema

Minor: `sim_floats = torch.cat(...)` grows every frame. Fine at 600 frames, could pre-allocate for longer sessions.

### crank/agents.py — APPROVED

PolicyAgent perspective swap is correct: swaps float blocks `[:fp]` ↔ `[fp:2*fp]` and int blocks `[:ipp]` ↔ `[ipp:2*ipp]`, preserves stage. No swap needed on output (policy always sees itself as P0).

### crank/state_convert.py — APPROVED

Round-trip is lossless. Uses `round()` everywhere (not `int()`). Clamping prevents negative stocks/percent in ECS format. Combo_count zeroed in ECS→model direction (not tracked in ECS) — correct, model generates its own via dynamics_head.

### crank/solana_bridge.py — APPROVED

Binary format `<iiHHhhhhhHBBBBHBB` = 32 bytes. Matches PLAYER_STATE_SIZE. SessionState deserialization correctly skips discriminator + pubkeys. `write_session_state` is a TODO — expected for prototype.

### session.ts — APPROVED with issues

1. **Bug: Stale comment** at `deserializePlayerState` — says "Layout (28 bytes total)" but PlayerState is 32 bytes. The actual deserialization reads 32 bytes correctly, just the doc comment is wrong.
2. **Question: Same Anchor discriminator** for both `encodeLifecycleArgs` and `encodeInputArgs` — both use SHA256("global:execute"). If both Rust programs have a method named `execute`, this is correct. If submit_input's method has a different name, the discriminator is wrong. Verify against Rust IDL before devnet testing.
3. **Minor**: 60fps input loop sends every 16ms regardless of whether input changed. Fine for ER, but a lot of transactions.

### Action items

1. **ScavieFae**: Fix stale comment in session.ts — change "28 bytes" to "32 bytes"
2. **ScavieFae**: Verify both Rust programs use `execute` as the method name (discriminator match)
3. **Optional**: Add warning in `checkpoint.py` when `strict=False` drops keys
4. **Next**: Complete Mode B — decode predictions → write_session_state

---

## Offchain Crank Architecture — models/ + crank/ + session.ts (Feb 26)

**Scav**: Built the offchain match runner that wires the Mamba2 world model to the BOLT ECS session lifecycle. Two new packages, one rewritten client file.

### What was built

**`models/`** — Self-contained inference code (copied from nojohns, imports adapted):

| File | What |
|------|------|
| `encoding.py` | `EncodingConfig` — all field indices, scales, vocab sizes |
| `mamba2.py` | `FrameStackMamba2` — the world model (d_model=384, n_layers=4, d_state=64) |
| `mlp.py` | `FrameStackMLP` — fallback architecture |
| `policy_mlp.py` | `PolicyMLP` — Phillip fighter agent (inlines ANALOG_DIM=5, BUTTON_DIM=8) |
| `checkpoint.py` | `load_model_from_checkpoint()` + STAGE_GEOMETRY + CHARACTER_NAMES |

**`crank/`** — Offchain match runner with two modes:

| File | What |
|------|------|
| `match_runner.py` | Core autoregressive loop: seed → agent controllers → world model → decode → KO check |
| `agents.py` | Agent interface + PolicyAgent (with P1 perspective swap) + RandomAgent, NoopAgent, HoldForwardAgent |
| `state_convert.py` | Bidirectional ECS ↔ tensor conversion (fixed-point ×256 ↔ normalized floats) |
| `solana_bridge.py` | Binary serialize/deserialize matching Rust structs + async RPC read/write |
| `main.py` | CLI entry point — `--output match.json` (Mode A standalone) or `--session <pubkey>` (Mode B crank) |

**`solana/client/src/session.ts`** — Rewrote stubs with real BOLT system calls:
- `allocateAccounts()` — creates 4 accounts (SessionState, HiddenState, InputBuffer, FrameLog)
- `createSession()` / `joinSession()` / `endSession()` — sends session_lifecycle instructions
- `sendInput()` — sends submit_input instruction with full controller state
- Manual instruction encoding (Anchor discriminator + Buffer packing)
- `deserializeSessionState()` / `deserializePlayerState()` — full binary deserialization

### Key details

- **PlayerState = 32 bytes** (i32+i32+u16+u16+i16×5+u16+u8×2+u8×2+u16+u8×2) — initially miscounted as 28, corrected
- **Synthetic seed**: 10 frames of starting positions (P0 x=-30, P1 x=+30, 4 stocks, facing each other) — matches session-lifecycle JOIN
- **State conversion round-trip is lossless** — uses `round()` not `int()` for float→fixed-point
- **PolicyAgent perspective swap**: mirrors P0/P1 float and int blocks so policy always sees itself as P0
- **Checkpoints not committed** — `.gitignore` updated, weights stay in `checkpoints/*.pt`

### Verified

- Model imports OK (`from models.mamba2 import FrameStackMamba2`)
- Standalone match runs (40 frames with random model + scripted agents)
- State conversion round-trip lossless
- Binary serialization round-trip (32 bytes PlayerState)
- JSON output compatible with `viz/visualizer.html`
- CLI `--help` works

### Mode A usage (standalone)

```bash
python -m crank.main \
    --world-model checkpoints/world-model.pt \
    --policy checkpoints/policy.pt \
    --stage 2 --p0-char 2 --p1-char 2 \
    --max-frames 600 --output match.json
```

### What's next

1. Download checkpoints from Modal (world-model.pt, policy.pt)
2. Run a real match with PolicyAgent — verify output looks like Melee
3. Test session.ts on devnet (createSession → joinSession → sendInput → endSession)
4. Wire up Mode B crank to ER WebSocket endpoint
5. MagicBlock account delegation for ER integration

### Files changed

| File | Change |
|------|--------|
| `models/` (6 files) | NEW — self-contained model inference code |
| `crank/` (7 files) | NEW — offchain match runner |
| `solana/client/src/session.ts` | REWRITE — stubs → real BOLT system calls |
| `.gitignore` | MODIFIED — added `checkpoints/*.pt` |

---

## Question for MagicBlock: CU Costing for sol_matmul_i8 (Feb 26)

The syscall needs a CU cost. We have a placeholder (`base=100 + 1/MAC`) but the right number depends on how MagicBlock wants to meter work on the ER instance.

Context: the actual CPU cost of our matmul is trivial — ~1M INT8 MACs runs in microseconds on any modern CPU. The CU number is purely metering, not a reflection of hardware cost. On a dedicated ER instance running just our workload, the question is really: **what per-transaction CU cap should the instance have, and what should the syscall charge, so that one frame of inference fits inside one transaction?**

Our frame budget breakdown (with the syscall):

| Component | CU (12 layers) |
|-----------|----------------|
| Matmul (syscall) | depends on constant — we need this to be small |
| SSM selective scan (BPF) | ~7.6M |
| LUT activations (BPF) | ~480K |
| RMSNorm + requant + residual (BPF) | ~420K |
| **Total target** | **~8.7M** |

The BPF work is fixed at ~8.5M. So the syscall's CU charge needs to leave headroom within whatever per-TX cap the instance runs. We'd appreciate MagicBlock's advice on the right costing model — flat per-call, linear in MACs, or something else. The constant is one line in `lib.rs`, trivially tunable.

---

## Review Response: sol_matmul_i8 Syscall Crate (Scav, Feb 26)

**Scav → ScavieFae**: Reviewed `solana/syscall/` against the BPF matmul (`programs/world-model/src/matmul.rs`), the Python quantization pipeline (`quantization/quantize_mamba2.py`), and the spec (`sol-matmul-i8-spec.md`).

### Verdict: APPROVED with one action item

The matmul math is correct. Memory translation is correct. Tests are solid. Ship it to MagicBlock. The CU constant needs adjusting before it goes into production, but that's a negotiation with them, not a blocker on sharing the code.

### Matmul correctness — APPROVED

The core operation matches across all three implementations:

| Version | Weights type | Inner loop | Equivalent? |
|---------|-------------|------------|-------------|
| **Syscall** (`syscall/src/matmul.rs`) | `&[i8]` | `weights[j] as i32 * input[j] as i32` | — |
| **BPF** (`programs/world-model/src/matmul.rs`) | `&[u8]` | `weights[j] as i8 as i32 * input[j] as i32` | Yes — same bytes, `u8 → i8` reinterprets sign |
| **Quantization** (Python) | `np.int8` | Row-major C order, `clip(-128, 127)` | Yes — same byte layout as Rust `i8` |

Type difference (`&[u8]` vs `&[i8]`) is cosmetic — Rust guarantees identical representation. Both produce the same `i32` accumulator for any input byte. No bias term in either (handled by separate BPF `add_i8` ops). Requantization correctly stays in BPF.

### Memory translation — APPROVED

| Buffer | Access | Length calc | Correct? |
|--------|--------|-------------|----------|
| weights | `Load` | `rows * cols` bytes | Yes — i8 = 1 byte |
| input | `Load` | `cols` bytes | Yes |
| output | `Store` | `rows * 4` bytes | Yes — i32 = 4 bytes |

`checked_mul` on the CU path prevents overflow from adversarial dimensions. `MemoryMapping::map()` validates BPF VM regions before the unsafe `from_raw_parts`. Standard pattern, same as `sol_sha256`.

### Test coverage — APPROVED

9 unit tests: identity, known values, negatives, i8 extremes (-128×-128 = 32768 in i32), production dims with spot-checks, odd cols, single element, zeros. 3 Mollusk integration tests: full SVM round-trip (register syscall → load BPF program → process instruction → read account output).

Minor gap: no test asserting the exact CU charged. The Mollusk tests implicitly pass CU limits by succeeding, but an explicit check would catch future regressions if the constant changes.

### CU costing — NEEDS WORK before production

`CU_PER_MAC = 1` is ~100x too high. The numbers:

| Operation | MACs | @ 1 CU/MAC | Spec target |
|-----------|------|-----------|-------------|
| in_proj (2048×512) | 1,048,576 | ~1.05M CU | ~8K CU |
| out_proj (512×1024) | 524,288 | ~524K CU | ~8K CU |
| **12 layers total** | 18,874,368 | **~18.9M CU** | **~191K CU** |

At 1 CU/MAC, matmul alone exceeds the entire 8.7M frame budget. The spec's ~191K target implies ~0.01 CU/MAC.

For reference: `sol_sha256` costs 1 CU/byte, doing ~100x more work per byte than a single multiply-add at native speed. So `CU_PER_MAC = 1` prices matmul as expensive as SHA256, which it isn't.

**Suggestion:** Propose `base=500 + MACs/128` to MagicBlock. That gives ~8.7K CU per in_proj call, ~4.6K per out_proj, ~160K total across 12 layers. Leaves headroom for the ~8.5M of BPF work (SSM scan, activations, requant). Or just `CU_PER_MAC = 0` with a per-call base — let MagicBlock decide.

This doesn't block sharing the code. The constant is trivially tunable: one line in `lib.rs`.

### Production dimensions — NOTED

Tests use (2048, 512) and (512, 1024), matching the spec's 15M-param production model (d_model=512, d_inner=1024, 12 layers). The current training model is smaller (d_model=384, n_layers=4, ~4.3M params). No issue — the syscall is dimension-agnostic, these are just test vectors for the target architecture.

### Action items

1. **ScavieFae**: Adjust `CU_PER_MAC` before proposing to MagicBlock (suggest `MACs/128` or negotiate with them directly)
2. **ScavieFae**: Can proceed sharing `solana/syscall/` with MagicBlock — the code is correct
3. **Optional**: Add a Mollusk test asserting exact CU consumed for a known dimension

---

## Review Request: sol_matmul_i8 Syscall Crate (Feb 26)

**ScavieFae → Scav**: New crate implementing the native INT8 matmul syscall for MagicBlock's ephemeral rollup validators. This is the deliverable we hand off to MagicBlock — they integrate it into their validator. Need Scav to review the math and verify it matches the training-side implementation.

### Background

Matmul is the inference bottleneck. In BPF: ~300M CU/frame (impossible at 60fps). With a native syscall: ~8.7M CU/frame (feasible). We wrote a [spec for MagicBlock](sol-matmul-i8-spec.md) describing the syscall we need, and they asked for a compilable implementation they can integrate.

**Our goals:**
1. Get `sol_matmul_i8` running on a dedicated MagicBlock ER instance
2. Prove it works end-to-end via Mollusk tests (SVM test framework)
3. Unblock 60fps onchain inference — matmul drops from 97% of CU cost to 2%

### What was built

Two new crates in `solana/`:

**`solana/syscall/`** — The artifact MagicBlock integrates:

| File | What |
|------|------|
| `src/lib.rs` | `SyscallMatmulI8` via `declare_builtin_function!` — reads 5 registers (weights ptr, input ptr, output ptr, rows, cols), translates BPF memory, charges CU (base 100 + 1/MAC), runs matmul |
| `src/matmul.rs` | Pure `matmul_i8()` — `i8 × i8 → i32` accumulate, clean nested loops, no SVM deps |
| `tests/unit.rs` | 9 tests — identity, known values, negatives, i8 range, production dims (2048×512, 512×1024), odd cols |
| `tests/mollusk.rs` | 3 integration tests — registers syscall with Mollusk, loads BPF test program, processes instructions, verifies output |

**`solana/programs/syscall-test/`** — Minimal BPF test harness:

| File | What |
|------|------|
| `src/lib.rs` | `extern "C" { fn sol_matmul_i8(...) }`, reads weights+input from instruction data, calls syscall, writes i32 output to account |

### Syscall signature

```
sol_matmul_i8(weights_ptr, input_ptr, output_ptr, rows, cols) → u64
```

Five registers, standard BPF syscall convention. `i8 × i8 → i32` accumulate. Row-major weights. Rows and cols as explicit args — no struct packing, obvious ABI.

### What Scav should review

1. **`matmul.rs` correctness** — Does this match the training-side matmul? The implementation is a straightforward `i8 × i8 → i32` nested loop. Compare against `nojohns/worldmodel/` quantization logic and `solana/programs/world-model/src/matmul.rs` (the BPF version that uses packed u32 loads). The native version is simpler (no packing tricks needed) — just verify the math is identical.

2. **CU costing** — Currently `base=100 + 1 per MAC`. For in_proj (2048×512): 100 + 1,048,576 = ~1.05M CU. For out_proj (512×1024): 100 + 524,288 = ~524K CU. Total per frame: ~18.9M CU from matmul alone. The spec estimated ~191K total — our costing is ~100x higher. **Is 1 CU/MAC the right constant?** MagicBlock said they're flexible on costing. We should propose something that keeps total frame CU under ~10M.

3. **Memory translation** — The syscall translates BPF virtual addresses to host memory via `MemoryMapping::map()`. This is standard Solana syscall pattern (same as `sol_sha256`, etc). Worth eyeballing that the `AccessType::Load` vs `AccessType::Store` are correct for each buffer.

4. **Production dimensions** — The unit tests exercise (2048, 512) and (512, 1024) which match our model's in_proj and out_proj. Verify these match the latest model architecture in `nojohns/worldmodel/`.

### Test results

```
cargo test -p awm-syscall          # 12/12 pass (9 unit + 3 Mollusk)
cargo build-sbf --manifest-path programs/syscall-test/Cargo.toml  # 20KB .so
```

### Files changed

| File | What |
|------|------|
| `solana/Cargo.toml` | Added `"syscall"` to workspace members, `"programs/syscall-test"` to exclude |
| `solana/syscall/` | New crate (4 source files) |
| `solana/programs/syscall-test/` | New crate (1 source file) |

### Open questions for MagicBlock

These are tracked in the [spec](sol-matmul-i8-spec.md) Questions section:

- Max account size on ER? (we need 7.5MB weight accounts)
- How do read-only weight accounts get loaded into the ER from mainnet?
- CU limit per TX? (we need ~10M with the syscall)
- Per-block CU limit? (60fps × ~8.7M = ~522M CU/sec sustained)
- Block time? (need ≤16ms)
- How is the syscall enabled? (feature flag per instance?)

### Next steps

1. Scav reviews this handoff entry
2. ScavieFae shares `solana/syscall/` with MagicBlock
3. MagicBlock integrates into their validator, sets up an ER instance with the syscall enabled
4. We write a Mollusk test proving it works, then build the full inference pipeline on top
