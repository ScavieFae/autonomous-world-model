---
name: experiment-launch
description: Pre-flight checklist for launching a training experiment on Modal — verifies run card, config, data contract, local smoke test, budget, observability, and stop conditions before burning GPU time. Use when kicking off any new experiment, Mamba2 or JEPA. Counterpart to /experiment-complete.
disable-model-invocation: true
user-invocable: true
---

# /experiment-launch — Pre-flight Checklist for a Training Run

Burning GPU time on a run that was doomed before it started is the expensive failure mode. This skill is the pre-flight checklist that catches the boring stuff — missing run card, wrong encoding file, stale config, forgotten wandb secret, H100 when L4 would do — before we pay for it.

**Launch is a gate, not a ritual.** Each step below can produce a "no-go" that stops the process. Don't skip steps to save time; a failed smoke test five minutes before launch saves hours downstream.

## Inputs

You need to know:

- **Experiment id** (e.g., `e030a`)
- **GPU class** (L4 / A100 / H100) — H100 requires explicit sign-off, see Step 6
- **Expected runtime** (hours) — for budget check
- **Dataset** — the pre-encoded `.pt` filename on the Modal volume (e.g., `encoded-e012-fd-top5.pt`)

If the user doesn't give these, ask before proceeding.

## Step 1: Run Card Exists and Is Valid

```
Read: docs/run-cards/{experiment-id}*.md
```

**Gate — all must be true:**

- [ ] File exists
- [ ] Frontmatter has `id`, `created`, `status`, `type`, `base_build`, `built_on`
- [ ] `status` is `proposed` or `running` (not `kept` / `discarded` — those are closed)
- [ ] `id` in frontmatter matches the filename stem
- [ ] `id` is globally unique — check `grep -r "^id: {id}$" docs/run-cards/` returns exactly one match
- [ ] `prior_best_rc` is filled (unless this is the very first experiment in a new lineage — if so, card should state that explicitly)
- [ ] `Launch command` section exists and points to the right `scripts/modal_train*.py`

**Go/no-go**: if any of these fail, stop and fix before proceeding.

## Step 2: Experiment Config Present

```
Read: experiments/{experiment-id}*.yaml
```

**Gate:**

- [ ] File exists
- [ ] YAML parses (use `python -c "import yaml; yaml.safe_load(open('experiments/{id}*.yaml'))"` if unsure)
- [ ] `model.arch` is set — dispatches which Modal script to use:
  - `mamba2` / `mlp` → `scripts/modal_train.py`
  - `jepa` → `scripts/modal_train_jepa.py`
- [ ] `training.num_epochs` matches the run card's epoch policy (instrument-don't-cap vs fixed)
- [ ] `training.batch_size` is reasonable for the GPU class
- [ ] `encoding.*` flags match the run card's data contract section
- [ ] Any flags explicitly called out as "probable levers" in the run card are present and commented

## Step 3: Data Contract Sanity

The most common silent failure: training on the wrong encoded data.

**Gate:**

- [ ] `encoded_file` path is the specific pre-encoded `.pt` the run card specifies — not guessed, not a near-neighbor, not a `max_games`-capped version of a different file
- [ ] Fingerprint, if known, matches prior runs on the same dataset
- [ ] For JEPA: `encoding.lookahead == 0` and `encoding.press_events == false` (the `JEPAFrameDataset` asserts these at init; if the YAML sets either, the run will fail fast but noisily)
- [ ] `modal_train*.py` will run the `saved_cfg` mismatch check at load time (both `modal_train.py:175-184` and `modal_train_jepa.py` have this) — **trust it, but grep the script once to confirm the block is still there**:

```bash
grep -n "Encoding config mismatches" scripts/modal_train*.py
```

Should return a hit in both files. If it doesn't, that's a blocker — the data contract is un-enforced.

## Step 4: Local Smoke Test

Never go to Modal cold. Run a small local pass first to prove the pipeline works end-to-end with the actual code and config. For CUDA-only models (Mamba2 with `chunk_size`), this means shape-preflight only — for JEPA and MLP it means an actual mini-run.

**For JEPA** (`arch: jepa`):

```bash
# Verify imports + shape check via the diagnostic CLI
python -m scripts.run_jepa_diagnostics --smoke --batch-size 32

# If there's a local parsed dataset available, mini-run:
python -m scripts.train_jepa \
    --config experiments/{experiment-id}*.yaml \
    --dataset {local-parsed-path} \
    --max-games 10 \
    --epochs 2 \
    --no-wandb
```

**For Mamba2 / MLP** (`arch: mamba2` / `arch: mlp`):

```bash
python -m scripts.train \
    --config experiments/{experiment-id}*.yaml \
    --dataset {local-parsed-path} \
    --max-games 10 \
    --epochs 1 \
    --no-wandb
```

**Gate:**

- [ ] Imports resolve
- [ ] Shape preflight passes (the Trainer's `_verify_shapes` runs first)
- [ ] Loss is finite on the first batch
- [ ] At least one epoch completes without error
- [ ] For JEPA: diagnostic batch is pulled and per-epoch `swap/`, `probe/`, `emergent/` metrics show up in the history

If the pipeline crashes here, it will crash on Modal too, just more expensively. Fix locally.

## Step 5: Modal Script and Run Name

**Gate:**

- [ ] `--config` points at the exact experiment YAML from Step 2
- [ ] `--encoded-file` matches Step 3
- [ ] `--run-name` matches the experiment id (or a clear variant — `e030a-jepa-baseline`, not `test` or `jepa-1`)
- [ ] `--gpu` matches the plan (default is A100; use L4 for cheap work, H100 needs the sign-off in Step 6)
- [ ] Use `--detach` for anything over ~30 min — otherwise a dropped connection kills the run

**Dry-run the exact command** before running it. No shell variables, no untested substitutions:

```bash
# JEPA example
modal run --detach scripts/modal_train_jepa.py \
    --config experiments/e030a-jepa-baseline.yaml \
    --encoded-file /encoded-e012-fd-top5.pt \
    --gpu A100 \
    --run-name e030a-jepa-baseline

# Mamba2 example
modal run --detach scripts/modal_train.py \
    --config experiments/e031a-something.yaml \
    --encoded-file /encoded-e012-fd-top5.pt \
    --gpu A100 \
    --run-name e031a-something
```

## Step 6: Budget Gate

Read the budget once: daily $30, weekly $150 (see `docs/autoresearch-plan.md` or memory). GPU rates on Modal:

| GPU | Hourly | Use for |
|-----|--------|---------|
| L4 | $0.80 | Smoke tests, eval-only runs, cheap sweeps |
| A100-40GB | $2.10 | Default — fits 2K-7.7K dataset comfortably |
| H100 | $3.95 | 7.7K scaling, high-throughput — **needs Mattie sign-off** |

**Compute the estimate**: `expected_runtime_hours * gpu_rate`. Target $10 or less for an exploratory run, $30 max for a bet.

**Gate:**

- [ ] Estimated cost ≤ run card's stated budget (usually in the Training section)
- [ ] Daily-total projection (this run + anything already running) ≤ $30
- [ ] H100: explicit "go" in-session from Mattie, not assumed
- [ ] For detached runs: check `.loop/state/running.json` if it exists to confirm no conflicting budget reservations

## Step 7: Observability

**Gate:**

- [ ] `wandb-key` Modal secret exists (`modal secret list` if unsure)
- [ ] `WANDB_API_KEY` env var set if running locally
- [ ] wandb project is `melee-worldmodel` (the default in both scripts)
- [ ] Tags include the experiment series (e.g., `jepa`, `e030`) — the scripts do this automatically but double-check
- [ ] Run name is searchable — matches the id so `wandb runs search e030a` finds it

If wandb is unavailable for any reason, the run will still work but metrics won't be visible. For short experiments that's OK; for multi-hour runs, fix wandb first.

## Step 8: Known Blockers

Open `docs/HANDOFF.md` and scan the top 2-3 entries. Anything marked as blocking this experiment?

**Gate:**

- [ ] No open "blocking" items in HANDOFF for this experiment id
- [ ] Any TODOs in the run card's "Known blocking fixes" section are resolved
- [ ] No conflicting in-flight experiments for the same base build / dataset (check `git log --oneline -5` and any running Modal apps)

## Step 9: Watch Plan

Before launching, decide what you're watching and when to intervene. Write this down in your head or in the launch message — you'll thank yourself if the run starts drifting at 3 AM.

- **Primary metrics**: which 2-3 numbers tell you the experiment is working? (RC@5, swap/ditto_cosine_sim, pred_loss, etc.)
- **Stop conditions**: what would make you kill the run early? (loss NaN, swap/ditto > 0.9 after epoch 5, pred_loss plateaued and diverging)
- **First-hour expectation**: by epoch N, we should see metric M in range R. If we don't, something's wrong.
- **Completion expectation**: what's a "success" number vs a "discard" number?

For exploratory runs (new architecture, new base build): the watch plan is usually "does the loss go down" and "do the probes show signal." For sweeps: it's the specific metric the sweep is moving.

## Step 10: Go / No-Go

Summarize pre-flight results in a single message to the user:

```
e030a-jepa-baseline pre-flight summary:

✅ Run card: valid, status=proposed, prior_best_rc=4.798
✅ Config: experiments/e030a-jepa-baseline.yaml, arch=jepa
✅ Data contract: /encoded-e012-fd-top5.pt, mismatch check present
✅ Local smoke: 10 games × 2 epochs, pred_loss finite, diagnostics ran
✅ Modal args: scripts/modal_train_jepa.py, A100, --detach, run-name matches
✅ Budget: ~$5-10 on A100 × 50 epochs, well under $30/day
✅ Observability: wandb-key secret exists, project=melee-worldmodel, tags=jepa,e030
✅ Blockers: none
📋 Watch plan: pred_loss decreasing by epoch 5, swap/ditto < 0.9 by epoch 10, RC@5 by epoch 20

Go / no-go?
```

**Do not launch without an explicit go from the user**, even if all gates pass. Launch is the one action that burns money — it deserves a human in the loop.

After go, run the command and report back the Modal run URL and wandb run URL for live watching.

## Verification

After launch, within the first ~5 minutes:

- [ ] Modal app shows the run as active (`modal app list`)
- [ ] wandb run exists and is logging
- [ ] First batch loss is finite
- [ ] Data loaded correctly (check log for `Dataset: X games, Y frames`)
- [ ] Encoding mismatch warnings absent (or understood)

If any of these are off, you have about 30 seconds to decide whether to kill the run or let it cook. Default: kill and re-plan. Wasted $0.10 is cheaper than 8 hours of corrupted metrics.

## Anti-patterns

Things this skill exists to stop:

- **"Just launch and watch"** — no smoke test, burns $20 to discover a shape mismatch.
- **"Same config as last time"** — the last config was for a different dataset; always verify the encoded file.
- **"Detached overnight without a watch plan"** — wakes up to a diverged run with no reason to look at the logs.
- **"Run name is `test`"** — unsearchable in wandb, can't be closed cleanly by `/experiment-complete`.
- **"H100 because why not"** — costs 5x more than A100, usually unnecessary. Reserve for legitimate scale runs.
- **"Skipping the run card because it's a quick sweep"** — sweeps still need cards. Unrecorded experiments disappear from history.

## Related skills

- `/experiment-complete` — the closeout counterpart. Every launch should eventually feed into a complete.
- `/consult-empirical` — use before launch to check whether this axis has been explored already.
- `/consult-architecture` — use before launch if the experiment touches model structure.
- `/update-program` — after a run is kept, propose program.md updates.
