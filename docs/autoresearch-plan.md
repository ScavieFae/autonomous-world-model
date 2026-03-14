# Autoresearch Plan

Autonomous overnight experiment loops for the world model. Based on Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) pattern, adapted for our setup.

Research notes and source analysis: `~/claude-projects/rnd-2026/autoresearch/`

## What This Is

An agent runs experiments in a loop overnight. It reads prior results, hypothesizes a change, trains for a fixed time budget, evaluates, keeps or discards, writes a run card, repeats. No human in the loop.

The human's only input is a `program.md` — a prose document describing what good research looks like for this project. The agent handles everything else.

## What We Need to Build

Three things, in order of priority:

### 1. Rollout Coherence Eval

**Why:** Our current metrics (val_loss, action_change_acc) are teacher-forced. They measure "given the exact right state, predict next state." Onchain, the model runs autoregressively — errors compound. We've repeatedly seen models that score well on TF metrics but produce obviously wrong AR rollouts (see e017c/d: TF metrics barely moved, but characters oscillate and heads decouple).

**What:**

```python
def rollout_coherence(model, val_data, N=300, K=20):
    """
    Sample N starting frames from val set.
    From each, autoregress K steps feeding predictions back in.
    Measure divergence from ground truth at each horizon.
    """
    divergence = {k: [] for k in range(1, K+1)}
    for start_frame in sample(val_data, N):
        state = start_frame
        for k in range(1, K+1):
            ctrl = ground_truth_controller[start + k]  # real inputs
            predicted = model(state, ctrl)
            divergence[k].append(distance(predicted, ground_truth[start + k]))
            state = predicted  # autoregressive
    return {k: mean(divs) for k, divs in divergence.items()}
```

**Output:** Divergence curve (pos_mae, action_acc, vel_mae at t+1, t+5, t+10, t+20).

**Summary metric for the autoresearch loop:** Mean position MAE over horizons 1-20. Single number. Lower is better. This is our val_bpb equivalent.

**Constraints:**
- Must run in ~30-60 seconds (fast enough to run after every experiment)
- N=300, K=20 is a starting point — tune for speed/signal tradeoff
- Controller inputs from ground truth replays (testing physics, not player intent)
- Deterministic: same checkpoint + same starting frames = same score

**Where it lives:** `scripts/eval_rollout.py` — standalone script, callable from training or autoresearch loop. Takes a checkpoint path, returns the divergence curve + summary number.

**Reference:** The e017 series proved this is needed. TF metrics said e017c/d were fine. AR demos showed head decoupling, oscillation, characters yoyoing. The rollout coherence eval quantifies what Mattie currently catches by eye.

### 2. Run Card Citations (`built_on:`)

**Why:** Run cards already reference prior experiments informally ("extending E017a"). Making this explicit lets us trace which findings actually compound — and eventually, lets autoresearch agents read the citation graph to decide what to try next.

**What:** Add a `built_on:` field to run cards:

```markdown
# Run Card: e018a-example

**Created**: 2026-03-10
**Config**: `experiments/e018a-example.yaml`
**Status**: KEPT
**Built on**: e017a, e016
**Rollout coherence**: 0.83 (mean pos MAE, K=20)
**Prior best rollout coherence**: 0.91 (e017a)
```

Fields to add:
- `Built on` — which prior experiments this one builds on (explicit citations)
- `Rollout coherence` — the AR metric, not just TF metrics
- `Prior best rollout coherence` — what we're comparing against

The rest of the run card format stays the same.

**Over time:** The citation chain becomes a graph. Findings cited by many successful experiments have high signal. Findings that get cited but don't lead to rollout improvements are teacher-forced flatterers. An autoresearch agent can read this graph to prioritize directions.

### 3. program.md — The Research Direction Doc

**Why:** The autoresearch agent needs to know *what kind of research to do*. Not which experiments to run — that's the agent's job. What the project cares about, what's been tried, what the constraints are.

**What:** A single markdown file that an autoresearch agent reads before starting. Contains:

- Current best checkpoint and its rollout coherence score
- The eval protocol (train for X minutes, run rollout coherence, compare)
- Research directions worth exploring (from the research diary)
- What's been tried and failed (summary, with pointers to run cards)
- Hard constraints (MPS GPU, memory budget, must remain quantization-compatible for onchain deployment)
- Encoding of taste: "we care about AR rollout quality over TF metrics," "simpler is better if metrics are close," "state observations with hit rates, not editorials"

This file is the human's only lever. Everything else is agent-driven.

**Where it lives:** `program.md` in repo root (matching Karpathy's convention).

## Experiment Workflow: PRs + Run Cards

Each experiment is a PR. The PR branch is the experiment's workspace. The run card is the persistent artifact.

```
Agent reads program.md + recent run cards (merged or open PRs)
    ↓
Creates branch: exp/e018a-description
    ↓
Hypothesizes an experiment (writes YAML config or edits code)
    ↓
Trains for fixed time budget (~10-15 min on M3 Max)
    ↓
Runs rollout coherence eval (~30-60s)
    ↓
Writes run card with built_on citations + rollout coherence score
    ↓
Opens PR with run card as description, then closes it
    ↓
Next experiment reads prior PRs for context, loops
```

PRs are always records, never merges. A successful experiment means "this worked in this context" — not "this should be the new default." Main stays stable.

Canon emerges from the citation graph. If 12 experiments cite "batch halving helps" and 10 of them improve rollout coherence, that finding is canonical — not because anyone merged it, but because the evidence accumulated. Bottom-up consensus, not top-down decree.

This matters for AWM specifically: an experiment that improves rollout coherence on Fox dittos on FD might not generalize to the full roster. The finding needs to be cited and confirmed across multiple contexts before it becomes standard.

PRs give you: branch isolation, reviewable diffs, persistent history for all experiments (kept and discarded), and a natural place for the run card summary. Closed PRs with positive results and closed PRs with negative results are both valuable — both citable.

~5-6 experiments/hour on M3 Max. ~50 overnight. Two machines = ~100/night.

## Integration with simple-loop

The autoresearch loop maps to simple-loop's brief/worker/conductor pattern:

| simple-loop concept | autoresearch equivalent |
|---|---|
| Brief | "Run N autoresearch iterations against current best" |
| Worker | Fresh Claude session: read program.md → hypothesize → train → eval → run card → open PR |
| Conductor | Evaluate: did rollout coherence improve? Close PR either way (PRs are records, not merges). |
| Notifications | Push on new best, on failures, on exhausted brief |

## What NOT to change

- **`prepare.py` / data pipeline** — fixed. The eval must be stable for experiments to be comparable.
- **Run card format** (except adding the new fields above) — append-only history.
- **Existing experiment configs** — they're the record. New experiments get new files.

## Epistemic Standards for Run Cards

State findings as observations with hit rates. Not editorials.

- No: "Weight decay on embeddings is a big deal"
- Yes: "WD 0.001 on embeddings improved rollout coherence in 3/3 experiments (e018a, e017a, e016). WD 0.005 regressed in 1/1."

The citation graph aggregates observations into confidence levels. A finding cited 10 times with 8 improvements has a 0.8 hit rate. That's the epistemic state.

## Future: Stable Builds

As experiments accumulate, a pattern will emerge: certain findings always appear together in `built_on:` chains, nobody re-tests them, they're treated as given. That's a **stable build** — a snapshot of canonical findings that new experiments build on top of.

```
stable-v1: e012 (FD top5 data) + e015 (true SS) + e017a (absolute y) + ...
stable-v2: stable-v1 + e021 (whatever survives next) + ...
```

An experiment cites `built_on: [stable-v3]` instead of listing 15 individual findings. The stable build is a package version for the research.

**How a stable build crystallizes:** Someone (human or agent) looks at the citation graph and identifies findings that always co-occur, have high hit rates, and aren't being questioned. Snapshot them as a named baseline. This is the one moment of editorial judgment in an otherwise bottom-up process.

**Competing stable builds can coexist.** Maybe stable-v3 uses absolute y targets and stable-v3-alt uses all-delta with a different loss function. Both have citation support. Experiments that cite one vs the other test which *package* of assumptions holds up. The canon forks — and that's fine. It resolves when one fork consistently produces better rollout coherence.

This isn't something to build yet. It's what naturally emerges once the citation graph has enough density. Recognize it when it shows up.
