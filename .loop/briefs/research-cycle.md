# Research Cycle Brief

Run one cycle of the autoresearch loop: hypothesize → review → execute → evaluate.

## Context

Read `program.md` for research directions and current state. Read `.loop/state/budget.json` for spend limits.

## Cycle Steps

### Step 1: Hypothesize
Spawn a **hypothesis** agent. It reads program.md, recent run cards, and source papers. It returns a structured hypothesis with claim, mechanism, falsification, and a draft run card.

### Step 2: Director Review
Spawn a **research-director** agent. It evaluates the hypothesis against the quality bar: single-variable, falsifiable, not a dead end repeat, correct cost tier. Returns APPROVE, REVISE, or REJECT with reasoning.

If REVISE: feed the Director's feedback back to the hypothesis agent for one revision. If still not approved, REJECT.

If REJECT: log the rejection and reasoning to `docs/RESEARCH-LOG.md`. This cycle is done — the rejected hypothesis is still valuable data.

### Step 3: Execute
If approved, spawn an **executor** agent with the approved run card. It writes the config, launches on Modal, monitors batch heartbeats, captures results, and updates the card.

Wait for the Modal run to complete. This takes 30-60 min for Scout experiments on 1.9K data.

### Step 4: Evaluate
Spawn a **research-director** agent with the results. It evaluates whether the improvement is real, checks the divergence curve, identifies confounds. Returns KEPT or DISCARDED with reasoning.

### Step 5: Close Out
- Update run card (the executor already did the mechanical part; the director adds the evaluation)
- Run `python scripts/docs_prebuild.py`
- Log to `docs/RESEARCH-LOG.md`
- Update `.loop/state/budget.json` with actual cost
- If KEPT: the Director proposes program.md updates (queued for Mattie)

## Budget Gate

Before launching Step 3, check `.loop/state/budget.json`:
- If `daily_spent + estimated_cost > daily_limit`: queue the experiment, don't run
- If `weekly_spent + estimated_cost > weekly_limit`: escalate to Mattie

## Completion

This cycle is complete when:
- A hypothesis was proposed (even if rejected — that's a valid outcome)
- The run card is updated with results or rejection
- RESEARCH-LOG.md has an entry
- Budget state is updated

## Cadence

One cycle per conductor heartbeat. The conductor decides whether to start a new cycle or wait based on: budget remaining, experiments in flight, time of day.
