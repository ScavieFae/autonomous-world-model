---
name: research-cycle
description: Run one autoresearch cycle — hypothesize, Director review, execute on Modal, evaluate. Use to manually trigger a research cycle or to test the loop.
disable-model-invocation: true
user-invocable: true
argument-hint: "[scout|confirm|dry-run]"
---

# /research-cycle — One Hypothesis → Experiment Cycle

Run one cycle of the autoresearch loop. Can be run manually or by the simple-loop conductor.

## Modes

- `/research-cycle` or `/research-cycle scout` — full cycle, Scout tier (<$2)
- `/research-cycle confirm` — full cycle, Confirm tier ($2-10)
- `/research-cycle dry-run` — hypothesis + Director review only, no Modal execution

## The Cycle

### 1. Check Budget
Read `.loop/state/budget.json`. If daily or weekly limit would be exceeded, report and stop.

### 2. Hypothesize
Spawn an Explore agent with the hypothesis agent prompt (`.loop/agents/hypothesis.md`). It reads program.md and recent cards, returns a structured hypothesis + draft run card.

### 3. Director Review
Spawn an Explore agent with the research-director prompt (`.loop/agents/research-director.md`). Pass it the hypothesis. It returns APPROVE / REVISE / REJECT with reasoning.

Present the hypothesis AND the Director's review to the user. If dry-run mode, stop here.

If REJECT: log to RESEARCH-LOG.md. Done.

### 4. Execute (if approved)
Ask the user for confirmation before spending money: "Director approved [experiment]. Estimated cost: $X. Launch on Modal?"

If confirmed:
- Write the experiment config
- Launch on Modal via `modal run --detach`
- Report the wandb URL and Modal app ID
- Note: results come asynchronously. Use `/experiment-complete` when the run finishes.

### 5. Update State
- Log the cycle to `.loop/state/log.jsonl`
- Update `.loop/state/budget.json`
- Update RESEARCH-LOG.md
