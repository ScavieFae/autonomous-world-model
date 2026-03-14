---
name: pull
description: Post-pull orientation — absorb what changed, scan HANDOFF and RUNNING for new context, check run cards and program.md, report to user. Use when starting a session, after a git pull, or when the user says "pull", "catch me up", "what changed", or "orient".
disable-model-invocation: true
user-invocable: true
---

# /pull — Orient, Absorb, Align

Post-pull intake workflow. The mirror of /push. Every pull is a reorientation — absorb what changed while we were away, pick up context, align on priorities.

This is a read-heavy workflow. The only output is a concise briefing to the user.

## Step 1: Pull + Diff

Pull from origin and understand what came in.

```bash
git pull
git log --oneline HEAD@{1}..HEAD    # what landed
git diff --stat HEAD@{1}..HEAD      # scope of changes
```

If already up to date, skip to Step 2 (the scan is still valuable at session start).

**Categorize the incoming changes:**
- **research** — models/, training/, scripts/eval*, scripts/train*, experiments/, run cards
- **infrastructure** — scripts/ (non-training), data/, configs/
- **onchain** — solana/, crank/
- **site** — site/
- **docs-only** — docs/, *.md only

## Step 2: Spawn Scanners

Launch two subagents **in parallel** to scan the living documents. These documents grow over time — the primary agent should NOT read them directly. The subagents read, filter, and return concise digests.

### Scanner A: HANDOFF Scanner

Spawn an **Explore** agent with this prompt:

> Read `docs/HANDOFF.md`. Find all entries that are new since [last pull timestamp from `.claude/last_pull.json`, or "the last 7 days" if no state file exists].
>
> For each new entry, extract:
> - **Who → Who**: direction of communication
> - **Summary**: 1-2 sentence summary of what changed
> - **Action items for us**: anything we need to do, review, or respond to
> - **Interface contract changes**: any shared boundary changes flagged
>
> Also check: are there any unanswered review requests or open questions directed at us?
>
> Return a structured digest. Be concise — the primary agent will relay this to the user.

### Scanner B: State Scanner

Spawn an **Explore** agent with this prompt:

> Scan the project state for changes and current status. Check these files:
>
> 1. **`docs/RUNNING.md`** — Read the 3 most recent entries. Summarize: what was worked on, key findings, open items.
>
> 2. **Run cards** (`docs/run-cards/e*.md`) — Read the YAML frontmatter of ALL run cards. Report:
>    - Any cards with `status: running` (active experiments)
>    - Any cards whose status changed recently
>    - Any cards with `rollout_coherence` values filled in (completed evals)
>    - Count by status: how many proposed, running, kept, discarded
>
> 3. **`program.md`** — Check git log for recent modifications: `git log --oneline -5 -- program.md`. If modified since last pull, summarize what changed.
>
> 4. **New checkpoints or eval results** — Check: `ls checkpoints/` for any new directories. Check for any `eval_result*.json` files. Report what's new.
>
> Return a structured situation report. Be concise.

## Step 3: Update Last-Pull State

Write `.claude/last_pull.json`:

```json
{
  "timestamp": "2026-03-14T12:00:00",
  "commit": "abc1234",
  "branch": "main"
}
```

This is used by future /pull runs to filter "what's new."

## Step 4: Report

Combine the scanner results into a concise briefing for the user:

```
## Pull Briefing

**Incoming:** [N commits / up to date]
**Category:** [research / infrastructure / onchain / site / mixed]

### HANDOFF
[Digest from Scanner A — new entries, action items, review requests]

### State
[Digest from Scanner B — active experiments, run card status, program.md changes, new results]

### Suggested Focus
[Based on what's new: what should we work on this session?]
```

Keep it tight. The user wants to orient in 30 seconds, not read a report.

## Step 5: Interface Contract Alert

If either scanner found changes to shared boundaries (encoding.py, wire format, SDK types, JSON frame format), call this out explicitly at the top of the briefing with a warning. These need attention before we start writing code that depends on the old contract.
