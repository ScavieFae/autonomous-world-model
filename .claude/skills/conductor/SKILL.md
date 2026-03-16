---
name: conductor
description: Run one conductor heartbeat — check experiment state, evaluate completed runs, dispatch new research cycles. The autoresearch loop entry point. Use with `/loop 60m /conductor` for autonomous operation, or manually for single heartbeats.
user-invocable: true
---

# /conductor — Research Loop Heartbeat

One tick of the autoresearch loop. Read state → decide → act → log → exit.

Execute ALL steps yourself. Do not ask the user for confirmation — the budget, Director, and signals are your safety gates.

## Decision Tree

```
READ STATE
  ├── Signal: pause.json active? → Log "paused", exit
  ├── Signal: escalate.json active? → Report to user, exit
  │
  ├── Experiment in flight? (running.json has in_flight != null)
  │   ├── Run: .venv/bin/python scripts/check_run.py {wandb_run_id}
  │   ├── state == "finished"? → EVALUATE (Step A)
  │   ├── state == "crashed"/"failed"? → Log error, clear lock, treat as nothing running
  │   ├── stale? (started_at > stale_timeout_hours ago AND state != "running") → Clear, log warning
  │   └── state == "running" → Log to .loop/state/log.jsonl ONLY (no Matrix), exit
  │
  └── Nothing running
      ├── Budget exhausted? → Log, exit
      └── Budget available → NEW CYCLE (Step B)
```

## Step A: Evaluate Completed Experiment

1. **Get metrics:**
   ```bash
   .venv/bin/python scripts/check_run.py {wandb_run_id}
   ```

2. **Spawn Director agent** (Explore subagent) with the results and the prior best RC from running.json. Ask: KEPT or DISCARDED? The Director reads program.md, the run card, and the metrics.

3. **Close out** based on Director verdict:
   - Update run card frontmatter: `status: kept/discarded`, `rollout_coherence: X.XX`
   - Add Results section with metrics table and Director evaluation
   - Append to `docs/RESEARCH-LOG.md`
   - Update `.loop/state/budget.json` (actual cost from runtime)
   - Clear `running.json` (set `in_flight: null`, `lock: false`)
   - If KEPT: merge the experiment's PR (`gh pr merge {pr_number} --merge`)
   - If DISCARDED: close the PR (`gh pr close {pr_number}`)
   - Run `python scripts/docs_prebuild.py`
   - Commit closeout changes on main, push

4. **Notify Matrix:**
   ```bash
   .venv/bin/python scripts/notify_matrix.py --room experiment-results "{full evaluation}"
   ```

5. **Continue to Step B** (start next cycle in same heartbeat).

## Step B: Start New Research Cycle

1. **Budget check:** `daily_spent + 4.0 <= daily_limit`

2. **Spawn hypothesis agent** (Explore subagent with `.loop/agents/hypothesis.md` prompt):
   > "Read program.md and recent run cards. Propose one experiment."

3. **Spawn Director agent** (Explore subagent with `.loop/agents/research-director.md` prompt):
   > "Evaluate this hypothesis: [paste full hypothesis]. APPROVE or REJECT."

4. **Notify Matrix** with the full deliberation (hypothesis + Director review):
   ```bash
   .venv/bin/python scripts/notify_matrix.py --room research "{hypothesis text}\n\n{director review}"
   ```

5. **If REJECTED:** log to log.jsonl, done.

6. **If APPROVED — spawn Coder agent** in an isolated worktree:
   ```
   Agent tool with:
     subagent_type: general-purpose
     isolation: "worktree"
     prompt: [from .loop/agents/coder.md + approved hypothesis + director conditions]
   ```
   The Coder agent:
   - Creates a branch named `{experiment-id}` (e.g., `e018c-context-k30`)
   - Implements the change (config and/or code)
   - Writes the run card with `status: running`
   - Commits on the branch
   - Returns the branch name and worktree path

7. **Create PR** from the experiment branch:
   ```bash
   git push -u origin {branch}
   gh pr create --title "{experiment-id}: {short description}" --body "{hypothesis + director review}"
   ```

8. **Launch on Modal** from the branch:
   ```bash
   modal run --detach scripts/modal_train.py --config experiments/{id}.yaml --encoded-file /encoded-e012-fd-top5.pt --run-name {id}
   ```
   Capture the wandb run ID from Modal output or poll wandb for the run name.

9. **Update state:**
   - Set `running.json`: `lock: true`, `in_flight: {id, wandb_url, pr_number, branch, ...}`
   - Reserve budget in `budget.json`
   - Log to `log.jsonl`

10. **Notify Matrix:**
    ```bash
    .venv/bin/python scripts/notify_matrix.py --room conductor-log "Launched {id}: {description}. PR: {url}. wandb: {url}"
    ```

## Matrix Notification Rules

**DO post** (state changes only):
- Experiment launched (with PR + wandb links)
- Experiment completed (with full Director evaluation)
- Hypothesis proposed + Director review (full deliberation text)
- Errors, crashes, budget exhaustion, escalations

**DO NOT post** routine "still running" heartbeats. Log those to log.jsonl only.

## State Files

| File | Purpose |
|------|---------|
| `.loop/state/running.json` | Experiment lock, in-flight tracking, history |
| `.loop/state/budget.json` | Daily/weekly spend limits and tracking |
| `.loop/state/log.jsonl` | Append-only decision log |
| `.loop/state/signals/pause.json` | Pause signal |
| `.loop/state/signals/escalate.json` | Escalation signal |

## Agent Roles

| Role | Subagent type | Prompt file | When |
|------|--------------|-------------|------|
| Hypothesis | Explore | `.loop/agents/hypothesis.md` | New cycle — propose experiment |
| Director | Explore | `.loop/agents/research-director.md` | Review hypothesis, evaluate results |
| Coder | general-purpose (worktree) | `.loop/agents/coder.md` | Implement approved experiment on branch |

## Error Recovery

- **Modal app not found / wandb crashed:** Clear `in_flight`, log error, close PR, continue
- **Agent timeout:** Log, exit. Next heartbeat retries.
- **Budget corrupted:** Reset daily_spent to 0, log warning
- **Stuck lock:** If wandb shows finished/crashed but lock is set, clear it
- **Coder agent fails:** Log error, close PR if created, continue to next heartbeat

## Starting Autonomous Mode

```bash
# Run one heartbeat manually:
/conductor

# Run every 60 minutes autonomously:
/loop 60m /conductor
```
