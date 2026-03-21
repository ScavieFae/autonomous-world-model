---
name: conductor
description: Run one conductor heartbeat — check experiment state, evaluate completed runs, dispatch new research cycles. Supports parallel experiments. Use with `/loop 60m /conductor` for autonomous operation.
user-invocable: true
---

# /conductor — Research Loop Heartbeat

One tick of the autoresearch loop. Read state → decide → act → log → exit.

Execute ALL steps yourself. Do not ask the user for confirmation — the budget, Director, and signals are your safety gates.

**Supports parallel experiments.** Multiple experiments can be in flight simultaneously, up to `max_concurrent`. Each heartbeat checks all in-flight experiments, evaluates any that finished, and launches new ones if slots and budget are available.

## Decision Tree

```
READ STATE
  ├── Signal: pause.json active? → Log "paused", exit
  ├── Signal: escalate.json active? → Report to user, exit
  │
  ├── CHECK ALL IN-FLIGHT EXPERIMENTS (running.json → experiments.in_flight[])
  │   For each experiment in the list:
  │   ├── Run: .venv/bin/python scripts/check_run.py {wandb_run_id}
  │   ├── state == "finished"? → EVALUATE (Step A) — remove from in_flight
  │   ├── state == "crashed"/"failed"? → Log error, remove from in_flight, close PR
  │   ├── stale? → Clear, log warning, remove from in_flight
  │   └── state == "running" → Log to log.jsonl ONLY (no Matrix)
  │
  ├── After processing all in-flight:
  │   ├── Open slots? (len(in_flight) < max_concurrent, default 3)
  │   │   ├── Budget available? (daily_spent + 5.0 <= daily_limit)
  │   │   │   └── YES → NEW CYCLE (Step B) — repeat until slots or budget exhausted
  │   │   └── NO → Log "budget exhausted", done
  │   └── No open slots → done (wait for next heartbeat)
```

## Step A: Evaluate Completed Experiment

1. **Get metrics:**
   ```bash
   .venv/bin/python scripts/check_run.py {wandb_run_id}
   ```

2. **Spawn Director agent** (Explore subagent) with the results and the prior best RC. Ask: KEPT or DISCARDED? The Director reads program.md, the run card, and the metrics.

3. **Post Director evaluation to Matrix immediately:**
   ```bash
   .venv/bin/python scripts/notify_matrix.py --room experiment-results "🎯 [Director] {FULL evaluation output}"
   ```

4. **Close out** based on Director verdict:
   - Update run card frontmatter: `status: kept/discarded`, `rollout_coherence: X.XX`
   - Add Results section with metrics table and Director evaluation
   - Append to `docs/RESEARCH-LOG.md`
   - Update `.loop/state/budget.json`: increment `daily_spent` and `weekly_spent` by actual cost, increment `experiments_run`, increment `experiments_kept` or `experiments_discarded`, update `total_spent`
   - Remove this experiment from `running.json → experiments.in_flight[]`, add to `history`
   - If KEPT: merge the experiment's PR (`gh pr merge {pr_number} --merge`)
   - If KEPT and RC < prior_best_rc: update `.loop/state/best.json` (new current_best, append to history) and `running.json → prior_best_rc`. Do NOT update program.md — propose changes to Mattie via the Director evaluation instead.
   - If DISCARDED: close the PR (`gh pr close {pr_number}`)
   - Run `python scripts/docs_prebuild.py`
   - Commit closeout changes on main, push

4. **Notify Matrix:**
   ```bash
   .venv/bin/python scripts/notify_matrix.py --room experiment-results "{full evaluation}"
   ```

## Step B: Start New Research Cycle

1. **Budget check:** `daily_spent + 5.0 <= daily_limit`

2. **Spawn hypothesis agent** (Explore subagent with `.loop/agents/hypothesis.md` prompt):
   > "Read program.md and recent run cards. Propose one experiment. Currently {N} experiments in flight: {list ids}. Propose something on a DIFFERENT axis from what's already running."

3. **Post hypothesis to Matrix immediately** (before Director reviews):
   ```bash
   .venv/bin/python scripts/notify_matrix.py --room research "🔬 [Hypothesis Agent] {FULL agent output — paste the entire proposal, not a summary}"
   ```

4. **Spawn Director agent** (Explore subagent with `.loop/agents/research-director.md` prompt):
   > "Evaluate this hypothesis: [paste full hypothesis]. APPROVE or REJECT."

5. **Post Director review to Matrix immediately:**
   ```bash
   .venv/bin/python scripts/notify_matrix.py --room research "🎯 [Director] {FULL agent output — paste the entire review, not a summary}"
   ```

6. **Log the decision durably** to `docs/decisions/{date}-cycle{N}.md`:
   - Full hypothesis text
   - Full Director reasoning (not summarized)
   - Verdict: APPROVED / REJECTED
   - "What would change the verdict" (for rejections)
   This is the permanent record that future agents can review.

7. **If REJECTED:** log to log.jsonl. Try ONE more hypothesis (max 2 attempts per heartbeat). If second also rejected, done.

8. **If APPROVED — spawn Coder agent** in an isolated worktree:
   ```
   Agent tool with:
     subagent_type: general-purpose
     isolation: "worktree"
     prompt: [from .loop/agents/coder.md + approved hypothesis + director conditions]
   ```
   The Coder agent:
   - Creates a branch named `{experiment-id}`
   - Implements the change (config and/or code)
   - Writes the run card with `status: running`
   - Commits on the branch
   - Returns the branch name

9. **Post Coder output to Matrix:**
   ```bash
   .venv/bin/python scripts/notify_matrix.py --room worker-updates "🔧 [Coder] {agent output — what was implemented, branch name, files changed}"
   ```

10. **Create PR** from the experiment branch:
   ```bash
   git push -u origin {branch}
   gh pr create --title "{experiment-id}: {short description}" --body "{hypothesis + director review}"
   ```

8. **Copy config to main and launch on Modal:**
   ```bash
   # Config must exist on main for Modal to see it
   git checkout {branch} -- experiments/{id}.yaml
   modal run --detach scripts/modal_train.py --config experiments/{id}.yaml \
     --encoded-file /encoded-e012-fd-top5.pt --run-name {id}
   ```
   Poll wandb for the run ID (filter by display_name).

9. **Update state:**
   - Append to `running.json → experiments.in_flight[]`: `{id, wandb_url, pr_number, branch, started_at, ...}`
   - Reserve budget in `budget.json`
   - Log to `log.jsonl`

10. **Notify Matrix:**
    ```bash
    .venv/bin/python scripts/notify_matrix.py --room conductor-log "Launched {id}. PR: {url}. wandb: {url}. Slots: {used}/{max}"
    ```

11. **Loop back to check slots** — if more slots and budget available, run Step B again to fill them.

## State Files

### running.json schema (parallel)

```json
{
  "experiments": {
    "in_flight": [
      {"id": "e020a", "wandb_run_id": "abc123", "pr_number": 4, "branch": "e020a-sf-ratio-10", "started_at": "...", "estimated_cost": 5.0},
      {"id": "e020b", "wandb_run_id": "def456", "pr_number": 5, "branch": "e020b-sf-ratio-30", "started_at": "...", "estimated_cost": 5.0}
    ],
    "max_concurrent": 3,
    "stale_timeout_hours": 4
  },
  "history": [...],
  "prior_best_rc": 6.03
}
```

### Other state files

| File | Purpose |
|------|---------|
| `.loop/state/best.json` | Current best checkpoint, config, metrics, and RC history. Read this for baseline comparisons — do NOT hardcode RC values. |
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

## Matrix Notification Rules

**DO post** (state changes only):
- Experiment launched (with PR + wandb links + slot count)
- Experiment completed (with full Director evaluation)
- Hypothesis proposed + Director review (full deliberation text)
- Errors, crashes, budget exhaustion, escalations

**DO NOT post** routine "still running" heartbeats. Log those to log.jsonl only.

## Error Recovery

- **Modal app not found / wandb crashed:** Remove from in_flight, log error, close PR
- **Agent timeout:** Log, exit. Next heartbeat retries.
- **Budget corrupted:** Reset daily_spent to 0, log warning
- **Stuck experiment:** If wandb shows finished/crashed but still in in_flight, remove it
- **Coder agent fails:** Log error, close PR if created, continue to next slot

## Starting Autonomous Mode

```bash
/conductor                    # One heartbeat
/loop 60m /conductor          # Every hour
```
