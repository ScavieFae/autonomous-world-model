---
name: conductor
description: Run one conductor heartbeat — check experiment state, evaluate completed runs, dispatch new research cycles. The autoresearch loop entry point. Use with `/loop 60m /conductor` for autonomous operation, or manually for single heartbeats.
user-invocable: true
---

# /conductor — Research Loop Heartbeat

One tick of the autoresearch loop. Read state → decide → act → log → exit.

## Decision Tree

```
READ STATE
  ├── Signal: pause.json active? → Log "paused", exit
  ├── Signal: escalate.json active? → Report to user, exit
  │
  ├── Experiment in flight? (running.json has in_flight != null)
  │   ├── Run: .venv/bin/python scripts/check_run.py {wandb_run_id}
  │   ├── Check: is it stale? (started_at > stale_timeout_hours ago AND state != "running")
  │   │   └── YES → Mark stale, release budget, log warning. Treat as "nothing running."
  │   ├── Check: state == "finished"?
  │   │   └── YES → EVALUATE (Step A below)
  │   ├── Check: state == "crashed" or "failed"?
  │   │   └── YES → Log error, clear in_flight, treat as "nothing running."
  │   └── state == "running" → Log "waiting for {experiment_id} ({progress_pct}%)", exit
  │
  └── Nothing running, no pending eval
      ├── Budget exhausted? (daily_spent >= daily_limit)
      │   └── YES → Log "budget exhausted", exit
      └── Budget available → NEW CYCLE (Step B below)
```

## Step A: Evaluate Completed Experiment

1. Read wandb results for the completed run:
   ```bash
   .venv/bin/python scripts/check_run.py {wandb_run_id}
   # Returns JSON: {"state": "finished", "metrics": {"eval/summary_pos_mae": ..., ...}}
   ```

2. Spawn an **Explore** agent as Research Director (prompt from `.loop/agents/research-director.md`):
   > "Evaluate this experiment result: [paste metrics]. The prior best rollout coherence is [X]. Is this improvement real? KEPT or DISCARDED?"

3. Based on Director's evaluation:
   - Update the run card (frontmatter + results section)
   - Run `python scripts/docs_prebuild.py`
   - Log to `docs/RESEARCH-LOG.md`
   - Update `.loop/state/budget.json` with actual cost
   - Clear `running.json` experiment (set `in_flight: null`, `lock: false`)
   - If KEPT and improvement is significant: queue program.md proposal for Mattie

## Step B: Start New Research Cycle

1. Check budget: `daily_spent + 2.0 <= daily_limit` (reserve $2 for Scout)

2. Spawn an **Explore** agent as Researcher (prompt from `.loop/agents/hypothesis.md`):
   > "Read program.md and recent run cards. Propose one hypothesis for a Scout experiment."

3. Spawn an **Explore** agent as Director (prompt from `.loop/agents/research-director.md`):
   > "Evaluate this hypothesis: [paste]. APPROVE or REJECT with reasoning."

4. If REJECTED: log to RESEARCH-LOG.md, log to `.loop/state/log.jsonl`. This heartbeat is done.

5. If APPROVED:
   - Set `running.json`: `lock: true`, `in_flight: {experiment_id, ...}`
   - Reserve budget: `daily_spent += estimated_cost`
   - Spawn an agent to execute:
     - Write the experiment YAML config
     - Launch on Modal: `modal run --detach scripts/modal_train.py --config ... --encoded-file /encoded-e012-fd-top5.pt`
     - Record the modal_app_id and wandb_url in `running.json`
   - Log to `.loop/state/log.jsonl`

## Step C: Log and Exit

Append to `.loop/state/log.jsonl`:
```json
{"timestamp": "2026-03-14T22:00:00Z", "action": "...", "experiment": "...", "reasoning": "...", "budget_spent": 0.0}
```

## Error Recovery

- **Modal app not found:** Clear `in_flight`, log error, continue to next cycle
- **wandb run crashed:** Same — clear, log, continue
- **Agent timeout:** Log timeout, exit. Next heartbeat will retry.
- **Budget file corrupted:** Reset daily_spent to 0, log warning
- **Lock stuck (in_flight set but experiment finished):** If wandb shows finished/crashed, clear the lock

## Starting Autonomous Mode

```bash
# Run one heartbeat manually:
/conductor

# Run every 60 minutes autonomously:
/loop 60m /conductor
```
