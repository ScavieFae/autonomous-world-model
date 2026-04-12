---
name: experiment-complete
description: Close out an experiment — update run card with results, update frontmatter, rebuild docs index, close PR, log to research log, propose program.md updates. Use when an experiment has finished training and eval.
disable-model-invocation: true
user-invocable: true
---

# /experiment-complete — Close Out an Experiment

The card is the permanent record. Everything flows into the card. This skill ensures that happens consistently.

## Inputs

You need to know:
- **Which experiment** (e.g., e019)
- **The results** — rollout coherence at K=5 **and** K=20, the full multi-metric suite (pos_mae, vel_mae, action_acc, percent_mae) at K=5/K=10/K=20, TF metrics, any qualitative findings
- **The decision** — kept or discarded, and why

**K=5 is the primary metric going forward.** K=20 is retained for historical continuity with older runs but is drift-dominated — per-step signal lives at K=5. Compare to prior experiments on K=5 first, K=20 second. See e029a-7k-scaling's Layer A/B analysis for why: a run can be meaningfully better at K=5 and indistinguishable at K=20, hiding real regime shifts.

If the user doesn't provide these, ask before proceeding.

## Step 1: Read Current State

```
Read the run card: docs/run-cards/{experiment-id}*.md
Read the experiment index: docs/experiments/index.md (for prior_best_rc context)
Check for a PR: gh pr list --search "{experiment-id}" --state open
```

## Step 2: Update Frontmatter

Update the YAML frontmatter in the run card. Both K=5 and K=20 fields are required for new runs:

```yaml
status: kept  # or discarded
rollout_coherence: {K=20 pos_mae — legacy metric, keep for continuity}
rollout_coherence_k5: {K=5 pos_mae — primary metric going forward}
prior_best_rc: {K=20 of the best prior experiment at time of eval}
prior_best_rc_k5: {K=5 of the best prior experiment at time of eval}
```

`prior_best_rc` / `prior_best_rc_k5` are the best existing experiment values *before* this one ran. This lets future readers understand the improvement without needing to cross-reference.

If you only have K=20 (e.g., reviewing an old closeout), leave `rollout_coherence_k5` out and note it in the results section. For new runs, both must be populated — `modal_train.py::eval_checkpoint` saves `per_horizon` so you can compute K=5 from the saved eval JSON.

## Step 3: Append Results Section

Add a `## Results` section to the card body. Follow this structure — **lead with the multi-metric K=5/K=10/K=20 suite**, not a single scalar:

```markdown
## Results

### Matched rollout eval vs {prior best ID}

| Metric | K=5 ({prior}) | K=5 ({new}) | K=10 ({prior}) | K=10 ({new}) | K=20 ({prior}) | K=20 ({new}) |
|--------|---------------|-------------|----------------|--------------|----------------|--------------|
| pos_mae | ... | ... | ... | ... | ... | ... |
| action_acc | ... | ... | ... | ... | ... | ... |
| percent_mae | ... | ... | ... | ... | ... | ... |

| Headline | Prior | This | Delta |
|----------|-------|------|-------|
| rollout_coherence_k5 (primary) | ... | ... | ... |
| rollout_coherence (K=20, legacy) | ... | ... | ... |

{Any qualitative observations — AR demo quality, failure modes, surprises.}
{If K=5 and K=20 tell different stories (Layer A/B tradeoff), call it out.}

## Decision

**{Kept/Discarded}.** {One-sentence rationale grounded in the K=5 numbers first, K=20 as secondary.}

{If kept: what this enables or changes. If discarded: what this rules out.}
```

**Rules for the results section:**
- State findings as observations with numbers. Not editorials.
- **Never celebrate or regret based on a single scalar.** If you only have K=20 improvement, check K=5. If K=5 improves but action_acc or percent_mae regress, that's the Layer A/B tradeoff and must be documented.
- Compare to the specific prior best on K=5, not to all experiments ever.
- Include the delta — raw numbers without context are useless.
- Note surprises. If K=5 and K=20 disagree, or if TF metrics improved but AR quality didn't (or vice versa), say so.
- Keep it short. 15-25 lines, not a paper.

## Step 4: Rebuild Docs Index

```bash
python scripts/docs_prebuild.py
```

This regenerates `experiments/index.md` and `run-cards/index.md` with the updated metrics.

Verify the experiment shows up correctly in the index with its new status and rollout coherence value.

## Step 5: Close PR

If there's an open PR for this experiment:

1. Post a summary comment on the PR:

```markdown
## {Experiment ID} — {Kept/Discarded}

**Rollout coherence:** {value} (prior best: {prior_best_rc})

{One-sentence summary of finding.}

Full results in the [run card](link to card on main).
```

2. Close the PR (do NOT merge — experiment PRs are for reference, not merging):

```bash
gh pr close {number} --comment "Experiment complete. Results recorded in run card."
```

If there's no PR, skip this step.

## Step 6: Research Log Entry

Append a brief entry to `docs/RESEARCH-LOG.md`:

```markdown
## {Date} — {Experiment ID}: {Kept/Discarded}

Rollout coherence {value} (prior best {prior_best_rc}, {delta}%).
{One sentence on what this means for the research direction.}
```

This is the session-level log entry. Keep it to 2-3 lines. The card has the details.

## Step 7: Propose program.md Updates

**NEVER modify program.md directly.** Propose changes to the user.

Check if any of these apply:

- **New baseline:** If this experiment is now the best rollout coherence, propose updating the "Current Best" table in program.md.
- **New dead end:** If discarded, propose adding to the "Dead ends" table with the reason.
- **New proven improvement:** If kept and the technique is novel, propose adding to "What We Know."
- **Direction change:** If the result changes which research direction is highest priority, propose reordering.

Present the proposed changes as diffs: quote the current text, show the replacement, explain why.

## Step 8: Verify

Quick sanity check:

- [ ] Card frontmatter has `status`, `rollout_coherence`, `rollout_coherence_k5`, `prior_best_rc`, `prior_best_rc_k5` filled in
- [ ] Card body has `## Results` with the K=5/K=10/K=20 multi-metric table and `## Decision` sections
- [ ] `experiments/index.md` shows the updated metrics in both K=5 and K=20 leaderboards
- [ ] PR is closed (if one existed)
- [ ] Research log has an entry
- [ ] program.md updates proposed (if applicable)

Report completion to the user with the key numbers.
