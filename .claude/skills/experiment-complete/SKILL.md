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
- **The results** — rollout coherence score, TF metrics, any qualitative findings
- **The decision** — kept or discarded, and why

If the user doesn't provide these, ask before proceeding.

## Step 1: Read Current State

```
Read the run card: docs/run-cards/{experiment-id}*.md
Read the experiment index: docs/experiments/index.md (for prior_best_rc context)
Check for a PR: gh pr list --search "{experiment-id}" --state open
```

## Step 2: Update Frontmatter

Update the YAML frontmatter in the run card:

```yaml
status: kept  # or discarded
rollout_coherence: {measured value}
prior_best_rc: {value from the best prior experiment at time of eval}
```

`prior_best_rc` is the rollout coherence of the best existing experiment *before* this one ran. This lets future readers understand the improvement without needing to cross-reference.

## Step 3: Append Results Section

Add a `## Results` section to the card body. Follow this structure:

```markdown
## Results

| Metric | {Prior Best ID} | {This Experiment} | Delta |
|--------|-----------------|-------------------|-------|
| rollout_coherence | {prior} | {new} | {change} |
| change_acc | ... | ... | ... |
| pos_mae | ... | ... | ... |
| val_loss | ... | ... | ... |

{Any qualitative observations — AR demo quality, failure modes, surprises.}

## Decision

**{Kept/Discarded}.** {One-sentence rationale grounded in the numbers.}

{If kept: what this enables or changes. If discarded: what this rules out.}
```

**Rules for the results section:**
- State findings as observations with numbers. Not editorials.
- Compare to the specific prior best, not to all experiments ever.
- Include the delta — raw numbers without context are useless.
- Note surprises. If TF metrics improved but AR quality didn't (or vice versa), say so.
- Keep it short. 10-20 lines, not a paper.

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

- [ ] Card frontmatter has `status`, `rollout_coherence`, `prior_best_rc` filled in
- [ ] Card body has `## Results` and `## Decision` sections
- [ ] `experiments/index.md` shows the updated metrics
- [ ] PR is closed (if one existed)
- [ ] Research log has an entry
- [ ] program.md updates proposed (if applicable)

Report completion to the user with the key numbers.
