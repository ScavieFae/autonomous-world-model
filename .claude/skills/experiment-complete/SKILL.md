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

## Step 4: Verify Numbers via Fresh-Context Subagent (MANDATORY)

**This step is not optional.** It exists because a previous closeout (e031b) almost shipped three fabricated K-summary values to the permanent record. The numbers looked plausible, were within ~1–2% of the correct values, and were typed from memory rather than read from source. Plausible fabrication is the single worst failure mode of this skill, and the only reliable defense is a fresh context that has no narrative stake in the numbers being right.

Spawn a subagent via the Agent tool (`subagent_type: general-purpose`). Give it:
- The path to the draft run card
- The paths to every `eval_rollout.json` the card references (this run's + any prior-best comparison runs)
- **Nothing else.** The point is fresh context with no story loaded.

Use this prompt verbatim — do not paraphrase or summarize, the prompt's rigidity is the point:

> Verify every number in the Results section of `{card_path}` against the linked eval JSON files at `{json_paths}`.
>
> For each numeric claim (pos_mae, vel_mae, action_acc, action_change_acc, percent_mae, K=5/K=10/K=20 summaries, per-horizon rows, deltas, percentages, pp differences), confirm it either:
>
> 1. **Appears verbatim** in one of the linked eval_rollout.json files (check `k_summary.K5.pos_mae`, `per_horizon.{h}.{metric}`, `summary_pos_mae`, etc.), OR
> 2. **Is recomputable** from `per_horizon` via the standard K-summary formula (mean over t=1..K), OR
> 3. **Is a standard arithmetic derivation** from verified numbers (delta = A − B, percentage = delta / B × 100, pp = (A − B) × 100 for accuracy fields).
>
> Be pedantic about precision — 1.447 vs 1.446 counts as a mismatch, and so does a sign error in a delta.
>
> Return structured findings in this form:
>
> ```
> VERIFIED (N numbers):
>   - K=5 pos_mae 1.487 → e031b/eval_rollout.json::k_summary.K5.pos_mae ✓
>   - delta −0.157 (10.8%) → 1.289 − 1.446 = −0.157 ✓
>   ...
>
> UNVERIFIED (M numbers):
>   - Action_acc K=5 0.950 → CLAIMED in card, but k_summary.K5.action_acc = 0.9437 in source JSON. MISMATCH.
>   - ...
> ```
>
> UNVERIFIED is the most important category — those are fabrications or errors. Report under 400 words total.

If the subagent reports anything UNVERIFIED, fix the numbers in the card against the source JSONs and re-run verification. **Do not proceed to Step 5 (docs rebuild) until verification is clean.**

Paste the subagent's verification output into the commit message or PR description when you ship the closeout, so future-you can see it actually ran.

## Step 5: Rebuild Docs Index

```bash
python scripts/docs_prebuild.py
```

This regenerates `experiments/index.md` and `run-cards/index.md` with the updated metrics.

Verify the experiment shows up correctly in the index with its new status and rollout coherence value.

## Step 6: Close PR

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

## Step 7: Research Log Entry

Append a brief entry to `docs/RESEARCH-LOG.md`:

```markdown
## {Date} — {Experiment ID}: {Kept/Discarded}

Rollout coherence {value} (prior best {prior_best_rc}, {delta}%).
{One sentence on what this means for the research direction.}
```

This is the session-level log entry. Keep it to 2-3 lines. The card has the details.

## Step 8: Propose program.md Updates

**NEVER modify program.md directly.** Propose changes to the user.

Check if any of these apply:

- **New baseline:** If this experiment is now the best rollout coherence, propose updating the "Current Best" table in program.md.
- **New dead end:** If discarded, propose adding to the "Dead ends" table with the reason.
- **New proven improvement:** If kept and the technique is novel, propose adding to "What We Know."
- **Direction change:** If the result changes which research direction is highest priority, propose reordering.

Present the proposed changes as diffs: quote the current text, show the replacement, explain why.

## Step 9: Verify

Quick sanity check:

- [ ] Card frontmatter has `status`, `rollout_coherence`, `rollout_coherence_k5`, `prior_best_rc`, `prior_best_rc_k5` filled in
- [ ] Card body has `## Results` with the K=5/K=10/K=20 multi-metric table and `## Decision` sections
- [ ] **Step 4 verification subagent ran and returned clean** (paste its output in the commit message)
- [ ] `experiments/index.md` shows the updated metrics in both K=5 and K=20 leaderboards
- [ ] PR is closed (if one existed)
- [ ] Research log has an entry
- [ ] program.md updates proposed (if applicable)

Report completion to the user with the key numbers.
