# Decision Log

Every hypothesis — approved, rejected, or revised — gets recorded here. This is the durable record of what was considered and why. Run cards capture what ran; this captures what *didn't* run and the reasoning behind it.

## Purpose

1. **Prevent repeated proposals.** Agents can check if an idea was already considered.
2. **Enable revisitation.** When context changes (new data, new code, new findings), a previously rejected idea may become viable. The reasoning here tells you what would need to change.
3. **Calibrate the Director.** Review past rejections to catch overcorrection. If the Director consistently rejects ideas that later prove correct (or approves ideas that consistently fail), that's signal for recalibration.

## File format

One file per decision cycle. Named `{date}-{cycle}.md`. Each entry:

```markdown
### {experiment-id}: {short title}
**Verdict:** APPROVED / REJECTED / REVISED
**Reasoning:** {Director's full reasoning, not summarized}
**What would change the verdict:** {conditions under which this should be reconsidered}
**Hypothesis:** {the full proposal that was evaluated}
```

## How agents use this

- **Hypothesis agent:** Before proposing, scan recent decisions for rejected ideas that match your direction. If context has changed, explicitly reference the prior rejection and explain what's different.
- **Director agent:** When evaluating, check if this idea was previously rejected. If so, has anything changed? Apply the "what would change the verdict" criteria.
- **Review agent (future):** Periodically scan decisions for patterns — is the Director too conservative? Too aggressive? Are there rejected ideas that new evidence supports?
