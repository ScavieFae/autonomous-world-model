---
name: push
description: Pre-push workflow — communicate with ScavieFae, capture notes in RUNNING.md, check hygiene, then commit and push. Use when the user says "push", "ship it", "send it", or asks to commit and push changes.
disable-model-invocation: true
user-invocable: true
---

# /push — Communicate, Capture, Ship

Pre-push workflow for the Autonomous World Model. Every push is a communication touchpoint — with ScavieFae, with future sessions, with the running record.

Do NOT skip steps or treat this as a formality. Each step exists because we got burned without it. But DO skip steps that genuinely don't apply (marked with trigger conditions).

## Step 1: Scope Audit

Read the current state. Understand what's going out.

```
git diff --stat origin/main..HEAD   (committed but not pushed)
git diff --stat                     (unstaged changes to potentially include)
git status                          (untracked files)
git log --oneline origin/main..HEAD (commits to push)
```

**Categorize the changes:**
- **research** — models/, training/, scripts/eval*, scripts/train*, experiments/, run cards
- **infrastructure** — scripts/ (non-training), data/, configs/
- **onchain** — solana/, crank/
- **site** — site/
- **docs-only** — docs/, *.md only

Report the category and scope to the user before proceeding.

## Step 2: HANDOFF.md — Communicate with ScavieFae

**Trigger:** Changes touch any of: `scripts/`, `models/`, `training/`, `data/`, `crank/`, `solana/`, or interface contracts (encoding.py, state_convert.py, solana_bridge.py, solana/client/src/, JSON frame format). Skip for docs-only changes that don't affect the offchain/onchain interface.

Update `docs/HANDOFF.md` with a new entry at the top (below the header, above the first `---`).

**Entry format** (match the existing style in HANDOFF.md):
```markdown
## Verb: Description (Author, Date)

**Author → ScavieFae**: [What changed and why, written for someone who wasn't here]

### What changed
[Substantive description. Not just file names — what the change DOES and WHY it matters for the recipient.]

### Cross-boundary implications
[How this affects ScavieFae's work. Interface contracts, deployment, onchain behavior, SDK changes. If none, say "None — offchain only" or similar.]

### Files changed
| File | Change |
|------|--------|
| ... | ... |

### Next steps
[What happens next. Action items for either side.]

---
```

**Write for someone who wasn't here.** They don't have your context. Include the "why" not just the "what."

## Step 3: RUNNING.md — Capture Notes and Findings

**Trigger:** The session involved research reasoning, design decisions, implementation findings, dead ends, or anything worth preserving for future agents and sessions. Skip only for truly mechanical changes (typo fixes, dependency bumps).

Update `docs/RUNNING.md` with a new entry at the top. If the file doesn't exist, create it with a header:

```markdown
# Running Log — Autonomous World Model

Running notes from work sessions. Newest entries at top. Append-only.

---
```

**Entry format:**
```markdown
## [Date] [Time] — [What you were doing]

[What you encountered. Decisions made and why. Dead ends hit. Surprises found. The texture of the work — not just outcomes, but the process.]

[If you changed approach mid-task, explain the pivot.]
```

**Rules:**
- Append-only. Never edit or delete existing entries.
- Write in first person. This is the running log. "Found that...", "Decided to..."
- Include friction and dead ends, not just successes.
- Note decisions with reasoning. "Chose approach A over B because..."
- Be honest about uncertainty. "I think this works but haven't verified edge case Z"
- State findings as observations with hit rates, not editorials.

## Step 4: Run Card Hygiene

**Trigger:** Changes to `docs/run-cards/`, new experiment infrastructure, or experiment status changes.

Check that run card YAML frontmatter `status` fields match reality:
- `proposed` → `running` if implementation started
- `running` → `kept` or `discarded` if results are in
- `rollout_coherence` and `prior_best_rc` filled in if eval has been run

Also check: does any new run card need to be created for work done in this session?

## Step 5: Interface Contract Check

**Trigger:** Changes to `models/encoding.py`, `crank/state_convert.py`, `crank/solana_bridge.py`, `solana/client/src/`, or the JSON frame format consumed by `viz/` and `site/`.

These are review-gated per CLAUDE.md:

| What changed | Who reviews | Why |
|---|---|---|
| Onchain programs, syscall, ECS | Model side reviews math/format | Hardest to undo |
| Weight format / encoding changes | Codex reviews | Must match onchain structs |
| Binary wire format (PlayerState) | All sides | Shared boundary |

If any of these changed: flag it explicitly in the HANDOFF entry and note the review requirement. Do NOT push without acknowledging the review gate — ask the user.

## Step 6: program.md — Propose Only

**Trigger:** New baseline numbers, new findings that change research direction, experiment status changes, new dead ends discovered.

**NEVER modify program.md autonomously.** This is Mattie's document — it shapes how agents operate. It is the human's lever on the autoresearch loop.

If the session produced findings that should update program.md:
1. **Propose** the specific changes (quote the current text, suggest the replacement)
2. **Explain why** — what finding or result motivates the change
3. **Wait for Mattie** to approve, modify, or reject

Examples of things to propose:
- New rollout coherence baselines for the "Current Best" table
- New entries in "What We Know" (proven improvements or dead ends)
- Reordering research directions based on results
- Updating the eval protocol with actual timing numbers

## Step 7: Commit + Push

1. **Stage specific files.** Never `git add -A` or `git add .`. Name each file. Don't sweep in unrelated changes, secrets (.env), or large binaries.
2. **Write the commit message.** Lead with why, not what. Follow the repo's style (see `git log --oneline -10`). End with the co-author line.
3. **Commit.**
4. **Push** to origin. Only because the user explicitly asked.
5. **Verify** with `git status` after push.

Report what was pushed (commit hash, files, branch) so the user has a record.
