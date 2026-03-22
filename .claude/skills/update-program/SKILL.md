# /update-program — Checklist for updating program.md

Run through this checklist whenever program.md needs updating (after kept experiments, base build mints, new research directions, or when sections feel stale).

Read `program.md`, `.loop/state/best.json`, `.loop/state/running.json`, and `docs/base-builds/` before starting. Update each section that's out of date, skip sections that are current.

## Checklist

### 1. Current Best table
- [ ] New best experiment at top of table?
- [ ] RC, change_acc, pos_mae match `best.json`?
- [ ] Notes column references correct base build?
- [ ] Narrative paragraph below table describes the latest result and cumulative improvement?

### 2. Proven improvements tables
- [ ] "Canonized in base build" table matches latest base build YAML (`docs/base-builds/b00X.yaml`)?
- [ ] "On top of base build" table lists all KEPT experiments not yet in a base build?
- [ ] No kept experiments missing from either table?

### 3. What we've tested (observations table)
- [ ] All experiment axes represented with current data points?
- [ ] Hit rates updated (e.g., "1/1 regressed" → "2/3 regressed" if new data)?
- [ ] Confidence levels still accurate?

### 4. Dead ends table
- [ ] New dead ends added (failed experiments with clear mechanisms)?
- [ ] No dead ends that should be revisited given new context (e.g., multi-phase regime switching)?

### 5. Engineering directions
- [ ] Blockers updated (e.g., data loader was blocked, now unblocked)?
- [ ] Tested directions marked with results (e.g., "TESTED, dead end" or "TESTED, needs different approach")?
- [ ] New directions added from research landscape report or recent insights?
- [ ] Cost estimates still accurate?

### 6. Config directions
- [ ] Closed axes marked as closed with data points?
- [ ] Open axes reflect current state of knowledge?
- [ ] Architecture grid updated with all results?

### 7. Hard Constraints
- [ ] Still accurate? (GPU tiers, memory budget, etc.)

### 8. Taste / Eval Protocol
- [ ] Any methodology changes? (e.g., multi-phase eval protocol from issue #17)
- [ ] Budget or cost norms changed?

### 9. Source Papers
- [ ] New papers from research landscape report (issue #16) added?
- [ ] Paper summaries in `research/sources/` for all cited papers?

### 10. Cross-references
- [ ] `docs/program.md` matches root `program.md` (run `python scripts/docs_prebuild.py`)?
- [ ] Base build YAML consistent with proven improvements table?
- [ ] `best.json` consistent with Current Best table?

## After updating

1. Run `python scripts/docs_prebuild.py` to sync docs/program.md
2. Commit: "Update program.md: {brief description of what changed}"
3. Push to main
