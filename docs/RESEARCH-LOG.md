# Research Log

Running notes from research and work sessions. Newest entries at top. Append-only.

*Continues the record started in [Research Diary (Archive)](RESEARCH-DIARY.md).*

---

## 2026-03-14 — Migration fixes, b001 tightened, E012 baseline, E019 in flight

First rollout coherence number: **E012 = 6.8448** (mean pos_mae, K=20, N=300). This is the pre-cascade, pre-SS checkpoint on 1.9K data.

Hit a wall trying to eval E016 and E017a — both use cascaded_heads which isn't ported to this repo. Asked: has cascaded_heads earned a place in b001? Reviewed the evidence: E014 showed promise at 1.9K (AR damage drift fix) but TF metrics regressed, and it was never isolated at 7.7K. E016 was a 3-variable omnibus — can't attribute. Decision: remove from b001, test as a separate experiment.

Tightened b001 to [e008c, e010b, e010c, e010d, e012] — only findings proven in isolation. Removed scheduled sampling too (E015 code not ported, E016 omnibus confounded).

Hit a class of migration bugs from nojohns→AWM: hardcoded dimensions everywhere assuming default encoding config (binary_dim=3, ctrl_dim=26). With v3 encoding (state_flags=true → binary_dim=43), things broke. Fixed in dataset.py, metrics.py, trainer.py, checkpoint.py, ar_utils.py. Mattie flagged that this exact loop happened before in nojohns — researched nojohns TROUBLESHOOTING and issues to avoid repeating mistakes.

E019 training launched on Modal A100 40GB with the fixed code. Waiting for results.

---

## 2026-03-14 — Base builds, Modal training, E019 baseline config

Surveyed all experiment history via subagent to ground dataset and config decisions in data rather than vibes. Key findings from the survey:

- 7.7K FD top-5 (`encoded-v3-ranked-fd-top5.pt`) is the proven dataset. E016 beat E012 on every metric with 4x data.
- The cumulative stack of proven improvements (E008c through E016) is well-defined. Eight experiments, all kept, super-additive when combined.
- Fox-only vs top-5 diversity on AR quality is untested. Flagged as a future experiment axis, not a baseline decision.

Introduced `base_build` as a frontmatter field to separate "stable foundation" from "thing being tested." b001 packages all eight proven experiments. This was already described as "Future: Stable Builds" in autoresearch-plan.md — now it's real with `docs/base-builds/b001.yaml`.

Built `scripts/modal_train.py` for Modal cloud training. Key decisions:
- A100 40GB ($2.10/hr) — fits batch_size up to ~8192, leaves room for batch size as experiment axis
- Pre-encoded `.pt` files loaded from Modal volume (14s load time on A100 per prior research)
- Rollout coherence eval runs automatically after training, results saved as JSON alongside checkpoint
- Encoding config validation — warns if YAML disagrees with encoded file

Checked Modal volume: `encoded-v3-ranked-fd-top5.pt` (7.7K games) exists alongside all checkpoints (E012, E016, E017a). Ready to run.

---

## 2026-03-14 — Added /pull skill with subagent scanners

Built the mirror to /push. The /pull skill spawns two Explore subagents in parallel — one to scan HANDOFF.md for new entries and action items, one to scan run cards, RUNNING.md, program.md, and checkpoint directories for state changes. Primary agent never reads the full living documents directly; gets concise digests back.

Added `.claude/last_pull.json` as local state (gitignored) so the scanners know what's "new since last pull." Seeded it with the current commit.

Design decision: subagents over agent teams. Two parallel disposable Explore agents are the right weight — read-only, fast, no persistent coordination needed. Agent teams would be overkill for a scan-and-report pattern.

---

## 2026-03-14 — Built rollout coherence eval, /push skill, operating doc catchup

First session in the standalone autonomous-world-model repo focused on the autoresearch infrastructure.

Started by surveying the status of the Karpathy autoresearch implementation. Found: program.md and autoresearch-plan.md were well-designed, run card schema was in place with `built_on` citations, but zero implementation — all four e018 cards still `proposed`, no eval script, no research briefs.

Built `scripts/eval_rollout.py` (the rollout coherence eval). Key decision: factored the AR reconstruction out of `rollout.py` into `scripts/ar_utils.py` as a shared module. This was motivated by discovering that `rollout.py` used hardcoded indices (`FPP=29`, `13:16` for binary, `16:29` for ctrl) which are wrong for non-default encoding configs — E017a-style configs have `float_per_player=69`, not 29. The refactored version uses `EncodingConfig` properties everywhere.

The eval is batched (20 forward passes of N=300, not 6000 sequential) and deterministic (seeded RNG, sorted indices, `@torch.no_grad()`). Metrics are in game units after denormalization. Summary metric is mean pos_mae over K=20 horizons.

Found that `absolute_y` from E017a's config doesn't exist in this repo's Python code — the flag is in the YAML but `EncodingConfig` doesn't have it. Likely lived in the nojohns repo before migration. Not a blocker: the eval matches rollout.py's behavior, and both would need updating together if we test an absolute_y model.

Built the `/push` skill as a communication workflow. The design: scope audit → HANDOFF → RUNNING → run card hygiene → interface contract check → program.md (propose only) → commit + push. Each step has trigger conditions so small changes don't demand full ceremony.

Key feedback from Mattie: program.md is the human's lever — never modify autonomously, only propose. RUNNING.md (not RESEARCH-DIARY.md) is the right place for session-level notes and findings. Saved these as persistent memories.

Still need: baseline rollout coherence numbers for E012 and E017a. That's the next session's first task — run the eval, get numbers, update program.md (with Mattie), then Self-Forcing can start.
