# Running Log — Autonomous World Model

Running notes from work sessions. Newest entries at top. Append-only.

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
