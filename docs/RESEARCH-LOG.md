# Research Log

Running notes from research and work sessions. Newest entries at top. Append-only.

*Continues the record started in [Research Diary (Archive)](RESEARCH-DIARY.md).*

---

## 2026-03-15 — E018a Self-Forcing launched (first autoresearch cycle)

First manual `/research-cycle dry-run` then live launch. Hypothesis agent proposed cascaded_heads (e020) — Director rejected: code not ported to this repo, and Self-Forcing is priority #1 in program.md. Both correct calls. Re-ran targeting Self-Forcing.

**Implementation:** Added `_self_forcing_step()` and `_build_sf_targets()` to `training/trainer.py` (~100 lines). SF interleaves AR unroll steps into the training loop: every 5th batch, the model unrolls 3 AR steps using `reconstruct_frame()` from `scripts/ar_utils.py`, computes loss at each step vs ground truth, truncated BPTT (detach between steps). Ground-truth controller inputs. The model sees its own drift and learns to handle it.

**Config:** `experiments/e018a-sf-minimal.yaml` — identical to E019 except: `self_forcing.enabled=true`, `ratio=4` (20% SF), `unroll_length=3`, `batch_size=512` (matching E019 actual run, not the YAML's 4096 which had data loading bottleneck).

**Director approved as Scout** ($2). Key concern: log `batch/sf_loss` vs `batch/tf_loss` separately to diagnose objective conflict. Expect 2-5pp val_change_acc regression from SF gradient budget steal — acceptable if AR improves.

**Launched:** Modal app `ap-BM6IK0190kM58s5lGxsngQ`, 1.9K data, 1 epoch, A100 40GB. Rollout coherence eval automatic after epoch 1. Baseline to beat: RC = 6.77 (E019).

**Results: KEPT.** Rollout coherence **6.26** (prior best 6.77, **-7.5%**). TF metrics regressed as predicted: change_acc 61.6% (-17pp), pos_mae 0.825 (+9%). SF loss (0.38) was 2.3× TF loss (0.16) — the model faces substantially harder predictions from its own state. This is the largest single-experiment improvement in the E008-E019 series and empirically confirms program.md's core insight: teacher-forced improvements don't predict AR quality. Self-Forcing directly addresses exposure bias.

Cost: $3.90 (1.85hr A100). wandb: https://wandb.ai/shinewave/melee-worldmodel/runs/mtpaj930

Next directions: longer unroll (N=5), higher SF ratio, horizon-weighted loss (e018d), full BPTT.

**E018b DISCARDED (N=5 unroll).** RC regressed to 6.45 (+3.0% worse than e018a's 6.26). SF loss nearly doubled (0.38→0.74) — the model couldn't learn from 5 steps of drift with truncated BPTT. Gradient signal degraded rather than improved. Key finding: truncated BPTT saturates at N=3 (~150ms at 60fps). Longer horizons need full BPTT or horizon-weighted loss. Cost: $5.25.

**E018d DISCARDED (horizon-weighted loss).** RC regressed to 6.81 (+8.8% worse than e018a's 6.26, worse than E019 baseline). Linear ramp [0.5, 1.25, 2.0] over-weighted step 3 where truncated BPTT has weakest gradient signal. SF loss +50%, TF loss +84%. The assumption that "later steps are more informative" was wrong under truncated BPTT — later steps have worse gradients, weighting them more amplifies noise. Cost: $4.00.

**Session summary (2026-03-15):** 3 experiments, $13.15 total. E018a (uniform SF, N=3) is the winner. Both refinements (longer horizon, weighted loss) regressed. The SF axis within truncated BPTT is explored.

**E018c KEPT (K=30 context, overnight run).** RC **6.03** (-3.7% vs e018a's 6.26). New best. Longer context (500ms vs 167ms) compounds with Self-Forcing — no TF metric regression, all secondary metrics improved. h10_action_acc +1.8pp. K=10 was undershooting; K=30 covers a full move sequence. Cost: $5.20. The two proven improvements (SF + longer context) are orthogonal and additive: 6.77 → 6.26 (SF) → 6.03 (context).

**E019a DISCARDED (K=50 context).** RC 5.97 (-1.0% vs e018c's 6.03). Marginal, barely outside ±0.05 noise band. Diminishing returns: K=10→K=30 gave -3.7%, K=30→K=50 gives -1.0%. SF loss jumped 58% (0.37→0.58). K=30 is the sweet spot — K=50 overshoots useful temporal structure. Context axis explored. Cost: $5.90. PR #3 closed.

**Autonomous loop note:** Director rejected 2 SF refinement hypotheses (ratio 30%, velocity-weighted loss) before approving K=50. Three rejections and one discarded result = the loop is correctly filtering bad ideas and exploring the frontier. Budget: $24.25/$30. Axes explored: SF refinements (closed), context length (ceiling found at K=30).

---

## 2026-03-14 — Trainer port, E019 baseline 6.77, autoresearch orchestration

Ported the full nojohns trainer. The AWM version was missing per-batch logging, velocity/dynamics/combat-context losses (6 heads!), num_workers, and epoch callbacks. The model was training on only ~40% of its output heads. No wonder the old migration runs were underperforming — the velocity and dynamics predictions were getting zero gradient.

E019 epoch 1 completed on 1.9K data: **rollout coherence = 6.77** (vs E012 = 6.84). Marginal improvement despite only 1 epoch vs E012's 2 — the full loss suite is helping. change_acc = 78.7% (lower than E012's 91.1%, expected with 1 epoch).

Discovered vel_mae=0 bug in rollout eval — `reconstruct_frame` wasn't applying velocity_delta or dynamics_pred to the AR frames. Velocities were frozen from the seed context. Fixed: AR reconstruction now uses all three prediction types (position deltas, velocity deltas, dynamics absolute).

Spent significant time debugging Modal data loading. The 7.7K dataset (82M frames, 43GB tensors) hangs with `num_workers=4` (fork OOM) and takes 9hr/epoch with `num_workers=0`. `share_memory_()` fails on Modal containers (shm too small). Pragmatic decision: use 1.9K data for Scout/Confirm experiments, fix 7.7K data loading as a separate infrastructure task.

Built the autoresearch orchestration: three agent roles (hypothesis/director/executor), research cycle brief, conductor prompt, budget tracking ($30/day), tiered cost gates. The design principle: the Director evaluates, doesn't hypothesize. The Researcher proposes, doesn't execute. Separation of concerns prevents the "agent that proposed the idea also evaluates its own results" failure mode.

Key feedback from Mattie: dead ends are observations not prohibitions — agents CAN revisit if they have specific reasoning for why context changed. Also: everything the agents need should be in program.md, since that's the single document Mattie operates from.

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
