# Autonomous World Model

A learned world model deployed onchain as an autonomous world. Trained on Melee replay data via the No Johns project, deployed on Solana using MagicBlock's ephemeral rollups and BOLT ECS.

## Architecture

This repo is the full pipeline: train → quantize → deploy → render. Model research, training infrastructure, and deployment all live here.

## Project Structure

```
autonomous-world-model/
├── models/           # Model definitions — Mamba2, MLP, PolicyMLP, encoding
├── data/             # Dataset loading — parse replays, build tensors
├── training/         # Training loops — Trainer, PolicyTrainer, metrics
├── scripts/          # Training & eval scripts — train.py, rollout.py, etc.
├── experiments/      # YAML experiment configs
├── configs/          # Match configs (fox_ditto_fd.yaml, etc.)
├── crank/            # Offchain match runner — standalone + Solana
├── quantization/     # INT8 quantization + accuracy testing
├── viz/              # State visualizer — renders world model output
├── research/         # Research notes, source papers, hitbox data
│   └── sources/      # Downloaded PDFs (gitignored) + summary .md files
├── checkpoints/      # Model weights — .pt files (gitignored)
├── site/             # "The Wire" — Next.js arena website
├── solana/           # Onchain code (Codex)
│   ├── syscall/      # sol_matmul_i8 native syscall implementation
│   ├── programs/     # Solana programs (world-model, cu-benchmark, syscall-test)
│   ├── programs-ecs/ # BOLT ECS components + systems
│   ├── client/       # TypeScript SDK (@awm/client)
│   ├── cli/          # Upload CLI tool
│   └── tests/        # Integration tests (Mocha)
├── docs/             # Architecture, specs, handoff, run cards
│   └── run-cards/    # Per-experiment run cards
└── RUNBOOK.md        # Training guide — how to run experiments
```

## Key Concepts

- **The model IS the world.** Learned rules become ground truth. Errors aren't bugs — they're the physics of a new world.
- **Arcade, not MMO.** Persistent weights/rules, not persistent world state. Sessions spin up in ephemeral rollups.
- **INT8 determinism for free.** Integer math is identical everywhere. Quantization solves both size and determinism.

## Development

### Agents

This project uses multiple Claude Code agents. No rigid directory ownership — any agent can edit any file. Use good judgment about what you're touching and coordinate via `docs/HANDOFF.md` when changes cross boundaries (especially onchain ↔ offchain).

**Codex** (OpenAI) owns `solana/` — if you need onchain changes, describe them in `docs/HANDOFF.md`.

### Interface Contracts

These are shared boundaries. Changes require coordination.

- **Binary wire format**: `PlayerState` = 32 bytes. `crank/solana_bridge.py` and `solana/programs-ecs/components/session-state/` must agree byte-for-byte. `models/encoding.py` (`EncodingConfig`) defines normalization scales and vocab sizes. Fixed-point: positions/velocities × 256, percent stored directly as u16.
- **TypeScript SDK**: `solana/client/src/` exports consumed by `site/`. Function signatures and type shapes are the contract.
- **JSON frame format**: `{ meta, stage_geometry, frames[] }` consumed by `viz/` and `site/`. Stable. See `viz/visualizer.html`.

### Review Gates

| What changed | Who reviews | Why |
|--------------|-------------|-----|
| Onchain programs, syscall, ECS | Model side reviews math/format | Hardest to undo, must match model |
| Weight format / encoding changes | Codex reviews | Must match onchain structs |
| Binary wire format (PlayerState) | All sides | Shared boundary |
| Everything else | No gate | Offchain, testable, reversible |

## Experiment Workflow

### Branching & PRs

- Clean branch names: `e018a-self-forcing`, not `scav/e018a` or `scaviefae/pr123`
- One PR per experiment, for reference not merging. PRs document what was tried and what happened.
- Close the PR when the experiment is complete (kept or discarded).
- Run cards live on `main`. Cards are the permanent record; PRs are the discussion.

### Run Card Schema

Cards live in `docs/run-cards/`. YAML frontmatter is machine-parseable for autoresearch agents.

```yaml
---
id: e018a
created: 2026-03-10
status: proposed | running | kept | discarded
type: hyperparameter | architectural | training-regime | data
base_build: b001           # versioned package of canonical findings (see docs/base-builds/)
built_on: [e017a]           # experiments on top of the base build, not yet canonized
source_paper: null          # arXiv ID if derived from a paper (e.g., 2508.13009)
rollout_coherence: null     # mean pos MAE over K=20 horizons (filled after eval)
prior_best_rc: null         # rollout coherence of the best prior experiment
---
```

**Required fields:** `id`, `created`, `status`, `type`, `base_build`, `built_on`
**Filled after eval:** `rollout_coherence`, `prior_best_rc`
**Optional:** `source_paper`

**`base_build`** is a versioned package of canonical findings — a stable set of design decisions that experiments build on. Defined in `docs/base-builds/{id}.yaml`. When enough experiments accumulate on top of a base build and prove out, mint a new one (b002, etc.). This is the one moment of editorial judgment in an otherwise bottom-up process.

**Status lifecycle:** `proposed` → `running` → `kept` or `discarded`

**Types:**
- `hyperparameter` — learning rate, loss weights, batch size, etc.
- `architectural` — model structure changes (new heads, different backbone, etc.)
- `training-regime` — how the model is trained (scheduled sampling, curriculum, etc.)
- `data` — dataset changes (more data, different filtering, new features)

The rest of the card body follows the existing format: Goal, Target Metrics, Data, Model, Training, etc.

### Paper → Experiment Pipeline

When a research paper has relevant techniques:

1. **Summary** lives in `research/sources/{arxiv-id}-summary.md` (summary, takeaways, applications, glossary)
2. **Proposed experiments** get cards with `status: proposed` and `source_paper: {arxiv-id}`
3. Chunk papers into independent, testable experiments. Each card tests one idea.
4. Cards reference the paper summary for motivation; the card itself specifies the concrete change.

### Epistemic Standards

State findings as observations with hit rates. Not editorials.

- No: "Weight decay on embeddings is a big deal"
- Yes: "WD 0.001 on embeddings improved rollout coherence in 3/3 experiments (e018a, e017a, e016). WD 0.005 regressed in 1/1."

## Reference Docs

| Doc | What's in it |
|-----|-------------|
| [docs/HANDOFF.md](docs/HANDOFF.md) | Active handoff — review requests, responses, status |
| [docs/autoresearch-plan.md](docs/autoresearch-plan.md) | Autoresearch loop design — rollout eval, program.md, citation graph |
| [docs/sol-matmul-i8-spec.md](docs/sol-matmul-i8-spec.md) | `sol_matmul_i8` syscall spec (shared with MagicBlock) |
| [docs/architecture-overview.md](docs/architecture-overview.md) | System architecture |
| [docs/cu-benchmark-findings.md](docs/cu-benchmark-findings.md) | CU measurements for INT8 ops |
| [docs/design-arena-mechanics.md](docs/design-arena-mechanics.md) | "The Wire" arena design |
| [docs/design-visual-ux.md](docs/design-visual-ux.md) | Visual/UX design for The Wire |
| [RUNBOOK.md](RUNBOOK.md) | Training guide — experiments, configs, data pipeline |
| [docs/MAMBA2-EXPLAINER.md](docs/MAMBA2-EXPLAINER.md) | Mamba-2 architecture explanation |
| [docs/RESEARCH-DIARY.md](docs/RESEARCH-DIARY.md) | Chronological research log |
| [docs/run-cards/](docs/run-cards/) | Per-experiment run cards (e008a–e019) |
| [docs/base-builds/](docs/base-builds/) | Base build definitions (b001+) — versioned canonical findings |
| [research/sources/](research/sources/) | Paper summaries (PDFs gitignored, .md summaries committed) |

## Related Projects

- **nojohns** — agent competition infrastructure, arena, Melee integration (tournament/community platform)
- **nojohns-training** — parsed replay data, training run outputs
- **rnd-2026** — research docs (`llms/world-models.md`, `projects/autonomous-world-model/README.md`)

## Model Output Format

The world model outputs one frame per timestep. Each frame contains per-player state matching the v2 encoding:

**Continuous (regression heads):** x, y, percent, shield_strength, speed_air_x, speed_y, speed_ground_x, speed_attack_x, speed_attack_y, state_age, hitlag, stocks

**Binary (classification):** facing, on_ground

**Categorical (classification heads):** action_state (400-class), jumps_left (8-class), character (33-class pass-through)

**Per-frame:** stage (33-class pass-through)

See `viz/visualizer.html` for the exact JSON shape consumed by the rendering layer.

## Onchain Target

- **Solana** via MagicBlock BOLT ECS + Ephemeral Rollups
- Weights stored on mainnet (permanent, forkable)
- Sessions in ephemeral rollups (10ms blocks, configurable CU, zero fees)
- 60fps achievable: 16.67ms frame budget - ~0.5-2ms inference - ~10ms block time

<!-- simple-loop:research -->

## Research Module

This project has the simple-loop research module installed. It enables autonomous research — iterative search, reading, synthesis, and coverage evaluation.

### How it works

Research briefs in `.loop/briefs/research-*.md` define questions to investigate. The research loop iterates: search → read → synthesize → evaluate coverage → repeat until questions are answered or max iterations reached.

### Key files

- `.loop/modules/research/state/findings.md` — accumulated findings (the output)
- `.loop/modules/research/state/coverage.json` — which questions are answered
- `.loop/modules/research/state/sources.json` — sources examined
- `.loop/briefs/research-*.md` — research briefs

### Persistent docs

When doing research work:
- **RESEARCH-LOG.md** — log what you searched, what you found, decisions made
- **HANDOFF.md** — summarize findings when a research brief completes
- **TROUBLESHOOTING.md** — if you hit errors during research (API failures, broken URLs, etc.)

<!-- /simple-loop:research -->

<!-- simple-loop:docs -->

## Docs Module

This project has a living documentation site built with [Zensical](https://github.com/squidfunk/zensical) (Material for MkDocs-compatible, Rust+Python, fast).

### Quick reference

```bash
# Build the site
python scripts/docs_prebuild.py && uvx zensical build

# Serve locally with live reload
uvx zensical serve --dev-addr 0.0.0.0:8000
```

### Key files

- `zensical.toml` — site config (nav, theme, extensions). The nav tree defines site structure.
- `docs/` — markdown source files. Any agent can write here.
- `scripts/docs_prebuild.py` — generates `docs/experiments/index.md` from run card frontmatter. Run before build.
- `.loop/modules/docs/config.json` — module config (port, dirs, prebuild script)
- `.loop/modules/docs/state/manifest.json` — page manifest mapping doc pages to source files (for change-driven regeneration)
- `.loop/modules/docs/state/site/` — build output (gitignored)

### Writing docs

- One concept per page. Use Mermaid fenced blocks for diagrams.
- Update `nav` in `zensical.toml` when adding/removing pages.
- Run `python scripts/docs_prebuild.py` to regenerate the experiment index after changing run cards.

### Upstream

This module is being upstreamed to `simple-loop/modules/docs/`. The AWM installation is the first deployment.

<!-- /simple-loop:docs -->
