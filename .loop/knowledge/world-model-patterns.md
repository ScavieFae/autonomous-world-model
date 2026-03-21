# World Model Training Patterns

Patterns from the world model literature relevant to our architecture (state-space world model for game frame prediction).

## The Train/Inference Gap

The central problem. During training the model sees ground truth history. During inference it sees its own predictions. Errors compound.

**Approaches in literature:**
- **Scheduled Sampling** (Bengio et al., 2015) — gradually replace teacher inputs with model predictions during training. Our E012b noise-based variant was a null result. True SS (using actual model predictions) is the mechanism that works.
- **Self-Forcing** (Adobe/UT Austin, 2025) — train on own predictions from the start. Our best results (e018a RC=6.26, 7.5% improvement). The key insight: the model must learn to recover from errors, not avoid them.
- **DAgger-style** (Ross et al., 2011) — iteratively collect data under the learned policy. Not applicable to our offline dataset.

**Our empirical finding:** The lever isn't *what* we predict — it's *how we train*. TF metrics don't predict AR quality (e016 had 38% better val_loss than e012 but worse AR rollouts).

## Autoregressive Drift

Model errors accumulate over rollout horizons. Position drifts, velocities diverge, action states desynchronize.

**What we know:**
- Rollout coherence (mean position MAE over K=20 horizons) is the north star metric
- Current best: RC=6.03 (e018c, rolling context K=30 + Self-Forcing)
- Absolute y encoding (e017a) traded drift for jitter — structural improvement
- Absolute x (e017c) caused oscillation. Absolute velocities (e017d) caused head decoupling.

## Curriculum and Horizon

**From literature:**
- DreamerV3 uses imagination horizons of 15 steps for policy learning
- IRIS trains on sequences of 20 frames
- DIAMOND uses diffusion over 4-frame windows

**Our findings:**
- Truncated BPTT saturated at N=3 (e018b)
- Horizon weighting made it worse (e018d)
- Rolling context window K=30 is current best (e018c)
- Context K=50 was marginal (e019a, RC=5.97 but discarded)

## Multi-Head Prediction

Our model predicts 10+ output types simultaneously: continuous (position, velocity), binary (facing, grounded), categorical (action_state, jumps_left).

**Patterns:**
- Loss weighting across heads matters. action_change_weight=5.0 is proven (+9.9pp)
- Cascaded heads (e014) — using position predictions to inform action predictions — fixed damage drift but overfitted at 1.9K games
- Velocity and dynamics heads were getting zero gradient early on (caught and fixed)

## Deep Dives

- `docs/research-canon.md` — full research canon with paper list
- `research/sources/` — paper summaries (PDFs gitignored, .md summaries committed)
- `program.md` — research directions and core insights
