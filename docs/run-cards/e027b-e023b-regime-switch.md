---
id: e027b
created: 2026-03-22
status: discarded
type: training-regime
base_build: b002
built_on: [e026c]
source_paper: null
rollout_coherence: 5.225
prior_best_rc: 4.965
---

# Run Card: e027b-e023b-regime-switch

## Goal

Test whether resuming from e023b and running a SF curriculum at N=4/5 for epoch 2 can achieve comparable results to e026c's full progressive ramp. This would validate a shortcut: skip the N=1/N=2 curriculum stages and jump straight to harder horizons, leveraging e023b's pre-trained weights as the foundation.

If successful, this would mean existing checkpoints can be upgraded to longer SF horizons without full retraining.

## What Changes

Resume from e023b checkpoint (d_model=768, 1 epoch standard training). Run epoch 2 with SF curriculum [4, 5] — two stages at longer horizons than e026c's [1, 2, 3].

## Target Metrics

- **Keep:** RC < 4.965 (improvement over e026c)
- **Kill:** RC > 5.2 (regression beyond prior best)

## Model

Identical to b002: d_model=768, d_state=64, n_layers=4, headdim=64, 15,794,548 params.

## Training

- Resume from: e023b checkpoint
- Epoch 2: SF curriculum [4, 5]
- lr: 5e-4, weight_decay: 1e-5, batch_size: 512
- AMP: enabled
- Self-Forcing: ratio=4 (20%)

## Cost

~$5 (A100 40GB with AMP, ~3hr).

## Confounds

- e023b was trained without warmup, unimix, or curriculum — it lacks the progressive foundation that e026c built.
- Jumping from no SF curriculum to N=4/5 is a bigger regime shift than e026c's smooth N=1->2->3 ramp.
- The e023b checkpoint may have converged to a basin that isn't compatible with high-N SF horizons.

## Results

| Metric | E026c (prior best) | E027b | Delta |
|--------|-------------------|-------|-------|
| **rollout_coherence** | **4.965** | **5.225** | **+5.2%** |
| change_acc | 80.2% | 77.2% | -3.0pp |
| pos_mae | 0.642 | 0.692 | +7.8% |

- **wandb:** 6jps8j6w
- **Cost:** ~$5

## Director Evaluation

**Verdict: DISCARDED — clear regression**

RC 5.225 is a 5.2% regression from e026c (4.965). All metrics worse: change_acc -3.0pp, pos_mae +7.8%.

Jumping from N=3 to N=4/5 without the N=1/N=2 foundation does not work. The full progressive ramp (e026c: N=1->2->3) is needed — each curriculum stage builds error-correction capacity that later stages depend on. The e023b checkpoint, trained without any SF curriculum, lacks this foundation entirely.

This strengthens the curriculum hypothesis from e026c: progressive SF works because each stage prepares the model for the next, not just because longer horizons are trained. Skipping stages breaks the chain.

**Implication:** To extend the curriculum to N=4/5, the path is e026c + continued curriculum [4, 5], not cold-starting from a pre-SF checkpoint.
