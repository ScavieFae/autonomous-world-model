---
id: e026a
created: 2026-03-21
status: proposed
type: training-regime
base_build: b002
built_on: []
source_paper: null
rollout_coherence: null
prior_best_rc: 5.146
---

# Run Card: e026a-muon

## Goal

Test whether the Muon optimizer improves convergence over AdamW for 2D weight matrices. Muon uses Newton-Schulz orthogonalized SGD — 3 iterations to approximate the matrix square root inverse of G@G^T, giving orthogonalized gradient updates. Embeddings, biases, and 1D parameters remain on AdamW. Reference: https://github.com/KellerJordan/Muon

Muon has shown faster convergence in language model training. This tests whether the same applies to our Mamba-2 world model at d_model=768.

## What Changes

One change: `training.optimizer: muon` with `muon_lr: 0.02` (Muon default) and `adamw_lr: 3e-4` for non-weight params. The base AdamW lr (5e-4) is still passed but only used as fallback. All other hyperparameters identical to b002.

**Parameter split:** 2D+ weight matrices (in_proj, out_proj, conv1d, frame_proj, ctrl_proj, all heads) get Muon updates. Embeddings (action, jumps, character, stage, etc.) and biases get AdamW.

## Target Metrics

- **Keep:** RC < 5.146 (improvement over b002)
- **Kill:** RC > 5.3 or training instability (NaN, loss spikes)

## Model

Identical to b002: d_model=768, d_state=64, n_layers=4, headdim=64, 15,648,882 params.

## Training

- **optimizer: muon** (Muon for 2D+ weights, AdamW for rest)
- muon_lr: 0.02, adamw_lr: 3e-4, weight_decay: 1e-5
- batch_size: 512, 1 epoch, warmup_pct: 0.05, AMP: true
- Self-Forcing: ratio=4 (20%), unroll_length=3
- context_len=30, chunk_size=15

## Cost

~$5 (A100 40GB, ~3hr).

## Confounds

- Muon LR (0.02) is the default from the reference implementation. This is ~40x higher than the AdamW LR (5e-4). If results are bad, it may be a LR scaling issue rather than the optimizer itself.
- Warmup interacts differently with Muon vs AdamW — the warmup schedule only modifies the base LR, not Muon's internal momentum.
- Newton-Schulz iteration adds ~3 matrix multiplications per 2D parameter per step. May increase step time slightly on A100.

## Results

*Pending — status: proposed*
