# Run Card: E008a — Tap Focal Hidden State

**Created**: 2026-02-28
**Config**: `worldmodel/research/experiments/E008-focal-context/config.yaml`
**Branch**: scav/E008a-tap-focal-hidden
**Status**: COMPLETE

## Goal

Test whether tapping the SSM hidden state at a position *before* future context frames (focal_offset=3) produces better representations than reading the last position. The causal SSM guarantees position 6 can't see positions 7-9, but gradients from position 6's loss flow through weights that process future frames.

## Results

| Metric | Value | Baseline | Delta |
|--------|-------|----------|-------|
| change_acc | **93.0%** | 67.3% | **+25.7pp** |
| action_acc | 96.3% | — | — |
| pos_mae | 0.65 | 0.65 | 0 |
| val_loss | 0.314 | 0.291 | +0.023 |

**Verdict**: Not cheating (causal SSM verified), but 93% is teacher-forced with real future context at positions 7-9. Autoregressive mode was "better but still a mess" — train/eval mismatch. The model expects future frames that don't exist at inference.

## Data

| Field | Value |
|-------|-------|
| Encoded file | `encoded-2k.pt` (pre-v3) |
| Games | 2,000 |
| Encoding | state_flags=false, hitstun=false, projectiles=false |

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD) |
| Parameters | 4,282,386 |
| d_model / n_layers | 384 / 4 |
| context_len (K) | 10 |
| focal_offset (D) | 3 |

## Training

| Field | Value |
|-------|-------|
| GPU | H100 80GB |
| Epochs | 2 |
| Batch size | 4096 |
| LR | 5e-4 |
| Wall time | ~35 min |

## Key Findings

- Causal SSM information flow verified: SSD chunked computation is mathematically equivalent to sequential scan with lower-triangular mask on Q*K^T.
- 93% change_acc is the highest of any non-cheating experiment, but doesn't survive at inference.
- E008b trained with the same approach but evaluated without future context scored 17.8% — the cheater model was worse than baseline when you remove the cheat sheet.
- Graduated to E009 combined with E008c (multi-position) to address train/eval mismatch.
