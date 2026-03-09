# Run Card: E008b — Positional Conditioning

**Created**: 2026-02-28
**Branch**: scav/E008b-positional-conditioning
**Status**: COMPLETE — CHEATING

## Goal

Test whether adding a positional scalar to ctrl conditioning (telling the model which position to predict) enables accurate prediction from the last hidden state with future context visible.

## Results

| Metric | Value |
|--------|-------|
| change_acc | 99.4% |
| action_acc | ~100% |
| val_loss | 0.006 |

**Verdict**: Cheating. The SSM hidden state at position 9 has already processed frames 7-9 including the target. The model copies the answer from its input.

**Diagnostic value**: When evaluated without focal context (standard 10-frame past window), scored 17.8% change_acc — *worse* than E008a's 25.9% autoregressive. Proves the model learned to copy, not to predict physics.

## Data

| Field | Value |
|-------|-------|
| Encoded file | `encoded-2k.pt` (pre-v3) |
| Games | 2,000 |

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD) |
| Parameters | ~4.3M |
| context_len (K) | 10 |
| focal_offset (D) | 3 |

## Training

| Field | Value |
|-------|-------|
| GPU | H100 80GB |
| Epochs | 2 |
| Batch size | 4096 |
