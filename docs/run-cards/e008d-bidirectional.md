# Run Card: E008d — Bidirectional Mamba

**Created**: 2026-02-28
**Branch**: scav/E008d-bidirectional
**Status**: COMPLETE — CHEATING

## Goal

Add a reverse Mamba pass so each position's hidden state combines forward (past) and backward (future) information.

## Results

| Metric | Value |
|--------|-------|
| change_acc | ~99% |

**Verdict**: Cheating, same mechanism as E008b. The backward pass explicitly gives the hidden state access to future frames including the target. Useful only as a diagnostic — confirms the SSM architecture can propagate temporal information.

## Data

| Field | Value |
|-------|-------|
| Encoded file | `encoded-2k.pt` (pre-v3) |
| Games | 2,000 |

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 + backward pass |
| context_len (K) | 10 |
| focal_offset (D) | 3 |

## Training

| Field | Value |
|-------|-------|
| GPU | H100 80GB |
| Epochs | 2 |
