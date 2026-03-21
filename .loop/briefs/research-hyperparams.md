# Research Brief: Orthogonal Hyperparameters

Sweep batch size, dropout, and evaluate the Muon optimizer. These are independent of the SF and architecture axes — orthogonal improvements that could compound.

## Context

Read `program.md` sections on batch size sweep (#6), dropout (in architecture #7), and Muon optimizer (#4).

## Experiments (priority order)

### 1. Batch Size Sweep (highest priority)
Karpathy's autoresearch found halving batch size was his single largest improvement. We've locked bs=512 without ever testing alternatives.

- **E-bs256**: bs=256, lr=2.5e-4 (linear scaling rule: halve BS → halve LR)
- **E-bs1024**: bs=1024, lr=1e-3 (double BS → double LR)
- Base: E018c config (SF + K=30, bs=512, lr=5e-4)
- Both are ~$5, config-only changes

### 2. Dropout Sweep
Currently dropout=0.1. Never tested alternatives.

- **E-drop0**: dropout=0.0
- **E-drop03**: dropout=0.3
- Base: E018c config
- ~$5 each

### 3. Muon Optimizer (requires code change)
Newton-Schulz orthogonalized SGD for weight matrices, AdamW for embeddings/scalars. Strong results in Karpathy's autoresearch on transformers. Open question whether it helps Mamba2 (different weight matrix structure).

- **Research first**: search for Muon + SSM/Mamba results in the literature. If no evidence it helps SSMs, deprioritize.
- **If promising**: implement in training loop, test against AdamW baseline
- Estimated ~$10 (implementation time + run)

## Protocol

- All experiments start from `experiments/e018c-context-k30.yaml`
- Change ONE thing per experiment
- 1.9K data, 1 epoch
- Compare RC to 6.03 (E018c)
- Run batch size experiments first — they're config-only and most likely to show signal based on external evidence

## Decision Rules

- If bs=256 improves RC: test bs=128
- If bs=1024 improves RC: test bs=2048 (if memory allows)
- If dropout=0.0 improves RC: regularization is hurting at this data scale
- If dropout=0.3 improves RC: model is overfitting at 1.9K
