# Cloud GPU Operations

How we run training experiments on cloud GPUs. This is a living guide — update it as we learn.

## Current setup

**Platform:** Modal (modal.com)
**Default GPU:** L4 (24GB, $0.80/hr)
**Training script:** `scripts/modal_train.py`
**Data:** Pre-encoded `.pt` files on Modal volume `melee-training-data`

## GPU tiers tested

| GPU | VRAM | $/hr | Batch time | Est. epoch (1.9K) | Cost/experiment | Status |
|-----|------|------|-----------|-------------------|----------------|--------|
| **L4** | 24 GB | **$0.80** | **371ms** | **3.1hr** | **$2.48** | **Default** — validated via diagnostic |
| A100 40GB | 40 GB | $2.10 | ~290ms | ~2.5hr | $5.25 | Works — previous default, expensive |
| T4 | 16 GB | $0.59 | unknown | unknown | ~$1.50 est. | Likely works (entered training loop, not fully validated) |
| A10G | 24 GB | $1.10 | unknown | unknown | unknown | Untested |
| A100 80GB | 80 GB | $2.50 | — | — | — | Overkill for our model |

**Our model uses 5.9 GB peak VRAM.** Any GPU with 16+ GB works. The cost difference is entirely about compute speed, not memory.

## How to test a new GPU

**Always use the diagnostic script first.** Never switch the production training script to an untested GPU.

```bash
# Test on a specific GPU:
modal run scripts/gpu_diagnostic.py --gpu T4
modal run scripts/gpu_diagnostic.py --gpu L4
modal run scripts/gpu_diagnostic.py --gpu A10G

# Run in foreground (not --detach) so errors are visible
```

The diagnostic tests each pipeline stage independently:
1. GPU info + basic CUDA ops
2. Load 3.5GB pre-encoded data
3. Reconstruct dataset + share_memory_()
4. Build DataLoader (num_workers=0)
5. First batch
6. Build model + move to GPU
7. Forward pass
8. Backward pass
9. Optimizer step
10. 10 full training batches with timing
11. DataLoader with num_workers=4 (with 60s timeout)

If any step fails, you know exactly where the issue is.

## How to switch GPUs

1. Run `scripts/gpu_diagnostic.py` on the target GPU
2. Confirm all steps pass and note the batch time
3. Edit `scripts/modal_train.py` line 65: `gpu="L4"` → `gpu="T4"` (appears twice — training and eval functions)
4. Launch a real experiment and verify wandb receives batch data within 10 minutes

## Critical lesson: Modal --detach swallows errors

**This cost us 3 days.** See [issue #7](https://github.com/ScavieFae/autonomous-world-model/issues/7).

When you launch with `modal run --detach`, the local process disconnects. If the remote function crashes, you see nothing — the wandb run shows `state: running` with zero data, which looks identical to a hang. You can't tell the difference between "crashed silently" and "actually hanging" from the outside.

**Rules:**
- Diagnose with `modal run` (foreground), never `--detach`
- Test GPU changes with `scripts/gpu_diagnostic.py` before switching production
- If wandb shows `state: running` + `_runtime: 0` + empty config: it crashed, it's not hanging
- Use `modal app logs <app-id>` to fetch remote logs (may not work for detached apps)

## Data access

Training data lives on the Modal volume `melee-training-data`:

| File | Size | Contents |
|------|------|----------|
| `encoded-e012-fd-top5.pt` | ~3.5 GB | 1.9K games, FD top-5 characters |
| `encoded-v3-ranked-fd-top5.pt` | ~15 GB | 7.7K games (data loading bottleneck — see below) |

### Data loading bottleneck on 7.7K dataset

The 7.7K dataset (`encoded-v3-ranked-fd-top5.pt`) takes 9hr/epoch with `num_workers=0`. With `num_workers=4`, it OOMs or deadlocks on some containers. This is a known blocker for data scaling experiments.

**Possible fixes (untested):**
- Pre-shard the dataset into smaller chunks
- Use streaming/memory-mapped loading instead of loading entire .pt into RAM
- Increase Modal container memory allocation

### Downloading data locally

`modal volume get` corrupts large files (3.5GB). Multiple attempts all produce invalid .pt files. **Don't download data to local.** Keep it on Modal or upload to S3/R2 from a Modal function.

## Cost optimization

At 5 experiments/day on L4:
- **Daily:** $12.40
- **Monthly:** $372
- **Annually:** $4,464

Compared to A100:
- **Daily:** $26.25
- **Monthly:** $788
- **Annually:** $9,450

**Savings: 53% ($5,000/year)**

### Future cost reduction paths

| Platform | GPU | $/hr | Est. savings vs L4 | Status |
|----------|-----|------|--------------------|--------|
| Modal T4 | T4 16GB | $0.59 | 26% cheaper | Likely works, needs validation |
| Vast.ai 4090 | RTX 4090 24GB | $0.25-0.40 | 50-69% cheaper | Untested, needs Docker image |
| Nosana | A100 40GB | $0.61 | 24% cheaper | Untested, Solana-native |
| Local MPS | M3 Pro | $0 | 100% cheaper | Script ready, data download blocked |

## Monitoring

- **wandb:** All experiments log to `shinewave/melee-worldmodel`
- **Modal dashboard:** modal.com/apps/scaviefae/main
- **Budget tracking:** `.loop/state/budget.json`
- **Matrix:** #conductor-log for launches, #experiment-results for outcomes

## Files

| File | Purpose |
|------|---------|
| `scripts/modal_train.py` | Modal training function (GPU selection on line 65) |
| `scripts/gpu_diagnostic.py` | Step-by-step GPU diagnostic |
| `scripts/train_local.py` | Local MPS training (data download needed) |
| `scripts/check_run.py` | Check wandb run status |
| `docs/TROUBLESHOOTING.md` | Known issues and resolutions |
