# Cloud GPU Operations

How we run training experiments on cloud GPUs. This is a living guide — update it as we learn.

## Current setup

**Platform:** Modal (modal.com)
**Default GPU:** L4 (24GB, $0.80/hr)
**Training script:** `scripts/modal_train.py`
**Data:** Pre-encoded `.pt` files on Modal volume `melee-training-data`

## GPU tiers tested

| GPU | VRAM | $/hr | TF batch | SF batch | Max bs (with SF) | Cost/epoch (1.9K) | Status |
|-----|------|------|----------|----------|-------------------|-------------------|--------|
| **L4** | 24 GB | **$0.80** | 371ms | ~1,500ms | **256** | **$5.22** (bs=256) | **Default** — validated, SF OOMs at bs=512 |
| T4 | 16 GB | $0.59 | 452ms | untested | ≤256 est. | ~$6.40 est. | Diagnostic passes, full epoch untested |
| A100 40GB | 40 GB | $2.10 | ~290ms | ~870ms | **512** | **$5.25** | Works end-to-end at bs=512 |
| A10G | 24 GB | $1.10 | untested | untested | 256 est. | untested | Same VRAM as L4, slightly faster |
| A100 80GB | 80 GB | $2.50 | — | — | 512+ | — | Overkill |

**Key finding:** L4 at bs=256 costs the same as A100 at bs=512 (~$5.22 vs $5.25). The cheaper GPU doesn't save money because Self-Forcing requires halving the batch size to fit in 24GB VRAM.

## VRAM breakdown

Our model uses **5.9 GB peak VRAM** for a single TF forward+backward pass at bs=512. But Self-Forcing (3 forward+backward passes per SF batch) pushes peak VRAM higher.

| Component | VRAM |
|-----------|------|
| Model parameters (4.35M × 4 bytes) | ~17 MB |
| Gradients | ~17 MB |
| AdamW optimizer states | ~35 MB |
| **Static total** | **~70 MB** |
| Activations (bs=512, 1 forward pass) | **~5.8 GB** |
| **TF batch peak** | **~5.9 GB** |
| SF batch (3 forward+backward, per-step) | **~12-18 GB est.** |

The activation memory dominates. It scales linearly with batch size:
- bs=512: ~5.9 GB per forward pass
- bs=256: ~3.0 GB per forward pass
- bs=128: ~1.5 GB per forward pass

### Why Mamba2 uses more VRAM than expected

The SSD (Structured State Space Duality) scan in Mamba2 has disproportionate memory overhead at small model sizes. This is documented in [Mamba GitHub issues #439](https://github.com/state-spaces/mamba/issues/439) and [#495](https://github.com/state-spaces/mamba/issues/495):

- Mamba2 uses ~57% more VRAM than Mamba1 for small models
- The 4-tensor einsum in `_ssd_scan` creates large intermediate tensors
- A comparable transformer at seq_len=30 would use ~3-4 GB (30×30 attention matrix is trivially small)
- At large scale this overhead amortizes; at 4.35M params it does not

### Self-Forcing VRAM issue

SF batches do 3 sequential forward+backward passes. Even with per-step backward (freeing the computation graph after each step), the SSD scan intermediates during a single forward pass can push past 24GB at bs=512. This is why **L4 requires bs≤256 for SF training**.

See [issue #7](https://github.com/ScavieFae/autonomous-world-model/issues/7) for Scav's detailed VRAM analysis including the einsum decomposition fix.

## Training cost breakdown

At L4 bs=256, with Self-Forcing enabled (20% of batches):

| Component | Batches | Time/batch | Total time | Cost | % of total |
|-----------|---------|-----------|-----------|------|------------|
| TF batches (80%) | ~48,800 | 386ms | 5.2hr | $4.18 | 80% |
| SF batches (20%) | ~12,200 | ~1,500ms | 5.1hr | — | — |
| **Total** | **~61,000** | — | **~6.5hr** | **$5.22** | |

SF is 20% of batches but contributes ~50% of wall time because each SF batch does 3 forward+backward passes plus CPU-side reconstruction.

## How we compare to other world models

Our $2-5/epoch is **cheap for the world model space**:

| Project | Params | Data type | Training cost | Notes |
|---------|--------|-----------|---------------|-------|
| **AWM (ours)** | **4.35M** | **Game state tensors** | **$2-5/epoch** | State-based, not pixels |
| DreamerV3 (Atari) | 12M | Game pixels | $2-4/run | Similar scale |
| DIAMOND (Atari) | ~4M | Game pixels | $17-28/run | 2.9 days on 4090 |
| IRIS (Atari) | ~60-100M | Game pixels | ~$1,400/game | 8×A100, 3.5 days |
| DreamerV3 (Minecraft) | 200M | Game pixels | ~$856 | 17 GPU-days |

**Why we're cheap:** We train on compact game state tensors (~300 floats per frame), not pixels or video. Pixel-based world models spend 90%+ of compute on visual encoding/decoding.

**SSM vs Transformer at our scale:** A transformer of the same size would use ~30% less VRAM and be ~10-20% faster for training at seq_len=30 (attention is trivially cheap at that length). But the SSM is correct for our deployment target — constant-time inference with fixed 200KB state for onchain execution.

## Training throughput

| Metric | L4 | A100 | Notes |
|--------|-----|------|-------|
| Samples/sec (TF) | ~1,380 | ~1,766 | 512 / batch_time |
| Samples/sec (mixed TF+SF) | ~787 | ~1,000 est. | Accounting for SF overhead |
| Batches/epoch (bs=512) | 30,500 | 30,500 | Same data, same batches |
| Batches/epoch (bs=256) | 61,000 | — | Double batches at half size |

For reference, a simple MLP of the same parameter count would achieve 5,000-50,000 samples/sec. The SSM overhead is the price of temporal modeling.

## How to test a new GPU

**Always use the diagnostic script first.** Never switch the production training script to an untested GPU.

```bash
# Test on a specific GPU:
modal run scripts/gpu_diagnostic.py --gpu T4
modal run scripts/gpu_diagnostic.py --gpu L4
modal run scripts/gpu_diagnostic.py --gpu A10G

# Run in foreground (not --detach) so errors are visible
```

The diagnostic tests each pipeline stage independently with timing and VRAM measurements. **Note:** The diagnostic only tests TF batches. SF batches use more VRAM — a GPU passing the diagnostic may still OOM during real training with SF.

## How to switch GPUs

1. Run `scripts/gpu_diagnostic.py` on the target GPU
2. Confirm all steps pass and note the batch time
3. Check VRAM headroom: if peak < 50% of total VRAM, bs=512 with SF should work. If peak > 50%, use bs=256.
4. Edit `scripts/modal_train.py`: `gpu="L4"` (appears twice — training and eval functions)
5. Launch a real experiment and verify wandb receives batch data within 10 minutes

## Critical lesson: Modal --detach swallows errors

**This cost us 3 days.** See [issue #7](https://github.com/ScavieFae/autonomous-world-model/issues/7).

When you launch with `modal run --detach`, the local process disconnects. If the remote function crashes, you see nothing — the wandb run shows `state: running` with zero data, which looks identical to a hang.

**Rules:**
- Diagnose with `modal run` (foreground), never `--detach`
- Test GPU changes with `scripts/gpu_diagnostic.py` before switching production
- If wandb shows `state: running` + `_runtime: 0` + empty config: it crashed, it's not hanging

## Data access

Training data lives on the Modal volume `melee-training-data`:

| File | Size | Contents | L4 compatible? |
|------|------|----------|---------------|
| `encoded-e012-fd-top5.pt` | ~3.5 GB | 1.9K games, FD top-5 | Yes |
| `encoded-v3-ranked-fd-top5.pt` | ~15 GB | 7.7K games | **No** — crashes on L4 (system RAM OOM) |

### 7.7K data scaling blocker

The 7.7K dataset crashes on L4 before training starts — system RAM (not GPU VRAM) runs out loading the 15GB `.pt` file plus dataset construction overhead. A100 containers have more system RAM and can load it, but previously had 9hr/epoch with `num_workers=0`.

**Status:** The fast batch loader (`get_batch`) should fix the data loading bottleneck on A100. Untested.

**Possible fixes for L4:**
- Memory-mapped loading (`torch.load` with `mmap=True`)
- Pre-shard the dataset into smaller chunks
- Use A100 only for 7.7K experiments

## Cost projections

At current rates with L4 bs=256 (~$5/experiment):

| Volume | Daily | Monthly | Annually |
|--------|-------|---------|----------|
| 3 experiments/day | $15 | $450 | $5,400 |
| 5 experiments/day | $25 | $750 | $9,000 |
| 10 experiments/day (parallel) | $50 | $1,500 | $18,000 |

### Optimization opportunities

| Optimization | Est. savings | Difficulty |
|-------------|-------------|------------|
| **SSD einsum decomposition** (reduce SF VRAM → bs=512 on L4) | ~20% (fewer batches) | Medium — code change in mamba2.py |
| **Gradient checkpointing** (recompute forward during backward) | bs=512 on L4 | Medium — wrap Mamba2Block |
| **A100 with fast loader for 7.7K** | Enables data scaling | Low — test existing code |
| **Vast.ai / Nosana** | 30-70% cheaper GPU hours | Medium — Docker image + data access |
| **Local MPS training** | $0 compute | Low — needs data download to external drive |

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
