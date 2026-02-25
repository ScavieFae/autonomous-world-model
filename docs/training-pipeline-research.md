# Training Pipeline Research: Cloud GPU + Data Best Practices

*Feb 25, 2026 — Research into why cloud GPU training has been painful and what a pro pipeline looks like.*

> **Status check (end of Feb 25):** We have never completed an epoch on a cloud GPU. The pre-encode → `.pt` → `torch.load` approach got us further than anything before (14s data load on A100, GPU activity visible), but training completion, checkpoint saving, and metric logging are all unverified. Treat everything below as "best available plan" not "proven pipeline."

## The Data

**22K games, 3.4GB compressed.** Each game is a zlib-compressed parquet file (~154KB avg), containing per-frame state for two players across ~9,400 frames. After encoding, the full dataset loads into ~6GB of RAM as contiguous tensors. Even at 230K games, you're looking at maybe 30-50GB — comfortably small by any ML standard.

For context: ImageNet is 150GB. Common Crawl is petabytes. People train 7B-param LLMs on terabytes. Our 4.3M-param Mamba-2 on 3.4GB of game state is a *tiny* workload by industry standards.

### Per-Game Data Shape

Per-player per-frame:
- **Continuous (float32):** x, y, percent, shield_strength, speed_air_x, speed_y, speed_ground_x, speed_attack_x, speed_attack_y, state_age, hitlag, stocks, combo_count (13 fields)
- **Binary (bool→float32):** facing, invulnerable, on_ground
- **Categorical (int64):** action (400 classes), jumps_left (8), character (33), l_cancel (3), hurtbox_state (3), ground (32), last_attack_landed (64)
- **Controller (float32):** main_stick_x/y, c_stick_x/y, shoulder, 8 button states

After encoding with embeddings + context window K=10: **1,820+ input dims** per sample.

---

## Why Cloud GPU Has Been Painful

**The problem isn't data scale. It's infrastructure friction on every single step.**

The actual kill list from RunPod sessions:

1. **RunPod's SSH proxy forces PTY allocation** — rsync and scp don't work through it. We ended up using croc (P2P file transfer) to get 3.4GB of data onto a pod. That's a workaround, not a workflow.

2. **Pip packages don't survive pod restarts** — installed to container disk, not the network volume. Every resume means reinstalling dependencies.

3. **Direct TCP ports were connection-refused** — the ip:port shown in pod details just didn't work on that host. No clear reason.

4. **wandb crashes hard on missing API key** — newer versions (0.25+) throw instead of falling back gracefully. One missing env var = dead training run.

5. **Data loading on network volumes is slow** — 10 minutes to load 2,200 games from RunPod's network volume. That's the network filesystem penalty on many small files.

6. **The pre-encode-upload-train loop is manual** — pre-encode locally to `.pt`, `modal volume put` to upload, then `modal run` to train. Three separate steps, each with its own failure modes.

Every one of these is a papercut. None of them are fundamental. But stacked together, they turn a 2-hour training run into a full-day debugging session.

---

## What a Pro Pipeline Looks Like

The gap between our current setup and what labs actually do comes down to three things.

### 1. Sharded Data, Not Individual Files

22K games as 22K individual parquet files is the root of the I/O pain. Every file open is a syscall; on network filesystems, that gets catastrophically slow.

**The fix:** Pack games into ~200MB tar shards (~400-500 games each). WebDataset reads these sequentially, converting 22K random reads into ~50 sequential reads. Works identically on local disk, Modal volumes, or S3.

```
shards/
  train-000000.tar   # ~500 games, ~200MB
  train-000001.tar
  ...
  train-000044.tar
```

**But:** At 3.4GB total, pre-encoding the whole thing into a single `.pt` file (which `modal_train.py` already does) skips the streaming complexity entirely. The pre-encoded file for 22K games fits in RAM on any GPU instance. Even 230K games as a single tensor would be ~30-40GB — fits on an A100 80GB.

**Sharding becomes necessary at 230K+ games.** Don't build it until then.

### 2. Infrastructure-as-Code, Not SSH Sessions

This is where Modal shines vs. RunPod. The `combat-context-heads` branch already has `modal_train.py` — the bones are right:

```python
@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("wandb")],
)
def train(epochs: int, encoded_file: str):
    # Data is already on the volume. Just load and train.
```

No SSH. No pip installs that vanish. No croc transfers. No connection-refused mysteries. The image definition pins dependencies, the volume persists data, the secret injects wandb. One command: `modal run modal_train.py`.

### 3. Parallel Experiment Sweeps

The killer feature for our use case: `train.starmap(configs)` — spin up 8 A100s simultaneously, each running a different experiment config. Round 1 experiments (1a, 2a, 3a and their stacks) could all run in parallel instead of sequentially on ScavieFae.

```python
@app.local_entrypoint()
def sweep():
    configs = [
        {"name": "exp-1a", "state_age_as_embed": True},
        {"name": "exp-2a", "press_events": True},
        {"name": "exp-3a", "lookahead": 1},
        {"name": "exp-1a-2a", "state_age_as_embed": True, "press_events": True},
    ]
    results = list(train.starmap(configs))
```

---

## Modal.com Deep Dive

### How It Works

Modal is serverless cloud compute defined in Python. You decorate functions with what they need (GPU, image, volumes, secrets), and Modal provisions ephemeral containers to run them. No Dockerfiles, no YAML, no Kubernetes.

```python
import modal

app = modal.App("melee-world-model")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.0", "wandb", "numpy", "pyarrow"
)

vol = modal.Volume.from_name("melee-training-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("melee-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",
    timeout=14400,
    volumes={"/data": vol, "/checkpoints": ckpt_vol},
    secrets=[modal.Secret.from_name("wandb")],
)
def train(config: dict):
    import torch
    import wandb
    wandb.init(project="melee-worldmodel", config=config)
    # ... training loop ...
    torch.save(model.state_dict(), "/checkpoints/best.pt")
    ckpt_vol.commit()
```

### Data Handling

**Modal Volumes** — distributed filesystem, mounted into functions:
- Create: `modal volume create melee-training-data`
- Upload: `modal volume put melee-training-data ./data/encoded-22k.pt /encoded-22k.pt`
- Mount: `volumes={"/data": vol}` in function decorator
- Persist writes: `vol.commit()` after writing
- **Free storage** — no charges for volume data, only compute

**Consistency model:** Not POSIX. Must call `vol.commit()` to persist writes, `vol.reload()` to see changes from other containers. Last-write-wins for concurrent modifications.

**V1 vs V2 volumes:**
- V1 (stable): max ~500K files, recommend <50K, ~5 concurrent writers
- V2 (beta): unlimited files, hundreds of concurrent writers, up to 2.5 GB/s

**CloudBucketMount** — mount S3/R2/GCS directly into functions:
```python
s3_mount = modal.CloudBucketMount("my-bucket", secret=modal.Secret.from_name("aws"), read_only=True)

@app.function(volumes={"/data": s3_mount})
def train():
    # read from /data/ as if local
```

Cloudflare R2 recommended for large datasets (zero egress fees).

### Available GPUs + Pricing

| GPU | VRAM | $/hr | Notes |
|-----|------|------|-------|
| T4 | 16GB | $0.59 | Budget option |
| L4 | 24GB | $0.80 | Good value for small models |
| A10 | 24GB | $1.10 | |
| A100 40GB | 40GB | $2.10 | Sweet spot for our model |
| A100 80GB | 80GB | $2.50 | Needed at 230K+ games |
| H100 | 80GB | $3.95 | Overkill for us right now |

Per-second billing. $30/mo free credits on Starter plan.

### Non-Obvious Advantages

1. **Memory snapshots (CRIU-based)** — snapshot container state after `import torch` + model load. Next cold start restores in ~1s instead of re-importing. Matters when iterating fast.

2. **Detached runs** — `modal run --detach` starts the job and disconnects your terminal. No SSH sessions dying at 3am.

3. **Image layer caching** — `pip_install("torch")` only runs once. Subsequent deploys that only change your code rebuild in seconds.

4. **Secrets management** — `modal.Secret.from_name("wandb")` injects env vars. No `.env` files, no hardcoded keys.

5. **Free volume storage** — unusual. 10-50GB of parsed replay data stored at no cost.

6. **Pre-encode on Modal** — instead of encoding locally then uploading, run encoding on a beefy CPU instance that reads raw data from a volume and writes the encoded `.pt` back to the same volume. Eliminates local → cloud transfer entirely.

---

## Pricing Comparison

Modal is ~2-2.5x more expensive per GPU-hour than bare metal:

| Platform | A100 80GB/hr | 8hr run | Notes |
|----------|-------------|---------|-------|
| Modal | $2.50 | ~$22 | Per-second, no setup |
| RunPod pod | $1.10 | ~$9 | Hourly, manage yourself |
| Lambda Labs | $1.29 | ~$10 | Hourly, best DX for bare metal |

**But:** We're not running 24-hour production training. We're running 2-hour experiment iterations where half the wall-clock time has been debugging infrastructure. The DX premium pays for itself in the first session where SSH doesn't break.

**Where Modal pricing wins:**
- Bursty, experimental workloads (spin up 10 GPUs for 30 min each)
- Scale-to-zero (pay nothing between experiments)
- No forgotten instances burning money overnight

**For the big run** (100K+ games, 10 epochs, ~$80-90 on RunPod): could do it on Modal (~$150-180) or drop to RunPod/Lambda for that single long run. Many labs use Modal for experimentation and cheaper instances for final training.

---

## Data Format Best Practices (For When We Scale)

### Current: Pre-Encoded `.pt` Files

Works great at 22K games. Pre-encode locally or on Modal, load entire tensor into GPU memory. Zero I/O during training. This is the right choice right now.

### At 230K+ Games: Sharded Streaming

| Format | Strengths | Best For |
|--------|-----------|----------|
| **WebDataset (sharded tar)** | Sequential I/O, cloud-native, works with S3/volumes | Streaming from cloud storage |
| **Mosaic Streaming (MDS)** | Deterministic resume, built-in caching, random access within shards | Mid-epoch resume, multi-node |
| **HDF5** | Fast for array data | Single-machine, fixed-shape data |
| **NumPy memmap** | Zero-copy, OS handles caching | Local SSD only |

**Recommendation when we get there:** WebDataset. Simpler, more mature, wider adoption. ~200MB shards, ~500 games each. Stream directly from S3 or Modal volume.

```python
import webdataset as wds

dataset = (
    wds.WebDataset("/data/shards/train-{000000..000044}.tar")
    .shuffle(5000)
    .decode()
    .to_tuple("state.npy", "meta.json")
    .batched(64)
)
```

### PyTorch DataLoader Settings

```python
loader = DataLoader(
    dataset,
    batch_size=512,
    num_workers=4,           # 2-4 for local SSD, 8+ for network
    pin_memory=True,         # always True on GPU
    persistent_workers=True, # avoid respawn cost per epoch
    prefetch_factor=4,       # hide network latency
)
```

### Non-Obvious Bottlenecks

- **CPU-GPU transfer is almost never the bottleneck** — a 512-batch of our data is ~3MB, PCIe transfer takes microseconds. The bottleneck is getting data to CPU (disk I/O, network, Python overhead).
- **`torch.tensor(arr)` copies, `torch.from_numpy(arr)` shares memory** — using the wrong constructor silently doubles memory traffic.
- **GPU at 40-70% utilization = data starvation**, not compute inefficiency.
- **Each DataLoader worker gets a full copy of the dataset object** — at 230K games, per-worker memory duplication matters.

---

## Recommendation

Given where we are — iterating on experiments, not running production training:

1. **Keep pre-encoding to `.pt`** — dataset fits in RAM, sharding complexity isn't justified yet.

2. **Make Modal the primary training path.** The `modal_train.py` on `combat-context-heads` is 80% there. Missing pieces:
   - A `pre_encode` function that runs *on Modal* (CPU instance, reads raw data from volume, writes encoded `.pt` to same volume)
   - Sweep support via `starmap` for parallel experiments
   - Checkpoint download script (`modal volume get`)

3. **Upload parsed data to Modal volume once.** Free storage, persistent, fast reads from any training function.

4. **For the 230K-game scale run:** Shard into WebDataset tars, upload to S3 or Modal volume, switch to streaming. Build this when we need it, not before.

**The core insight: our bottleneck has never been data or compute. It's the gap between "I want to run an experiment" and "the experiment is actually running." Modal closes that gap to a single command.**

---

## What's Actually Proven vs. Assumed (Feb 25 EOD)

### Proven (directly observed)

| Claim | Evidence |
|-------|----------|
| `torch.load` of 7GB `.pt` takes ~14s on A100 | Modal run log, timestamp |
| Pre-encode 2K games locally → 7GB in ~4 min | Local terminal output |
| `modal volume put` upload works | Volume contents verified via `check_volume()` |
| GPU shows training activity | Modal metrics: power 100-250W, temp 40-46°C, util 20-30% |
| Per-file loading from network volume hangs | Multiple attempts, never completed |

### Not Proven (inferred, assumed, or untested)

| Claim | Risk |
|-------|------|
| A training epoch completes on Modal | Never happened. Could hang, OOM, or error silently. |
| Reconstructed `MeleeDataset` via `__new__` is correct | Manual attribute assignment — no validation against direct construction. |
| 20-30% GPU util = DataLoader bottleneck | Could also be small model with fast forward passes. No A/B test with `num_workers`. |
| `vol.commit()` saves checkpoints correctly | Never tested in practice. |
| wandb logs full epoch metrics from Modal | Never observed — no epoch has completed. |
| `starmap` sweep works for parallel experiments | Entirely theoretical. |
| Pricing estimates are accurate | Research numbers, no actual invoices. |

### Known Issue: Pre-Encode Config Validation Gap

`modal_train.py` reconstructs `MeleeDataset` via `__new__` + manual attribute assignment. The tensor data is correct (saved from a real `MeleeDataset.__init__` call), but the `EncodingConfig` used at training time is reconstructed from YAML — **not** read from the saved payload.

`pre_encode.py` saves `encoding_config` in the `.pt` file (line 60), but `modal_train.py` never checks it. If the YAML changes between pre-encoding and training, tensors and config silently disagree.

**Immediate fix:** Add config assertion in `modal_train.py` after loading:
```python
saved_cfg = payload.get("encoding_config", {})
if saved_cfg and saved_cfg != enc_cfg_dict:
    raise ValueError(f"Config mismatch! Encoded with {saved_cfg}, training with {enc_cfg_dict}")
```

**Structural fix:** Add `MeleeDataset.from_tensors()` classmethod in `dataset.py` so the reconstruction logic lives next to `__init__` and stays in sync when attributes change.

**Runbook item:** The review runbook should verify:
1. Config saved in `.pt` matches config used for training
2. A sample from the reconstructed dataset matches a sample from a fresh `MeleeDataset(games, cfg)` (even just one game, offline)
3. Shape preflight in Trainer passes (it does catch gross dimension mismatches)

---

*Runbook review feedback written to `nojohns/worldmodel/docs/MODAL-REVIEW-RUNBOOK.md` (Scav-2, Feb 25).*

### Next Milestone

**Close the loop: one complete epoch, checkpoint saved, metrics logged to wandb, checkpoint downloadable.** Until that happens, everything else is planning.
