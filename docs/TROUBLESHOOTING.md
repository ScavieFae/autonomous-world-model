# Troubleshooting

Known issues and their resolutions. If you hit something new, add it here.

---

## Modal training silently fails on non-A100 GPUs

**Symptoms:** Training launches on Modal, wandb run creates, but zero batches complete. No error visible in `--detach` mode. Looks like a hang.

**Root cause:** Not a GPU issue. Modal's `--detach` mode swallows errors. The actual failures were trivial Python bugs (wrong attribute names, missing pip packages) that crashed the container silently. The wandb run was created before the crash, so it appeared to be "running" with zero progress.

**How we found it:** Built `scripts/gpu_diagnostic.py` — a step-by-step diagnostic that tests each pipeline stage independently. Errors surfaced immediately.

**Resolution:**
- Fixed `torch.cuda.get_device_properties(0).total_mem` → `.total_memory`
- Added `pyarrow` to the diagnostic image (was already in `modal_train.py` image)
- L4 ($0.80/hr) now works. T4 ($0.59/hr) likely works too (untested after fixes).

**Prevention:**
- **Never diagnose Modal issues with `--detach`.** Use `modal run` (foreground) or the diagnostic script to see actual errors.
- **Always test GPU changes with `scripts/gpu_diagnostic.py` first** before switching the production training script.
- When wandb shows `state: running` but `_runtime: 0` and empty config, the container crashed before wandb.init() completed — it's not a hang.

**Issue:** [#7](https://github.com/ScavieFae/autonomous-world-model/issues/7)

---

## Modal `volume get` produces corrupt .pt files

**Symptoms:** `modal volume get melee-training-data encoded-e012-fd-top5.pt ./local/path` downloads a file but `torch.load()` fails with `PytorchStreamReader failed reading zip archive: failed finding central directory`.

**Root cause:** Unknown. Multiple download attempts all produce corrupt files. May be a file size issue (3.5GB) or a network interruption during download.

**Workaround:** Don't download data to local. Keep data on Modal volume and access it from Modal functions. For local training, consider uploading to S3/R2 from a Modal function (datacenter-to-datacenter transfer, no home bandwidth needed).

**Status:** Unresolved. External hard drive arriving for local storage as alternative.

---

## DataLoader deadlock on Modal containers with small /dev/shm

**Symptoms:** Training hangs at the first batch when using `num_workers > 0`. No error, no output, just silence.

**Root cause:** Modal containers have 64MB default `/dev/shm`. When `MeleeDataset` is reconstructed via `__new__()` (bypassing `__init__`), `share_memory_()` isn't called. DataLoader workers try to pass batches through `/dev/shm`, the queue fills, everything deadlocks.

**Resolution:**
- `modal_train.py` now calls `share_memory_()` explicitly after dataset reconstruction
- Falls back to `num_workers=0` if `share_memory_()` fails
- The L4 diagnostic confirmed `share_memory_()` works (42s) and `num_workers=4` works after it

**Note:** The A100 containers have larger `/dev/shm`, which is why this only appeared when testing cheaper GPUs. The fix applies to all GPUs.

---

## Modal heartbeat errors on local client

**Symptoms:** `Modal Client → Modal Worker Heartbeat attempt failed` errors during `--detach` runs.

**Root cause:** The local Modal CLI (Python 3.9 system install) had incompatible `h2`/`grpclib` versions. The SSL/gRPC stack was broken.

**Resolution:** Reinstalled Modal via `uv tool install modal` which uses an isolated Python environment with compatible deps. Removed old binary from `~/Library/Python/3.9/bin/modal`.

**Note:** The remote function runs fine regardless — heartbeat errors only affect the local client's ability to stream logs. Training continues on Modal's servers.

---

## Mamba2 SSD scan with nchunks=1

**Symptoms:** Unnecessary compute overhead when `context_len == chunk_size` (e.g., both set to 30).

**Root cause:** The SSD chunked algorithm with `nchunks=1` does useless inter-chunk propagation (a no-op) while creating large intermediate tensors.

**Resolution:** Added guard in `models/mamba2.py`: when `seqlen == chunk_size`, fall back to sequential scan which is faster for the single-chunk case. Also changed `e018c` config to `chunk_size=15` so SSD provides benefit with `nchunks=2`.

---

## vel_mae reads zero in rollout eval

**Symptoms:** `eval/vel_mae` is always 0.0 in wandb and eval results.

**Root cause:** `reconstruct_frame()` wasn't applying velocity_delta or dynamics_pred to the AR frames during eval. Velocities were frozen from the seed context.

**Resolution:** Fixed in `scripts/ar_utils.py` — AR reconstruction now uses all three prediction types (position deltas, velocity deltas, dynamics absolute). Code is fixed but hasn't been validated with a training run that explicitly checks velocity metrics.

**Status:** Fix deployed, not yet validated.
