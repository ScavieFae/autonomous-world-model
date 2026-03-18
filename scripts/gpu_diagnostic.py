#!/usr/bin/env python3
"""GPU diagnostic for debugging training hangs on non-A100 GPUs.

Tests each stage of the training pipeline independently with timing,
so we can see exactly where it hangs.

Usage:
    # Test on T4:
    modal run scripts/gpu_diagnostic.py --gpu T4

    # Test on L4:
    modal run scripts/gpu_diagnostic.py --gpu L4

    # Test on A100 (known working, for comparison):
    modal run scripts/gpu_diagnostic.py --gpu A100-40GB
"""

import modal
import time

app = modal.App("awm-gpu-diagnostic")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "pyyaml", "pyarrow")
    .add_local_dir(".", remote_path="/app", ignore=[
        ".git", "site", "solana", ".loop", ".obsidian", ".claude",
        "checkpoints", "node_modules", "__pycache__", "*.pyc",
    ])
)

vol = modal.Volume.from_name("melee-training-data")


def run_diagnostic(gpu_name: str):
    """Run step-by-step diagnostic on the specified GPU."""
    import sys
    sys.path.insert(0, "/app")

    results = {}

    def step(name):
        """Print and timestamp a diagnostic step."""
        t = time.time()
        print(f"\n{'='*60}")
        print(f"STEP: {name}")
        print(f"{'='*60}")
        return t

    def done(name, t0):
        elapsed = time.time() - t0
        results[name] = elapsed
        print(f"  ✓ {name}: {elapsed:.2f}s")
        return elapsed

    # --- Step 1: GPU info ---
    t = step("GPU info")
    import torch
    print(f"  torch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Compute capability: {torch.cuda.get_device_capability()}")
        print(f"  CUDA version: {torch.version.cuda}")
    done("GPU info", t)

    # --- Step 2: Basic CUDA ops ---
    t = step("Basic CUDA operations")
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"  matmul result shape: {y.shape}, sum: {y.sum().item():.2f}")
    done("Basic CUDA ops", t)

    # --- Step 3: Load data ---
    t = step("Load pre-encoded data")
    data_path = "/data/encoded-e012-fd-top5.pt"
    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    print(f"  Keys: {list(payload.keys())}")
    print(f"  Floats shape: {payload['floats'].shape}")
    print(f"  Ints shape: {payload['ints'].shape}")
    done("Load data", t)

    # --- Step 4: Reconstruct dataset ---
    t = step("Reconstruct MeleeDataset")
    import numpy as np
    from data.dataset import MeleeDataset
    from models.encoding import EncodingConfig
    import yaml

    with open("/app/experiments/e018c-context-k30.yaml") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = EncodingConfig(**{k: v for k, v in cfg['encoding'].items()
                                if v is not None and hasattr(EncodingConfig, k)})

    dataset = MeleeDataset.__new__(MeleeDataset)
    dataset.cfg = enc_cfg
    dataset.floats = payload["floats"]
    dataset.ints = payload["ints"]
    dataset.game_offsets = payload["game_offsets"]
    if isinstance(dataset.game_offsets, torch.Tensor):
        dataset.game_offsets = dataset.game_offsets.numpy()
    dataset.num_games = len(dataset.game_offsets) - 1
    dataset.total_frames = dataset.game_offsets[-1]
    dataset.game_lengths = [
        int(dataset.game_offsets[i + 1] - dataset.game_offsets[i])
        for i in range(dataset.num_games)
    ]
    print(f"  Games: {dataset.num_games}, Frames: {dataset.total_frames}")
    done("Reconstruct dataset", t)

    # --- Step 5: share_memory_() ---
    t = step("share_memory_()")
    try:
        dataset.floats.share_memory_()
        dataset.ints.share_memory_()
        print(f"  ✓ share_memory_() succeeded")
    except RuntimeError as e:
        print(f"  ✗ share_memory_() FAILED: {e}")
    done("share_memory_()", t)

    # --- Step 6: Build frame dataset ---
    t = step("Build FrameDataset")
    context_len = cfg['model']['context_len']
    train_ds = dataset.get_frame_dataset(context_len=context_len, train=True, train_split=0.9)
    print(f"  Train examples: {len(train_ds)}")
    done("Build FrameDataset", t)

    # --- Step 7: DataLoader (num_workers=0) ---
    t = step("DataLoader (num_workers=0)")
    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0, drop_last=True)
    done("DataLoader init", t)

    # --- Step 8: First batch ---
    t = step("First batch load")
    batch = next(iter(loader))
    float_ctx, int_ctx, next_ctrl, float_tgt, int_tgt = batch
    print(f"  float_ctx: {float_ctx.shape}")
    print(f"  int_ctx: {int_ctx.shape}")
    done("First batch load", t)

    # --- Step 9: Build model ---
    t = step("Build model")
    from models.mamba2 import FrameStackMamba2
    model_cfg = cfg['model']
    model = FrameStackMamba2(
        cfg=enc_cfg,
        context_len=context_len,
        d_model=model_cfg.get("d_model", 384),
        d_state=model_cfg.get("d_state", 64),
        n_layers=model_cfg.get("n_layers", 4),
        headdim=model_cfg.get("headdim", 64),
        dropout=model_cfg.get("dropout", 0.1),
        chunk_size=model_cfg.get("chunk_size"),
    )
    model = model.to("cuda")
    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params:,}")
    print(f"  GPU memory after model: {torch.cuda.memory_allocated()/1e6:.0f} MB")
    done("Build model", t)

    # --- Step 10: Forward pass ---
    t = step("Forward pass (1 batch)")
    float_ctx = float_ctx.to("cuda")
    int_ctx = int_ctx.to("cuda")
    next_ctrl = next_ctrl.to("cuda")
    preds = model(float_ctx, int_ctx, next_ctrl)
    torch.cuda.synchronize()
    print(f"  Output keys: {list(preds.keys())}")
    print(f"  continuous_delta shape: {preds['continuous_delta'].shape}")
    print(f"  GPU memory after forward: {torch.cuda.memory_allocated()/1e6:.0f} MB")
    done("Forward pass", t)

    # --- Step 11: Backward pass ---
    t = step("Backward pass")
    loss = sum(v.mean() for v in preds.values())
    loss.backward()
    torch.cuda.synchronize()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  GPU memory after backward: {torch.cuda.memory_allocated()/1e6:.0f} MB")
    done("Backward pass", t)

    # --- Step 12: Optimizer step ---
    t = step("Optimizer step")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    optimizer.step()
    optimizer.zero_grad()
    done("Optimizer step", t)

    # --- Step 13: 10 batches ---
    t = step("10 training batches")
    model.train()
    batch_times = []
    for i, (fc, ic, nc, ft, it) in enumerate(loader):
        if i >= 10:
            break
        bt = time.time()
        fc, ic, nc, ft, it = fc.to("cuda"), ic.to("cuda"), nc.to("cuda"), ft.to("cuda"), it.to("cuda")
        optimizer.zero_grad()
        preds = model(fc, ic, nc)
        loss = sum(v.mean() for v in preds.values())
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        batch_times.append(time.time() - bt)
        print(f"  Batch {i+1}/10: {batch_times[-1]*1000:.0f}ms, loss={loss.item():.4f}")
    avg_ms = sum(batch_times) / len(batch_times) * 1000
    done("10 batches", t)

    # --- Step 14: DataLoader with num_workers=4 ---
    t = step("DataLoader (num_workers=4)")
    try:
        loader4 = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4,
                             drop_last=True, persistent_workers=True, prefetch_factor=4)
        print("  DataLoader created. Attempting first batch...")
        # Set a timeout — if this hangs, we know it's the multiprocess issue
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("DataLoader with num_workers=4 timed out after 60s")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        batch = next(iter(loader4))
        signal.alarm(0)
        print(f"  ✓ First batch with num_workers=4 succeeded: {batch[0].shape}")
    except TimeoutError as e:
        print(f"  ✗ TIMEOUT: {e}")
        print(f"  This confirms the DataLoader multiprocess deadlock on this GPU tier")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
    done("DataLoader (num_workers=4)", t)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY — {gpu_name}")
    print(f"{'='*60}")
    total_batches = 30500  # approximate for 1 epoch
    est_epoch_s = avg_ms / 1000 * total_batches
    est_epoch_hr = est_epoch_s / 3600
    print(f"  Average batch time: {avg_ms:.0f}ms")
    print(f"  Est. epoch time: {est_epoch_hr:.1f}hr")
    print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")
    print(f"\n  Step timings:")
    for name, elapsed in results.items():
        print(f"    {name}: {elapsed:.2f}s")

    return {
        "gpu": gpu_name,
        "avg_batch_ms": avg_ms,
        "est_epoch_hr": est_epoch_hr,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1e6,
        "steps": results,
    }


# Create functions for each GPU tier
@app.function(image=image, gpu="T4", timeout=1800, volumes={"/data": vol})
def diagnostic_t4():
    return run_diagnostic("T4")

@app.function(image=image, gpu="L4", timeout=1800, volumes={"/data": vol})
def diagnostic_l4():
    return run_diagnostic("L4")

@app.function(image=image, gpu="A10G", timeout=1800, volumes={"/data": vol})
def diagnostic_a10g():
    return run_diagnostic("A10G")

@app.function(image=image, gpu="A100-40GB", timeout=1800, volumes={"/data": vol})
def diagnostic_a100():
    return run_diagnostic("A100-40GB")


@app.local_entrypoint()
def main(gpu: str = "L4"):
    import json
    gpu_map = {"T4": diagnostic_t4, "L4": diagnostic_l4, "A10G": diagnostic_a10g, "A100-40GB": diagnostic_a100}
    fn = gpu_map.get(gpu)
    if not fn:
        print(f"Unknown GPU: {gpu}. Options: {list(gpu_map.keys())}")
        return
    print(f"Running diagnostic on {gpu}...")
    result = fn.remote()
    print(f"\nResult: {json.dumps(result, indent=2)}")
