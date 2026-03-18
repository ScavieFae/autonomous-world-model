#!/usr/bin/env python3
"""Modal training script for the world model.

Runs training on Modal cloud GPUs with pre-encoded data from volumes.
Supports single runs and parallel sweeps.

Usage:
    # Single run with pre-encoded data:
    modal run scripts/modal_train.py --config experiments/e019-baseline.yaml \
        --encoded-file /encoded-v3-ranked-fd-top5.pt

    # Detached (background) run:
    modal run --detach scripts/modal_train.py --config experiments/e019-baseline.yaml \
        --encoded-file /encoded-v3-ranked-fd-top5.pt

    # With CLI overrides:
    modal run scripts/modal_train.py --config experiments/e019-baseline.yaml \
        --encoded-file /encoded-v3-ranked-fd-top5.pt \
        --batch-size 8192 --epochs 3

    # Run rollout coherence eval on an existing checkpoint:
    modal run scripts/modal_train.py::eval_checkpoint \
        --checkpoint /checkpoints/e019-baseline/best.pt \
        --encoded-file /encoded-v3-ranked-fd-top5.pt \
        --config experiments/e019-baseline.yaml
"""

import modal

# --- Modal app definition ---

app = modal.App("awm-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "numpy",
        "pyarrow",
        "pyyaml",
        "wandb",
    )
    .add_local_dir(".", remote_path="/app", ignore=[
        ".git",
        "site",
        "solana",
        ".loop",
        ".obsidian",
        ".claude",
        "checkpoints",
        "node_modules",
        "inspo",
        "__pycache__",
        "*.pyc",
    ])
)

vol = modal.Volume.from_name("melee-training-data")


# --- Training function ---

@app.function(
    image=image,
    gpu="L4",  # 24GB, $0.80/hr — benchmarking (T4 works but slow)
    timeout=14400,  # 4 hours
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("wandb-key")],
)
def train(
    config_path: str,
    encoded_file: str,
    run_name: str = "",
    epochs: int = 0,
    batch_size: int = 0,
    lr: float = 0.0,
):
    """Train a world model on Modal with pre-encoded data.

    Args:
        config_path: Path to experiment YAML (relative to repo root).
        encoded_file: Path to pre-encoded .pt on the Modal volume (e.g. /encoded-v3-ranked-fd-top5.pt).
        run_name: Override run name (default: derived from config filename).
        epochs: Override num_epochs (0 = use config).
        batch_size: Override batch_size (0 = use config).
        lr: Override learning rate (0 = use config).
        run_eval: Run rollout coherence eval after training (default True).
    """
    import json
    import logging
    import sys
    import time

    sys.path.insert(0, "/app")

    import numpy as np
    import torch
    import yaml

    from data.dataset import MeleeDataset, MeleeFrameDataset
    from models.encoding import EncodingConfig
    from models.mlp import FrameStackMLP
    from models.mamba2 import FrameStackMamba2
    from training.metrics import LossWeights
    from training.trainer import Trainer

    try:
        import wandb
    except ImportError:
        wandb = None

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    with open(f"/app/{config_path}") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    enc_cfg_dict = cfg.get("encoding", {})

    # CLI overrides
    if epochs > 0:
        train_cfg["num_epochs"] = epochs
    if batch_size > 0:
        train_cfg["batch_size"] = batch_size
    if lr > 0:
        train_cfg["lr"] = lr

    if not run_name:
        import pathlib
        run_name = pathlib.Path(config_path).stem

    # Build encoding config
    enc_cfg = EncodingConfig(**{k: v for k, v in enc_cfg_dict.items()
                                if v is not None and hasattr(EncodingConfig, k)})

    # Load pre-encoded data
    t0 = time.time()
    data_path = f"/data{encoded_file}"
    logging.info("Loading pre-encoded data from %s", data_path)
    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    load_time = time.time() - t0
    logging.info("Data loaded in %.1fs", load_time)

    # Validate encoding config matches what was used to encode
    saved_cfg = payload.get("encoding_config", {})
    if saved_cfg:
        mismatches = []
        for k, v in saved_cfg.items():
            if hasattr(enc_cfg, k) and getattr(enc_cfg, k) != v:
                mismatches.append(f"  {k}: encoded={v}, config={getattr(enc_cfg, k)}")
        if mismatches:
            logging.warning("Encoding config mismatches:\n%s", "\n".join(mismatches))
            logging.warning("Using config from YAML, not from encoded file. Verify this is intentional.")

    # Reconstruct MeleeDataset from saved tensors
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

    # Move tensors to shared memory so DataLoader workers (num_workers>0)
    # can access them without copying across the fork boundary.
    # Without this, workers deadlock on Modal containers where /dev/shm is
    # small (64MB default). The MeleeDataset.__init__ path does this
    # automatically, but __new__ bypasses it.
    try:
        dataset.floats.share_memory_()
        dataset.ints.share_memory_()
        logging.info("Tensors moved to shared memory")
    except RuntimeError as e:
        # share_memory_() can fail if /dev/shm is too small for the tensors.
        # Fall back to num_workers=0 (set below).
        logging.warning("share_memory_() failed (%s) — will use num_workers=0", e)
        train_cfg["num_workers"] = 0

    logging.info("Dataset: %d games, %d frames", dataset.num_games, dataset.total_frames)

    # Build train/val splits
    context_len = model_cfg.get("context_len", 10)
    train_split = train_cfg.get("train_split", 0.9)
    train_ds = dataset.get_frame_dataset(context_len=context_len, train=True, train_split=train_split)
    val_ds = dataset.get_frame_dataset(context_len=context_len, train=False, train_split=train_split)

    logging.info("Train: %d examples, Val: %d examples", len(train_ds), len(val_ds))

    # Build model
    arch = model_cfg.get("arch", "mlp")
    if arch == "mamba2":
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
    else:
        model = FrameStackMLP(
            cfg=enc_cfg,
            context_len=context_len,
            hidden_dim=model_cfg.get("hidden_dim", 512),
            trunk_dim=model_cfg.get("trunk_dim", 256),
            dropout=model_cfg.get("dropout", 0.1),
        )

    param_count = sum(p.numel() for p in model.parameters())
    logging.info("Model: %s, %s params", arch, f"{param_count:,}")

    # Loss weights (filter to known fields — config may have extras from nojohns)
    loss_cfg = cfg.get("loss_weights", {})
    if loss_cfg:
        import dataclasses
        lw_fields = {f.name for f in dataclasses.fields(LossWeights)}
        loss_weights = LossWeights(**{k: v for k, v in loss_cfg.items() if k in lw_fields})
    else:
        loss_weights = None

    # Save dir on the volume
    save_dir = f"/data/checkpoints/{run_name}"

    # Init wandb
    run_config = {
        "model": model_cfg,
        "training": train_cfg,
        "loss_weights": loss_cfg,
        "encoding": enc_cfg_dict,
        "data": {
            "encoded_file": encoded_file,
            "num_games": dataset.num_games,
            "total_frames": int(dataset.total_frames),
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
        },
        "model_params": param_count,
        "gpu": "A100-40GB",
    }

    if wandb:
        import os
        if os.environ.get("WANDB_API_KEY"):
            wandb.init(project="melee-worldmodel", name=run_name, config=run_config)
            logging.info("Wandb run: %s", wandb.run.url)
        else:
            logging.info("No WANDB_API_KEY — logging to stdout only")
            wandb = None

    # Self-Forcing config
    sf_cfg = cfg.get("self_forcing", {})

    # Train (pass dataset for per-epoch rollout coherence eval + Self-Forcing)
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg=enc_cfg,
        lr=train_cfg.get("lr", 5e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        batch_size=train_cfg.get("batch_size", 4096),
        num_epochs=train_cfg.get("num_epochs", 2),
        loss_weights=loss_weights,
        save_dir=save_dir,
        device="cuda",
        num_workers=train_cfg.get("num_workers"),  # None = auto (4 for CUDA), 0 = no multiprocessing
        dataset=dataset,
        epoch_callback=lambda: vol.commit(),
        sf_enabled=sf_cfg.get("enabled", False),
        sf_ratio=sf_cfg.get("ratio", 4),
        sf_unroll_length=sf_cfg.get("unroll_length", 3),
        sf_horizon_weights=sf_cfg.get("horizon_weights", False),
        sf_selective_bptt=sf_cfg.get("selective_bptt", False),
    )

    history = trainer.train()

    # Save manifest
    manifest = {
        "config": run_config,
        "results": history[-1] if history else {},
        "all_epochs": history,
    }
    import pathlib
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{save_dir}/manifest.json", "w") as f:
        def _default(obj):
            if hasattr(obj, "item"):
                return obj.item()
            raise TypeError(f"{type(obj).__name__} not serializable")
        json.dump(manifest, f, indent=2, default=_default)

    # Persist checkpoints to volume
    vol.commit()
    logging.info("Checkpoints saved to %s", save_dir)

    # Summary
    if history:
        final = history[-1]
        logging.info("--- Training complete ---")
        for k in ["loss/total", "metric/p0_action_acc", "metric/position_mae",
                   "metric/action_change_acc", "eval/summary_pos_mae"]:
            if k in final:
                logging.info("  %s: %.4f", k, final[k])

    # Save rollout eval results if available
    if history and "eval/summary_pos_mae" in history[-1]:
        eval_out = {"summary_pos_mae": history[-1]["eval/summary_pos_mae"]}
        with open(f"{save_dir}/eval_rollout.json", "w") as f:
            json.dump(eval_out, f, indent=2)
        vol.commit()

    if wandb and wandb.run:
        wandb.finish()

    return {
        "run_name": run_name,
        "results": history[-1] if history else {},
        "checkpoint": f"{save_dir}/best.pt",
    }


# --- Eval-only function (for existing checkpoints) ---

@app.function(
    image=image,
    gpu="L4",  # 24GB, $0.80/hr — benchmarking (T4 works but slow)
    timeout=3600,
    volumes={"/data": vol},
    secrets=[],
)
def eval_checkpoint(
    checkpoint: str,
    encoded_file: str,
    config: str = "",
):
    """Run rollout coherence eval on an existing checkpoint.

    Args:
        checkpoint: Path to checkpoint on volume (e.g. /checkpoints/e019-baseline/best.pt).
        encoded_file: Path to pre-encoded .pt on volume.
        config: Experiment YAML for encoding config (optional if checkpoint has it).
    """
    import json
    import logging
    import sys
    import time

    sys.path.insert(0, "/app")

    import numpy as np
    import torch
    import yaml

    from data.dataset import MeleeDataset
    from models.checkpoint import load_model_from_checkpoint
    from models.encoding import EncodingConfig
    from scripts.eval_rollout import evaluate_rollout_coherence, sample_starting_points, format_table

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load model
    ckpt_path = f"/data{checkpoint}"
    model, cfg, context_len, arch = load_model_from_checkpoint(ckpt_path, "cuda")
    logging.info("Model: %s, context_len=%d", arch, context_len)

    # Load data
    data_path = f"/data{encoded_file}"
    payload = torch.load(data_path, map_location="cpu", weights_only=False)

    dataset = MeleeDataset.__new__(MeleeDataset)
    dataset.cfg = cfg
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

    # Val split
    train_split = 0.9
    if config:
        with open(f"/app/{config}") as f:
            exp_cfg = yaml.safe_load(f)
        train_split = exp_cfg.get("training", {}).get("train_split", 0.9)

    split_idx = max(1, int(dataset.num_games * train_split))
    val_game_range = range(split_idx, dataset.num_games)

    starting_points = sample_starting_points(
        dataset, val_game_range, context_len, 20, 300, seed=42,
    )

    t0 = time.time()
    results = evaluate_rollout_coherence(
        model, dataset, starting_points, context_len, 20, cfg, "cuda",
    )
    eval_time = time.time() - t0

    print()
    print("Rollout Coherence Eval")
    print("=" * 56)
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Architecture: {arch}")
    print(f"  Val games  : {len(val_game_range)}")
    print(f"  Samples    : {len(starting_points)}")
    print(f"  Eval time  : {eval_time:.1f}s")
    print()
    print(format_table(results, 20))
    print()
    print(f"  ** summary_pos_mae = {results['summary_pos_mae']:.4f} **")
    print()

    # Save to volume alongside checkpoint
    import pathlib
    ckpt_dir = str(pathlib.Path(ckpt_path).parent)
    eval_path = f"{ckpt_dir}/eval_rollout.json"
    output = {
        "checkpoint": checkpoint,
        "num_samples": len(starting_points),
        "horizon": 20,
        "seed": 42,
        "eval_time_s": round(eval_time, 2),
        "summary_pos_mae": results["summary_pos_mae"],
        "per_horizon": {str(k): v for k, v in results["per_horizon"].items()},
    }
    with open(eval_path, "w") as f:
        json.dump(output, f, indent=2)
    vol.commit()
    logging.info("Results saved to %s", eval_path)

    return output


# --- Local entrypoint ---

@app.local_entrypoint()
def main(
    config: str = "experiments/e019-baseline.yaml",
    encoded_file: str = "/encoded-v3-ranked-fd-top5.pt",
    run_name: str = "",
    epochs: int = 0,
    batch_size: int = 0,
    lr: float = 0.0,
):
    result = train.remote(
        config_path=config,
        encoded_file=encoded_file,
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    print(f"\nDone: {result['run_name']}")
    print(f"Checkpoint: {result['checkpoint']}")
    if result.get("results"):
        r = result["results"]
        for k in ["loss/total", "metric/action_change_acc", "metric/position_mae"]:
            if k in r:
                print(f"  {k}: {r[k]:.4f}")
