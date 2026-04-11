#!/usr/bin/env python3
"""Modal training script for the JEPA world model.

Mirrors scripts/modal_train.py but uses the JEPA stack (JEPAFrameDataset,
JEPAWorldModel, JEPATrainer, per-epoch linear probe + temporal straightness).

Usage:
    modal run scripts/modal_train_jepa.py \\
        --config experiments/e030a-jepa-baseline.yaml \\
        --encoded-file /encoded-e012-fd-top5.pt

    modal run --detach scripts/modal_train_jepa.py \\
        --config experiments/e030a-jepa-baseline.yaml \\
        --encoded-file /encoded-e012-fd-top5.pt \\
        --gpu A100
"""

import modal

# --- Modal app definition ---

app = modal.App("awm-train-jepa")

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

_train_kwargs = dict(
    image=image,
    timeout=86400,  # 24 hours (budget gate is the real cost control)
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("wandb-key")],
)


@app.function(gpu="L4", **_train_kwargs)           # $0.80/hr
def train_l4(config_path, encoded_file, run_name="", epochs=0, batch_size=0, lr=0.0, resume=""):
    return _train_impl(config_path, encoded_file, run_name, epochs, batch_size, lr, resume)


@app.function(gpu="A100-40GB", **_train_kwargs)    # $2.10/hr
def train_a100(config_path, encoded_file, run_name="", epochs=0, batch_size=0, lr=0.0, resume=""):
    return _train_impl(config_path, encoded_file, run_name, epochs, batch_size, lr, resume)


@app.function(gpu="H100", **_train_kwargs)         # $3.95/hr — needs Mattie approval
def train_h100(config_path, encoded_file, run_name="", epochs=0, batch_size=0, lr=0.0, resume=""):
    return _train_impl(config_path, encoded_file, run_name, epochs, batch_size, lr, resume)


GPU_FUNCS = {"L4": train_l4, "A100": train_a100, "H100": train_h100}


def _train_impl(
    config_path: str,
    encoded_file: str,
    run_name: str = "",
    epochs: int = 0,
    batch_size: int = 0,
    lr: float = 0.0,
    resume: str = "",
):
    """Train a JEPA world model on Modal with pre-encoded data.

    Args:
        config_path: Path to experiment YAML (relative to repo root).
        encoded_file: Path to pre-encoded .pt on the Modal volume.
        run_name: Override run name (default: derived from config filename).
        epochs: Override num_epochs (0 = use config).
        batch_size: Override batch_size (0 = use config).
        lr: Override learning rate (0 = use config).
        resume: Path to checkpoint on volume to resume from.
    """
    import dataclasses
    import json
    import logging
    import pathlib
    import sys
    import time

    sys.path.insert(0, "/app")

    import numpy as np
    import torch
    import yaml

    from data.dataset import MeleeDataset
    from data.jepa_dataset import JEPAFrameDataset
    from models.encoding import EncodingConfig
    from models.jepa.model import JEPAWorldModel
    from training.jepa_trainer import JEPATrainer

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
        run_name = pathlib.Path(config_path).stem

    # Build encoding config (filter to known fields)
    enc_cfg = EncodingConfig(**{k: v for k, v in enc_cfg_dict.items()
                                if v is not None and hasattr(EncodingConfig, k)})

    # Load pre-encoded data
    t0 = time.time()
    data_path = f"/data{encoded_file}"
    use_mmap = train_cfg.get("mmap", True)
    logging.info("Loading pre-encoded data from %s (mmap=%s)", data_path, use_mmap)
    if use_mmap:
        try:
            payload = torch.load(data_path, map_location="cpu", weights_only=False, mmap=True)
            logging.info("Loaded with mmap=True")
        except TypeError:
            payload = torch.load(data_path, map_location="cpu", weights_only=False)
            logging.info("Loaded without mmap (PyTorch < 2.1 fallback)")
    else:
        payload = torch.load(data_path, map_location="cpu", weights_only=False)
        logging.info("Loaded with mmap=False (full RAM load)")
    logging.info("Data loaded in %.1fs", time.time() - t0)

    # Validate encoding config matches what was used to encode the file.
    # This is load-bearing — the whole JEPA critique hinges on not silently
    # inheriting an encoding that doesn't match the YAML. Copied from
    # modal_train.py:175-184.
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
    game_offsets = payload["game_offsets"]
    if isinstance(game_offsets, torch.Tensor):
        game_offsets = game_offsets.clone().numpy()
    dataset.game_offsets = game_offsets
    dataset.num_games = len(dataset.game_offsets) - 1
    dataset.total_frames = dataset.game_offsets[-1]
    dataset.game_lengths = [
        int(dataset.game_offsets[i + 1] - dataset.game_offsets[i])
        for i in range(dataset.num_games)
    ]
    logging.info(
        "Dataset: %d games, %d frames, floats=%s ints=%s",
        dataset.num_games, dataset.total_frames,
        tuple(dataset.floats.shape), tuple(dataset.ints.shape),
    )

    # Build JEPA train/val splits
    history_size = model_cfg.get("history_size", 3)
    num_preds = model_cfg.get("num_preds", 1)
    train_split = train_cfg.get("train_split", 0.9)
    split_idx = max(1, int(dataset.num_games * train_split))

    train_ds = JEPAFrameDataset(
        dataset, range(0, split_idx),
        history_size=history_size, num_preds=num_preds,
    )
    val_ds = JEPAFrameDataset(
        dataset, range(split_idx, dataset.num_games),
        history_size=history_size, num_preds=num_preds,
    )
    logging.info("Train: %d examples, Val: %d examples", len(train_ds), len(val_ds))

    # Build JEPA model
    model = JEPAWorldModel(
        cfg=enc_cfg,
        embed_dim=model_cfg.get("embed_dim", 192),
        history_size=history_size,
        encoder_hidden_dim=model_cfg.get("encoder_hidden_dim", 512),
        predictor_layers=model_cfg.get("predictor_layers", 6),
        predictor_heads=model_cfg.get("predictor_heads", 16),
        predictor_dim_head=model_cfg.get("predictor_dim_head", 64),
        predictor_mlp_dim=model_cfg.get("predictor_mlp_dim", 2048),
        predictor_dropout=model_cfg.get("predictor_dropout", 0.1),
        sigreg_lambda=model_cfg.get("sigreg_lambda", 0.1),
        sigreg_knots=model_cfg.get("sigreg_knots", 17),
        sigreg_projections=model_cfg.get("sigreg_projections", 1024),
    )
    param_count = sum(p.numel() for p in model.parameters())
    logging.info("JEPA model parameters: %s", f"{param_count:,}")

    # Save dir on the volume
    save_dir = f"/data/checkpoints/{run_name}"

    # Wandb config
    run_config = {
        "arch": "jepa",
        "model": model_cfg,
        "training": train_cfg,
        "encoding": enc_cfg_dict,
        "data": {
            "encoded_file": encoded_file,
            "num_games": dataset.num_games,
            "total_frames": int(dataset.total_frames),
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
        },
        "model_params": param_count,
    }

    if wandb:
        import os
        if os.environ.get("WANDB_API_KEY"):
            wandb.init(
                project="melee-worldmodel",
                name=run_name,
                config=run_config,
                tags=["jepa", "e030"],
            )
            logging.info("Wandb run: %s", wandb.run.url)
        else:
            logging.info("No WANDB_API_KEY — logging to stdout only")
            wandb = None

    # Build trainer
    trainer = JEPATrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=train_cfg.get("lr", 5e-5),
        weight_decay=train_cfg.get("weight_decay", 1e-3),
        batch_size=train_cfg.get("batch_size", 128),
        num_epochs=train_cfg.get("num_epochs", 50),
        save_dir=save_dir,
        device="cuda",
        use_amp=train_cfg.get("use_amp", train_cfg.get("amp", False)),
        warmup_pct=train_cfg.get("warmup_pct", 0.0),
        gradient_clip=train_cfg.get("gradient_clip", 1.0),
        num_workers=train_cfg.get("num_workers"),
        diagnostic_every=train_cfg.get("diagnostic_every", 1),
        diagnostic_batch_size=train_cfg.get("diagnostic_batch_size", 256),
        epoch_callback=lambda: vol.commit(),
    )

    history = trainer.train()

    # Save manifest
    manifest = {
        "config": run_config,
        "results": history[-1] if history else {},
        "all_epochs": history,
    }
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{save_dir}/manifest.json", "w") as f:
        def _default(obj):
            if hasattr(obj, "item"):
                return obj.item()
            raise TypeError(f"{type(obj).__name__} not serializable")
        json.dump(manifest, f, indent=2, default=_default)
    vol.commit()
    logging.info("Checkpoints saved to %s", save_dir)

    # Summary
    if history:
        final = history[-1]
        logging.info("--- JEPA Training complete ---")
        for k in [
            "pred_loss", "sigreg_loss", "total_loss",
            "swap/mean_cosine_sim", "swap/ditto_cosine_sim",
            "probe/p0_x_r2", "probe/p1_x_r2",
            "probe/p0_percent_r2", "probe/p1_percent_r2",
            "probe/rel_x_r2", "probe/rel_percent_r2",
            "emergent/straightness",
        ]:
            if k in final:
                logging.info("  %s: %.4f", k, final[k])

    if wandb and wandb.run:
        wandb.finish()

    return {
        "run_name": run_name,
        "results": history[-1] if history else {},
        "checkpoint": f"{save_dir}/best.pt",
    }


# --- Local entrypoint ---

@app.local_entrypoint()
def main(
    config: str = "experiments/e030a-jepa-baseline.yaml",
    encoded_file: str = "/encoded-e012-fd-top5.pt",
    run_name: str = "",
    epochs: int = 0,
    batch_size: int = 0,
    lr: float = 0.0,
    gpu: str = "A100",
    resume: str = "",
):
    train_fn = GPU_FUNCS.get(gpu)
    if train_fn is None:
        raise ValueError(f"Unknown GPU '{gpu}'. Options: {list(GPU_FUNCS.keys())}")
    print(f"GPU: {gpu}")
    if resume:
        print(f"Resume: {resume}")
    result = train_fn.remote(
        config_path=config,
        encoded_file=encoded_file,
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        resume=resume,
    )
    print(f"\nDone: {result['run_name']}")
    print(f"Checkpoint: {result['checkpoint']}")
    if result.get("results"):
        r = result["results"]
        for k in [
            "pred_loss", "total_loss",
            "swap/mean_cosine_sim", "swap/ditto_cosine_sim",
            "probe/p0_x_r2", "probe/p1_x_r2", "probe/rel_x_r2",
            "emergent/straightness",
        ]:
            if k in r:
                print(f"  {k}: {r[k]:.4f}")
