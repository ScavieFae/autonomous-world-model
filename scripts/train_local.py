#!/usr/bin/env python3
"""Local training script using pre-encoded .pt data.

Same as modal_train.py but runs locally on MPS/CPU. No Modal, no cloud.
Designed for overnight experiments on Apple Silicon.

Usage:
    python scripts/train_local.py --config experiments/e018c-context-k30.yaml \
        --encoded-file data/encoded/encoded-e012-fd-top5.pt \
        --run-name e022-local-test

    # Without wandb:
    python scripts/train_local.py --config experiments/e018c-context-k30.yaml \
        --encoded-file data/encoded/encoded-e012-fd-top5.pt \
        --no-wandb
"""

import argparse
import dataclasses
import json
import logging
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import yaml

from data.dataset import MeleeDataset
from models.encoding import EncodingConfig
from models.mamba2 import FrameStackMamba2
from models.mlp import FrameStackMLP
from training.metrics import LossWeights
from training.trainer import Trainer

try:
    import wandb
except ImportError:
    wandb = None


def main():
    parser = argparse.ArgumentParser(description="Train locally with pre-encoded data")
    parser.add_argument("--config", required=True, help="Experiment YAML config")
    parser.add_argument("--encoded-file", required=True, help="Path to pre-encoded .pt file")
    parser.add_argument("--run-name", default="", help="Run name (default: config filename)")
    parser.add_argument("--device", default=None, help="Override device (mps/cpu/cuda)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    enc_cfg_dict = cfg.get("encoding", {})
    sf_cfg = cfg.get("self_forcing", {})

    if args.batch_size > 0:
        train_cfg["batch_size"] = args.batch_size
    if args.epochs > 0:
        train_cfg["num_epochs"] = args.epochs

    if not args.run_name:
        args.run_name = pathlib.Path(args.config).stem

    # Build encoding config (filter unknown fields)
    enc_cfg = EncodingConfig(**{k: v for k, v in enc_cfg_dict.items()
                                if v is not None and hasattr(EncodingConfig, k)})

    # Load pre-encoded data
    t0 = time.time()
    logging.info("Loading pre-encoded data from %s", args.encoded_file)
    payload = torch.load(args.encoded_file, map_location="cpu", weights_only=False)
    logging.info("Data loaded in %.1fs", time.time() - t0)

    # Reconstruct MeleeDataset
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
            cfg=enc_cfg, context_len=context_len,
            d_model=model_cfg.get("d_model", 384),
            d_state=model_cfg.get("d_state", 64),
            n_layers=model_cfg.get("n_layers", 4),
            headdim=model_cfg.get("headdim", 64),
            dropout=model_cfg.get("dropout", 0.1),
            layer_dropout=model_cfg.get("layer_dropout", 0.0),
            chunk_size=model_cfg.get("chunk_size"),
        )
    else:
        model = FrameStackMLP(
            cfg=enc_cfg, context_len=context_len,
            hidden_dim=model_cfg.get("hidden_dim", 512),
            trunk_dim=model_cfg.get("trunk_dim", 256),
            dropout=model_cfg.get("dropout", 0.1),
        )
    logging.info("Model: %s, %s params", arch, f"{sum(p.numel() for p in model.parameters()):,}")

    # Loss weights (filter unknown fields)
    loss_cfg = cfg.get("loss_weights", {})
    if loss_cfg:
        lw_fields = {f.name for f in dataclasses.fields(LossWeights)}
        loss_weights = LossWeights(**{k: v for k, v in loss_cfg.items() if k in lw_fields})
    else:
        loss_weights = None

    save_dir = cfg.get("save_dir", f"checkpoints/{args.run_name}")

    # wandb
    run_config = {
        "model": model_cfg, "training": train_cfg,
        "loss_weights": loss_cfg, "encoding": enc_cfg_dict,
        "data": {"encoded_file": args.encoded_file, "num_games": dataset.num_games},
        "platform": "local-mps",
    }
    if wandb and not args.no_wandb:
        wandb.init(project="melee-worldmodel", name=args.run_name, config=run_config)
        logging.info("wandb: %s", wandb.run.url)

    # Train
    trainer = Trainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=enc_cfg,
        lr=train_cfg.get("lr", 5e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        batch_size=train_cfg.get("batch_size", 512),
        num_epochs=train_cfg.get("num_epochs", 1),
        loss_weights=loss_weights,
        save_dir=save_dir,
        device=args.device,
        num_workers=0,  # MPS doesn't benefit from multiprocess loading
        dataset=dataset,
        sf_enabled=sf_cfg.get("enabled", False),
        sf_ratio=sf_cfg.get("ratio", 4),
        sf_unroll_length=sf_cfg.get("unroll_length", 3),
        sf_horizon_weights=sf_cfg.get("horizon_weights", False),
        use_amp=train_cfg.get("amp", False),
        warmup_pct=train_cfg.get("warmup_pct", 0.0),
    )

    history = trainer.train()

    # Save manifest
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    manifest = {"config": run_config, "results": history[-1] if history else {}, "all_epochs": history}
    with open(f"{save_dir}/manifest.json", "w") as f:
        def _default(obj):
            if hasattr(obj, "item"):
                return obj.item()
            raise TypeError(f"{type(obj).__name__} not serializable")
        json.dump(manifest, f, indent=2, default=_default)

    if history:
        final = history[-1]
        logging.info("--- Training complete ---")
        for k in ["loss/total", "metric/p0_action_acc", "metric/position_mae",
                   "metric/action_change_acc", "eval/summary_pos_mae"]:
            if k in final:
                logging.info("  %s: %.4f", k, final[k])

    if wandb and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
