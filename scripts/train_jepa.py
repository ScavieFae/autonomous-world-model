#!/usr/bin/env python3
"""Train the JEPA world model.

Usage:
    python -m scripts.train_jepa --dataset /path/to/parsed/dataset
    python -m scripts.train_jepa --config experiments/e030a-jepa-baseline.yaml --dataset /path
"""

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import MeleeDataset
from data.jepa_dataset import JEPAFrameDataset
from data.parse import load_games_from_dir
from models.encoding import EncodingConfig
from models.jepa.model import JEPAWorldModel
from training.jepa_trainer import JEPATrainer

try:
    import wandb
except ImportError:
    wandb = None


def load_config(config_path: str | None) -> dict:
    if config_path:
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {
        "encoding": {},
        "model": {},
        "training": {},
    }


def main():
    parser = argparse.ArgumentParser(description="Train JEPA world model")
    parser.add_argument("--dataset", required=True, help="Path to parsed dataset directory")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", default=None, help="Override device (cpu/mps/cuda)")
    parser.add_argument("--save-dir", default=None, help="Override checkpoint save dir")
    parser.add_argument("--max-games", type=int, default=None, help="Max games to load")
    parser.add_argument("--stage", type=int, default=None, help="Filter by stage ID")
    parser.add_argument("--character", type=int, default=None, help="Filter by character ID")
    parser.add_argument("--run-name", default=None, help="Name for this run")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)

    # CLI overrides
    data_cfg = cfg.get("data", {})
    if args.max_games is not None:
        data_cfg["max_games"] = args.max_games
    if args.stage is not None:
        data_cfg["stage_filter"] = args.stage
    if args.character is not None:
        data_cfg["character_filter"] = args.character

    train_cfg = cfg.get("training", {})
    if args.epochs is not None:
        train_cfg["num_epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.device is not None:
        train_cfg["device"] = args.device

    experiment_name = None
    if args.config:
        experiment_name = Path(args.config).stem

    base_save_dir = args.save_dir or cfg.get("save_dir", "checkpoints")
    if experiment_name and not args.save_dir:
        save_dir = str(Path(base_save_dir) / experiment_name)
    else:
        save_dir = base_save_dir

    if args.run_name is None and experiment_name:
        args.run_name = experiment_name

    if experiment_name:
        logging.info("Experiment: %s", experiment_name)

    # Build encoding config
    enc_cfg_dict = cfg.get("encoding", {})
    enc_cfg = EncodingConfig(**{k: v for k, v in enc_cfg_dict.items() if v is not None})

    # Load data
    logging.info("Loading games from %s", args.dataset)
    games = load_games_from_dir(
        args.dataset,
        max_games=data_cfg.get("max_games"),
        stage_filter=data_cfg.get("stage_filter"),
        character_filter=data_cfg.get("character_filter"),
    )

    if not games:
        logging.error("No games loaded! Check dataset path and filters.")
        sys.exit(1)

    logging.info("Loaded %d games", len(games))

    game_md5s = [g.meta["slp_md5"] for g in games if g.meta]
    data_fingerprint = hashlib.sha256("|".join(sorted(game_md5s)).encode()).hexdigest()[:12]
    logging.info("Data fingerprint: %s (%d games)", data_fingerprint, len(game_md5s))

    dataset = MeleeDataset(games, enc_cfg)

    # Build JEPA datasets
    model_cfg = cfg.get("model", {})
    history_size = model_cfg.get("history_size", 3)
    train_split = train_cfg.get("train_split", 0.9)
    split_idx = max(1, int(dataset.num_games * train_split))

    train_ds = JEPAFrameDataset(
        dataset, range(0, split_idx),
        history_size=history_size, num_preds=1,
    )
    val_ds = JEPAFrameDataset(
        dataset, range(split_idx, dataset.num_games),
        history_size=history_size, num_preds=1,
    )

    logging.info("Train: %d examples, Val: %d examples", len(train_ds), len(val_ds))

    # Build model
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

    # Run config for tracking
    run_config = {
        "arch": "jepa",
        "model": model_cfg,
        "training": train_cfg,
        "encoding": enc_cfg_dict,
        "data": {
            "dataset_path": args.dataset,
            "num_games": len(game_md5s),
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "total_frames": dataset.total_frames,
            "fingerprint": data_fingerprint,
        },
        "model_params": param_count,
        "machine": platform.node(),
        "device": train_cfg.get("device", "auto"),
    }

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[1]),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        run_config["git_commit"] = git_hash
    except Exception:
        pass

    if wandb and not args.no_wandb:
        wandb.init(
            project="melee-worldmodel",
            name=args.run_name,
            config=run_config,
            tags=["jepa"],
        )
        logging.info("Wandb run: %s", wandb.run.url)

    # Build trainer
    trainer = JEPATrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=train_cfg.get("lr", 5e-5),
        weight_decay=train_cfg.get("weight_decay", 1e-3),
        batch_size=train_cfg.get("batch_size", 128),
        num_epochs=train_cfg.get("num_epochs", 100),
        save_dir=save_dir,
        device=train_cfg.get("device"),
        use_amp=train_cfg.get("use_amp", False),
        warmup_pct=train_cfg.get("warmup_pct", 0.0),
        gradient_clip=train_cfg.get("gradient_clip", 1.0),
    )

    # Train
    history = trainer.train()

    # Summary
    if history:
        final = history[-1]
        logging.info("--- JEPA Training complete ---")
        logging.info("Final pred_loss: %.4f", final.get("pred_loss", 0))
        logging.info("Final sigreg_loss: %.4f", final.get("sigreg_loss", 0))
        logging.info("Final total_loss: %.4f", final.get("total_loss", 0))

    # Save manifest
    manifest_dir = Path(save_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": run_config,
        "data_fingerprint": data_fingerprint,
        "game_md5s": game_md5s,
        "results": history[-1] if history else {},
        "all_epochs": history,
    }

    def _json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(manifest_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
    logging.info("Saved run manifest: %s", manifest_dir / "manifest.json")

    if wandb and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
