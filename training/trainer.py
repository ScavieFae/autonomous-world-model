"""Training loop for the world model."""

import dataclasses
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, IterableDataset

from data.dataset import MeleeFrameDataset
from models.encoding import EncodingConfig
from training.metrics import EpochMetrics, LossWeights, MetricsTracker

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_dataset: MeleeFrameDataset,
        val_dataset: Optional[MeleeFrameDataset],
        cfg: EncodingConfig,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        num_epochs: int = 50,
        loss_weights: Optional[LossWeights] = None,
        save_dir: Optional[str | Path] = None,
        device: Optional[str] = None,
        resume_from: Optional[str | Path] = None,
        num_workers: Optional[int] = None,
        log_interval: Optional[int] = None,
        epoch_callback: Optional[Callable[[], None]] = None,
        dataset: Optional["MeleeDataset"] = None,
        rollout_eval_every: int = 1,
        rollout_eval_samples: int = 300,
        rollout_eval_horizon: int = 20,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        logger.info("Using device: %s", self.device)

        self.model = model.to(self.device)
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir) if save_dir else None
        self.start_epoch = 0
        self._log_interval_override = log_interval
        self._epoch_callback = epoch_callback

        # DataLoader workers: CUDA benefits from multi-process loading.
        # MPS/CPU stays single-process.
        if num_workers is None:
            num_workers = 4 if device == "cuda" else 0
        loader_kwargs: dict = {}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4
        logger.info("DataLoader: num_workers=%d", num_workers)

        is_iterable = isinstance(train_dataset, IterableDataset)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=not is_iterable,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=(device != "cpu"),
            **loader_kwargs,
        )
        self.val_loader = None
        if val_dataset and len(val_dataset) > 0:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=(device != "cpu"),
                **loader_kwargs,
            )

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        self.metrics = MetricsTracker(cfg, loss_weights)
        self.history: list[dict] = []

        # Rollout coherence eval setup
        self._rollout_dataset = dataset
        self._rollout_eval_every = rollout_eval_every
        self._rollout_eval_samples = rollout_eval_samples
        self._rollout_eval_horizon = rollout_eval_horizon
        self._rollout_starts = None
        if dataset is not None and rollout_eval_every > 0:
            from scripts.eval_rollout import sample_starting_points
            train_split = max(1, int(dataset.num_games * 0.9))
            val_range = range(train_split, dataset.num_games)
            try:
                self._rollout_starts = sample_starting_points(
                    dataset, val_range, model.context_len,
                    rollout_eval_horizon, rollout_eval_samples, seed=42,
                )
                logger.info(
                    "Rollout eval: %d samples, K=%d, every %d epoch(s)",
                    len(self._rollout_starts), rollout_eval_horizon, rollout_eval_every,
                )
            except ValueError:
                logger.warning("Rollout eval: not enough val data for sampling, disabled")

        # Shape preflight
        self._verify_shapes(train_dataset)

        if resume_from:
            self._load_checkpoint(resume_from)

    def _compute_log_interval(self, num_batches: int) -> int:
        """Compute batch logging interval. Uses override if set, else ~10x per epoch."""
        if self._log_interval_override is not None:
            return max(1, self._log_interval_override)
        return max(1, num_batches // 10)

    def _verify_shapes(self, dataset) -> None:
        """Preflight check: verify first sample's tensor shapes match config."""
        try:
            sample = dataset[0] if hasattr(dataset, '__getitem__') else next(iter(dataset))
        except Exception:
            logger.warning("Shape preflight: couldn't read sample, skipping check")
            return

        float_ctx, int_ctx, next_ctrl, float_tgt, int_tgt = sample
        cfg = self.cfg
        K = self.model.context_len
        expected_float = cfg.float_per_player * 2
        expected_int = cfg.int_per_frame
        expected_ctrl = cfg.ctrl_conditioning_dim
        expected_float_tgt = (cfg.core_continuous_dim * 2 + cfg.velocity_dim * 2
                              + cfg.predicted_binary_dim + cfg.predicted_dynamics_dim)

        errors = []
        if float_ctx.shape != (K, expected_float):
            errors.append(f"float_ctx: got {tuple(float_ctx.shape)}, expected ({K}, {expected_float})")
        if int_ctx.shape != (K, expected_int):
            errors.append(f"int_ctx: got {tuple(int_ctx.shape)}, expected ({K}, {expected_int})")
        if next_ctrl.shape != (expected_ctrl,):
            errors.append(f"next_ctrl: got {tuple(next_ctrl.shape)}, expected ({expected_ctrl},)")
        if float_tgt.shape != (expected_float_tgt,):
            errors.append(f"float_tgt: got {tuple(float_tgt.shape)}, expected ({expected_float_tgt},)")
        if int_tgt.shape != (12,):
            errors.append(f"int_tgt: got {tuple(int_tgt.shape)}, expected (12,)")

        if errors:
            msg = "Shape preflight FAILED — data/config mismatch:\n  " + "\n  ".join(errors)
            msg += f"\n  Config: state_age_as_embed={cfg.state_age_as_embed}, press_events={cfg.press_events}"
            raise ValueError(msg)

        logger.info(
            "Shape preflight OK: float_ctx=(%d,%d) int_ctx=(%d,%d) next_ctrl=(%d,) float_tgt=(%d,) int_tgt=(12,)",
            K, expected_float, K, expected_int, expected_ctrl, expected_float_tgt,
        )

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        epoch_metrics = EpochMetrics()
        num_batches = len(self.train_loader)
        log_interval = self._compute_log_interval(num_batches)

        for batch_idx, (float_ctx, int_ctx, next_ctrl, float_tgt, int_tgt) in enumerate(self.train_loader):
            float_ctx = float_ctx.to(self.device)
            int_ctx = int_ctx.to(self.device)
            next_ctrl = next_ctrl.to(self.device)
            float_tgt = float_tgt.to(self.device)
            int_tgt = int_tgt.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(float_ctx, int_ctx, next_ctrl)
            loss, batch_metrics = self.metrics.compute_loss(predictions, float_tgt, int_tgt, int_ctx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            epoch_metrics.update(batch_metrics)

            if (batch_idx + 1) % log_interval == 0:
                pct = 100.0 * (batch_idx + 1) / num_batches
                logger.info(
                    "  batch %d/%d (%.0f%%) loss=%.4f",
                    batch_idx + 1, num_batches, pct, batch_metrics.total_loss,
                )
                if wandb and wandb.run:
                    wandb.log({
                        "batch/loss": batch_metrics.total_loss,
                        "batch/step": batch_idx + 1,
                        "batch/pct": pct,
                    })

        return epoch_metrics.averaged()

    @torch.no_grad()
    def _val_epoch(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        epoch_metrics = EpochMetrics()
        num_batches = len(self.val_loader)
        log_interval = self._compute_log_interval(num_batches)

        for batch_idx, (float_ctx, int_ctx, next_ctrl, float_tgt, int_tgt) in enumerate(self.val_loader):
            float_ctx = float_ctx.to(self.device)
            int_ctx = int_ctx.to(self.device)
            next_ctrl = next_ctrl.to(self.device)
            float_tgt = float_tgt.to(self.device)
            int_tgt = int_tgt.to(self.device)

            predictions = self.model(float_ctx, int_ctx, next_ctrl)
            _, batch_metrics = self.metrics.compute_loss(predictions, float_tgt, int_tgt, int_ctx)
            epoch_metrics.update(batch_metrics)

            if (batch_idx + 1) % log_interval == 0:
                pct = 100.0 * (batch_idx + 1) / num_batches
                logger.info(
                    "  val batch %d/%d (%.0f%%) loss=%.4f",
                    batch_idx + 1, num_batches, pct, batch_metrics.total_loss,
                )

        return {f"val_{k}": v for k, v in epoch_metrics.averaged().items()}

    @torch.no_grad()
    def _rollout_eval(self) -> dict[str, float]:
        """Run rollout coherence eval on val data. Returns metrics dict."""
        if self._rollout_starts is None or self._rollout_dataset is None:
            return {}
        from scripts.eval_rollout import evaluate_rollout_coherence

        self.model.eval()
        t0 = time.time()
        results = evaluate_rollout_coherence(
            self.model, self._rollout_dataset, self._rollout_starts,
            self.model.context_len, self._rollout_eval_horizon,
            self.cfg, str(self.device),
        )
        elapsed = time.time() - t0

        rc = results["summary_pos_mae"]
        logger.info("  rollout coherence = %.4f (%.1fs)", rc, elapsed)

        metrics = {"eval/summary_pos_mae": rc, "eval/time_s": elapsed}
        for k, m in results["per_horizon"].items():
            for name, val in m.items():
                metrics[f"eval/h{k}_{name}"] = val
        return metrics

    def train(self) -> list[dict]:
        logger.info(
            "Starting training: epochs %d→%d, %d train examples, batch_size=%d",
            self.start_epoch + 1, self.num_epochs,
            len(self.train_loader.dataset), self.batch_size,
        )

        best_val_loss = float("inf")

        for epoch in range(self.start_epoch, self.num_epochs):
            t0 = time.time()
            train_metrics = self._train_epoch()
            val_metrics = self._val_epoch()
            self.scheduler.step()

            # Rollout coherence eval
            rollout_metrics = {}
            if (self._rollout_starts is not None
                    and self._rollout_eval_every > 0
                    and (epoch + 1) % self._rollout_eval_every == 0):
                rollout_metrics = self._rollout_eval()

            elapsed = time.time() - t0
            combined = {**train_metrics, **val_metrics, **rollout_metrics, "epoch": epoch, "time": elapsed}
            self.history.append(combined)

            loss_str = f"loss={train_metrics['loss/total']:.4f}"
            acc_str = f"action_acc={train_metrics['metric/p0_action_acc']:.3f}"
            pos_str = f"pos_mae={train_metrics['metric/position_mae']:.2f}"

            val_str = ""
            if val_metrics:
                val_str = f" | val_loss={val_metrics['val_loss/total']:.4f}"
                val_str += f" val_acc={val_metrics['val_metric/p0_action_acc']:.3f}"

            change_str = ""
            if "metric/action_change_acc" in train_metrics:
                change_str = f" change_acc={train_metrics['metric/action_change_acc']:.3f}"

            logger.info(
                "Epoch %d/%d [%.1fs]: %s %s %s%s%s",
                epoch + 1, self.num_epochs, elapsed,
                loss_str, acc_str, pos_str, change_str, val_str,
            )

            if wandb and wandb.run:
                wandb.log({**combined, "lr": self.scheduler.get_last_lr()[0]})

            if self.save_dir and val_metrics:
                val_loss = val_metrics.get("val_loss/total", float("inf"))
                self._save_checkpoint("latest.pt", epoch, val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best.pt", epoch, val_loss)

            if self._epoch_callback:
                self._epoch_callback()

        if self.save_dir:
            self._save_checkpoint("final.pt", self.num_epochs - 1)

        return self.history

    def _load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing:
            logger.warning("Checkpoint missing %d keys: %s", len(missing), ", ".join(missing))
        if unexpected:
            logger.warning("Checkpoint has %d unexpected keys: %s", len(unexpected), ", ".join(unexpected))
        if missing or unexpected:
            logger.warning("Architecture changed — reinitializing optimizer")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        for _ in range(self.start_epoch):
            self.scheduler.step()
        logger.info(
            "Resumed from %s (epoch %d, val_loss=%.4f)",
            path, checkpoint["epoch"] + 1, checkpoint.get("val_loss", 0),
        )

    def _save_checkpoint(self, name: str, epoch: int, val_loss: float = 0.0) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "encoding_config": dataclasses.asdict(self.cfg),
                "context_len": self.model.context_len,
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)
