"""Training loop for the JEPA world model.

Simpler than the main Trainer — no per-field loss heads, no self-forcing.
Loss is MSE in latent space + SIGReg. Matches LeWM's training regime.

Reference: le-wm/train.py — lejepa_forward, training loop.
"""

import dataclasses
import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from data.jepa_dataset import JEPAFrameDataset
from models.jepa.model import JEPAWorldModel
from training.jepa_eval import run_jepa_eval

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


class JEPATrainer:
    """Training loop for the JEPA world model.

    Optimizer: AdamW, lr=5e-5, wd=1e-3, gradient clip 1.0 (matching LeWM).
    Loss: MSE in latent space + λ * SIGReg.
    """

    def __init__(
        self,
        model: JEPAWorldModel,
        train_dataset: JEPAFrameDataset,
        val_dataset: Optional[JEPAFrameDataset],
        *,
        lr: float = 5e-5,
        weight_decay: float = 1e-3,
        batch_size: int = 128,
        num_epochs: int = 100,
        save_dir: Optional[str | Path] = None,
        device: Optional[str] = None,
        use_amp: bool = False,
        warmup_pct: float = 0.0,
        gradient_clip: float = 1.0,
        num_workers: Optional[int] = None,
        probe_eval_every: int = 1,
        probe_samples: int = 1024,
        straightness_trajectories: int = 128,
        straightness_length: int = 20,
        epoch_callback: Optional[callable] = None,
    ):
        # Device setup (same pattern as main Trainer)
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        logger.info("Using device: %s", self.device)
        if self.device.type == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("GPU VRAM: %.1f GB", vram_gb)

        self._use_amp = use_amp and (device == "cuda")
        self._amp_dtype = torch.float16
        self._scaler = GradScaler(enabled=self._use_amp)
        if self._use_amp:
            logger.info("AMP enabled (float16 autocast + GradScaler)")
        self.gradient_clip = gradient_clip

        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir) if save_dir else None

        if self.device.type == "cuda":
            logger.info("VRAM after model: %.2f GB", torch.cuda.memory_allocated() / 1e9)

        param_count = sum(p.numel() for p in model.parameters())
        logger.info("Model parameters: %s", f"{param_count:,}")

        # Optimizer: AdamW matching LeWM
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # LR schedule: optional warmup + cosine decay
        self._use_fast_loader = hasattr(train_dataset, "get_batch")
        if self._use_fast_loader:
            steps_per_epoch = len(train_dataset) // batch_size
        else:
            steps_per_epoch = max(1, len(train_dataset) // batch_size)
        total_steps = steps_per_epoch * num_epochs

        if warmup_pct > 0.0:
            warmup_steps = int(warmup_pct * total_steps)

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return (step + 1) / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
            logger.info(
                "LR: warmup %d steps (%.1f%%) + cosine (%d total)",
                warmup_steps, warmup_pct * 100, total_steps,
            )
        else:
            def lr_lambda(step: int) -> float:
                progress = step / max(1, total_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
            logger.info("LR: cosine decay over %d steps", total_steps)

        # DataLoader
        if num_workers is None:
            num_workers = 4 if device == "cuda" else 0
        loader_kwargs = {}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers if not self._use_fast_loader else 0,
            drop_last=True,
            pin_memory=False,
            **({} if self._use_fast_loader else loader_kwargs),
        )
        self.val_loader = None
        if val_dataset and len(val_dataset) > 0:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=False,
                **loader_kwargs,
            )

        self.history: list[dict] = []

        # Probe eval config
        self._probe_eval_every = probe_eval_every
        self._probe_samples = probe_samples
        self._straightness_trajectories = straightness_trajectories
        self._straightness_length = straightness_length
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._epoch_callback = epoch_callback

    def _fast_batch_iter(self):
        """Yield batches using vectorized get_batch (same pattern as main Trainer)."""
        ds = self.train_loader.dataset
        n = len(ds)
        indices = np.random.permutation(n)
        bs = self.batch_size
        for start in range(0, n - bs + 1, bs):
            batch_indices = indices[start : start + bs]
            yield ds.get_batch(batch_indices)

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_pred = 0.0
        total_sigreg = 0.0
        total_loss = 0.0
        num_batches = 0

        if self._use_fast_loader:
            batch_iter = self._fast_batch_iter()
            num_batches_expected = len(self.train_loader.dataset) // self.batch_size
        else:
            batch_iter = iter(self.train_loader)
            num_batches_expected = len(self.train_loader)

        log_interval = max(1, num_batches_expected // 10)

        for batch_idx, batch in enumerate(batch_iter):
            float_frames, int_frames, ctrl_inputs = [
                x.to(self.device) for x in batch
            ]

            self.optimizer.zero_grad()

            with autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
                output = self.model(float_frames, int_frames, ctrl_inputs)
                loss = output["total_loss"]

            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self._scaler.step(self.optimizer)
            self._scaler.update()
            self.scheduler.step()

            total_pred += output["pred_loss"].item()
            total_sigreg += output["sigreg_loss"].item()
            total_loss += output["total_loss"].item()
            num_batches += 1

            if batch_idx == 0 and self.device.type == "cuda":
                logger.info(
                    "VRAM peak: %.2f GB (allocated: %.2f GB)",
                    torch.cuda.max_memory_allocated() / 1e9,
                    torch.cuda.memory_allocated() / 1e9,
                )

            if (batch_idx + 1) % log_interval == 0:
                pct = 100.0 * (batch_idx + 1) / num_batches_expected
                logger.info(
                    "  batch %d/%d (%.0f%%) pred=%.4f sigreg=%.4f total=%.4f",
                    batch_idx + 1, num_batches_expected, pct,
                    output["pred_loss"].item(),
                    output["sigreg_loss"].item(),
                    output["total_loss"].item(),
                )
                if wandb and wandb.run:
                    wandb.log({
                        "batch/pred_loss": output["pred_loss"].item(),
                        "batch/sigreg_loss": output["sigreg_loss"].item(),
                        "batch/total_loss": output["total_loss"].item(),
                        "batch/step": batch_idx + 1,
                    })

        n = max(1, num_batches)
        return {
            "pred_loss": total_pred / n,
            "sigreg_loss": total_sigreg / n,
            "total_loss": total_loss / n,
        }

    @torch.no_grad()
    def _val_epoch(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        total_pred = 0.0
        total_sigreg = 0.0
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            float_frames, int_frames, ctrl_inputs = [
                x.to(self.device) for x in batch
            ]
            output = self.model(float_frames, int_frames, ctrl_inputs)
            total_pred += output["pred_loss"].item()
            total_sigreg += output["sigreg_loss"].item()
            total_loss += output["total_loss"].item()
            num_batches += 1

        n = max(1, num_batches)
        return {
            "val_pred_loss": total_pred / n,
            "val_sigreg_loss": total_sigreg / n,
            "val_total_loss": total_loss / n,
        }

    def train(self) -> list[dict]:
        logger.info(
            "Starting JEPA training: %d epochs, %d train examples, batch=%d",
            self.num_epochs, len(self.train_loader.dataset), self.batch_size,
        )

        best_val_loss = float("inf")

        for epoch in range(self.num_epochs):
            t0 = time.time()
            train_metrics = self._train_epoch()
            val_metrics = self._val_epoch()
            probe_metrics = self._probe_eval(epoch)
            elapsed = time.time() - t0

            combined = {
                **train_metrics, **val_metrics, **probe_metrics,
                "epoch": epoch, "time": elapsed,
            }
            self.history.append(combined)

            val_str = ""
            if val_metrics:
                val_str = f" | val_pred={val_metrics['val_pred_loss']:.4f}"
            probe_str = ""
            if probe_metrics:
                probe_str = (
                    f" | pos_r2={probe_metrics.get('probe/pos_r2', float('nan')):.3f}"
                    f" pos_mae={probe_metrics.get('probe/pos_mae', float('nan')):.2f}"
                    f" act_acc={probe_metrics.get('probe/action_acc', float('nan')):.3f}"
                    f" straight={probe_metrics.get('straightness/cosine', float('nan')):.3f}"
                )

            logger.info(
                "Epoch %d/%d [%.1fs]: pred=%.4f sigreg=%.4f total=%.4f%s%s",
                epoch + 1, self.num_epochs, elapsed,
                train_metrics["pred_loss"],
                train_metrics["sigreg_loss"],
                train_metrics["total_loss"],
                val_str, probe_str,
            )

            if wandb and wandb.run:
                wandb.log({
                    **combined,
                    "lr": self.scheduler.get_last_lr()[0],
                })

            if self.save_dir:
                val_loss = val_metrics.get("val_total_loss", train_metrics["total_loss"])
                self._save_checkpoint("latest.pt", epoch, val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best.pt", epoch, val_loss)

            if self._epoch_callback:
                self._epoch_callback()

        if self.save_dir:
            self._save_checkpoint("final.pt", self.num_epochs - 1)

        return self.history

    def _probe_eval(self, epoch: int) -> dict[str, float]:
        """Run linear probe + temporal straightness + embedding stats.

        Cheap per-epoch diagnostic. Gates when `probe_eval_every=0` or val
        dataset is missing.
        """
        if self._probe_eval_every <= 0 or self._val_dataset is None:
            return {}
        if (epoch + 1) % self._probe_eval_every != 0:
            return {}
        try:
            return run_jepa_eval(
                self.model,
                self._train_dataset,
                self._val_dataset,
                self.model.cfg,
                device=str(self.device),
                probe_samples=self._probe_samples,
                straightness_trajectories=self._straightness_trajectories,
                straightness_length=self._straightness_length,
            )
        except Exception as e:
            logger.warning("Probe eval failed: %s", e)
            return {}

    def _save_checkpoint(self, name: str, epoch: int, val_loss: float = 0.0) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "encoding_config": dataclasses.asdict(self.model.cfg),
                "history_size": self.model.history_size,
                "embed_dim": self.model.embed_dim,
                "arch": "jepa",
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)
