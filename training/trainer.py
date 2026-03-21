"""Training loop for the world model."""

import dataclasses
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, IterableDataset

from data.dataset import MeleeFrameDataset
from models.encoding import EncodingConfig
from scripts.ar_utils import build_ctrl_batch, reconstruct_frame, reconstruct_frame_differentiable
from training.metrics import BatchMetrics, EpochMetrics, LossWeights, MetricsTracker

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
        sf_enabled: bool = False,
        sf_ratio: int = 4,
        sf_unroll_length: int = 3,
        sf_horizon_weights: bool = False,
        sf_selective_bptt: bool = False,
        sf_full_bptt: bool = False,
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
        if self.device.type == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("GPU VRAM: %.1f GB", vram_gb)

        self.model = model.to(self.device)
        if self.device.type == "cuda":
            logger.info("VRAM after model.to(): %.2f GB", torch.cuda.memory_allocated() / 1e9)
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

        # Use fast vectorized batch loading if the dataset supports it.
        # This is ~100x faster than DataLoader's per-sample __getitem__ + collation
        # because it uses advanced indexing on contiguous tensors.
        self._use_fast_loader = hasattr(train_dataset, 'get_batch') and not is_iterable
        if self._use_fast_loader:
            logger.info("Using fast vectorized batch loader (bypasses DataLoader)")
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=not is_iterable,
            num_workers=num_workers if not self._use_fast_loader else 0,
            drop_last=True,
            pin_memory=False,  # pin_memory=True can hang on OOM (PyTorch #73003)
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
                pin_memory=False,  # pin_memory=True can hang on OOM (PyTorch #73003)
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

        # Self-Forcing setup
        self._sf_enabled = sf_enabled
        self._sf_ratio = sf_ratio
        self._sf_unroll = sf_unroll_length
        self._sf_horizon_weights = sf_horizon_weights
        self._sf_selective_bptt = sf_selective_bptt
        self._sf_full_bptt = sf_full_bptt
        self._sf_valid_starts = None
        if sf_enabled:
            if dataset is None:
                logger.warning("Self-Forcing enabled but no dataset provided — disabling")
                self._sf_enabled = False
            else:
                sf_K = self.model.context_len
                train_split_idx = max(1, int(dataset.num_games * 0.9))
                starts = []
                for gi in range(train_split_idx):
                    gs = dataset.game_offsets[gi]
                    ge = dataset.game_offsets[gi + 1]
                    for t in range(gs + sf_K, ge - sf_unroll_length):
                        starts.append(t)
                self._sf_valid_starts = np.array(starts, dtype=np.int64)
                logger.info(
                    "Self-Forcing: %d valid starts, ratio=1:%d, unroll=%d, selective_bptt=%s, full_bptt=%s",
                    len(self._sf_valid_starts), sf_ratio, sf_unroll_length,
                    sf_selective_bptt, sf_full_bptt,
                )

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

    def _build_sf_targets(
        self, frame_indices: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build ground truth (ctrl, float_tgt, int_tgt) for Self-Forcing.

        Mirrors MeleeFrameDataset.__getitem__ target construction, batched.
        """
        dataset = self._rollout_dataset
        cfg = self.cfg
        fp = cfg.float_per_player
        ccd = cfg.core_continuous_dim
        ipp = cfg.int_per_player

        fi = torch.from_numpy(frame_indices)

        tgt = dataset.floats[fi]          # (B, F)
        prev = dataset.floats[fi - 1]     # (B, F)
        tgt_ints = dataset.ints[fi]        # (B, I)

        # Controller input (with threshold features if enabled)
        ctrl = build_ctrl_batch(dataset.floats, fi, cfg)

        # Continuous deltas (4 per player)
        p0_cont_d = tgt[:, :ccd] - prev[:, :ccd]
        p1_cont_d = tgt[:, fp:fp + ccd] - prev[:, fp:fp + ccd]

        # Velocity deltas (5 per player)
        vs = ccd
        ve = vs + cfg.velocity_dim
        p0_vel_d = tgt[:, vs:ve] - prev[:, vs:ve]
        p1_vel_d = tgt[:, fp + vs:fp + ve] - prev[:, fp + vs:fp + ve]

        # Binary (absolute)
        bs = cfg.continuous_dim
        be = bs + cfg.binary_dim
        p0_bin = tgt[:, bs:be]
        p1_bin = tgt[:, fp + bs:fp + be]

        # Dynamics (absolute — hitlag, stocks, combo [, hitstun])
        ds = ve + (0 if cfg.state_age_as_embed else 1)
        dpp = cfg.predicted_dynamics_dim // 2
        p0_dyn_idx = list(range(ds, ds + dpp))
        p1_dyn_idx = [fp + i for i in p0_dyn_idx]
        p0_dyn = tgt[:, p0_dyn_idx]
        p1_dyn = tgt[:, p1_dyn_idx]

        float_tgt = torch.cat([
            p0_cont_d, p1_cont_d,
            p0_vel_d, p1_vel_d,
            p0_bin, p1_bin,
            p0_dyn, p1_dyn,
        ], dim=1)

        int_tgt = torch.stack([
            tgt_ints[:, 0],           # p0_action
            tgt_ints[:, 1],           # p0_jumps
            tgt_ints[:, 3],           # p0_l_cancel
            tgt_ints[:, 4],           # p0_hurtbox
            tgt_ints[:, 5],           # p0_ground
            tgt_ints[:, 6],           # p0_last_attack
            tgt_ints[:, ipp],         # p1_action
            tgt_ints[:, ipp + 1],     # p1_jumps
            tgt_ints[:, ipp + 3],     # p1_l_cancel
            tgt_ints[:, ipp + 4],     # p1_hurtbox
            tgt_ints[:, ipp + 5],     # p1_ground
            tgt_ints[:, ipp + 6],     # p1_last_attack
        ], dim=1)

        return ctrl, float_tgt, int_tgt

    def _self_forcing_step(self) -> tuple[float, "BatchMetrics"]:
        """One Self-Forcing step: unroll N AR steps, loss vs ground truth.

        Dispatches to full BPTT variant if enabled. Otherwise uses truncated
        BPTT (per-step backward, detached reconstruction).

        Returns (averaged_loss_scalar, last_step_metrics).
        """
        if self._sf_full_bptt:
            return self._self_forcing_step_full_bptt()
        dataset = self._rollout_dataset
        K = self.model.context_len
        N = self._sf_unroll
        B = self.batch_size

        # Sample starting points from pre-computed valid starts
        idx = np.random.choice(
            len(self._sf_valid_starts), size=B,
            replace=len(self._sf_valid_starts) < B,
        )
        starts = self._sf_valid_starts[idx]

        # Pre-build ALL context and targets on CPU, transfer once
        # Context: vectorized slice instead of per-sample loop
        offsets = np.arange(-K, 0)
        ctx_indices = starts[:, None] + offsets[None, :]  # (B, K)
        batch_floats = dataset.floats[ctx_indices]  # (B, K, F)
        batch_ints = dataset.ints[ctx_indices]  # (B, K, I)

        # Pre-build targets for all N steps at once
        all_ctrls = []
        all_float_tgts = []
        all_int_tgts = []
        for step in range(N):
            ctrl, float_tgt, int_tgt = self._build_sf_targets(starts + step)
            all_ctrls.append(ctrl)
            all_float_tgts.append(float_tgt)
            all_int_tgts.append(int_tgt)

        # Transfer targets to GPU once (not per-step)
        all_ctrls_gpu = [c.to(self.device) for c in all_ctrls]
        all_float_tgts_gpu = [f.to(self.device) for f in all_float_tgts]
        all_int_tgts_gpu = [i.to(self.device) for i in all_int_tgts]

        total_sf_loss = 0.0
        last_metrics = None

        for step in range(N):
            ctx_f = batch_floats[:, -K:, :].to(self.device)
            ctx_i = batch_ints[:, -K:, :].to(self.device)

            preds = self.model(ctx_f, ctx_i, all_ctrls_gpu[step])

            if self._sf_selective_bptt and step > 0:
                _CONTINUOUS_KEYS = {
                    'continuous_delta', 'velocity_delta',
                    'dynamics_pred', 'binary_logits',
                }
                preds = {
                    k: v if k in _CONTINUOUS_KEYS else v.detach()
                    for k, v in preds.items()
                }

            loss, metrics = self.metrics.compute_loss(
                preds, all_float_tgts_gpu[step], all_int_tgts_gpu[step], ctx_i,
            )
            (loss / N).backward()
            total_sf_loss += loss.item()
            last_metrics = metrics

            # Reconstruct next frame — on CPU (argmax is cheap, keeps GPU free)
            with torch.no_grad():
                preds_cpu = {k: v.cpu() for k, v in preds.items()}
                next_float, next_int = reconstruct_frame(
                    preds_cpu, batch_floats[:, -1, :],
                    batch_ints[:, -1, :], all_ctrls[step], self.cfg,
                )
            batch_floats = torch.cat(
                [batch_floats, next_float.unsqueeze(1)], dim=1,
            )
            batch_ints = torch.cat(
                [batch_ints, next_int.unsqueeze(1)], dim=1,
            )

        # Gradients already accumulated via per-step backward().
        # Return None for loss (caller should NOT call backward again).
        avg_loss = total_sf_loss / N
        return avg_loss, last_metrics

    def _self_forcing_step_full_bptt(self) -> tuple[float, "BatchMetrics"]:
        """Full BPTT Self-Forcing: gradients flow through reconstruction.

        Unlike truncated BPTT:
        - Differentiable reconstruction keeps float gradients flowing
        - Categorical predictions use soft embeddings (softmax @ embed.weight)
        - Single backward() at the end instead of per-step
        - Everything stays on GPU

        This lets the model learn multi-step error correction: "my position
        error at step 1 caused an action mispredict at step 3."
        """
        dataset = self._rollout_dataset
        K = self.model.context_len
        N = self._sf_unroll
        B = self.batch_size

        # Sample starting points
        idx = np.random.choice(
            len(self._sf_valid_starts), size=B,
            replace=len(self._sf_valid_starts) < B,
        )
        starts = self._sf_valid_starts[idx]

        # Pre-build context (CPU, then transfer once)
        offsets = np.arange(-K, 0)
        ctx_indices = starts[:, None] + offsets[None, :]
        batch_floats = dataset.floats[ctx_indices].to(self.device)  # (B, K, F)
        batch_ints = dataset.ints[ctx_indices].to(self.device)  # (B, K, I)

        # Pre-build targets for all N steps, transfer to GPU
        all_ctrls_gpu = []
        all_float_tgts_gpu = []
        all_int_tgts_gpu = []
        for step in range(N):
            ctrl, float_tgt, int_tgt = self._build_sf_targets(starts + step)
            all_ctrls_gpu.append(ctrl.to(self.device))
            all_float_tgts_gpu.append(float_tgt.to(self.device))
            all_int_tgts_gpu.append(int_tgt.to(self.device))

        total_loss = torch.tensor(0.0, device=self.device)
        last_metrics = None
        cat_logits = None  # soft categoricals from previous step's reconstruction

        for step in range(N):
            ctx_f = batch_floats[:, -K:, :]
            ctx_i = batch_ints[:, -K:, :]

            # Pass soft categoricals from previous reconstruction (step > 0)
            preds = self.model(ctx_f, ctx_i, all_ctrls_gpu[step],
                               soft_cat_last=cat_logits)

            loss, metrics = self.metrics.compute_loss(
                preds, all_float_tgts_gpu[step], all_int_tgts_gpu[step], ctx_i,
            )
            total_loss = total_loss + loss / N
            last_metrics = metrics

            # Differentiable reconstruction — gradients flow through
            next_float, next_int, cat_logits = reconstruct_frame_differentiable(
                preds, batch_floats[:, -1, :],
                batch_ints[:, -1, :], all_ctrls_gpu[step], self.cfg,
            )

            batch_floats = torch.cat(
                [batch_floats, next_float.unsqueeze(1)], dim=1,
            )
            batch_ints = torch.cat(
                [batch_ints, next_int.unsqueeze(1)], dim=1,
            )

        # Single backward for all steps — gradients flow through entire unroll
        total_loss.backward()

        avg_loss = total_loss.item()
        return avg_loss, last_metrics

    def _fast_batch_iter(self):
        """Yield batches using vectorized get_batch instead of DataLoader.

        Shuffles indices each epoch, slices into batch-sized chunks,
        and calls dataset.get_batch() which does one advanced-index
        operation instead of N __getitem__ calls.
        """
        ds = self.train_loader.dataset
        n = len(ds)
        indices = np.random.permutation(n)
        bs = self.batch_size
        for start in range(0, n - bs + 1, bs):
            batch_indices = indices[start:start + bs]
            yield ds.get_batch(batch_indices)

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        epoch_metrics = EpochMetrics()

        if self._use_fast_loader:
            batch_iter = self._fast_batch_iter()
            num_batches = len(self.train_loader.dataset) // self.batch_size
        else:
            batch_iter = iter(self.train_loader)
            num_batches = len(self.train_loader)
        log_interval = self._compute_log_interval(num_batches)

        sf_count = 0
        for batch_idx, (float_ctx, int_ctx, next_ctrl, float_tgt, int_tgt) in enumerate(batch_iter):
            self.optimizer.zero_grad()

            # Interleave: every (ratio+1)th batch is Self-Forcing
            is_sf = (
                self._sf_enabled
                and (batch_idx + 1) % (self._sf_ratio + 1) == 0
            )

            if is_sf:
                # SF does per-step backward internally (saves GPU memory).
                # Returns scalar loss, not tensor — don't call backward.
                loss_val, batch_metrics = self._self_forcing_step()
                sf_count += 1
            else:
                float_ctx = float_ctx.to(self.device)
                int_ctx = int_ctx.to(self.device)
                next_ctrl = next_ctrl.to(self.device)
                float_tgt = float_tgt.to(self.device)
                int_tgt = int_tgt.to(self.device)
                predictions = self.model(float_ctx, int_ctx, next_ctrl)
                loss, batch_metrics = self.metrics.compute_loss(
                    predictions, float_tgt, int_tgt, int_ctx,
                )
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            epoch_metrics.update(batch_metrics)

            # Log VRAM peak after first batch
            if batch_idx == 0 and self.device.type == "cuda":
                logger.info(
                    "VRAM peak after first batch: %.2f GB (allocated: %.2f GB)",
                    torch.cuda.max_memory_allocated() / 1e9,
                    torch.cuda.memory_allocated() / 1e9,
                )

            if (batch_idx + 1) % log_interval == 0:
                pct = 100.0 * (batch_idx + 1) / num_batches
                sf_tag = " [SF]" if is_sf else ""
                logger.info(
                    "  batch %d/%d (%.0f%%) loss=%.4f%s",
                    batch_idx + 1, num_batches, pct,
                    batch_metrics.total_loss, sf_tag,
                )
                if wandb and wandb.run:
                    log_dict = {
                        "batch/loss": batch_metrics.total_loss,
                        "batch/step": batch_idx + 1,
                        "batch/pct": pct,
                    }
                    if is_sf:
                        log_dict["batch/sf_loss"] = batch_metrics.total_loss
                    else:
                        log_dict["batch/tf_loss"] = batch_metrics.total_loss
                    wandb.log(log_dict)

        if sf_count > 0:
            logger.info("  Self-Forcing: %d SF batches this epoch", sf_count)
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
