#!/usr/bin/env python3
"""Rollout coherence evaluation — the primary metric for autoresearch.

Measures how fast the model diverges from reality when running
autoregressively (feeding its own predictions back as input).

Samples N starting frames from the val set, autoregeresses K steps from each,
and compares against ground truth at every horizon. The summary metric is
mean position MAE averaged over all K horizons — single number, lower is better.

Usage:
    python scripts/eval_rollout.py \
        --checkpoint checkpoints/e017a/best.pt \
        --dataset data/parsed-v2 \
        --config experiments/e017a-absolute-y.yaml

    # With JSON output for autoresearch agents:
    python scripts/eval_rollout.py \
        --checkpoint checkpoints/e017a/best.pt \
        --dataset data/parsed-v2 \
        --output eval_result.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import MeleeDataset
from data.parse import load_games_from_dir
from models.checkpoint import load_model_from_checkpoint
from models.encoding import EncodingConfig
from scripts.ar_utils import build_ctrl_batch, reconstruct_frame
from training.constraints import ConstraintChecker

logger = logging.getLogger(__name__)


def load_config(config_path: str | None) -> dict:
    if config_path:
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def sample_starting_points(
    dataset: MeleeDataset,
    val_game_range: range,
    context_len: int,
    horizon: int,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    """Sample deterministic starting frame indices from val games.

    Each starting point must have context_len frames before it and
    horizon frames after it, all within the same game.

    Returns global frame indices into dataset.floats/ints.
    """
    valid_starts = []
    for gi in val_game_range:
        game_start = dataset.game_offsets[gi]
        game_end = dataset.game_offsets[gi + 1]
        # Need context_len frames for seed + horizon frames for rollout
        for t in range(game_start + context_len, game_end - horizon):
            valid_starts.append(t)

    valid_starts = np.array(valid_starts, dtype=np.int64)

    if len(valid_starts) == 0:
        raise ValueError("No valid starting points in val set. Check data filters and horizon.")

    rng = np.random.RandomState(seed)
    n = min(num_samples, len(valid_starts))
    chosen = rng.choice(valid_starts, size=n, replace=False)
    chosen.sort()  # cache-friendly access order

    logger.info(
        "Sampled %d starting points from %d valid (%d val games)",
        n, len(valid_starts), len(val_game_range),
    )
    return chosen


@torch.no_grad()
def evaluate_rollout_coherence(
    model: torch.nn.Module,
    dataset: MeleeDataset,
    starting_points: np.ndarray,
    context_len: int,
    horizon: int,
    cfg: EncodingConfig,
    device: str,
) -> dict:
    """Run batched AR rollouts and compute divergence metrics.

    Returns dict with per-horizon metrics and summary scalar.
    """
    model.eval()
    N = len(starting_points)
    fp = cfg.float_per_player
    ipp = cfg.int_per_player

    # Constraint checker for violation detection
    constraint_checker = ConstraintChecker(cfg)

    # Pre-build batched context windows: (N, K, F) and (N, K, I).
    # .cpu() handles the gpu_resident=True case — dataset tensors may live
    # on GPU, but reconstruct_frame() and the metric computations below
    # expect CPU tensors. Rollout eval does CPU↔GPU roundtrips per step.
    batch_floats = torch.stack([
        dataset.floats[t - context_len:t] for t in starting_points
    ]).cpu()  # (N, K, F)
    batch_ints = torch.stack([
        dataset.ints[t - context_len:t] for t in starting_points
    ]).cpu()  # (N, K, I)

    # Metrics accumulators: [horizon] -> list of per-sample values
    pos_maes = [[] for _ in range(horizon)]
    vel_maes = [[] for _ in range(horizon)]
    action_accs = [[] for _ in range(horizon)]
    percent_maes = [[] for _ in range(horizon)]
    # action_change_acc needs special handling: at frames where GT action
    # differs from the PREVIOUS GT frame, did the model predict the new one?
    # Accumulator holds (n_correct, n_changes) per horizon so we can compute
    # accuracy at the end (avoiding division-by-zero when no changes happen).
    action_change_counts = [[0, 0] for _ in range(horizon)]  # [correct, total]
    horizon_violations: list[dict[str, float]] = [{} for _ in range(horizon)]

    starts_t = torch.from_numpy(starting_points)

    for k in range(horizon):
        # Context: last K frames from accumulated simulation
        ctx_f = batch_floats[:, -context_len:, :].to(device)
        ctx_i = batch_ints[:, -context_len:, :].to(device)

        # Controller inputs from ground truth for frame (start + k)
        frame_indices = starts_t + k
        ctrl = build_ctrl_batch(dataset.floats, frame_indices, cfg).to(device)

        # Batched forward pass
        preds = model(ctx_f, ctx_i, ctrl)

        # Move predictions to CPU for reconstruction
        preds_cpu = {key: val.cpu() for key, val in preds.items()}

        # Reconstruct next frame (batched)
        prev_float = batch_floats[:, -1, :]
        prev_int = batch_ints[:, -1, :]
        ctrl_cpu = ctrl.cpu()
        next_float, next_int = reconstruct_frame(
            preds_cpu, prev_float, prev_int, ctrl_cpu, cfg,
        )

        # Check constraint violations on reconstructed frame
        horizon_violations[k] = constraint_checker.check_batch_and_log(
            next_float, prev_float, cfg,
        )

        # Ground truth for this horizon step — .cpu() for gpu_resident mode
        gt_float = torch.stack([dataset.floats[t + k] for t in starting_points]).cpu()
        gt_int = torch.stack([dataset.ints[t + k] for t in starting_points]).cpu()

        # --- Compute metrics (in game units) ---

        # Position MAE: x(1), y(2) for both players, denormalized
        p0_x_err = (next_float[:, 1] - gt_float[:, 1]).abs() / cfg.xy_scale
        p0_y_err = (next_float[:, 2] - gt_float[:, 2]).abs() / cfg.xy_scale
        p1_x_err = (next_float[:, fp + 1] - gt_float[:, fp + 1]).abs() / cfg.xy_scale
        p1_y_err = (next_float[:, fp + 2] - gt_float[:, fp + 2]).abs() / cfg.xy_scale
        pos_mae = (p0_x_err + p0_y_err + p1_x_err + p1_y_err) / 4.0
        pos_maes[k] = pos_mae.tolist()

        # Velocity MAE: indices 4:9 for 5 velocity components per player
        vel_start, vel_end = 4, 9
        p0_vel = (next_float[:, vel_start:vel_end] - gt_float[:, vel_start:vel_end]).abs() / cfg.velocity_scale
        p1_vel = (next_float[:, fp + vel_start:fp + vel_end] - gt_float[:, fp + vel_start:fp + vel_end]).abs() / cfg.velocity_scale
        vel_mae = (p0_vel.mean(dim=1) + p1_vel.mean(dim=1)) / 2.0
        vel_maes[k] = vel_mae.tolist()

        # Action accuracy: action is int column 0 (p0) and ipp (p1)
        p0_act_match = (next_int[:, 0] == gt_int[:, 0]).float()
        p1_act_match = (next_int[:, ipp] == gt_int[:, ipp]).float()
        action_acc = (p0_act_match + p1_act_match) / 2.0
        action_accs[k] = action_acc.tolist()

        # Action CHANGE accuracy: the sharp diagnostic that catches the
        # "position improved but moves are now wrong" regime tradeoff.
        # Fetch previous GT frame's action (either frame start+k-1 from the
        # dataset, or for k=0 the seed context's last frame action).
        if k == 0:
            # Seed context ends at start-1 in global frame coords, so the
            # "previous" GT action is the last context frame's action.
            prev_gt_int = batch_ints[:, -1, :]  # (N, I) — last context frame
            prev_p0_act = prev_gt_int[:, 0]
            prev_p1_act = prev_gt_int[:, ipp]
        else:
            prev_gt_float_ignored = torch.stack(
                [dataset.ints[t + k - 1] for t in starting_points]
            ).cpu()
            prev_p0_act = prev_gt_float_ignored[:, 0]
            prev_p1_act = prev_gt_float_ignored[:, ipp]

        p0_changed = prev_p0_act != gt_int[:, 0]
        p1_changed = prev_p1_act != gt_int[:, ipp]
        n_p0_changes = int(p0_changed.sum().item())
        n_p1_changes = int(p1_changed.sum().item())
        if n_p0_changes > 0:
            p0_change_correct = int(
                ((next_int[:, 0] == gt_int[:, 0]) & p0_changed).sum().item()
            )
        else:
            p0_change_correct = 0
        if n_p1_changes > 0:
            p1_change_correct = int(
                ((next_int[:, ipp] == gt_int[:, ipp]) & p1_changed).sum().item()
            )
        else:
            p1_change_correct = 0
        action_change_counts[k][0] = p0_change_correct + p1_change_correct
        action_change_counts[k][1] = n_p0_changes + n_p1_changes

        # Percent MAE: index 0 for each player, denormalized
        p0_pct = (next_float[:, 0] - gt_float[:, 0]).abs() / cfg.percent_scale
        p1_pct = (next_float[:, fp] - gt_float[:, fp]).abs() / cfg.percent_scale
        pct_mae = (p0_pct + p1_pct) / 2.0
        percent_maes[k] = pct_mae.tolist()

        # Append predicted frame to batch for next step's context
        batch_floats = torch.cat([batch_floats, next_float.unsqueeze(1)], dim=1)
        batch_ints = torch.cat([batch_ints, next_int.unsqueeze(1)], dim=1)

    # --- Aggregate ---
    per_horizon = {}
    for k in range(horizon):
        entry = {
            "pos_mae": float(np.mean(pos_maes[k])),
            "vel_mae": float(np.mean(vel_maes[k])),
            "action_acc": float(np.mean(action_accs[k])),
            "percent_mae": float(np.mean(percent_maes[k])),
        }
        # Action change accuracy — undefined when no frames changed in this
        # horizon slice. Report as NaN in that case (rare at K=1, common at
        # later horizons if the rollout stalls).
        n_correct, n_total = action_change_counts[k]
        entry["action_change_acc"] = (
            float(n_correct / n_total) if n_total > 0 else float("nan")
        )
        entry["action_change_n"] = int(n_total)

        if horizon_violations[k]:
            entry["violation_rate"] = horizon_violations[k].get("total", 0.0)
            entry["violations"] = {
                name: rate for name, rate in horizon_violations[k].items()
                if name != "total"
            }
        per_horizon[k + 1] = entry

    # Summary helpers — mean a given per_horizon field over the first K
    # horizons. Used for both K=horizon (legacy north star) and K=5/K=10
    # (per ScavieFae review — K=20 is below the noise floor for
    # architecture discrimination at 60fps, K=5 is pure local-dynamics).
    def _mean_up_to(field: str, k: int) -> float:
        k = min(k, horizon)
        if k <= 0:
            return float("nan")
        vals = [per_horizon[i + 1].get(field, float("nan")) for i in range(k)]
        vals = [v for v in vals if v == v]  # drop NaN
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    # Legacy: pos_mae mean across full horizon (the Mamba2 north star)
    summary_pos_mae = _mean_up_to("pos_mae", horizon)

    # K=5 and K=10 summaries for every metric (per Mattie's multi-metric
    # directive — single-metric tracking hides regime tradeoffs like
    # "position improved but action_change_acc degraded").
    summary_pos_mae_k5 = _mean_up_to("pos_mae", 5)
    summary_pos_mae_k10 = _mean_up_to("pos_mae", 10)
    summary_vel_mae_k5 = _mean_up_to("vel_mae", 5)
    summary_vel_mae_k10 = _mean_up_to("vel_mae", 10)
    summary_action_acc_k5 = _mean_up_to("action_acc", 5)
    summary_action_acc_k10 = _mean_up_to("action_acc", 10)
    summary_action_change_acc_k5 = _mean_up_to("action_change_acc", 5)
    summary_action_change_acc_k10 = _mean_up_to("action_change_acc", 10)
    summary_percent_mae_k5 = _mean_up_to("percent_mae", 5)
    summary_percent_mae_k10 = _mean_up_to("percent_mae", 10)

    # Summary violation rate: mean across all horizons
    all_violation_rates = [
        per_horizon[k + 1].get("violation_rate", 0.0) for k in range(horizon)
    ]
    summary_violation_rate = float(np.mean(all_violation_rates))
    summary_violation_rate_k5 = _mean_up_to("violation_rate", 5)
    summary_violation_rate_k10 = _mean_up_to("violation_rate", 10)

    return {
        "per_horizon": per_horizon,
        # Legacy K=horizon summary (Mamba2 historical north star)
        "summary_pos_mae": summary_pos_mae,
        "violation_rate": summary_violation_rate,
        # K=5 suite (new primary)
        "summary_pos_mae_k5": summary_pos_mae_k5,
        "summary_vel_mae_k5": summary_vel_mae_k5,
        "summary_action_acc_k5": summary_action_acc_k5,
        "summary_action_change_acc_k5": summary_action_change_acc_k5,
        "summary_percent_mae_k5": summary_percent_mae_k5,
        "summary_violation_rate_k5": summary_violation_rate_k5,
        # K=10 suite (secondary)
        "summary_pos_mae_k10": summary_pos_mae_k10,
        "summary_vel_mae_k10": summary_vel_mae_k10,
        "summary_action_acc_k10": summary_action_acc_k10,
        "summary_action_change_acc_k10": summary_action_change_acc_k10,
        "summary_percent_mae_k10": summary_percent_mae_k10,
        "summary_violation_rate_k10": summary_violation_rate_k10,
    }


def format_table(results: dict, horizon: int) -> str:
    """Format results as a human-readable table."""
    lines = []
    lines.append("  Horizon | pos_mae | vel_mae | action_acc | pct_mae")
    lines.append("  --------+---------+---------+------------+--------")

    display_horizons = [1, 5, 10, 20]
    for h in display_horizons:
        if h > horizon:
            break
        m = results["per_horizon"][h]
        lines.append(
            f"  t+{h:<5d} | {m['pos_mae']:7.2f} | {m['vel_mae']:7.2f} "
            f"| {m['action_acc']:10.3f} | {m['percent_mae']:6.2f}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Rollout coherence evaluation for world model checkpoints",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to parsed dataset directory")
    parser.add_argument("--config", default=None, help="Experiment YAML (for data filters)")
    parser.add_argument("--num-samples", type=int, default=300, help="Starting points to sample (N)")
    parser.add_argument("--horizon", type=int, default=20, help="AR steps per rollout (K)")
    parser.add_argument("--device", default=None, help="Device (cpu/mps/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    # Load model + encoding config from checkpoint
    model, cfg, context_len, arch = load_model_from_checkpoint(args.checkpoint, device)
    logger.info("Model: %s, context_len=%d", arch, context_len)

    # Load experiment config for data filters
    exp_cfg = load_config(args.config)
    data_cfg = exp_cfg.get("data", {})
    train_split = exp_cfg.get("training", {}).get("train_split", 0.9)

    # Handle character_filter that might be a list (YAML) or int
    char_filter = data_cfg.get("character_filter")
    if isinstance(char_filter, list):
        # load_games_from_dir expects a single int; for multi-char,
        # we pass None and let all characters through (the dataset was
        # already filtered during training data prep)
        char_filter = None

    # Load val games
    t0 = time.time()
    games = load_games_from_dir(
        args.dataset,
        max_games=data_cfg.get("max_games"),
        stage_filter=data_cfg.get("stage_filter"),
        character_filter=char_filter,
    )
    if not games:
        logger.error("No games loaded! Check dataset path and filters.")
        sys.exit(1)

    dataset = MeleeDataset(games, cfg)
    load_time = time.time() - t0
    logger.info("Loaded %d games (%d frames) in %.1fs", dataset.num_games, dataset.total_frames, load_time)

    # Val split (same logic as training)
    split_idx = max(1, int(dataset.num_games * train_split))
    val_game_range = range(split_idx, dataset.num_games)
    val_frames = sum(
        dataset.game_offsets[gi + 1] - dataset.game_offsets[gi]
        for gi in val_game_range
    )
    logger.info("Val: %d games, %d frames", len(val_game_range), val_frames)

    # Sample starting points
    starting_points = sample_starting_points(
        dataset, val_game_range, context_len, args.horizon,
        args.num_samples, args.seed,
    )

    # Run evaluation
    t0 = time.time()
    results = evaluate_rollout_coherence(
        model, dataset, starting_points, context_len,
        args.horizon, cfg, device,
    )
    eval_time = time.time() - t0

    # Load checkpoint metadata for reporting
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch", -1)

    # Print results
    print()
    print("Rollout Coherence Eval")
    print("=" * 56)
    print(f"  Checkpoint : {args.checkpoint} (epoch {epoch})")
    print(f"  Architecture: {arch}")
    print(f"  Val games  : {len(val_game_range)}")
    print(f"  Samples    : {len(starting_points)}")
    print(f"  Horizon    : {args.horizon}")
    print(f"  Eval time  : {eval_time:.1f}s")
    print()
    print(format_table(results, args.horizon))
    print()
    print(f"  ** summary_pos_mae = {results['summary_pos_mae']:.4f} **")
    print(f"  ** violation_rate  = {results.get('violation_rate', 0.0):.4f} **")
    print()

    # JSON output
    if args.output:
        output = {
            "checkpoint": str(args.checkpoint),
            "epoch": epoch,
            "arch": arch,
            "num_samples": len(starting_points),
            "horizon": args.horizon,
            "seed": args.seed,
            "eval_time_s": round(eval_time, 2),
            "summary_pos_mae": results["summary_pos_mae"],
            "violation_rate": results.get("violation_rate", 0.0),
            "per_horizon": {
                str(k): v for k, v in results["per_horizon"].items()
            },
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
