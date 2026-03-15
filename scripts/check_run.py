#!/usr/bin/env python3
"""Check wandb run status and print summary metrics.

Used by the conductor heartbeat to detect completed experiments.

Usage:
    python scripts/check_run.py <wandb_run_id>
    python scripts/check_run.py wdv9wynz

Output (JSON):
    {"state": "finished", "runtime_s": 7200, "metrics": {"eval/summary_pos_mae": 6.26, ...}}
    {"state": "running", "runtime_s": 1800, "progress_pct": 50.0, "latest_loss": 0.31}
    {"state": "crashed", "runtime_s": 300, "error": "OOM"}
"""

import json
import sys

def check_run(run_id: str, project: str = "shinewave/melee-worldmodel") -> dict:
    import wandb
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")

    result = {
        "state": run.state,
        "runtime_s": round(run.summary.get("_runtime", 0)),
    }

    if run.state == "finished":
        result["metrics"] = {}
        for key in [
            "eval/summary_pos_mae",
            "loss/total",
            "metric/p0_action_acc",
            "metric/action_change_acc",
            "metric/position_mae",
            "batch/tf_loss",
            "batch/sf_loss",
            "eval/h10_pos_mae",
            "eval/h10_action_acc",
        ]:
            val = run.summary.get(key)
            if val is not None:
                result["metrics"][key] = round(float(val), 4)
    elif run.state == "running":
        result["progress_pct"] = round(run.summary.get("batch/pct", 0), 1)
        result["latest_loss"] = round(run.summary.get("batch/loss", 0), 4)
    elif run.state in ("crashed", "failed"):
        # Try to get error info from run notes or config
        result["error"] = run.summary.get("_error", "unknown")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_run.py <wandb_run_id>", file=sys.stderr)
        sys.exit(1)

    run_id = sys.argv[1]
    result = check_run(run_id)
    print(json.dumps(result, indent=2))
