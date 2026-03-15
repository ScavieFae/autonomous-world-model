#!/usr/bin/env python3
"""Post notifications to Matrix rooms via Synapse REST API.

No matrix-nio needed — just HTTP calls. Usable from the conductor,
executor, training scripts, or any agent that needs to notify.

Usage:
    # From Python:
    from bot.notify import post, post_experiment_started, post_experiment_result

    post("#conductor-log", "Loop started")
    post_experiment_started("e020", "batch size 256", "$2", "https://wandb.ai/...")
    post_experiment_result("e020", rc=6.10, prior_rc=6.26, kept=True, cost=1.80)

    # From shell:
    source bot/.env
    python -m bot.notify "#conductor-log" "Experiment e020 launched"
"""

import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

# Room aliases → room IDs (loaded from env)
ROOM_MAP = {
    "#conductor-log": os.environ.get("MATRIX_CONDUCTOR_ROOM", ""),
    "#experiment-results": os.environ.get("MATRIX_EXPERIMENT_ROOM", ""),
    "#escalations": os.environ.get("MATRIX_ESCALATION_ROOM", ""),
}

HOMESERVER = os.environ.get("MATRIX_HOMESERVER", "")
BOT_USER = os.environ.get("MATRIX_BOT_USER", "")
SERVER_NAME = os.environ.get("MATRIX_SERVER_NAME", "")

# Try to load access token from session file
BOT_DIR = Path(__file__).resolve().parent
SESSION_FILE = BOT_DIR / ".session.json"

def _get_token():
    """Get access token from session file."""
    try:
        session = json.loads(SESSION_FILE.read_text())
        return session.get("access_token", "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


def _resolve_room(room):
    """Resolve room alias to room ID."""
    if room.startswith("!"):
        return room
    return ROOM_MAP.get(room, room)


def post(room: str, message: str):
    """Post a message to a Matrix room via REST API."""
    token = _get_token()
    if not token or not HOMESERVER:
        print(f"[notify] No Matrix session. Would post to {room}: {message[:80]}")
        return

    room_id = _resolve_room(room)
    if not room_id:
        print(f"[notify] Unknown room: {room}")
        return

    import time
    txn_id = str(int(time.time() * 1000))
    url = f"{HOMESERVER}/_matrix/client/v3/rooms/{room_id}/send/m.room.message/{txn_id}"

    body = json.dumps({
        "msgtype": "m.text",
        "body": message,
    }).encode()

    req = Request(url, data=body, method="PUT")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")

    try:
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except URLError as e:
        print(f"[notify] Failed to post to {room}: {e}")


# --- Templated notifications ---

def post_cycle_started(hypothesis: str, cost_tier: str):
    post("#conductor-log",
         f"**New research cycle**\n\n"
         f"Hypothesis: {hypothesis}\n"
         f"Cost tier: {cost_tier}")


def post_hypothesis_rejected(hypothesis: str, reason: str):
    post("#conductor-log",
         f"**Hypothesis rejected**\n\n"
         f"{hypothesis}\n\n"
         f"Reason: {reason}")


def post_experiment_started(exp_id: str, description: str, cost: str, wandb_url: str = ""):
    msg = (f"**Experiment launched: {exp_id}**\n\n"
           f"{description}\n"
           f"Estimated cost: {cost}")
    if wandb_url:
        msg += f"\nwandb: {wandb_url}"
    post("#conductor-log", msg)
    post("#experiment-results", msg)


def post_batch_progress(exp_id: str, batch: int, total: int, loss: float, pct: float):
    post("#conductor-log",
         f"`{exp_id}` batch {batch}/{total} ({pct:.0f}%) loss={loss:.4f}")


def post_experiment_result(exp_id: str, rc: float, prior_rc: float,
                           kept: bool, cost: float, summary: str = ""):
    delta_pct = (rc - prior_rc) / prior_rc * 100
    direction = "better" if rc < prior_rc else "worse"
    verdict = "KEPT" if kept else "DISCARDED"
    emoji = "+" if kept else "-"

    msg = (f"**{exp_id}: {verdict}**\n\n"
           f"Rollout coherence: {rc:.2f} (was {prior_rc:.2f}, {delta_pct:+.1f}% {direction})\n"
           f"Cost: ${cost:.2f}")
    if summary:
        msg += f"\n\n{summary}"

    post("#experiment-results", msg)
    post("#conductor-log", f"[{emoji}] {exp_id}: RC={rc:.2f} ({delta_pct:+.1f}%) — {verdict}")


def post_escalation(message: str):
    post("#escalations", f"**Needs attention**\n\n{message}")


def post_daily_digest(experiments_run: int, kept: int, discarded: int,
                      daily_spent: float, best_rc: float, best_exp: str,
                      narrative: str = ""):
    msg = (f"**Daily digest**\n\n"
           f"Experiments: {experiments_run} run, {kept} kept, {discarded} discarded\n"
           f"Spent: ${daily_spent:.2f}\n"
           f"Best RC: {best_rc:.2f} ({best_exp})")
    if narrative:
        msg += f"\n\n{narrative}"
    post("#research", msg)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m bot.notify '#room-name' 'message'")
        sys.exit(1)
    post(sys.argv[1], sys.argv[2])
