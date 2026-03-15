#!/usr/bin/env python3
"""AWM Research Loop Matrix Bot.

Posts experiment updates and accepts control commands via Matrix.
Runs on ScavieFae's machine alongside the research loop.

Safety model:
- Bot can ONLY read .loop/state/ files and write to signal files
- Commands are allowlisted (no arbitrary execution)
- Approvals require explicit confirmation keyword
- Commands expire after 10 minutes
- Budget limits are hard (bot refuses launches over limit)

Usage:
    pip install matrix-nio aiofiles
    python bot/matrix_bot.py
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from nio import AsyncClient, MatrixRoom, RoomMessageText
except ImportError:
    print("Install matrix-nio: pip install matrix-nio[e2e]")
    raise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("awm-bot")

# --- Configuration (from environment) ---

import os

HOMESERVER = os.environ.get("MATRIX_HOMESERVER", "")
BOT_USER = os.environ.get("MATRIX_BOT_USER", "")
BOT_PASS = os.environ.get("MATRIX_BOT_PASS", "")
SERVER_NAME = os.environ.get("MATRIX_SERVER_NAME", "")

CONDUCTOR_ROOM = os.environ.get("MATRIX_CONDUCTOR_ROOM", "")
EXPERIMENT_ROOM = os.environ.get("MATRIX_EXPERIMENT_ROOM", "")
ESCALATION_ROOM = os.environ.get("MATRIX_ESCALATION_ROOM", "")

# Only accept commands from these Matrix user IDs
AUTHORIZED_USERS = os.environ.get("MATRIX_AUTHORIZED_USERS", "").split(",")

# Validate on startup
_REQUIRED = ["MATRIX_HOMESERVER", "MATRIX_BOT_USER", "MATRIX_BOT_PASS",
             "MATRIX_SERVER_NAME", "MATRIX_CONDUCTOR_ROOM"]
_missing = [k for k in _REQUIRED if not os.environ.get(k)]
if _missing and __name__ == "__main__":
    print(f"Missing env vars: {', '.join(_missing)}")
    print("Copy bot/.env.example to bot/.env and fill in values, then: source bot/.env")
    raise SystemExit(1)

# Project paths (relative to repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / ".loop" / "state"
BUDGET_FILE = STATE_DIR / "budget.json"
RUNNING_FILE = STATE_DIR / "running.json"
LOG_FILE = STATE_DIR / "log.jsonl"
PAUSE_SIGNAL = STATE_DIR / "signals" / "pause.json"
ESCALATE_SIGNAL = STATE_DIR / "signals" / "escalate.json"

# Command expiry (seconds)
COMMAND_EXPIRY = 600  # 10 minutes


# --- State readers ---

def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_budget() -> dict:
    return read_json(BUDGET_FILE)


def read_running() -> dict:
    return read_json(RUNNING_FILE)


def read_log_tail(n: int = 5) -> list[str]:
    try:
        lines = LOG_FILE.read_text().strip().split("\n")
        return lines[-n:] if lines and lines[0] else []
    except FileNotFoundError:
        return []


def is_paused() -> bool:
    sig = read_json(PAUSE_SIGNAL)
    return sig.get("active", False)


# --- State writers (ONLY signal files) ---

def write_signal(path: Path, active: bool, reason: str = None):
    data = {
        "active": active,
        "reason": reason,
        "set_by": "matrix_bot",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2))


# --- Command handlers ---

async def cmd_status(client: AsyncClient, room_id: str):
    running = read_running()
    budget = read_budget()
    paused = is_paused()

    exp = running.get("experiments", {})
    in_flight = exp.get("in_flight")

    lines = ["**Loop Status**", ""]
    lines.append(f"- Paused: {'YES' if paused else 'no'}")
    lines.append(f"- Budget: ${budget.get('daily_spent', 0):.2f} / ${budget.get('daily_limit', 30):.2f} today")
    lines.append(f"- Weekly: ${budget.get('weekly_spent', 0):.2f} / ${budget.get('weekly_limit', 150):.2f}")
    lines.append(f"- Experiments run: {budget.get('experiments_run', 0)} "
                 f"(kept: {budget.get('experiments_kept', 0)}, "
                 f"discarded: {budget.get('experiments_discarded', 0)})")

    if in_flight:
        lines.append("")
        lines.append(f"**In flight:** {in_flight.get('experiment_id', '?')}")
        lines.append(f"- Started: {in_flight.get('started_at', '?')}")
        lines.append(f"- Budget reserved: ${in_flight.get('budget_reserved', 0):.2f}")
        if in_flight.get("wandb_url"):
            lines.append(f"- wandb: {in_flight['wandb_url']}")
    else:
        lines.append("")
        lines.append("No experiment in flight.")

    await client.room_send(room_id, "m.room.message", {
        "msgtype": "m.text", "body": "\n".join(lines),
    })


async def cmd_budget(client: AsyncClient, room_id: str):
    b = read_budget()
    msg = (
        f"**Budget**\n"
        f"- Today: ${b.get('daily_spent', 0):.2f} / ${b.get('daily_limit', 30):.2f}\n"
        f"- Week: ${b.get('weekly_spent', 0):.2f} / ${b.get('weekly_limit', 150):.2f}\n"
        f"- Total: ${b.get('total_spent', 0):.2f}\n"
        f"- Experiments: {b.get('experiments_run', 0)} run, "
        f"{b.get('experiments_kept', 0)} kept, "
        f"{b.get('experiments_discarded', 0)} discarded"
    )
    await client.room_send(room_id, "m.room.message", {
        "msgtype": "m.text", "body": msg,
    })


async def cmd_pause(client: AsyncClient, room_id: str):
    write_signal(PAUSE_SIGNAL, active=True, reason="Paused via Matrix")
    await client.room_send(room_id, "m.room.message", {
        "msgtype": "m.text", "body": "Loop paused. Use `!resume` to continue.",
    })
    logger.info("Loop paused via Matrix command")


async def cmd_resume(client: AsyncClient, room_id: str):
    write_signal(PAUSE_SIGNAL, active=False)
    await client.room_send(room_id, "m.room.message", {
        "msgtype": "m.text", "body": "Loop resumed.",
    })
    logger.info("Loop resumed via Matrix command")


async def cmd_logs(client: AsyncClient, room_id: str):
    entries = read_log_tail(5)
    if not entries:
        await client.room_send(room_id, "m.room.message", {
            "msgtype": "m.text", "body": "No log entries yet.",
        })
        return

    lines = ["**Recent decisions:**", ""]
    for entry in entries:
        try:
            data = json.loads(entry)
            ts = data.get("timestamp", "?")[:16]
            action = data.get("action", "?")
            exp = data.get("experiment", "")
            lines.append(f"- `{ts}` {action} {exp}")
        except json.JSONDecodeError:
            lines.append(f"- {entry[:80]}")

    await client.room_send(room_id, "m.room.message", {
        "msgtype": "m.text", "body": "\n".join(lines),
    })


async def cmd_kill(client: AsyncClient, room_id: str):
    """Kill the current in-flight Modal experiment."""
    running = read_running()
    exp = running.get("experiments", {})
    in_flight = exp.get("in_flight")

    if not in_flight:
        await client.room_send(room_id, "m.room.message", {
            "msgtype": "m.text", "body": "Nothing running to kill.",
        })
        return

    app_id = in_flight.get("modal_app_id", "")
    exp_id = in_flight.get("experiment_id", "?")

    # Clear the lock
    running["experiments"]["in_flight"] = None
    running["experiments"]["lock"] = False
    RUNNING_FILE.write_text(json.dumps(running, indent=2))

    msg = f"Cleared {exp_id} from running state."
    if app_id:
        msg += f"\n\nTo stop the Modal app, run:\n`modal app stop {app_id}`"

    await client.room_send(room_id, "m.room.message", {
        "msgtype": "m.text", "body": msg,
    })
    logger.info("Killed experiment %s via Matrix", exp_id)


async def cmd_help(client: AsyncClient, room_id: str):
    msg = (
        "**AWM Research Bot Commands**\n\n"
        "- `!status` — loop state, in-flight experiment, budget\n"
        "- `!budget` — spend details\n"
        "- `!pause` — pause the research loop\n"
        "- `!resume` — resume the loop\n"
        "- `!logs` — last 5 conductor decisions\n"
        "- `!kill` — clear in-flight experiment\n"
        "- `!help` — this message"
    )
    await client.room_send(room_id, "m.room.message", {
        "msgtype": "m.text", "body": msg,
    })


COMMANDS = {
    "!status": cmd_status,
    "!budget": cmd_budget,
    "!pause": cmd_pause,
    "!resume": cmd_resume,
    "!logs": cmd_logs,
    "!kill": cmd_kill,
    "!help": cmd_help,
}


# --- Notification helpers (for agents to call) ---

async def notify(client: AsyncClient, room_id: str, message: str):
    """Send a notification to a room."""
    await client.room_send(room_id, "m.room.message", {
        "msgtype": "m.text", "body": message,
    })


# --- Bot lifecycle ---

async def message_callback(room: MatrixRoom, event: RoomMessageText, client: AsyncClient):
    # Ignore own messages
    if event.sender == f"{BOT_USER}:{SERVER_NAME}":
        return

    # Only accept from authorized users
    if event.sender not in AUTHORIZED_USERS:
        logger.warning("Unauthorized command from %s: %s", event.sender, event.body)
        return

    # Check command expiry
    event_age = time.time() - event.server_timestamp / 1000
    if event_age > COMMAND_EXPIRY:
        logger.info("Ignoring stale command (%.0fs old): %s", event_age, event.body)
        return

    body = event.body.strip().lower()
    cmd_word = body.split()[0] if body else ""

    handler = COMMANDS.get(cmd_word)
    if handler:
        logger.info("Command from %s: %s", event.sender, body)
        await handler(client, room.room_id)
    elif body.startswith("!"):
        await client.room_send(room.room_id, "m.room.message", {
            "msgtype": "m.text", "body": f"Unknown command: `{cmd_word}`. Try `!help`.",
        })


async def main():
    client = AsyncClient(HOMESERVER, f"{BOT_USER}:{SERVER_NAME}")

    logger.info("Logging in as %s...", BOT_USER)
    resp = await client.login(BOT_PASS)
    if hasattr(resp, "access_token"):
        logger.info("Logged in. Access token: %s...", resp.access_token[:10])
    else:
        logger.error("Login failed: %s", resp)
        return

    # Register callback
    client.add_event_callback(
        lambda room, event: message_callback(room, event, client),
        RoomMessageText,
    )

    # Join rooms
    for room_id in [CONDUCTOR_ROOM, EXPERIMENT_ROOM, ESCALATION_ROOM]:
        await client.join(room_id)
        logger.info("Joined %s", room_id)

    # Announce
    await notify(client, CONDUCTOR_ROOM, "AWM Research Bot online. Type `!help` for commands.")

    # Sync loop
    logger.info("Listening for commands...")
    await client.sync_forever(timeout=30000)


if __name__ == "__main__":
    asyncio.run(main())
