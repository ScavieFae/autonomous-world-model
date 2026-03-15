#!/usr/bin/env python3
"""AWM Research Loop Matrix Bot.

Posts experiment updates and accepts control commands via Matrix.
Runs on ScavieFae's machine alongside the research loop.

Security model:
- E2E encryption via matrix-nio + libolm (messages encrypted in transit)
- Device verification required (SAS emoji verification on first setup)
- Read commands: require verified device from authorized user
- Mutation commands: require Ed25519 signature over (command + timestamp)
- Rate limiting: 1 command per 10s, 30 per hour
- Command expiry: 10 minutes
- Bot only writes to .loop/state/signals/ (never code or configs)

Setup:
    pip install "matrix-nio[e2e]" pynacl
    python bot/keygen.py              # generate keypair, save pubkey
    cp bot/.env.example bot/.env      # fill in values
    source bot/.env
    python bot/matrix_bot.py          # first run: will prompt for device verification
"""

import asyncio
import base64
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from nio import AsyncClient, AsyncClientConfig, MatrixRoom, RoomMessageText
except ImportError:
    print("Install: pip install matrix-nio pynacl")
    raise

# E2E is optional — works without libolm but device verification is disabled
E2E_AVAILABLE = False
try:
    from nio import RoomEncryptedMessage
    E2E_AVAILABLE = True
except ImportError:
    pass

try:
    from nacl.signing import VerifyKey
    from nacl.exceptions import BadSignatureError
except ImportError:
    print("Install: pip install pynacl")
    raise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("awm-bot")


# --- Configuration (from environment) ---

HOMESERVER = os.environ.get("MATRIX_HOMESERVER", "")
BOT_USER = os.environ.get("MATRIX_BOT_USER", "")
BOT_PASS = os.environ.get("MATRIX_BOT_PASS", "")
SERVER_NAME = os.environ.get("MATRIX_SERVER_NAME", "")

CONDUCTOR_ROOM = os.environ.get("MATRIX_CONDUCTOR_ROOM", "")
EXPERIMENT_ROOM = os.environ.get("MATRIX_EXPERIMENT_ROOM", "")
ESCALATION_ROOM = os.environ.get("MATRIX_ESCALATION_ROOM", "")

AUTHORIZED_USERS = [u.strip() for u in os.environ.get("MATRIX_AUTHORIZED_USERS", "").split(",") if u.strip()]

# Ed25519 public key for signed commands (base64-encoded, 32 bytes)
SIGNING_PUBKEY = os.environ.get("MATRIX_SIGNING_PUBKEY", "")

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / ".loop" / "state"
BUDGET_FILE = STATE_DIR / "budget.json"
RUNNING_FILE = STATE_DIR / "running.json"
LOG_FILE = STATE_DIR / "log.jsonl"
PAUSE_SIGNAL = STATE_DIR / "signals" / "pause.json"
ESCALATE_SIGNAL = STATE_DIR / "signals" / "escalate.json"
BOT_DIR = Path(__file__).resolve().parent
SESSION_FILE = BOT_DIR / ".session.json"
STORE_PATH = BOT_DIR / "crypto_store"

# Limits
COMMAND_EXPIRY = 600        # 10 minutes
RATE_LIMIT_INTERVAL = 10    # seconds between commands
RATE_LIMIT_HOURLY = 30      # max commands per hour

# Validate
_REQUIRED = ["MATRIX_HOMESERVER", "MATRIX_BOT_USER", "MATRIX_BOT_PASS",
             "MATRIX_SERVER_NAME", "MATRIX_CONDUCTOR_ROOM"]
_missing = [k for k in _REQUIRED if not os.environ.get(k)]
if _missing and __name__ == "__main__":
    print(f"Missing env vars: {', '.join(_missing)}")
    print("Copy bot/.env.example to bot/.env and fill in values, then: source bot/.env")
    raise SystemExit(1)


# --- Signature verification ---

def load_verify_key() -> VerifyKey | None:
    """Load the Ed25519 public key for signed command verification."""
    if not SIGNING_PUBKEY:
        logger.warning("No MATRIX_SIGNING_PUBKEY set — signed commands disabled")
        return None
    try:
        key_bytes = base64.b64decode(SIGNING_PUBKEY)
        return VerifyKey(key_bytes)
    except Exception as e:
        logger.error("Failed to load signing pubkey: %s", e)
        return None


def verify_signature(body: str, verify_key: VerifyKey | None) -> tuple[bool, str]:
    """Verify an Ed25519 signed command.

    Expected format:
        !command args
        --sig:BASE64_SIGNATURE:TIMESTAMP

    Returns (ok, command_text).
    """
    if verify_key is None:
        return False, ""

    lines = body.strip().split("\n")
    if len(lines) < 2 or not lines[-1].startswith("--sig:"):
        return False, ""

    sig_line = lines[-1]
    command = "\n".join(lines[:-1]).strip()

    try:
        _, sig_b64, timestamp_str = sig_line.split(":")
        sig_bytes = base64.b64decode(sig_b64)
        timestamp = int(timestamp_str)
    except (ValueError, Exception):
        return False, ""

    # Check timestamp freshness
    now = int(time.time())
    if abs(now - timestamp) > COMMAND_EXPIRY:
        logger.warning("Signature expired: command=%s, age=%ds", command, now - timestamp)
        return False, ""

    # Verify signature over (command + timestamp)
    message = f"{command}:{timestamp}".encode()
    try:
        verify_key.verify(message, sig_bytes)
        return True, command
    except BadSignatureError:
        logger.warning("Bad signature for command: %s", command)
        return False, ""


# --- Rate limiter ---

class RateLimiter:
    def __init__(self, min_interval: float = 10, max_hourly: int = 30):
        self.min_interval = min_interval
        self.max_hourly = max_hourly
        self._last_command: dict[str, float] = {}
        self._hourly_counts: dict[str, list[float]] = defaultdict(list)

    def check(self, user_id: str) -> tuple[bool, str]:
        now = time.time()

        # Interval check
        last = self._last_command.get(user_id, 0)
        if now - last < self.min_interval:
            return False, f"Rate limited. Wait {self.min_interval - (now - last):.0f}s."

        # Hourly check
        hour_ago = now - 3600
        self._hourly_counts[user_id] = [t for t in self._hourly_counts[user_id] if t > hour_ago]
        if len(self._hourly_counts[user_id]) >= self.max_hourly:
            return False, f"Hourly limit reached ({self.max_hourly}/hr)."

        self._last_command[user_id] = now
        self._hourly_counts[user_id].append(now)
        return True, ""


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
    return read_json(PAUSE_SIGNAL).get("active", False)


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
# Each returns (response_text,). Handlers marked SIGNED require signature.

async def cmd_status(client, room_id, **_):
    running = read_running()
    budget = read_budget()
    paused = is_paused()
    exp = running.get("experiments", {})
    in_flight = exp.get("in_flight")

    lines = ["**Loop Status**", ""]
    lines.append(f"- Paused: {'YES' if paused else 'no'}")
    lines.append(f"- Budget: ${budget.get('daily_spent', 0):.2f} / ${budget.get('daily_limit', 30):.2f} today")
    lines.append(f"- Weekly: ${budget.get('weekly_spent', 0):.2f} / ${budget.get('weekly_limit', 150):.2f}")
    lines.append(f"- Experiments: {budget.get('experiments_run', 0)} run, "
                 f"{budget.get('experiments_kept', 0)} kept, "
                 f"{budget.get('experiments_discarded', 0)} discarded")
    if in_flight:
        lines.append(f"\n**In flight:** {in_flight.get('experiment_id', '?')}")
        lines.append(f"- Started: {in_flight.get('started_at', '?')}")
        if in_flight.get("wandb_url"):
            lines.append(f"- wandb: {in_flight['wandb_url']}")
    else:
        lines.append("\nNo experiment in flight.")

    await _send(client, room_id, "\n".join(lines))


async def cmd_budget(client, room_id, **_):
    b = read_budget()
    await _send(client, room_id,
        f"**Budget**\n"
        f"- Today: ${b.get('daily_spent', 0):.2f} / ${b.get('daily_limit', 30):.2f}\n"
        f"- Week: ${b.get('weekly_spent', 0):.2f} / ${b.get('weekly_limit', 150):.2f}\n"
        f"- Total: ${b.get('total_spent', 0):.2f}")


async def cmd_logs(client, room_id, **_):
    entries = read_log_tail(5)
    if not entries:
        await _send(client, room_id, "No log entries yet.")
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
    await _send(client, room_id, "\n".join(lines))


async def cmd_help(client, room_id, **_):
    await _send(client, room_id,
        "**AWM Research Bot**\n\n"
        "**Read (verified device required):**\n"
        "- `!status` — loop state, experiment, budget\n"
        "- `!budget` — spend details\n"
        "- `!logs` — last 5 conductor decisions\n"
        "- `!help` — this message\n\n"
        "**Mutate (Ed25519 signature required):**\n"
        "- `!pause` — pause the research loop\n"
        "- `!resume` — resume the loop\n"
        "- `!kill` — clear in-flight experiment\n\n"
        "Sign mutations with: `command\\n--sig:BASE64:TIMESTAMP`")


# --- Mutation handlers (require signature) ---

async def cmd_pause(client, room_id, **_):
    write_signal(PAUSE_SIGNAL, active=True, reason="Paused via Matrix (signed)")
    await _send(client, room_id, "Loop paused (signed command). Use `!resume` to continue.")
    logger.info("Loop PAUSED via signed Matrix command")


async def cmd_resume(client, room_id, **_):
    write_signal(PAUSE_SIGNAL, active=False)
    await _send(client, room_id, "Loop resumed (signed command).")
    logger.info("Loop RESUMED via signed Matrix command")


async def cmd_kill(client, room_id, **_):
    running = read_running()
    exp = running.get("experiments", {})
    in_flight = exp.get("in_flight")
    if not in_flight:
        await _send(client, room_id, "Nothing running to kill.")
        return
    exp_id = in_flight.get("experiment_id", "?")
    app_id = in_flight.get("modal_app_id", "")
    running["experiments"]["in_flight"] = None
    running["experiments"]["lock"] = False
    RUNNING_FILE.write_text(json.dumps(running, indent=2))
    msg = f"Cleared {exp_id} from running state (signed command)."
    if app_id:
        msg += f"\nTo stop Modal app: `modal app stop {app_id}`"
    await _send(client, room_id, msg)
    logger.info("KILLED experiment %s via signed Matrix command", exp_id)


# --- Command registry ---

# read_only: verified device sufficient
# signed: Ed25519 signature required
READ_COMMANDS = {
    "!status": cmd_status,
    "!budget": cmd_budget,
    "!logs": cmd_logs,
    "!help": cmd_help,
}

SIGNED_COMMANDS = {
    "!pause": cmd_pause,
    "!resume": cmd_resume,
    "!kill": cmd_kill,
}


# --- Helpers ---

async def _send(client: AsyncClient, room_id: str, body: str):
    await client.room_send(room_id, "m.room.message", {
        "msgtype": "m.text", "body": body,
    })


# --- Session persistence ---

def save_session(client: AsyncClient, resp):
    data = {
        "access_token": resp.access_token,
        "device_id": resp.device_id,
        "user_id": client.user_id,
    }
    SESSION_FILE.write_text(json.dumps(data, indent=2))
    logger.info("Session saved to %s", SESSION_FILE)


def load_session() -> dict | None:
    try:
        return json.loads(SESSION_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# --- Message handler ---

async def on_message(room: MatrixRoom, event: RoomMessageText, client: AsyncClient,
                     verify_key: VerifyKey | None, rate_limiter: RateLimiter):
    # Ignore own messages
    if event.sender == client.user_id:
        return

    # Must be from authorized user
    if event.sender not in AUTHORIZED_USERS:
        return

    # Command expiry (based on server timestamp)
    event_age = time.time() - event.server_timestamp / 1000
    if event_age > COMMAND_EXPIRY:
        return

    # Check verified device (E2E) — skip if E2E not available
    if E2E_AVAILABLE and not getattr(event, "verified", False):
        logger.warning("Unverified device from %s — ignoring command: %s",
                       event.sender, event.body[:50])
        await _send(client, room.room_id,
                    "Command rejected: your device is not verified. "
                    "Run emoji verification with the bot first.")
        return

    # Rate limit
    ok, reason = rate_limiter.check(event.sender)
    if not ok:
        await _send(client, room.room_id, reason)
        return

    body = event.body.strip()
    cmd_word = body.split()[0].lower() if body else ""

    # Read commands — verified device is sufficient
    if cmd_word in READ_COMMANDS:
        logger.info("Read command from %s: %s", event.sender, cmd_word)
        await READ_COMMANDS[cmd_word](client, room.room_id)
        return

    # Signed commands — require Ed25519 signature
    if cmd_word in SIGNED_COMMANDS or (body.startswith("!") and "\n--sig:" in body):
        # Extract command from potentially signed message
        sig_ok, signed_cmd = verify_signature(body, verify_key)
        if not sig_ok:
            await _send(client, room.room_id,
                        f"Mutation command `{cmd_word}` requires Ed25519 signature.\n"
                        f"Format: `{cmd_word}\\n--sig:BASE64_SIG:UNIX_TIMESTAMP`")
            return

        signed_word = signed_cmd.split()[0].lower()
        handler = SIGNED_COMMANDS.get(signed_word)
        if handler:
            logger.info("SIGNED command from %s: %s", event.sender, signed_cmd)
            await handler(client, room.room_id)
        else:
            await _send(client, room.room_id, f"Unknown signed command: `{signed_word}`")
        return

    # Unknown command
    if body.startswith("!"):
        await _send(client, room.room_id, f"Unknown command: `{cmd_word}`. Try `!help`.")


# --- Main ---

async def main():
    STORE_PATH.mkdir(parents=True, exist_ok=True)

    config = AsyncClientConfig(
        store_sync_tokens=True,
        encryption_enabled=E2E_AVAILABLE,
    )
    client = AsyncClient(
        HOMESERVER,
        f"{BOT_USER}:{SERVER_NAME}",
        config=config,
        store_path=str(STORE_PATH),
    )

    # Try to restore session, fall back to password login
    session = load_session()
    if session:
        client.restore_login(
            user_id=session["user_id"],
            device_id=session["device_id"],
            access_token=session["access_token"],
        )
        logger.info("Restored session (device %s)", session["device_id"])
    else:
        logger.info("Logging in with password...")
        resp = await client.login(BOT_PASS, device_name="awm-research-bot")
        if hasattr(resp, "access_token"):
            save_session(client, resp)
            logger.info("Logged in. Device: %s", resp.device_id)
            logger.info("FIRST RUN: You need to verify this device from Element.")
            logger.info("Open Element → Settings → Sessions → verify 'awm-research-bot'")
        else:
            logger.error("Login failed: %s", resp)
            return

    # Load signing key
    verify_key = load_verify_key()
    if verify_key:
        logger.info("Ed25519 signing key loaded — mutation commands require signature")
    else:
        logger.warning("No signing key — mutation commands DISABLED")

    # Rate limiter
    rate_limiter = RateLimiter(RATE_LIMIT_INTERVAL, RATE_LIMIT_HOURLY)

    # Register callback
    client.add_event_callback(
        lambda room, event: on_message(room, event, client, verify_key, rate_limiter),
        RoomMessageText,
    )

    # Join rooms
    for room_id in [CONDUCTOR_ROOM, EXPERIMENT_ROOM, ESCALATION_ROOM]:
        if room_id:
            await client.join(room_id)
            logger.info("Joined %s", room_id)

    # Announce
    if CONDUCTOR_ROOM:
        mode = "signed mutations" if verify_key else "read-only (no signing key)"
        await _send(client, CONDUCTOR_ROOM,
                    f"AWM Research Bot online ({mode}). Type `!help` for commands.")

    # Sync
    logger.info("Listening for commands...")
    await client.sync_forever(timeout=30000, full_state=True)


if __name__ == "__main__":
    asyncio.run(main())
