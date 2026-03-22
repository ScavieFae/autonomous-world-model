#!/usr/bin/env python3
"""Hook script: post to Matrix when an agent spins down.

Handles both SubagentStop and TeammateIdle events.
Each agent posts as its own identity — hypothesis agent posts as researcher,
coder posts as worker, director posts as conductor.

Matrix becomes a read-only log of agent-to-agent interaction.

Wired via .claude/settings.local.json hooks.
Fails silently — a notification failure should never block an agent.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    from notify_matrix import send_message
except ImportError:
    sys.exit(0)

ROOM = "worker-updates"
MAX_MSG_LEN = 800

# Map agent types / names to bot identities and display names.
# The bot identity determines which Matrix account posts the message.
# The display name is what shows up in the message prefix.
AGENT_IDENTITY = {
    # Subagent types (from Agent tool)
    "Explore": ("researcher", "researcher"),
    "general-purpose": ("worker", "worker"),
    "Plan": ("researcher", "researcher"),

    # Agent names (from .loop/agents/ or teammate names)
    "hypothesis": ("researcher", "hypothesis agent"),
    "research-director": ("conductor", "director"),
    "director": ("conductor", "director"),
    "coder": ("worker", "coder"),
    "researcher": ("researcher", "researcher"),
    "reviewer": ("researcher", "reviewer"),
    "spec": ("researcher", "spec agent"),

    # Fallback
    "default": ("conductor", "agent"),
}


def _identify(event: dict) -> tuple[str, str]:
    """Return (bot_name, display_name) for an agent event."""
    # Try teammate name first (most specific)
    name = event.get("teammate_name", "")
    if name and name in AGENT_IDENTITY:
        return AGENT_IDENTITY[name]

    # Try agent type
    agent_type = event.get("agent_type", "")
    if agent_type and agent_type in AGENT_IDENTITY:
        return AGENT_IDENTITY[agent_type]

    # Try agent_id for partial matches
    agent_id = event.get("agent_id", "")
    for key in AGENT_IDENTITY:
        if key in agent_id.lower():
            return AGENT_IDENTITY[key]

    return AGENT_IDENTITY["default"]


def _truncate(text: str, limit: int = MAX_MSG_LEN) -> str:
    if len(text) <= limit:
        return text
    return text[:limit - 3] + "..."


def handle_subagent_stop(event: dict):
    bot, display = _identify(event)
    last_msg = event.get("last_assistant_message", "")

    if not last_msg or last_msg.strip() == "":
        return

    summary = _truncate(last_msg)
    body = f"{summary}"
    send_message(ROOM, body, bot=bot)


def handle_teammate_idle(event: dict):
    bot, display = _identify(event)

    # Post the idle notification as the teammate
    body = f"going idle"
    send_message(ROOM, body, bot=bot)


def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.exit(0)
        event = json.loads(raw)
    except (json.JSONDecodeError, IOError):
        sys.exit(0)

    hook = event.get("hook_event_name", "")

    if hook == "SubagentStop":
        handle_subagent_stop(event)
    elif hook == "TeammateIdle":
        handle_teammate_idle(event)

    # Always exit 0 — never block the agent from stopping
    sys.exit(0)


if __name__ == "__main__":
    main()
