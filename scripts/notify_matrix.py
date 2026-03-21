#!/usr/bin/env python3
"""Post messages to Matrix rooms for the autoresearch loop.

Usage:
    python scripts/notify_matrix.py --room conductor-log "Heartbeat: e018d running (30%)"
    python scripts/notify_matrix.py --room experiment-results --html "<b>E018a KEPT</b>: RC 6.26"
    python scripts/notify_matrix.py --room escalations "Budget limit reached — pausing"

Rooms: conductor-log, research, experiment-results, escalations, worker-updates
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

HOMESERVER = "http://100.93.8.111:8008"
SERVER_NAME = "scaviefae.matrix"

# Bot credentials — conductor bot
BOT_USER = "conductor"
BOT_PASSWORD = "conductor_bot"

# Room name → room ID mapping
ROOMS = {
    "conductor-log": "!UwmcnPRSnLKOhuFtVA:scaviefae.matrix",
    "research": "!AKPUHHdvAZsvrIANcJ:scaviefae.matrix",
    "experiment-results": "!WdnvlgezNiEAYhxbxL:scaviefae.matrix",
    "escalations": "!vzOSznMmGhfnnLYwJC:scaviefae.matrix",
    "worker-updates": "!eIBTJrcmgzagIMcwJq:scaviefae.matrix",
    "awm": "!zltalEmYxGVBwTmuha:scaviefae.matrix",
    "github-activity": "!YcELSmuaWiBUWpvBxr:scaviefae.matrix",
}

_token_cache = None


def _request(url: str, data: dict | None = None, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
    if body and not data:
        req.method = "PUT"
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(), "status": e.code}


def _login() -> str:
    global _token_cache
    if _token_cache:
        return _token_cache
    result = _request(
        f"{HOMESERVER}/_matrix/client/v3/login",
        {"type": "m.login.password", "user": BOT_USER, "password": BOT_PASSWORD},
    )
    if "access_token" not in result:
        raise RuntimeError(f"Matrix login failed: {result}")
    _token_cache = result["access_token"]
    return _token_cache


def send_message(room: str, body: str, html: str | None = None) -> dict:
    """Send a message to a Matrix room.

    Args:
        room: Room name (e.g., "conductor-log") or full room ID.
        body: Plain text message body.
        html: Optional HTML formatted body.

    Returns:
        {"event_id": "..."} on success, {"error": "..."} on failure.
    """
    room_id = ROOMS.get(room, room)
    token = _login()
    txn_id = str(int(time.time() * 1000))

    content = {"msgtype": "m.text", "body": body}
    if html:
        content["format"] = "org.matrix.custom.html"
        content["formatted_body"] = html

    url = f"{HOMESERVER}/_matrix/client/v3/rooms/{room_id}/send/m.room.message/{txn_id}"
    req = urllib.request.Request(
        url,
        data=json.dumps(content).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(), "status": e.code}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post to Matrix room")
    parser.add_argument("message", help="Message text")
    parser.add_argument("--room", default="conductor-log", help="Room name")
    parser.add_argument("--html", default=None, help="HTML formatted body")
    args = parser.parse_args()

    result = send_message(args.room, args.message, args.html)
    if "event_id" in result:
        print(f"Sent to #{args.room}: {result['event_id']}")
    else:
        print(f"Error: {result}", file=sys.stderr)
        sys.exit(1)
