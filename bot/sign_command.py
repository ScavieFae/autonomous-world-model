#!/usr/bin/env python3
"""Sign a command for the AWM Matrix bot.

Generates the signed message format that the bot expects.
Copy-paste the output into Element to send a signed command.

Usage:
    python bot/sign_command.py "!pause"
    python bot/sign_command.py "!kill"
    python bot/sign_command.py "!resume"

The output is ready to paste into Matrix:
    !pause
    --sig:BASE64SIGNATURE:1710000000
"""

import base64
import sys
import time
from pathlib import Path

try:
    from nacl.signing import SigningKey
except ImportError:
    print("Install: pip install pynacl")
    raise

BOT_DIR = Path(__file__).resolve().parent
PRIVKEY_PATH = BOT_DIR / ".signing_key"


def sign(command: str, key: SigningKey) -> str:
    timestamp = int(time.time())
    message = f"{command}:{timestamp}".encode()
    sig = key.sign(message).signature
    sig_b64 = base64.b64encode(sig).decode()
    return f"{command}\n--sig:{sig_b64}:{timestamp}"


def main():
    if not PRIVKEY_PATH.exists():
        print(f"No signing key found at {PRIVKEY_PATH}")
        print("Run: python bot/keygen.py")
        raise SystemExit(1)

    if len(sys.argv) < 2:
        print("Usage: python bot/sign_command.py '!pause'")
        raise SystemExit(1)

    privkey_b64 = PRIVKEY_PATH.read_text().strip()
    key = SigningKey(base64.b64decode(privkey_b64))

    command = sys.argv[1]
    signed = sign(command, key)

    print("--- Copy this into Matrix ---")
    print(signed)
    print("--- End ---")


if __name__ == "__main__":
    main()
