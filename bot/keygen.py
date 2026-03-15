#!/usr/bin/env python3
"""Generate an Ed25519 keypair for signing AWM bot commands.

The private key stays on your phone (or wherever you send commands from).
The public key goes in the bot's MATRIX_SIGNING_PUBKEY env var.

Usage:
    python bot/keygen.py

Outputs:
    bot/.signing_key     — private key (KEEP SECRET, never commit)
    bot/.signing_key.pub — public key (set as MATRIX_SIGNING_PUBKEY)

The private key file can be copied to your phone for use with a
signing helper (bot/sign_command.py or a mobile shortcut).
"""

import base64
from pathlib import Path

try:
    from nacl.signing import SigningKey
except ImportError:
    print("Install: pip install pynacl")
    raise

BOT_DIR = Path(__file__).resolve().parent
PRIVKEY_PATH = BOT_DIR / ".signing_key"
PUBKEY_PATH = BOT_DIR / ".signing_key.pub"


def main():
    if PRIVKEY_PATH.exists():
        print(f"Key already exists at {PRIVKEY_PATH}")
        print(f"Public key: {PUBKEY_PATH.read_text().strip()}")
        print("Delete .signing_key to regenerate.")
        return

    key = SigningKey.generate()
    pubkey = key.verify_key

    privkey_b64 = base64.b64encode(bytes(key)).decode()
    pubkey_b64 = base64.b64encode(bytes(pubkey)).decode()

    PRIVKEY_PATH.write_text(privkey_b64 + "\n")
    PRIVKEY_PATH.chmod(0o600)
    PUBKEY_PATH.write_text(pubkey_b64 + "\n")

    print(f"Generated Ed25519 keypair:")
    print(f"  Private: {PRIVKEY_PATH} (chmod 600)")
    print(f"  Public:  {PUBKEY_PATH}")
    print()
    print(f"Add to bot/.env:")
    print(f'  export MATRIX_SIGNING_PUBKEY="{pubkey_b64}"')
    print()
    print(f"Copy the private key to your phone for signing commands.")
    print(f"Use bot/sign_command.py to sign from a terminal,")
    print(f"or build an iOS Shortcut that signs and sends to Matrix.")


if __name__ == "__main__":
    main()
