"""Read/write Solana accounts for the crank loop.

Handles SessionState, HiddenState, InputBuffer, and FrameLog accounts
via Solana RPC. For prototype, uses direct account reads and simple
transaction construction.

This module is only used in Mode B (Solana crank). Mode A (standalone)
does not touch Solana at all.
"""

import asyncio
import logging
import struct
from typing import Optional

from crank.state_convert import (
    ControllerInput,
    PlayerState,
    SessionStateECS,
)

logger = logging.getLogger(__name__)

# Anchor discriminator size
DISCRIMINATOR_SIZE = 8

# Status constants (match Rust)
STATUS_CREATED = 0
STATUS_WAITING_PLAYERS = 1
STATUS_ACTIVE = 2
STATUS_ENDED = 3


def deserialize_player_state(data: bytes, offset: int) -> tuple[PlayerState, int]:
    """Deserialize a PlayerState from raw account bytes.

    Field order matches the Rust struct (AnchorSerialize):
    x(i32), y(i32), percent(u16), shield_strength(u16),
    speed_air_x(i16), speed_y(i16), speed_ground_x(i16),
    speed_attack_x(i16), speed_attack_y(i16),
    state_age(u16), hitlag(u8), stocks(u8),
    facing(u8), on_ground(u8), action_state(u16),
    jumps_left(u8), character(u8)
    """
    p = PlayerState()
    p.x = struct.unpack_from("<i", data, offset)[0]; offset += 4
    p.y = struct.unpack_from("<i", data, offset)[0]; offset += 4
    p.percent = struct.unpack_from("<H", data, offset)[0]; offset += 2
    p.shield_strength = struct.unpack_from("<H", data, offset)[0]; offset += 2
    p.speed_air_x = struct.unpack_from("<h", data, offset)[0]; offset += 2
    p.speed_y = struct.unpack_from("<h", data, offset)[0]; offset += 2
    p.speed_ground_x = struct.unpack_from("<h", data, offset)[0]; offset += 2
    p.speed_attack_x = struct.unpack_from("<h", data, offset)[0]; offset += 2
    p.speed_attack_y = struct.unpack_from("<h", data, offset)[0]; offset += 2
    p.state_age = struct.unpack_from("<H", data, offset)[0]; offset += 2
    p.hitlag = struct.unpack_from("<B", data, offset)[0]; offset += 1
    p.stocks = struct.unpack_from("<B", data, offset)[0]; offset += 1
    p.facing = struct.unpack_from("<B", data, offset)[0]; offset += 1
    p.on_ground = struct.unpack_from("<B", data, offset)[0]; offset += 1
    p.action_state = struct.unpack_from("<H", data, offset)[0]; offset += 2
    p.jumps_left = struct.unpack_from("<B", data, offset)[0]; offset += 1
    p.character = struct.unpack_from("<B", data, offset)[0]; offset += 1
    return p, offset


def serialize_player_state(p: PlayerState) -> bytes:
    """Serialize a PlayerState to raw bytes (matching Rust layout)."""
    return struct.pack(
        "<iiHHhhhhhHBBBBHBB",
        p.x, p.y, p.percent, p.shield_strength,
        p.speed_air_x, p.speed_y, p.speed_ground_x,
        p.speed_attack_x, p.speed_attack_y,
        p.state_age, p.hitlag, p.stocks,
        p.facing, p.on_ground, p.action_state,
        p.jumps_left, p.character,
    )


# PlayerState packed size: i32+i32+u16+u16+i16*5+u16+u8*2+u8*2+u16+u8*2 = 32 bytes
PLAYER_STATE_SIZE = 4 + 4 + 2 + 2 + 2*5 + 2 + 1*2 + 1*2 + 2 + 1*2  # = 32 bytes


def deserialize_session_state(data: bytes) -> SessionStateECS:
    """Deserialize a SessionState account from raw bytes.

    Layout (after 8-byte Anchor discriminator):
    status(u8), frame(u32), max_frames(u32),
    player1(Pubkey/32), player2(Pubkey/32),
    stage(u8), players[2](PlayerState×2),
    model(Pubkey/32), created_at(i64), last_update(i64), seed(u64)
    """
    offset = DISCRIMINATOR_SIZE

    status = struct.unpack_from("<B", data, offset)[0]; offset += 1
    frame = struct.unpack_from("<I", data, offset)[0]; offset += 4
    max_frames = struct.unpack_from("<I", data, offset)[0]; offset += 4
    offset += 32  # player1 pubkey
    offset += 32  # player2 pubkey
    stage = struct.unpack_from("<B", data, offset)[0]; offset += 1

    p0, offset = deserialize_player_state(data, offset)
    p1, offset = deserialize_player_state(data, offset)

    return SessionStateECS(
        status=status,
        frame=frame,
        max_frames=max_frames,
        stage=stage,
        players=(p0, p1),
    )


def deserialize_input_buffer(data: bytes) -> tuple[int, ControllerInput, ControllerInput, bool, bool]:
    """Deserialize InputBuffer account.

    Returns: (frame, p1_input, p2_input, p1_ready, p2_ready)
    """
    offset = DISCRIMINATOR_SIZE

    frame = struct.unpack_from("<I", data, offset)[0]; offset += 4

    # Player 1 controller
    c1 = ControllerInput()
    c1.stick_x = struct.unpack_from("<b", data, offset)[0]; offset += 1
    c1.stick_y = struct.unpack_from("<b", data, offset)[0]; offset += 1
    c1.c_stick_x = struct.unpack_from("<b", data, offset)[0]; offset += 1
    c1.c_stick_y = struct.unpack_from("<b", data, offset)[0]; offset += 1
    c1.trigger_l = struct.unpack_from("<B", data, offset)[0]; offset += 1
    c1.trigger_r = struct.unpack_from("<B", data, offset)[0]; offset += 1
    c1.buttons = struct.unpack_from("<B", data, offset)[0]; offset += 1
    c1.buttons_ext = struct.unpack_from("<B", data, offset)[0]; offset += 1

    # Player 2 controller
    c2 = ControllerInput()
    c2.stick_x = struct.unpack_from("<b", data, offset)[0]; offset += 1
    c2.stick_y = struct.unpack_from("<b", data, offset)[0]; offset += 1
    c2.c_stick_x = struct.unpack_from("<b", data, offset)[0]; offset += 1
    c2.c_stick_y = struct.unpack_from("<b", data, offset)[0]; offset += 1
    c2.trigger_l = struct.unpack_from("<B", data, offset)[0]; offset += 1
    c2.trigger_r = struct.unpack_from("<B", data, offset)[0]; offset += 1
    c2.buttons = struct.unpack_from("<B", data, offset)[0]; offset += 1
    c2.buttons_ext = struct.unpack_from("<B", data, offset)[0]; offset += 1

    p1_ready = bool(struct.unpack_from("<B", data, offset)[0]); offset += 1
    p2_ready = bool(struct.unpack_from("<B", data, offset)[0]); offset += 1

    return frame, c1, c2, p1_ready, p2_ready


# --- Async RPC operations (require solana-py or solders) ---

async def read_session_state(rpc_url: str, session_pubkey: str) -> Optional[SessionStateECS]:
    """Read and deserialize a SessionState account from Solana.

    Requires solana-py: `pip install solana`
    """
    try:
        from solana.rpc.async_api import AsyncClient
        from solders.pubkey import Pubkey  # type: ignore

        client = AsyncClient(rpc_url)
        pubkey = Pubkey.from_string(session_pubkey)
        resp = await client.get_account_info(pubkey)

        if resp.value is None:
            logger.warning("Session account not found: %s", session_pubkey)
            return None

        data = bytes(resp.value.data)
        return deserialize_session_state(data)
    except ImportError:
        logger.error("solana-py not installed. Run: pip install solana")
        return None


async def read_input_buffer(rpc_url: str, input_pubkey: str):
    """Read and deserialize an InputBuffer account from Solana."""
    try:
        from solana.rpc.async_api import AsyncClient
        from solders.pubkey import Pubkey  # type: ignore

        client = AsyncClient(rpc_url)
        pubkey = Pubkey.from_string(input_pubkey)
        resp = await client.get_account_info(pubkey)

        if resp.value is None:
            return None

        data = bytes(resp.value.data)
        return deserialize_input_buffer(data)
    except ImportError:
        logger.error("solana-py not installed. Run: pip install solana")
        return None


async def write_session_state(
    rpc_url: str,
    session_pubkey: str,
    session: SessionStateECS,
):
    """Write updated SessionState back to the Solana account.

    In production, this sends a BOLT system transaction (run_inference)
    that updates the SessionState component. For prototype, we'd use
    direct account writes via the ER's permissive mode.
    """
    # TODO: Implement via BOLT system call or direct ER write
    logger.info(
        "Would write session state: frame=%d, p0_stocks=%d, p1_stocks=%d",
        session.frame,
        session.players[0].stocks,
        session.players[1].stocks,
    )
