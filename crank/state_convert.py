"""Bidirectional conversion between model tensors and ECS account formats.

The critical glue between PyTorch world model and Solana BOLT ECS components.

Directions:
  SessionState → model tensor: ECS fixed-point (×256) → normalized floats
  model tensor → SessionState: decode predictions back to ECS fixed-point
  ControllerInput → model tensor: ECS i8 sticks/buttons → [0,1] float range
  HiddenState → model bytes: FP32 prototype stores raw f32 bytes
"""

import struct
from dataclasses import dataclass

import torch

from models.encoding import EncodingConfig


# --- ECS data structures (Python mirrors of Rust components) ---

@dataclass
class PlayerState:
    """Mirrors session_state::PlayerState from Rust."""
    x: int = 0            # i32, fixed-point ×256
    y: int = 0            # i32, fixed-point ×256
    percent: int = 0      # u16
    shield_strength: int = 0  # u16, fixed-point ×256
    speed_air_x: int = 0  # i16, fixed-point ×256
    speed_y: int = 0      # i16, fixed-point ×256
    speed_ground_x: int = 0  # i16, fixed-point ×256
    speed_attack_x: int = 0  # i16, fixed-point ×256
    speed_attack_y: int = 0  # i16, fixed-point ×256
    state_age: int = 0    # u16
    hitlag: int = 0       # u8
    stocks: int = 0       # u8
    facing: int = 0       # u8 (1=right, 0=left)
    on_ground: int = 0    # u8 (1=ground, 0=airborne)
    action_state: int = 0 # u16
    jumps_left: int = 0   # u8
    character: int = 0    # u8


@dataclass
class ControllerInput:
    """Mirrors input_buffer::ControllerInput from Rust."""
    stick_x: int = 0      # i8
    stick_y: int = 0      # i8
    c_stick_x: int = 0    # i8
    c_stick_y: int = 0    # i8
    trigger_l: int = 0    # u8
    trigger_r: int = 0    # u8
    buttons: int = 0      # u8 bitmask
    buttons_ext: int = 0  # u8 bitmask


@dataclass
class SessionStateECS:
    """Mirrors session_state::SessionState (subset needed for conversion)."""
    status: int = 0
    frame: int = 0
    max_frames: int = 0
    stage: int = 0
    players: tuple = (None, None)


# --- SessionState → model tensor ---

def player_state_to_floats(p: PlayerState, cfg: EncodingConfig) -> torch.Tensor:
    """Convert one ECS PlayerState to the model's float tensor format.

    ECS uses fixed-point (×256 for positions/velocities).
    Model uses normalized floats (×cfg.xy_scale etc).
    """
    floats = torch.zeros(cfg.float_per_player)
    cd = cfg.continuous_dim
    bd = cfg.binary_dim

    # Core continuous: percent, x, y, shield
    floats[0] = p.percent * cfg.percent_scale
    floats[1] = (p.x / 256.0) * cfg.xy_scale
    floats[2] = (p.y / 256.0) * cfg.xy_scale
    floats[3] = (p.shield_strength / 256.0) * cfg.shield_scale

    # Velocities
    floats[4] = (p.speed_air_x / 256.0) * cfg.velocity_scale
    floats[5] = (p.speed_y / 256.0) * cfg.velocity_scale
    floats[6] = (p.speed_ground_x / 256.0) * cfg.velocity_scale
    floats[7] = (p.speed_attack_x / 256.0) * cfg.velocity_scale
    floats[8] = (p.speed_attack_y / 256.0) * cfg.velocity_scale

    # Dynamics
    vel_end = cfg.core_continuous_dim + cfg.velocity_dim
    if not cfg.state_age_as_embed:
        floats[vel_end] = p.state_age * cfg.state_age_scale
        floats[vel_end + 1] = p.hitlag * cfg.hitlag_scale
        floats[vel_end + 2] = p.stocks * cfg.stocks_scale
        floats[vel_end + 3] = 0.0  # combo_count (not in ECS)
    else:
        floats[vel_end] = p.hitlag * cfg.hitlag_scale
        floats[vel_end + 1] = p.stocks * cfg.stocks_scale
        floats[vel_end + 2] = 0.0  # combo_count

    # Binary
    floats[cd] = float(p.facing)
    floats[cd + 1] = 0.0  # invulnerable (not tracked in simple ECS)
    floats[cd + 2] = float(p.on_ground)

    # Controller: neutral by default (0.5 for sticks, 0 for buttons)
    ctrl_start = cd + bd
    floats[ctrl_start] = 0.5
    floats[ctrl_start + 1] = 0.5
    floats[ctrl_start + 2] = 0.5
    floats[ctrl_start + 3] = 0.5

    return floats


def player_state_to_ints(p: PlayerState, cfg: EncodingConfig) -> torch.Tensor:
    """Convert one ECS PlayerState to the model's int tensor format."""
    ints = torch.zeros(cfg.int_per_player, dtype=torch.long)
    ints[0] = min(p.action_state, cfg.action_vocab - 1)
    ints[1] = min(p.jumps_left, cfg.jumps_vocab - 1)
    ints[2] = min(p.character, cfg.character_vocab - 1)
    # l_cancel, hurtbox, ground, last_attack default to 0
    if cfg.state_age_as_embed:
        ints[7] = min(p.state_age, cfg.state_age_embed_vocab - 1)
    return ints


def session_to_tensors(
    session: SessionStateECS,
    cfg: EncodingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a full ECS SessionState to model tensor format.

    Returns:
        float_frame: (float_per_player * 2,) — one frame
        int_frame: (int_per_frame,) — one frame
    """
    p0, p1 = session.players
    fp = cfg.float_per_player
    ipp = cfg.int_per_player

    float_frame = torch.zeros(fp * 2)
    float_frame[:fp] = player_state_to_floats(p0, cfg)
    float_frame[fp:2*fp] = player_state_to_floats(p1, cfg)

    int_frame = torch.zeros(cfg.int_per_frame, dtype=torch.long)
    int_frame[:ipp] = player_state_to_ints(p0, cfg)
    int_frame[ipp:2*ipp] = player_state_to_ints(p1, cfg)
    int_frame[ipp * 2] = min(session.stage, cfg.stage_vocab - 1)

    return float_frame, int_frame


# --- model tensor → SessionState ---

def floats_to_player_state(
    floats: torch.Tensor,
    ints: torch.Tensor,
    cfg: EncodingConfig,
) -> PlayerState:
    """Convert model output floats + ints back to ECS PlayerState.

    Inverse of player_state_to_floats/ints. Converts normalized values
    back to ECS fixed-point format.
    """
    cd = cfg.continuous_dim
    vel_end = cfg.core_continuous_dim + cfg.velocity_dim

    p = PlayerState()
    p.x = round((floats[1].item() / cfg.xy_scale) * 256)
    p.y = round((floats[2].item() / cfg.xy_scale) * 256)
    p.percent = max(0, round(floats[0].item() / cfg.percent_scale))
    p.shield_strength = max(0, round((floats[3].item() / cfg.shield_scale) * 256))
    p.speed_air_x = round((floats[4].item() / cfg.velocity_scale) * 256)
    p.speed_y = round((floats[5].item() / cfg.velocity_scale) * 256)
    p.speed_ground_x = round((floats[6].item() / cfg.velocity_scale) * 256)
    p.speed_attack_x = round((floats[7].item() / cfg.velocity_scale) * 256)
    p.speed_attack_y = round((floats[8].item() / cfg.velocity_scale) * 256)

    if not cfg.state_age_as_embed:
        p.state_age = max(0, round(floats[vel_end].item() / cfg.state_age_scale))
        p.hitlag = max(0, round(floats[vel_end + 1].item() / cfg.hitlag_scale))
        p.stocks = max(0, min(4, round(floats[vel_end + 2].item() / cfg.stocks_scale)))
    else:
        p.state_age = int(ints[7].item()) if len(ints) > 7 else 0
        p.hitlag = max(0, round(floats[vel_end].item() / cfg.hitlag_scale))
        p.stocks = max(0, min(4, round(floats[vel_end + 1].item() / cfg.stocks_scale)))

    p.facing = 1 if floats[cd].item() > 0.5 else 0
    p.on_ground = 1 if floats[cd + 2].item() > 0.5 else 0
    p.action_state = int(ints[0].item())
    p.jumps_left = int(ints[1].item())
    p.character = int(ints[2].item())

    return p


def tensors_to_session(
    float_frame: torch.Tensor,
    int_frame: torch.Tensor,
    cfg: EncodingConfig,
    frame_num: int = 0,
) -> SessionStateECS:
    """Convert model output tensors to ECS SessionState."""
    fp = cfg.float_per_player
    ipp = cfg.int_per_player

    p0 = floats_to_player_state(float_frame[:fp], int_frame[:ipp], cfg)
    p1 = floats_to_player_state(float_frame[fp:2*fp], int_frame[ipp:2*ipp], cfg)

    return SessionStateECS(
        status=2,  # STATUS_ACTIVE
        frame=frame_num,
        stage=int(int_frame[ipp * 2].item()),
        players=(p0, p1),
    )


# --- ControllerInput → model tensor ---

def controller_to_tensor(ctrl: ControllerInput, cfg: EncodingConfig) -> torch.Tensor:
    """Convert ECS ControllerInput to the model's 13-float controller format.

    ECS: i8 sticks (-128..127), u8 triggers (0..255), u8 button bitmask
    Model: [0,1] floats for all 13 values
    """
    t = torch.zeros(cfg.controller_dim)

    # Sticks: i8 → [0, 1] (center at 0.5)
    t[0] = (ctrl.stick_x + 128) / 255.0
    t[1] = (ctrl.stick_y + 128) / 255.0
    t[2] = (ctrl.c_stick_x + 128) / 255.0
    t[3] = (ctrl.c_stick_y + 128) / 255.0

    # Shoulder/trigger: max of L and R, normalized
    t[4] = max(ctrl.trigger_l, ctrl.trigger_r) / 255.0

    # Buttons: extract from bitmask
    t[5] = float(bool(ctrl.buttons & 0x01))   # A
    t[6] = float(bool(ctrl.buttons & 0x02))   # B
    t[7] = float(bool(ctrl.buttons & 0x04))   # X
    t[8] = float(bool(ctrl.buttons & 0x08))   # Y
    t[9] = float(bool(ctrl.buttons & 0x10))   # Z
    t[10] = float(bool(ctrl.buttons_ext & 0x04))  # L digital
    t[11] = float(bool(ctrl.buttons_ext & 0x08))  # R digital
    t[12] = float(bool(ctrl.buttons_ext & 0x01))  # D-up

    return t


# --- Hidden state serialization ---

def hidden_state_to_bytes(hidden: torch.Tensor) -> bytes:
    """Serialize Mamba2 hidden state tensor to raw bytes for onchain storage.

    For FP32 prototype, stores as raw little-endian f32 bytes.
    """
    return hidden.contiguous().numpy().tobytes()


def bytes_to_hidden_state(data: bytes, shape: tuple, device: str = "cpu") -> torch.Tensor:
    """Deserialize raw bytes back to Mamba2 hidden state tensor."""
    import numpy as np
    arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr.copy()).to(device)
