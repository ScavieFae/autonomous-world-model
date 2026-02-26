"""Core autoregressive match loop.

Runs two agents inside the world model. Each frame:
1. Both agents observe state → produce controller inputs
2. World model predicts next frame given both controllers
3. Predictions clamped to valid ranges, fed back as context
4. Check for KO

Adapted from nojohns/worldmodel/scripts/play_match.py and rollout.py.
"""

import logging

import torch

from models.encoding import EncodingConfig
from models.checkpoint import STAGE_GEOMETRY, resolve_character_name
from crank.agents import Agent

logger = logging.getLogger(__name__)


# Valid game state ranges (in normalized/scaled units)
CLAMP_RANGES = {
    "percent": (0.0, 999.0),
    "x": (-300.0, 300.0),
    "y": (-200.0, 300.0),
    "shield": (0.0, 60.0),
    "stocks": (0.0, 4.0),
    "hitlag": (0.0, 50.0),
    "combo_count": (0.0, 50.0),
}


def clamp_frame(next_float: torch.Tensor, cfg: EncodingConfig) -> torch.Tensor:
    """Clamp predicted frame values to physically valid ranges."""
    fp = cfg.float_per_player

    for player_offset in [0, fp]:
        lo, hi = CLAMP_RANGES["percent"]
        next_float[player_offset + 0].clamp_(lo * cfg.percent_scale, hi * cfg.percent_scale)
        lo, hi = CLAMP_RANGES["x"]
        next_float[player_offset + 1].clamp_(lo * cfg.xy_scale, hi * cfg.xy_scale)
        lo, hi = CLAMP_RANGES["y"]
        next_float[player_offset + 2].clamp_(lo * cfg.xy_scale, hi * cfg.xy_scale)
        lo, hi = CLAMP_RANGES["shield"]
        next_float[player_offset + 3].clamp_(lo * cfg.shield_scale, hi * cfg.shield_scale)

        vel_end = cfg.core_continuous_dim + cfg.velocity_dim
        dyn_start = vel_end + (0 if cfg.state_age_as_embed else 1)
        lo, hi = CLAMP_RANGES["hitlag"]
        next_float[player_offset + dyn_start].clamp_(lo * cfg.hitlag_scale, hi * cfg.hitlag_scale)
        lo, hi = CLAMP_RANGES["stocks"]
        next_float[player_offset + dyn_start + 1].clamp_(lo * cfg.stocks_scale, hi * cfg.stocks_scale)
        lo, hi = CLAMP_RANGES["combo_count"]
        next_float[player_offset + dyn_start + 2].clamp_(lo * cfg.combo_count_scale, hi * cfg.combo_count_scale)

    return next_float


def decode_continuous(normalized: torch.Tensor, cfg: EncodingConfig) -> dict:
    """Decode a player's continuous floats back to game units."""
    result = {
        "percent": normalized[0].item() / cfg.percent_scale,
        "x": normalized[1].item() / cfg.xy_scale,
        "y": normalized[2].item() / cfg.xy_scale,
        "shield_strength": normalized[3].item() / cfg.shield_scale,
        "speed_air_x": normalized[4].item() / cfg.velocity_scale,
        "speed_y": normalized[5].item() / cfg.velocity_scale,
        "speed_ground_x": normalized[6].item() / cfg.velocity_scale,
        "speed_attack_x": normalized[7].item() / cfg.velocity_scale,
        "speed_attack_y": normalized[8].item() / cfg.velocity_scale,
    }
    if not cfg.state_age_as_embed:
        result["state_age"] = normalized[9].item() / cfg.state_age_scale
        result["hitlag"] = normalized[10].item() / cfg.hitlag_scale
        result["stocks"] = normalized[11].item() / cfg.stocks_scale
        result["combo_count"] = normalized[12].item() / cfg.combo_count_scale
    else:
        result["state_age"] = 0  # stored as int embed
        result["hitlag"] = normalized[9].item() / cfg.hitlag_scale
        result["stocks"] = normalized[10].item() / cfg.stocks_scale
        result["combo_count"] = normalized[11].item() / cfg.combo_count_scale
    return result


def decode_frame(float_frame: torch.Tensor, int_frame: torch.Tensor, cfg: EncodingConfig) -> dict:
    """Decode a full frame tensor back to a visualizer-friendly dict."""
    fp = cfg.float_per_player
    cd = cfg.continuous_dim
    ipp = cfg.int_per_player

    p0_cont = decode_continuous(float_frame[0:cd], cfg)
    p0_cont["facing"] = 1.0 if float_frame[cd].item() > 0.5 else 0.0
    p0_cont["on_ground"] = 1.0 if float_frame[cd + 2].item() > 0.5 else 0.0
    p0_cont["action_state"] = int(int_frame[0].item())
    p0_cont["jumps_left"] = int(int_frame[1].item())
    p0_cont["character"] = int(int_frame[2].item())

    p1_cont = decode_continuous(float_frame[fp:fp + cd], cfg)
    p1_cont["facing"] = 1.0 if float_frame[fp + cd].item() > 0.5 else 0.0
    p1_cont["on_ground"] = 1.0 if float_frame[fp + cd + 2].item() > 0.5 else 0.0
    p1_cont["action_state"] = int(int_frame[ipp].item())
    p1_cont["jumps_left"] = int(int_frame[ipp + 1].item())
    p1_cont["character"] = int(int_frame[ipp + 2].item())

    return {
        "players": [p0_cont, p1_cont],
        "stage": int(int_frame[ipp * 2].item()),
    }


def generate_synthetic_seed(
    cfg: EncodingConfig,
    context_len: int,
    stage: int,
    p0_char: int,
    p1_char: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate K frames of synthetic seed data from starting positions.

    Uses the same starting positions as session-lifecycle JOIN:
    P0 at x=-30, P1 at x=+30, on ground, facing each other.
    """
    K = context_len
    fp = cfg.float_per_player
    ipp = cfg.int_per_player

    float_data = torch.zeros(K, fp * 2)
    int_data = torch.zeros(K, cfg.int_per_frame, dtype=torch.long)

    for t in range(K):
        # P0: left side
        float_data[t, 0] = 0.0  # percent = 0
        float_data[t, 1] = -30.0 * cfg.xy_scale  # x = -30
        float_data[t, 2] = 0.0  # y = 0 (on ground)
        float_data[t, 3] = 60.0 * cfg.shield_scale  # full shield
        # velocities 4-8 = 0
        # state_age / hitlag / stocks / combo
        vel_end = cfg.core_continuous_dim + cfg.velocity_dim
        if not cfg.state_age_as_embed:
            float_data[t, vel_end] = 0.0  # state_age
            float_data[t, vel_end + 1] = 0.0  # hitlag
            float_data[t, vel_end + 2] = 4.0 * cfg.stocks_scale  # 4 stocks
            float_data[t, vel_end + 3] = 0.0  # combo_count
        else:
            float_data[t, vel_end] = 0.0  # hitlag
            float_data[t, vel_end + 1] = 4.0 * cfg.stocks_scale  # 4 stocks
            float_data[t, vel_end + 2] = 0.0  # combo_count
        cd = cfg.continuous_dim
        float_data[t, cd] = 1.0  # facing right
        float_data[t, cd + 1] = 0.0  # not invulnerable
        float_data[t, cd + 2] = 1.0  # on ground
        # Controller: neutral sticks (0.5), no buttons
        ctrl_start = cd + cfg.binary_dim
        float_data[t, ctrl_start] = 0.5  # main x
        float_data[t, ctrl_start + 1] = 0.5  # main y
        float_data[t, ctrl_start + 2] = 0.5  # c x
        float_data[t, ctrl_start + 3] = 0.5  # c y

        # P1: right side
        float_data[t, fp + 0] = 0.0  # percent
        float_data[t, fp + 1] = 30.0 * cfg.xy_scale  # x = +30
        float_data[t, fp + 2] = 0.0  # y = 0
        float_data[t, fp + 3] = 60.0 * cfg.shield_scale  # full shield
        if not cfg.state_age_as_embed:
            float_data[t, fp + vel_end] = 0.0
            float_data[t, fp + vel_end + 1] = 0.0
            float_data[t, fp + vel_end + 2] = 4.0 * cfg.stocks_scale
            float_data[t, fp + vel_end + 3] = 0.0
        else:
            float_data[t, fp + vel_end] = 0.0
            float_data[t, fp + vel_end + 1] = 4.0 * cfg.stocks_scale
            float_data[t, fp + vel_end + 2] = 0.0
        float_data[t, fp + cd] = 0.0  # facing left
        float_data[t, fp + cd + 1] = 0.0
        float_data[t, fp + cd + 2] = 1.0  # on ground
        float_data[t, fp + ctrl_start] = 0.5
        float_data[t, fp + ctrl_start + 1] = 0.5
        float_data[t, fp + ctrl_start + 2] = 0.5
        float_data[t, fp + ctrl_start + 3] = 0.5

        # Int data: action=0 (idle), jumps=2, character, stage
        int_data[t, 0] = 0  # p0 action (idle)
        int_data[t, 1] = 2  # p0 jumps_left
        int_data[t, 2] = p0_char  # p0 character
        # l_cancel=0, hurtbox=0, ground=1, last_attack=0
        int_data[t, 3] = 0
        int_data[t, 4] = 0
        int_data[t, 5] = 1  # on ground surface
        int_data[t, 6] = 0

        int_data[t, ipp] = 0  # p1 action
        int_data[t, ipp + 1] = 2  # p1 jumps_left
        int_data[t, ipp + 2] = p1_char  # p1 character
        int_data[t, ipp + 3] = 0
        int_data[t, ipp + 4] = 0
        int_data[t, ipp + 5] = 1
        int_data[t, ipp + 6] = 0

        int_data[t, ipp * 2] = stage  # stage ID

    return float_data, int_data


@torch.no_grad()
def run_match(
    world_model: torch.nn.Module,
    cfg: EncodingConfig,
    p0_agent: Agent,
    p1_agent: Agent,
    stage: int,
    p0_char: int,
    p1_char: int,
    max_frames: int = 600,
    device: str = "cpu",
    no_early_ko: bool = False,
) -> dict:
    """Run a two-agent match inside the world model.

    Seeds with synthetic starting positions, then autoregressively predicts
    frames using both agents' controller inputs.

    Returns a complete match dict compatible with viz/visualizer.html.
    """
    world_model.eval()
    K = world_model.context_len

    # Generate synthetic seed
    sim_floats, sim_ints = generate_synthetic_seed(cfg, K, stage, p0_char, p1_char)

    fp = cfg.float_per_player
    ipp = cfg.int_per_player
    cd = cfg.continuous_dim
    bd = cfg.binary_dim
    ctrl_start = cd + bd
    ctrl_end = ctrl_start + cfg.controller_dim
    vel_start = cfg.core_continuous_dim
    vel_end = vel_start + cfg.velocity_dim
    dyn_start = vel_end + (0 if cfg.state_age_as_embed else 1)
    p0_dyn_idx = [dyn_start, dyn_start + 1, dyn_start + 2]
    p1_dyn_idx = [fp + i for i in p0_dyn_idx]

    if not cfg.state_age_as_embed:
        p0_state_age_idx = vel_end
        p1_state_age_idx = fp + vel_end

    # Collect frames
    frames = []
    for i in range(K):
        frames.append(decode_frame(sim_floats[i], sim_ints[i], cfg))

    for t in range(K, K + max_frames):
        ctx_f = sim_floats[-K:]
        ctx_i = sim_ints[-K:]

        # Get controller inputs from both agents
        p0_ctrl = p0_agent.get_controller(ctx_f, ctx_i, cfg, t)
        p1_ctrl = p1_agent.get_controller(ctx_f, ctx_i, cfg, t)

        # Combine into world model conditioning
        next_ctrl = torch.cat([p0_ctrl, p1_ctrl]).unsqueeze(0).to(device)

        # Run world model
        preds = world_model(
            ctx_f.unsqueeze(0).to(device),
            ctx_i.unsqueeze(0).to(device),
            next_ctrl,
        )

        # Build next frame from predictions
        curr_float = sim_floats[-1].clone()
        next_float = curr_float.clone()

        # Continuous deltas
        delta = preds["continuous_delta"][0].cpu()
        next_float[0:4] += delta[0:4]
        next_float[fp:fp + 4] += delta[4:8]

        # Velocity deltas
        if "velocity_delta" in preds:
            vel_d = preds["velocity_delta"][0].cpu()
            next_float[vel_start:vel_end] += vel_d[0:5]
            next_float[fp + vel_start:fp + vel_end] += vel_d[5:10]

        # Dynamics (absolute: hitlag, stocks, combo)
        if "dynamics_pred" in preds:
            dyn = preds["dynamics_pred"][0].cpu()
            for i, idx in enumerate(p0_dyn_idx):
                next_float[idx] = dyn[i]
            for i, idx in enumerate(p1_dyn_idx):
                next_float[idx] = dyn[3 + i]

        # Binary predictions
        binary = (preds["binary_logits"][0].cpu() > 0).float()
        next_float[cd:cd + bd] = binary[0:3]
        next_float[fp + cd:fp + cd + bd] = binary[3:6]

        # Store controller inputs in the frame
        next_float[ctrl_start:ctrl_end] = p0_ctrl
        next_float[fp + ctrl_start:fp + ctrl_end] = p1_ctrl

        # Categorical predictions
        next_int = sim_ints[-1].clone()
        next_int[0] = preds["p0_action_logits"][0].cpu().argmax()
        next_int[1] = preds["p0_jumps_logits"][0].cpu().argmax()
        next_int[ipp] = preds["p1_action_logits"][0].cpu().argmax()
        next_int[ipp + 1] = preds["p1_jumps_logits"][0].cpu().argmax()

        if "p0_l_cancel_logits" in preds:
            next_int[3] = preds["p0_l_cancel_logits"][0].cpu().argmax()
            next_int[4] = preds["p0_hurtbox_logits"][0].cpu().argmax()
            next_int[5] = preds["p0_ground_logits"][0].cpu().argmax()
            next_int[6] = preds["p0_last_attack_logits"][0].cpu().argmax()
            next_int[ipp + 3] = preds["p1_l_cancel_logits"][0].cpu().argmax()
            next_int[ipp + 4] = preds["p1_hurtbox_logits"][0].cpu().argmax()
            next_int[ipp + 5] = preds["p1_ground_logits"][0].cpu().argmax()
            next_int[ipp + 6] = preds["p1_last_attack_logits"][0].cpu().argmax()

        # State_age: rules-based
        if cfg.state_age_as_embed:
            sa_int_idx = cfg.int_per_player - 1
            for sa, act_col in [(sa_int_idx, 0), (ipp + sa_int_idx, ipp)]:
                if next_int[act_col] == sim_ints[-1][act_col]:
                    next_int[sa] = min(int(sim_ints[-1][sa].item()) + 1, cfg.state_age_embed_vocab - 1)
                else:
                    next_int[sa] = 0
        else:
            for sa_idx, act_col in [(p0_state_age_idx, 0), (p1_state_age_idx, ipp)]:
                if next_int[act_col] == sim_ints[-1][act_col]:
                    next_float[sa_idx] += 1.0 * cfg.state_age_scale
                else:
                    next_float[sa_idx] = 0.0

        # Clamp
        next_float = clamp_frame(next_float, cfg)

        # Append
        sim_floats = torch.cat([sim_floats, next_float.unsqueeze(0)], dim=0)
        sim_ints = torch.cat([sim_ints, next_int.unsqueeze(0)], dim=0)

        frame = decode_frame(next_float, next_int, cfg)
        frames.append(frame)

        # KO check
        if not no_early_ko:
            p0_stocks = frame["players"][0]["stocks"]
            p1_stocks = frame["players"][1]["stocks"]
            if p0_stocks < 0.5 or p1_stocks < 0.5:
                logger.info("KO detected at frame %d", t)
                break

    # Build output
    stage_geo = STAGE_GEOMETRY.get(stage, {
        "name": f"Stage {stage}",
        "ground_y": 0, "ground_x_range": [-85, 85],
        "platforms": [],
        "blast_zones": {"left": -240, "right": 240, "top": 200, "bottom": -140},
        "camera_bounds": {"left": -160, "right": 160, "top": 100, "bottom": -50},
    })

    return {
        "meta": {
            "mode": "agent-vs-agent",
            "total_frames": len(frames),
            "seed_frames": K,
            "stage": {"id": stage, "name": stage_geo["name"]},
            "characters": {
                "p0": {"id": p0_char, "name": resolve_character_name(p0_char)},
                "p1": {"id": p1_char, "name": resolve_character_name(p1_char)},
            },
        },
        "stage_geometry": stage_geo,
        "frames": frames,
    }
