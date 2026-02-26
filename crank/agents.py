"""Agent interfaces for the match runner.

Each agent observes the current game state (context window) and produces
a 13-float controller tensor: [main_x, main_y, c_x, c_y, shoulder, A..D_UP].

Adapted from nojohns/worldmodel/scripts/play_match.py.
"""

import logging

import torch

from models.encoding import EncodingConfig
from models.policy_mlp import PolicyMLP

logger = logging.getLogger(__name__)


class Agent:
    """Base class for agents that produce controller outputs."""

    def get_controller(
        self,
        float_ctx: torch.Tensor,
        int_ctx: torch.Tensor,
        cfg: EncodingConfig,
        t: int,
    ) -> torch.Tensor:
        """Return 13-float controller tensor: [main_x, main_y, c_x, c_y, shoulder, A..D_UP]."""
        raise NotImplementedError


class RandomAgent(Agent):
    """Random controller inputs — stick in random directions, random buttons."""

    def get_controller(self, float_ctx, int_ctx, cfg, t):
        ctrl = torch.zeros(cfg.controller_dim)
        ctrl[0:4] = torch.rand(4)
        ctrl[4] = torch.rand(1).item()
        ctrl[5:13] = (torch.rand(8) > 0.9).float()
        return ctrl


class NoopAgent(Agent):
    """Does nothing — neutral stick, no buttons."""

    def get_controller(self, float_ctx, int_ctx, cfg, t):
        ctrl = torch.zeros(cfg.controller_dim)
        ctrl[0] = 0.5  # neutral main stick x
        ctrl[1] = 0.5  # neutral main stick y
        ctrl[2] = 0.5  # neutral c stick x
        ctrl[3] = 0.5  # neutral c stick y
        return ctrl


class HoldForwardAgent(Agent):
    """Holds forward + A — basic aggressive approach."""

    def get_controller(self, float_ctx, int_ctx, cfg, t):
        ctrl = torch.zeros(cfg.controller_dim)
        ctrl[0] = 1.0  # main stick full right
        ctrl[1] = 0.5  # neutral y
        ctrl[2] = 0.5  # neutral c
        ctrl[3] = 0.5
        if t % 10 < 3:
            ctrl[5] = 1.0  # A button
        return ctrl


class PolicyAgent(Agent):
    """Trained imitation learning policy (Phillip).

    Loads a PolicyMLP checkpoint and runs it to produce controller outputs.
    Uses predict_player for P1 perspective swap (handled inside the model).

    Checkpoint: checkpoints/policy-22k-v2/best.pt
    Pull from Modal: modal volume get melee-training-data /checkpoints/policy-22k-v2/best.pt
    """

    def __init__(
        self,
        checkpoint_path: str,
        cfg: EncodingConfig,
        player: int = 0,
        device: str = "cpu",
    ):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]

        hidden_dim = state_dict["trunk.0.weight"].shape[0]
        trunk_dim = state_dict["trunk.3.weight"].shape[0]
        input_dim = state_dict["trunk.0.weight"].shape[1]

        embed_per_player = (
            cfg.action_embed_dim + cfg.jumps_embed_dim + cfg.character_embed_dim
            + cfg.l_cancel_embed_dim + cfg.hurtbox_embed_dim
            + cfg.ground_embed_dim + cfg.last_attack_embed_dim
        )
        if cfg.state_age_as_embed:
            embed_per_player += cfg.state_age_embed_dim
        frame_dim = cfg.float_per_player * 2 + 2 * embed_per_player + cfg.stage_embed_dim
        context_len = input_dim // frame_dim

        self.model = PolicyMLP(
            cfg=cfg,
            context_len=context_len,
            hidden_dim=hidden_dim,
            trunk_dim=trunk_dim,
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.player = player
        self.cfg = cfg

        logger.info(
            "Loaded policy from %s (player=%d, context=%d, hidden=%d, trunk=%d)",
            checkpoint_path, player, context_len, hidden_dim, trunk_dim,
        )

    @torch.no_grad()
    def get_controller(self, float_ctx, int_ctx, cfg, t):
        """Run policy model on context, return 13-float controller tensor."""
        ctx_f = float_ctx.unsqueeze(0).to(self.device)
        ctx_i = int_ctx.unsqueeze(0).to(self.device)

        preds = self.model(ctx_f, ctx_i, predict_player=self.player)

        analog = preds["analog_pred"][0].cpu()
        buttons = (preds["button_logits"][0].cpu() > 0).float()

        return torch.cat([analog, buttons])


def make_agent(
    spec: str,
    player: int,
    cfg: EncodingConfig,
    device: str = "cpu",
) -> Agent:
    """Create an agent from a specification string.

    Formats:
        "random"           — random controller
        "noop"             — neutral stick, no buttons
        "hold-forward"     — hold right + occasional A
        "policy:<path>"    — trained policy checkpoint
    """
    if spec == "random":
        return RandomAgent()
    elif spec == "noop":
        return NoopAgent()
    elif spec == "hold-forward":
        return HoldForwardAgent()
    elif spec.startswith("policy:"):
        return PolicyAgent(spec[7:], cfg=cfg, player=player, device=device)
    else:
        raise ValueError(
            f"Unknown agent spec: {spec!r}. "
            "Use: random, noop, hold-forward, policy:<path>"
        )
