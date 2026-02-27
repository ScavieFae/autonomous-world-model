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

    Handles config mismatches between world model and policy (e.g. the world
    model has projectiles=True but the policy was trained without them) by
    stripping extra floats at runtime.
    """

    def __init__(
        self,
        checkpoint_path: str,
        cfg: EncodingConfig,
        player: int = 0,
        device: str = "cpu",
    ):
        from dataclasses import fields, asdict

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]

        hidden_dim = state_dict["trunk.0.weight"].shape[0]
        trunk_dim = state_dict["trunk.3.weight"].shape[0]
        input_dim = state_dict["trunk.0.weight"].shape[1]

        # Detect whether the policy has state_age_embed from its weights
        has_sa_embed = "state_age_embed.weight" in state_dict

        # Build a policy-specific cfg by trying flag combinations to match input_dim.
        # The policy may have been trained with different flags than the world model.
        policy_cfg = None
        base_kwargs = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
        for proj in [cfg.projectiles, not cfg.projectiles]:
            for sf in [cfg.state_flags, not cfg.state_flags]:
                for hs in [cfg.hitstun, not cfg.hitstun]:
                    trial = EncodingConfig(
                        **{**base_kwargs,
                           "state_age_as_embed": has_sa_embed,
                           "projectiles": proj,
                           "state_flags": sf,
                           "hitstun": hs},
                    )
                    if input_dim % trial.frame_dim == 0:
                        ctx = input_dim // trial.frame_dim
                        if ctx >= 1 and ctx <= 120:
                            policy_cfg = trial
                            break
                if policy_cfg:
                    break
            if policy_cfg:
                break

        if policy_cfg is None:
            raise ValueError(
                f"Cannot match policy input_dim={input_dim} to any encoding config "
                f"(world model frame_dim={cfg.frame_dim})"
            )

        context_len = input_dim // policy_cfg.frame_dim
        self.policy_cfg = policy_cfg

        # Figure out which float indices to strip when world cfg != policy cfg.
        # We build per-player index masks mapping world model floats → policy floats.
        self._strip_indices = None
        wm_fpp = cfg.float_per_player
        pol_fpp = policy_cfg.float_per_player
        if wm_fpp != pol_fpp:
            # Build the keep-mask: which indices in the world model's per-player
            # float block correspond to the policy's per-player float block.
            # Layout: [continuous | binary | controller]
            # Differences are in continuous (projectiles, hitstun) and binary (state_flags).
            keep = []
            # Continuous: strip projectile floats if policy doesn't have them
            wm_cd = cfg.continuous_dim
            pol_cd = policy_cfg.continuous_dim
            # The projectile floats sit at the end of continuous
            keep_continuous = pol_cd  # take first pol_cd of continuous
            for i in range(keep_continuous):
                keep.append(i)
            # Binary: strip state_flags if policy doesn't have them
            wm_bd = cfg.binary_dim
            pol_bd = policy_cfg.binary_dim
            # Base binary (3) come first, state_flags (40) come after
            for i in range(pol_bd):
                keep.append(wm_cd + i)
            # Controller: always the same
            for i in range(cfg.controller_dim):
                keep.append(wm_cd + wm_bd + i)

            # Build full indices for both players
            indices = []
            for offset in [0, wm_fpp]:
                for k in keep:
                    indices.append(offset + k)
            self._strip_indices = torch.tensor(indices, dtype=torch.long)
            logger.info(
                "Float adapter: world %d/player → policy %d/player (stripping %d)",
                wm_fpp, pol_fpp, wm_fpp - pol_fpp,
            )

        self.model = PolicyMLP(
            cfg=policy_cfg,
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
        ctx_f = float_ctx
        # Strip extra float columns if world model has more features than policy
        if self._strip_indices is not None:
            ctx_f = ctx_f[..., self._strip_indices]
        ctx_f = ctx_f.unsqueeze(0).to(self.device)
        ctx_i = int_ctx.unsqueeze(0).to(self.device)

        preds = self.model(ctx_f, ctx_i, predict_player=self.player)

        analog = preds["analog_pred"][0].cpu()
        # Sample buttons stochastically instead of hard threshold.
        # Sigmoid converts logits to probabilities; Bernoulli samples.
        # This breaks determinism and adds natural behavioral variety.
        button_probs = torch.sigmoid(preds["button_logits"][0].cpu())
        buttons = torch.bernoulli(button_probs)

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
