"""Model loading from checkpoint files.

Handles both MLP and Mamba-2 architectures, old and new checkpoint formats.
Infers hyperparameters from weight shapes.

Adapted from nojohns/worldmodel/scripts/generate_demo.py.
"""

import logging
from pathlib import Path

import torch

from models.encoding import EncodingConfig
from models.mlp import FrameStackMLP
from models.mamba2 import FrameStackMamba2

logger = logging.getLogger(__name__)


# --- Stage geometry (legal tournament stages) ---

STAGE_GEOMETRY = {
    32: {
        "name": "Final Destination",
        "ground_y": 0, "ground_x_range": [-85.57, 85.57],
        "platforms": [],
        "blast_zones": {"left": -246, "right": 246, "top": 188, "bottom": -140},
        "camera_bounds": {"left": -160, "right": 160, "top": 100, "bottom": -50},
    },
    31: {
        "name": "Battlefield",
        "ground_y": 0, "ground_x_range": [-68.4, 68.4],
        "platforms": [
            {"x_range": [-57.6, -20], "y": 27.2},
            {"x_range": [20, 57.6], "y": 27.2},
            {"x_range": [-18.8, 18.8], "y": 54.4},
        ],
        "blast_zones": {"left": -224, "right": 224, "top": 200, "bottom": -108.8},
        "camera_bounds": {"left": -150, "right": 150, "top": 100, "bottom": -50},
    },
    3: {
        "name": "Pokemon Stadium",
        "ground_y": 0, "ground_x_range": [-87.75, 87.75],
        "platforms": [
            {"x_range": [-55, -25], "y": 25},
            {"x_range": [25, 55], "y": 25},
        ],
        "blast_zones": {"left": -230, "right": 230, "top": 200, "bottom": -111},
        "camera_bounds": {"left": -160, "right": 160, "top": 100, "bottom": -50},
    },
    8: {
        "name": "Yoshi's Story",
        "ground_y": 0, "ground_x_range": [-56, 56],
        "platforms": [
            {"x_range": [-60, -28], "y": 23.45},
            {"x_range": [28, 60], "y": 23.45},
            {"x_range": [-15.75, 15.75], "y": 42},
        ],
        "blast_zones": {"left": -175.7, "right": 173.6, "top": 168, "bottom": -91},
        "camera_bounds": {"left": -120, "right": 120, "top": 80, "bottom": -40},
    },
    28: {
        "name": "Dream Land N64",
        "ground_y": 0, "ground_x_range": [-77.27, 77.27],
        "platforms": [
            {"x_range": [-61.39, -31.73], "y": 30.14},
            {"x_range": [31.73, 63.03], "y": 30.14},
            {"x_range": [-19.02, 19.02], "y": 51.43},
        ],
        "blast_zones": {"left": -255, "right": 255, "top": 250, "bottom": -123},
        "camera_bounds": {"left": -170, "right": 170, "top": 100, "bottom": -50},
    },
    2: {
        "name": "Fountain of Dreams",
        "ground_y": 0, "ground_x_range": [-63.35, 63.35],
        "platforms": [
            {"x_range": [-50.5, -20.5], "y": 27.2},
            {"x_range": [20.5, 50.5], "y": 27.2},
            {"x_range": [-15, 15], "y": 42.75},
        ],
        "blast_zones": {"left": -198.75, "right": 198.75, "top": 202.5, "bottom": -146.25},
        "camera_bounds": {"left": -140, "right": 140, "top": 90, "bottom": -50},
    },
}


# --- Character name fallback ---

CHARACTER_NAMES = {
    0: "MARIO", 1: "FOX", 2: "CPTFALCON", 3: "DK", 4: "KIRBY",
    5: "BOWSER", 6: "LINK", 7: "SHEIK", 8: "NESS", 9: "PEACH",
    10: "POPO", 12: "PIKACHU", 13: "SAMUS", 14: "YOSHI",
    15: "JIGGLYPUFF", 16: "MEWTWO", 17: "LUIGI", 18: "MARTH",
    19: "ZELDA", 20: "YOUNGLINK", 21: "DOC", 22: "FALCO",
    23: "PICHU", 24: "GAMEANDWATCH", 25: "GANONDORF", 26: "ROY",
}


def resolve_character_name(char_id: int) -> str:
    return CHARACTER_NAMES.get(char_id, f"CHAR_{char_id}")


# --- Model loading ---

def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[torch.nn.Module, EncodingConfig, int, str]:
    """Load a world model from a checkpoint file.

    Handles both old and new checkpoint formats, detects architecture
    (MLP vs Mamba-2) from weight keys, infers hyperparams from weight shapes.

    Returns:
        (model, encoding_config, context_len, arch_name)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # --- Reconstruct EncodingConfig ---
    if "encoding_config" in checkpoint and checkpoint["encoding_config"]:
        cfg = EncodingConfig(**checkpoint["encoding_config"])
        context_len = checkpoint.get("context_len", 10)
    elif "config" in checkpoint and checkpoint["config"]:
        old_cfg = checkpoint["config"]
        context_len = old_cfg.pop("context_len", 10)
        enc_fields = {f.name for f in EncodingConfig.__dataclass_fields__.values()}
        enc_kwargs = {k: v for k, v in old_cfg.items() if k in enc_fields}
        cfg = EncodingConfig(**enc_kwargs)
    else:
        cfg = EncodingConfig()
        context_len = 10

    # --- Detect architecture from weight keys ---
    is_mamba2 = any("layers.0.mamba" in k for k in state_dict)

    if is_mamba2:
        d_model = state_dict["frame_proj.weight"].shape[0]
        layer_indices = set()
        for k in state_dict:
            if k.startswith("layers."):
                layer_indices.add(int(k.split(".")[1]))
        n_layers = len(layer_indices)

        d_inner = 2 * d_model  # expand=2 default
        nheads = state_dict["layers.0.mamba.A_log"].shape[0]
        headdim = d_inner // nheads
        conv_dim = state_dict["layers.0.mamba.conv1d.weight"].shape[0]
        d_state = (conv_dim - d_inner) // 2

        model = FrameStackMamba2(
            cfg=cfg,
            context_len=context_len,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            headdim=headdim,
            chunk_size=None,  # sequential for inference
        )
        arch = "mamba2"
        logger.info(
            "Detected Mamba-2: d_model=%d, d_state=%d, n_layers=%d, headdim=%d",
            d_model, d_state, n_layers, headdim,
        )
    else:
        hidden_dim = state_dict["trunk.0.weight"].shape[0]
        trunk_dim = state_dict["trunk.3.weight"].shape[0]
        model = FrameStackMLP(
            cfg=cfg,
            context_len=context_len,
            hidden_dim=hidden_dim,
            trunk_dim=trunk_dim,
        )
        arch = "mlp"
        logger.info(
            "Detected MLP: hidden_dim=%d, trunk_dim=%d",
            hidden_dim, trunk_dim,
        )

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", -1)
    logger.info("Loaded checkpoint: epoch %d, context_len=%d", epoch, context_len)

    return model, cfg, context_len, arch
