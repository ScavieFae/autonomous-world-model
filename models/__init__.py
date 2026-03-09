"""Model definitions — architecture, encoding, and checkpoint loading.

Canonical home for all model code. Training pipeline lives in data/, training/, scripts/.
"""

from models.encoding import EncodingConfig, encode_player_frames, StateEncoder
from models.mamba2 import FrameStackMamba2
from models.mlp import FrameStackMLP
from models.policy_mlp import PolicyMLP
from models.checkpoint import load_model_from_checkpoint, STAGE_GEOMETRY
