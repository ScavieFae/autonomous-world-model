"""Self-contained model code for inference.

Copied from nojohns/worldmodel/model/ — adapted imports to be standalone.
Training code stays in nojohns; this repo only needs inference.
"""

from models.encoding import EncodingConfig
from models.mamba2 import FrameStackMamba2
from models.mlp import FrameStackMLP
from models.policy_mlp import PolicyMLP
from models.checkpoint import load_model_from_checkpoint, STAGE_GEOMETRY
