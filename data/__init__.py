"""Data pipeline: .slp parsing → PyTorch datasets."""

from data.parse import load_game, load_games_from_dir
from data.dataset import MeleeDataset, MeleeFrameDataset

__all__ = ["load_game", "load_games_from_dir", "MeleeDataset", "MeleeFrameDataset"]
