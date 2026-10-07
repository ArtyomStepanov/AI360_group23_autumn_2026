from .abs_model import AbstractModel
from pathlib import Path

import torch


def load_model(model: AbstractModel, load_path: Path | str) -> AbstractModel:
    """Load parameters and buffers into an existing compatible model."""
    state_dict = torch.load(load_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model


def save_model(model: AbstractModel, save_path: Path | str) -> None:
    """Save parameters and buffers without training or optimizer state."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
