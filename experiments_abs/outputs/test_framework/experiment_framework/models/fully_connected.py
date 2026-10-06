from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


def build_model(
    *, input_shape: tuple[int, ...], num_classes: int,
    hidden_sizes: tuple[int, ...] = (256, 128),
) -> nn.Module:
    """Вернуть MLP с logits [batch, num_classes] и flatten входа внутри модели."""
    raise NotImplementedError("Implement the selected fully connected architecture")
