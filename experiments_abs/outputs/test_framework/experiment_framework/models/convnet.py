from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


def build_model(
    *, input_shape: tuple[int, ...], num_classes: int,
    channels: tuple[int, ...] = (32, 64),
) -> nn.Module:
    """Вернуть CNN с logits [batch, num_classes]; seed задаётся снаружи.

    Архитектуру по статье необходимо согласовать перед реализацией.
    """
    raise NotImplementedError("Implement the selected convolutional architecture")
