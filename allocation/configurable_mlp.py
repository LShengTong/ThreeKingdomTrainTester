from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn

from allocation.activation import build_activation

class ConfigurableMLP(nn.Module):
    """A generic MLP with configurable depth and common blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")

        hidden: List[int] = list(hidden_dims or [])
        dims = [input_dim] + hidden + [output_dim]

        layers: List[nn.Module] = []
        for idx in range(len(dims) - 1):
            in_dim, out_dim = dims[idx], dims[idx + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if idx != len(dims) - 2:
                layers.append(build_activation(activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
