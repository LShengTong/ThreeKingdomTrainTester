from typing import Iterable

import torch
from torch import nn

from common.configurable_mlp import ConfigurableMLP


class DeepSetEncoder(nn.Module):
    """
    DeepSets: ρ( sum_i φ(x_i) ) — permutation invariant over the set dimension.

    Input:  [batch, set_size, element_dim]
    Output: [batch, embed_dim]
    Optional mask: [batch, set_size] with 1 (or True) for valid positions, 0 for padding.
    """

    def __init__(
        self,
        element_dim: int,
        phi_hidden_dims: Iterable[int],
        phi_out_dim: int,
        rho_hidden_dims: Iterable[int],
        out_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.phi = ConfigurableMLP(
            input_dim=element_dim,
            output_dim=phi_out_dim,
            hidden_dims=phi_hidden_dims,
            activation=activation,
        )
        self.rho = ConfigurableMLP(
            input_dim=phi_out_dim,
            output_dim=out_dim,
            hidden_dims=rho_hidden_dims,
            activation=activation,
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected x of shape [batch, set_size, element_dim].")
        phi_x = self.phi(x)
        if mask.shape != x.shape[:2]:
            raise ValueError("mask must be [batch, set_size].")
        m = mask.to(dtype=phi_x.dtype, device=phi_x.device).unsqueeze(-1)
        phi_x = phi_x * m
        pooled = phi_x.sum(dim=1)
        rho = self.rho(pooled)
        return self.relu(rho)