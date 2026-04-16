from typing import Iterable

import torch
from torch import nn

from configurable_mlp import ConfigurableMLP
from deep_set_encoder import DeepSetEncoder


class HeroDevelopNet(nn.Module):
    def __init__(
        self,
        hero_feature_dim: int,
        hero_mlp_hidden: Iterable[int],
        hero_mlp_out: int,
        hero_phi_hidden: Iterable[int],
        hero_phi_out: int,
        hero_out_dim: int,

        first_hero_hidden: Iterable[int],

        develop_count: int,
        develop_feature_dim: int,
        num_develops: int,
        develop_mlp_hidden: Iterable[int],
        develop_mlp_out: int,
        develop_out_dim: int,

        fusion_hidden_dims: Iterable[int],
        output_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.hero_feature_mlp = ConfigurableMLP(
            input_dim=hero_feature_dim,
            hidden_dims=hero_mlp_hidden,
            output_dim=hero_mlp_out,
            activation=activation,
        )
        self.hero_deepset = DeepSetEncoder(
            element_dim=hero_mlp_out,
            phi_hidden_dims=hero_phi_hidden,
            phi_out_dim=hero_phi_out,
            out_dim=hero_out_dim,
            activation=activation,
        )
        self.first_hero_mlp = ConfigurableMLP(
            input_dim=hero_feature_dim,
            hidden_dims=first_hero_hidden,
            output_dim=hero_out_dim,
            activation=activation,
        )
        self.develop_feature_mlp = ConfigurableMLP(
            input_dim=develop_feature_dim,
            hidden_dims=develop_mlp_hidden,
            output_dim=develop_mlp_out,
            activation=activation,
        )
        self.develop_rho = ConfigurableMLP(
            input_dim=num_develops * develop_mlp_out,
            hidden_dims=[],
            output_dim=develop_out_dim,
            activation=activation,
        )
        self.fusion = ConfigurableMLP(
            input_dim=hero_out_dim + hero_out_dim + develop_count * develop_out_dim,
            hidden_dims=fusion_hidden_dims,
            output_dim=output_dim,
            activation=activation,
        )

    def forward(
        self,
        heroes: torch.Tensor,
        hero_mask: torch.Tensor,
        develops: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_heroes, hero_dim = heroes.shape
        other_heroes = heroes[:, 1:, :]
        hero_feat = self.hero_feature_mlp(other_heroes.reshape(-1, hero_dim))
        hero_feat = hero_feat.reshape(batch_size, num_heroes - 1, -1)
        hero_vec = self.hero_deepset(hero_feat, hero_mask[:, 1:])

        first_hero_vec = self.first_hero_mlp(heroes[:, 0, :])

        batch_size, num_develops, develop_dim = develops.shape
        develop_feat = self.develop_feature_mlp(develops.reshape(-1, develop_dim))
        develop_feat = develop_feat.reshape(batch_size, -1)
        develop_vec = self.develop_rho(develop_feat)

        return self.fusion(torch.cat([hero_vec, develop_vec, first_hero_vec], dim=-1))
