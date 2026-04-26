from typing import Iterable

import torch
from torch import nn

from allocation.net_observation import NetObservations
from configurable_mlp import ConfigurableMLP
from deep_set_encoder import DeepSetEncoder


class HeroDevelopNet(nn.Module):
    def __init__(
        self,
        hero_feature_dim: int,
        hero_phi_hidden: Iterable[int],
        hero_rho_hidden: Iterable[int],
        hero_phi_out: int,

        # hero_out_dim: int,

        work_phi_hidden: Iterable[int],
        work_phi_out: int,
        work_rho_hidden: Iterable[int],

        # first_hero_hidden: Iterable[int],

        # develop_count: int,
        develop_feature_dim: int,
        num_develops: int,
        # develop_mlp_hidden: Iterable[int],
        # develop_mlp_out: int,
        # develop_out_dim: int,

        fusion_hidden_dims: Iterable[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.hero_deepset = DeepSetEncoder(
            element_dim=hero_feature_dim,
            phi_hidden_dims=hero_phi_hidden,
            rho_hidden_dims=hero_rho_hidden,
            phi_out_dim=hero_phi_out,
            out_dim=hero_feature_dim,
            activation=activation,
        )
        # Per develop index a: only hero scalar at channel (a % hero_feature_dim), matching env i % 4.
        self.working_deepset = DeepSetEncoder(
            element_dim=1,
            phi_hidden_dims=work_phi_hidden,
            rho_hidden_dims=work_rho_hidden,
            phi_out_dim=work_phi_out,
            out_dim=1,
            activation=activation,
        )
        # self.first_hero_mlp = ConfigurableMLP(
        #     input_dim=hero_feature_dim,
        #     hidden_dims=first_hero_hidden,
        #     output_dim=hero_out_dim,
        #     activation=activation,
        # )
        # self.develop_feature_mlp = ConfigurableMLP(
        #     input_dim=develop_feature_dim,
        #     hidden_dims=develop_mlp_hidden,
        #     output_dim=develop_mlp_out,
        #     activation=activation,
        # )
        # self.develop_rho = ConfigurableMLP(
        #     input_dim=num_develops * develop_mlp_out,
        #     hidden_dims=[],
        #     output_dim=develop_out_dim,
        #     activation=activation,
        # )
        # hero side: [pool deepset | curr scalar at (d % H) per develop d] + develop features
        # (raw env columns plus [1]-[0] appended in forward) + work deepset
        self.fusion = ConfigurableMLP(
            input_dim=hero_feature_dim +
                      1 +
                      1 +
                      1,
            hidden_dims=fusion_hidden_dims,
            output_dim=1,
            activation=activation,
        )
        self.num_develops = num_develops
        self.hero_feature_dim = hero_feature_dim

    def forward(
        self,
        obs: NetObservations,
    ) -> torch.Tensor:
        shape = (obs.todo_heroes.shape[0], obs.develops.shape[1])
        heroes_pool = self.hero_deepset.forward(obs.todo_heroes, obs.todo_hero_mask)
        heroes_pool = heroes_pool.unsqueeze(1).repeat(1, obs.develops.shape[1], 1)
        working_heroes = obs.working_heroes.reshape(-1, obs.working_heroes.shape[-2], obs.working_heroes.shape[-1])
        working_heroes_mask = obs.working_heroes_mask.reshape(-1, obs.working_heroes_mask.shape[-1])
        working_heroes_pool = self.working_deepset.forward(working_heroes, working_heroes_mask)
        working_heroes_pool = working_heroes_pool.reshape(shape[0], shape[1], -1)
        fusion_in = torch.cat(
            [heroes_pool, obs.curr_hero, obs.develops, working_heroes_pool], dim=-1)

        flat = fusion_in.reshape(shape[0] * shape[1], -1)
        out = self.fusion(flat).view(shape[0], shape[1])
        return out
