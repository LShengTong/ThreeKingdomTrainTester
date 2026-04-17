from typing import Iterable

import torch
from torch import nn

from configurable_mlp import ConfigurableMLP
from deep_set_encoder import DeepSetEncoder
from observation import Observations


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
        output_dim: int,
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
        self.working_deepset = DeepSetEncoder(
            element_dim=hero_feature_dim + develop_feature_dim,
            phi_hidden_dims=work_phi_hidden,
            rho_hidden_dims=work_rho_hidden,
            phi_out_dim=work_phi_out,
            out_dim=hero_feature_dim + develop_feature_dim,
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
        self.fusion = ConfigurableMLP(
            input_dim=hero_feature_dim +
                      hero_feature_dim +
                      num_develops * develop_feature_dim +
                      hero_feature_dim + develop_feature_dim,
            hidden_dims=fusion_hidden_dims,
            output_dim=output_dim,
            activation=activation,
        )

    def forward(
        self,
        obs: Observations,
    ) -> torch.Tensor:
        working_count = (obs.working_assignments != -1).sum(dim=1)

        batch_size, num_heroes = obs.hero_mask.shape
        pos = torch.arange(num_heroes, device=obs.hero_mask.device).unsqueeze(0)
        cut = (working_count + 1).unsqueeze(1)
        hero_masks = obs.hero_mask * (pos >= cut).to(obs.hero_mask.dtype)
        hero_vec = self.hero_deepset(obs.heroes, hero_masks)

        # first_hero_vec = self.first_hero_mlp(heroes[:, 0, :])
        batch_idx = torch.arange(batch_size, device=obs.heroes.device)
        first_hero_vec = obs.heroes[batch_idx, working_count, :]

        batch_size, num_develops, develop_dim = obs.develops.shape
        # develop_feat = self.develop_feature_mlp(develops.reshape(-1, develop_dim))
        # develop_feat = develop_feat.reshape(batch_size, -1)
        # develop_vec = self.develop_rho(develop_feat)
        develop_vec = obs.develops.reshape(batch_size, -1)

        working_mask = (obs.working_assignments >= 0).to(obs.hero_mask.dtype)
        safe_working_assignments = obs.working_assignments.clamp(min=0)
        batch_idx = torch.arange(batch_size, device=obs.develops.device).unsqueeze(1).expand(-1, num_heroes)
        assigned_develop_vec = obs.develops[batch_idx, safe_working_assignments, :]
        working_features = torch.cat([obs.heroes, assigned_develop_vec], dim=-1)
        working_vec = self.working_deepset(working_features, working_mask)

        return self.fusion(torch.cat([hero_vec, develop_vec, first_hero_vec, working_vec], dim=-1))
