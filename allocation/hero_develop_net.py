import torch
from torch import nn

from allocation.config import NetworkConfig
from allocation.environment import Environment
from allocation.net_observation import NetObservations
from allocation.configurable_mlp import ConfigurableMLP
from allocation.deep_set_encoder import DeepSetEncoder


class HeroDevelopNet(nn.Module):
    def __init__(
        self,
        input_network_config: NetworkConfig | None = None,
    ) -> None:
        super().__init__()
        network_config: NetworkConfig
        if input_network_config is None:
            network_config = NetworkConfig()
        else:
            network_config = input_network_config
        hero_feature_dim = int(Environment.hero_dim)
        num_develops = int(Environment.develop_shape[0])
        self.hero_deepset = DeepSetEncoder(
            element_dim=hero_feature_dim,
            phi_hidden_dims=network_config.hero_phi_hidden,
            rho_hidden_dims=network_config.hero_rho_hidden,
            phi_out_dim=network_config.hero_phi_out,
            out_dim=hero_feature_dim,
            activation=network_config.activation,
        )
        # self.working_deepset = DeepSetEncoder(
        #     element_dim=1,
        #     phi_hidden_dims=network_config.work_phi_hidden,
        #     rho_hidden_dims=network_config.work_rho_hidden,
        #     phi_out_dim=network_config.work_phi_out,
        #     out_dim=1,
        #     activation=network_config.activation,
        # )
        self.fusion = ConfigurableMLP(
            input_dim=hero_feature_dim +
                      1 +
                      1 +
                      1,
            hidden_dims=network_config.fusion_hidden_dims,
            output_dim=1,
            activation=network_config.activation,
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
        # working_heroes = obs.working_heroes.reshape(-1, obs.working_heroes.shape[-2], obs.working_heroes.shape[-1])
        # working_heroes_mask = obs.working_heroes_mask.reshape(-1, obs.working_heroes_mask.shape[-1])
        # working_heroes_pool = self.working_deepset.forward(working_heroes, working_heroes_mask)
        # working_heroes_pool = working_heroes_pool.reshape(shape[0], shape[1], -1)
        working_heroes_pool = obs.working_heroes
        fusion_in = torch.cat(
            [heroes_pool, obs.curr_hero, obs.develops, working_heroes_pool], dim=-1)

        flat = fusion_in.reshape(shape[0] * shape[1], -1)
        out = self.fusion(flat).view(shape[0], shape[1])
        return out
