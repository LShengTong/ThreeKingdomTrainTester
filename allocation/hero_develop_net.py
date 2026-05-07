import torch
from torch import nn

from allocation.config import NetworkConfig
from allocation.environment import Environment
from allocation.net_observation import NetObservations
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
            phi_out_dim=network_config.hero_phi_out,
            rho_hidden_dims=network_config.hero_rho_hidden,
            out_dim=network_config.hero_rho_out,
            activation=network_config.activation,
        )
        self.num_develops = num_develops
        self.hero_feature_dim = hero_feature_dim

    def forward(
        self,
        obs: NetObservations,
    ) -> torch.Tensor:
        batch = obs.curr_hero.shape[0]
        heroes_pool = self.hero_deepset.forward(obs.todo_heroes, obs.todo_hero_mask)
        return torch.cat([
            heroes_pool.reshape(batch, -1),
            obs.develops.reshape(batch, -1),
            obs.curr_hero.reshape(batch, -1),
            obs.working_heroes.reshape(batch, -1),
        ], dim=-1)
