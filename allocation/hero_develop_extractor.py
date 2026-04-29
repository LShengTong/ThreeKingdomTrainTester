from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from allocation.environment import Environment
from allocation.hero_develop_net import HeroDevelopNet
from allocation.net_observation import NetObservations


class HeroDevelopExtractor(BaseFeaturesExtractor):
    """Use HeroDevelopNet directly inside DQN policy network."""

    def __init__(
        self,
        observation_space: spaces.Dict
    ) -> None:
        super().__init__(observation_space, features_dim=int(Environment.develop_shape[0]))
        self.hero_develop_net = HeroDevelopNet()

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        net_obs = NetObservations(
            todo_heroes=observations["todo_heroes"],
            todo_hero_mask=observations["todo_hero_mask"],
            develops=observations["develops"],
            working_heroes=observations["working_heroes"],
        )
        return self.hero_develop_net(net_obs)
