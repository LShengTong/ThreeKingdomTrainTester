from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from flyer.config import NetConfig
from flyer.flyer_net import FlyerNet


class FlyerFeatureExtractor(BaseFeaturesExtractor):
    """Single ``Box`` observation → features via full ``FlyerNet`` (embed + dual action heads)."""

    def __init__(
        self,
        observation_space: spaces.Box,
    ) -> None:
        super().__init__(observation_space, features_dim=NetConfig.feature_out_dim)
        self.flyer_net = FlyerNet()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.flyer_net(observations)
