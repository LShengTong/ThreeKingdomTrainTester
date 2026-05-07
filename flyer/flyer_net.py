import torch
from torch import nn

from common.configurable_mlp import ConfigurableMLP
from flyer.config import NetConfig
from flyer.gym_env import FLYER_SELF_DIM, FLYER_TARGET_DIM

class FlyerNet(nn.Module):
    """Encodes flyer self-features and blends target-conditioned vs idle actions."""

    def __init__(self) -> None:
        super().__init__()

        self.self_info_net = ConfigurableMLP(FLYER_SELF_DIM, NetConfig.self_info_hidden_dims, NetConfig.self_info_feature_dim, "relu")
        self.no_target_net = ConfigurableMLP(NetConfig.self_info_feature_dim, NetConfig.no_target_hidden_dims, NetConfig.feature_out_dim, "relu")
        self.has_target_net = ConfigurableMLP(NetConfig.self_info_feature_dim + FLYER_TARGET_DIM - 1, NetConfig.has_target_hidden_dims, NetConfig.feature_out_dim, "relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self_emb = self.self_info_net(x[:, :FLYER_SELF_DIM])
        no_target = self.no_target_net(self_emb)
        has_target_input = torch.cat((self_emb, x[:, FLYER_SELF_DIM + 1: ]), dim=1)
        has_target = self.has_target_net(has_target_input)
        mask = x[:, FLYER_SELF_DIM: FLYER_SELF_DIM + 1]
        output = has_target * mask + no_target * (1.0 - mask)
        return output
