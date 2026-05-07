from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class NetConfig:
    self_info_hidden_dims: Tuple[int, ...] = (32,)
    self_info_feature_dim: int = 32
    has_target_hidden_dims: Tuple[int, ...] = (32,)
    no_target_hidden_dims: Tuple[int, ...] = (32,)
    feature_out_dim: int = 32
    arch: Tuple[int, ...] = (64, 64)

@dataclass(frozen=True)
class FlyerPPOConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.3
    device: str = "auto"
    tensorboard_log: str = "./logs/flyer_ppo"
