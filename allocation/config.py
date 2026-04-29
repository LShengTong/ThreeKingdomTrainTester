from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class NetworkConfig:
    hero_phi_hidden: tuple[int, ...] = (64, )
    hero_phi_out: int = 64
    hero_rho_hidden: tuple[int, ...] = ()
    hero_rho_out:int = 16
    # work_phi_hidden: tuple[int, ...] = ()
    # work_phi_out: int = 4
    # work_rho_hidden: tuple[int, ...] = ()
    fusion_hidden_dims: tuple[int, ...] = (128, 128)
    activation: str = "relu"


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 50000
    learning_starts: int = 10000
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1000
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    exploration_fraction: float = 0.1
    device: str = "auto"
    total_timesteps: int = 100000
    tensorboard_log: str = "./logs/"


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    lr: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "auto"
    total_timesteps: int = 200000
    tensorboard_log: str = "./logs/"


@dataclass(frozen=True)
class EnvironmentConfig:
    max_heroes: int = 10


@dataclass(frozen=True)
class AllocationAgentConfig:
    algorithm: Literal["dqn", "ppo"] = "ppo"
    network: NetworkConfig = field(default_factory=NetworkConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)


DEFAULT_AGENT_CONFIG = AllocationAgentConfig()
