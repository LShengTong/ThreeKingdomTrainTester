from dataclasses import dataclass

import torch

from observation import Observation, Observations


@dataclass
class BatchTransition:
    obs: Observations
    action_id: torch.Tensor
    reward: torch.Tensor
    next_obs: Observations
    done: torch.Tensor
    real_return: torch.Tensor

@dataclass
class Transition:
    obs: Observation
    action_id: int
    reward: float
    next_obs: Observation
    done: bool
    real_return: float