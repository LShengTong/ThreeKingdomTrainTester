from dataclasses import dataclass

import torch

from allocation.net_observation import NetObservations, NetObservation


@dataclass
class BatchTransition:
    obs: NetObservations
    action_id: torch.Tensor
    reward: torch.Tensor
    next_obs: NetObservations
    done: torch.Tensor
    real_return: torch.Tensor

@dataclass
class NetTransition:
    obs: NetObservation
    action_id: int
    reward: float
    next_obs: NetObservation
    done: bool
    real_return: float