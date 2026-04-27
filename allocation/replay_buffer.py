import random
from collections import deque
from typing import Deque, List

import torch

from allocation.net_observation import NetObservations
from allocation.transition import BatchTransition, NetTransition


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.buffer: Deque[NetTransition] = deque(maxlen=capacity)

    def add(
        self,
        transition: NetTransition,
    ) -> None:
        self.buffer.append(transition)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> BatchTransition:
        transitions = random.sample(self.buffer, batch_size)
        return ReplayBuffer._transitions_to_batch(device, transitions)

    @staticmethod
    def _transitions_to_batch(
        device: torch.device,
        transitions: List[NetTransition],
    ) -> BatchTransition:
        obs = NetObservations.from_observations([t.obs for t in transitions], device)
        next_obs = NetObservations.from_observations([t.next_obs for t in transitions], device)

        actions = torch.tensor(
            [t.action_id for t in transitions], dtype=torch.long, device=device
        ).unsqueeze(1)
        rewards = torch.tensor(
            [t.reward for t in transitions], dtype=torch.float32, device=device
        ).unsqueeze(1)
        dones = torch.tensor(
            [t.done for t in transitions], dtype=torch.float32, device=device
        ).unsqueeze(1)
        real_returns = torch.tensor(
            [t.real_return for t in transitions], dtype=torch.float32, device=device
        ).unsqueeze(1)

        return BatchTransition(
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            real_returns,
        )

    def __len__(self) -> int:
        return len(self.buffer)
