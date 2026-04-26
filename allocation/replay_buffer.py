import random
from collections import deque
from typing import Deque, List

import torch

from allocation.net_observation import NetObservation, NetObservations
from transition import Transition, BatchTransition, NetTransition


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.buffer: Deque[NetTransition] = deque(maxlen=capacity)

    def add(
        self,
        transition: Transition,
    ) -> None:
        """Optionally run develop-span + dtype on ``preprocess_device`` (e.g. CUDA) before CPU storage."""
        stored = NetTransition(
            NetObservation.from_observation(transition.obs).detach_cpu(),
            int(transition.action_id),
            float(transition.reward),
            NetObservation.from_observation(transition.next_obs).detach_cpu(),
            bool(transition.done),
            float(transition.real_return),
        )

        self.buffer.append(stored)

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
        obs = NetObservations.from_observations([t.obs for t in transitions]).to(device)
        next_obs = NetObservations.from_observations([t.next_obs for t in transitions]).to(device)

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
