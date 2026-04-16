from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List

import numpy as np
import torch
from torch import nn

from environment import Environment
from hero_develop_net import HeroDevelopNet


Observation = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class Transition:
    heroes: torch.Tensor
    hero_mask: torch.Tensor
    develops: torch.Tensor
    action_id: int
    reward: float
    next_heroes: torch.Tensor
    next_hero_mask: torch.Tensor
    next_develops: torch.Tensor
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DeepQLearningAgent:
    """Project-specific DQN using Environment observation and HeroDevelopNet."""

    def __init__(
        self,
        env: Environment,
        hero_mlp_hidden: Iterable[int] = (16,),
        hero_mlp_out: int = 16,
        hero_phi_hidden: Iterable[int] = (16,),
        hero_phi_out: int = 16,
        hero_out_dim: int = 16,
        first_hero_hidden: Iterable[int] = (16,),
        develop_mlp_hidden: Iterable[int] = (16,),
        develop_mlp_out: int = 16,
        develop_out_dim: int = 16,
        fusion_hidden_dims: Iterable[int] = (32, 16),
        activation: str = "relu",
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_interval: int = 100,
        batch_size: int = 64,
        replay_capacity: int = 10000,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.num_actions = int(env.action_n)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval
        self.train_steps = 0

        net_kwargs = dict(
            hero_feature_dim=int(env.hero_dim),
            hero_mlp_hidden=hero_mlp_hidden,
            hero_mlp_out=hero_mlp_out,
            hero_phi_hidden=hero_phi_hidden,
            hero_phi_out=hero_phi_out,
            hero_out_dim=hero_out_dim,
            first_hero_hidden=first_hero_hidden,
            develop_count=1,
            develop_feature_dim=int(env.develop_dim),
            num_develops=int(env.action_n),
            develop_mlp_hidden=develop_mlp_hidden,
            develop_mlp_out=develop_mlp_out,
            develop_out_dim=develop_out_dim,
            fusion_hidden_dims=fusion_hidden_dims,
            output_dim=self.num_actions,
            activation=activation,
        )

        self.online_net = HeroDevelopNet(**net_kwargs).to(self.device)
        self.target_net = HeroDevelopNet(**net_kwargs).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay = ReplayBuffer(capacity=replay_capacity)

    def _obs_to_device(self, observation: Observation) -> Observation:
        heroes, hero_mask, develops = observation
        return (
            heroes.to(self.device, dtype=torch.float32),
            hero_mask.to(self.device, dtype=torch.float32),
            develops.to(self.device, dtype=torch.float32),
        )

    def act(self, observation: Observation, use_epsilon=True, print_q=False) -> int:
        if use_epsilon:
            if np.random.rand() < self.epsilon:
                return int(np.random.randint(self.num_actions))

        heroes, hero_mask, develops = self._obs_to_device(observation)
        with torch.no_grad():
            q_values = self.online_net(
                heroes.unsqueeze(0), hero_mask.unsqueeze(0), develops.unsqueeze(0))
            if print_q:
                print(f"q_value={q_values}")
        return int(torch.argmax(q_values, dim=1).item())

    def remember(
        self,
        obs: Observation,
        action_id: int,
        reward: float,
        next_obs: Observation,
        done: bool,
    ) -> None:
        if action_id < 0 or action_id >= self.num_actions:
            raise IndexError("action_id out of range.")

        heroes, hero_mask, develops = obs
        next_heroes, next_hero_mask, next_develops = next_obs

        self.replay.add(
            Transition(
                heroes=heroes.detach().cpu(),
                hero_mask=hero_mask.detach().cpu(),
                develops=develops.detach().cpu(),
                action_id=int(action_id),
                reward=float(reward),
                next_heroes=next_heroes.detach().cpu(),
                next_hero_mask=next_hero_mask.detach().cpu(),
                next_develops=next_develops.detach().cpu(),
                done=bool(done),
            )
        )

    def train_step(self, need_print=False) -> float | None:
        # if len(self.replay) < self.batch_size:
        #     return None
        if len(self.replay) < 300:
            return None

        batch = self.replay.sample(self.batch_size)
        heroes = torch.stack([t.heroes for t in batch]).to(self.device)
        hero_masks = torch.stack([t.hero_mask for t in batch], dim=0).to(self.device)
        develops = torch.stack([t.develops for t in batch], dim=0).to(self.device)

        next_heroes = torch.stack([t.next_heroes for t in batch], dim=0).to(self.device)
        next_hero_masks = torch.stack([t.next_hero_mask for t in batch], dim=0).to(self.device)
        next_develops = torch.stack([t.next_develops for t in batch], dim=0).to(self.device)

        actions = torch.tensor([t.action_id for t in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        q_pred = self.online_net(heroes, hero_masks, develops).gather(1, actions)
        with torch.no_grad():
            q_next = self.target_net(next_heroes, next_hero_masks, next_develops)
            q_next = q_next.max(dim=1, keepdim=True)[0]
            q_target = rewards + (1 - dones) * self.gamma * q_next
            if need_print:
                print(f"q pred={q_pred}, q target={q_target}")

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())
