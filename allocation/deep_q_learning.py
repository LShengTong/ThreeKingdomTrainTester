from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn

from allocation.net_observation import NetObservation, NetObservations
from environment import Environment
from hero_develop_net import HeroDevelopNet
from observation import Observation
from replay_buffer import ReplayBuffer
from transition import Transition

class DeepQLearningAgent:
    """Project-specific DQN using Environment observation and HeroDevelopNet."""

    def __init__(
        self,
        hero_phi_hidden: Iterable[int] = (),
        hero_phi_out: int = 4,
        hero_rho_hidden: Iterable[int] = (),
        # hero_out_dim: int = 32,
        work_phi_hidden: Iterable[int] = (),
        work_phi_out: int = 4,
        work_rho_hidden: Iterable[int] = (),
        # first_hero_hidden: Iterable[int] = (32,),
        # develop_mlp_hidden: Iterable[int] = (32,),
        # develop_mlp_out: int = 32,
        # develop_out_dim: int = 32,
        fusion_hidden_dims: Iterable[int] = (64, 64, 64),
        activation: str = "relu",
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_interval: int = 100,
        batch_size: int = 1024,
        replay_capacity: int = 10000,
        device: str = "",
    ) -> None:
        if device == "":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        print(f"using device: {device}")
        self.device = torch.device(device)
        self.num_actions = int(Environment.develop_shape[0])
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval
        self.train_steps = 0

        net_kwargs = dict(
            hero_feature_dim=int(Environment.hero_dim),
            hero_phi_hidden=hero_phi_hidden,
            hero_phi_out=hero_phi_out,
            hero_rho_hidden=hero_rho_hidden,
            # hero_out_dim=hero_out_dim,
            work_phi_hidden=work_phi_hidden,
            work_phi_out=work_phi_out,
            work_rho_hidden=work_rho_hidden,
            # first_hero_hidden=first_hero_hidden,
            # develop_count=env.action_n,
            develop_feature_dim=int(Environment.develop_shape[1]),
            num_develops=int(Environment.develop_shape[0]),
            # develop_mlp_hidden=develop_mlp_hidden,
            # develop_mlp_out=develop_mlp_out,
            # develop_out_dim=develop_out_dim,
            fusion_hidden_dims=fusion_hidden_dims,
            activation=activation,
        )

        self.online_net = HeroDevelopNet(**net_kwargs).to(self.device)
        self.target_net = HeroDevelopNet(**net_kwargs).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay = ReplayBuffer(capacity=replay_capacity)

    def act(self, observation: Observation, use_epsilon=True, print_q=False) -> int:
        if use_epsilon:
            if np.random.rand() < self.epsilon:
                return int(np.random.randint(self.num_actions))

        o = NetObservations.from_observations(
            [NetObservation.from_observation(observation)]
        ).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(o)
            if print_q:
                print(f"q_value={q_values}")
        return int(torch.argmax(q_values, dim=1).item())

    def remember(
        self,
        transition: Transition
    ) -> None:
        if transition.action_id < 0 or transition.action_id >= self.num_actions:
            raise IndexError("action_id out of range.")

        self.replay.add(transition)

    def train_step(self, need_print=False) -> float | None:
        # if len(self.replay) < self.batch_size:
        #     return None
        if len(self.replay) < 1024:
            return None

        batch_transition = self.replay.sample(self.batch_size, self.device)
        q_pred = self.online_net(batch_transition.obs).gather(1, batch_transition.action_id)
        with torch.no_grad():
            q_next = self.target_net(batch_transition.next_obs)
            q_next = q_next.max(dim=1, keepdim=True)[0]
            q_target = batch_transition.reward + (1 - batch_transition.done) * self.gamma * q_next
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
