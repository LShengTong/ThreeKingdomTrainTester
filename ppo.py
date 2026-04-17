from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.distributions import Categorical

from environment import Environment
from hero_develop_net import HeroDevelopNet
from observation import Observations, Observation


@dataclass
class _RolloutStep:
    obs: Observation
    next_obs: Observation
    action_id: int
    log_prob: float
    reward: float
    done: bool
    value: float


class PPOAgent:
    """On-policy actor–critic PPO using the same observation encoder as DQN."""

    def __init__(
        self,
        env: Environment,
        hero_phi_hidden: Iterable[int] = (8,),
        hero_phi_out: int = 4,
        hero_rho_hidden: Iterable[int] = (8,),
        work_phi_hidden: Iterable[int] = (8,),
        work_phi_out: int = 4,
        work_rho_hidden: Iterable[int] = (8,),
        fusion_hidden_dims: Iterable[int] = (64, 64),
        activation: str = "relu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_length: int = 512,
        epochs: int = 4,
        minibatch_size: int = 64,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device: {device}")
        self.device = torch.device(device)
        self.num_actions = int(env.action_n)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_length = rollout_length
        self.epochs = epochs
        self.minibatch_size = minibatch_size

        net_kwargs = dict(
            hero_feature_dim=int(env.hero_dim),
            hero_phi_hidden=hero_phi_hidden,
            hero_phi_out=hero_phi_out,
            hero_rho_hidden=hero_rho_hidden,
            work_phi_hidden=work_phi_hidden,
            work_phi_out=work_phi_out,
            work_rho_hidden=work_rho_hidden,
            develop_feature_dim=int(env.develop_dim),
            num_develops=int(env.action_n),
            fusion_hidden_dims=fusion_hidden_dims,
            activation=activation,
        )

        self.actor = HeroDevelopNet(**net_kwargs, output_dim=self.num_actions).to(self.device)
        self.critic = HeroDevelopNet(**net_kwargs, output_dim=1).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

        self._rollout: list[_RolloutStep] = []

    @property
    def online_net(self) -> nn.Module:
        """Alias for tools that expect a value / Q network (BiasCalculator uses critic for PPO)."""
        return self.critic

    def act(self, observation: Observation, use_epsilon: bool = True, print_q: bool = False) -> int:
        """use_epsilon=True: stochastic policy; False: greedy (argmax). print_q is unused (API parity with DQN)."""
        action, _, _ = self._forward_policy(observation, stochastic=use_epsilon)
        return action

    def act_with_details(self, observation: Observation, stochastic: bool = True) -> tuple[int, float, float]:
        """Returns action, log pi(a|s), V(s)."""
        return self._forward_policy(observation, stochastic=stochastic)

    def _forward_policy(
        self,
        observation: Observation,
        *,
        stochastic: bool,
    ) -> tuple[int, float, float]:
        o = Observations.stack([observation]).to(self.device, dtype=torch.float32)
        logits = self.actor(o)
        value = self.critic(o).squeeze(-1)
        dist = Categorical(logits=logits)
        if stochastic:
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=-1)
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def evaluate_actions(
        self,
        obs: Observations,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch evaluate log pi(a|s), entropy, V(s)."""
        logits = self.actor(obs)
        values = self.critic(obs).squeeze(-1)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.squeeze(-1))
        entropy = dist.entropy()
        return log_probs, entropy, values

    def remember_from_env_step(
        self,
        obs: Observation,
        next_obs: Observation,
        action_id: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        if action_id < 0 or action_id >= self.num_actions:
            raise IndexError("action_id out of range.")
        self._rollout.append(
            _RolloutStep(
                obs=obs,
                next_obs=next_obs,
                action_id=action_id,
                log_prob=log_prob,
                reward=reward,
                done=done,
                value=value,
            )
        )

    def train_step(self, need_print: bool = False) -> float | None:
        if len(self._rollout) < self.rollout_length:
            return None

        steps = self._rollout[: self.rollout_length]
        self._rollout = self._rollout[self.rollout_length :]

        rewards = torch.tensor([s.reward for s in steps], dtype=torch.float32, device=self.device)
        dones = torch.tensor([s.done for s in steps], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor([s.log_prob for s in steps], dtype=torch.float32, device=self.device)
        old_values = torch.tensor([s.value for s in steps], dtype=torch.float32, device=self.device)
        actions = torch.tensor([s.action_id for s in steps], dtype=torch.long, device=self.device).unsqueeze(1)

        obs_batch = Observations.stack([s.obs for s in steps]).to(self.device, dtype=torch.float32)
        next_obs_batch = Observations.stack([s.next_obs for s in steps]).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            next_values = self.critic(next_obs_batch).squeeze(-1)
            next_values = next_values * (1.0 - dones)

        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in range(len(steps) - 1, -1, -1):
            delta = rewards[t] + self.gamma * next_values[t] - old_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        total_loss = 0.0
        n_updates = 0
        indices = torch.arange(len(steps), device=self.device)

        for _ in range(self.epochs):
            perm = indices[torch.randperm(len(steps), device=self.device)]
            for start in range(0, len(steps), self.minibatch_size):
                idx = perm[start : start + self.minibatch_size]
                mb_obs = Observations(
                    obs_batch.heroes[idx],
                    obs_batch.hero_mask[idx],
                    obs_batch.develops[idx],
                    obs_batch.working_assignments[idx],
                )
                mb_actions = actions[idx]
                mb_old_log = old_log_probs[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]

                log_probs, entropy, values = self.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(log_probs - mb_old_log)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, mb_ret)
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_loss += float(loss.item())
                n_updates += 1
                if need_print and n_updates == 1:
                    print(f"ppo loss={loss.item():.6f}")

        return total_loss / max(n_updates, 1)
