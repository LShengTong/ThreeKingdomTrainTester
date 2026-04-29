from __future__ import annotations

import torch
from stable_baselines3 import DQN

from allocation.allocation_gym_env import AllocationGymEnv
from allocation.config import AllocationAgentConfig, DEFAULT_AGENT_CONFIG
from allocation.environment import Environment
from allocation.hero_develop_extractor import HeroDevelopExtractor
from allocation.observation import Observation

class DeepQLearningAgent:
    """DQN agent implemented with stable_baselines3."""

    def __init__(self, config: AllocationAgentConfig | None = None) -> None:
        self.config = config or DEFAULT_AGENT_CONFIG
        Environment.max_heroes = self.config.environment.max_heroes
        self.gamma = self.config.dqn.gamma
        self.train_env = AllocationGymEnv()
        policy_kwargs = dict(
            net_arch=[],
            features_extractor_class=HeroDevelopExtractor,
        )
        self.model = DQN(
            policy="MultiInputPolicy",
            env=self.train_env,
            learning_rate=self.config.dqn.lr,
            buffer_size=self.config.dqn.replay_capacity,
            learning_starts=self.config.dqn.learning_starts,
            batch_size=self.config.dqn.batch_size,
            tau=1.0,
            gamma=self.config.dqn.gamma,
            train_freq=(self.config.dqn.train_freq, "step"),
            gradient_steps=self.config.dqn.gradient_steps,
            target_update_interval=self.config.dqn.target_update_interval,
            exploration_initial_eps=self.config.dqn.exploration_initial_eps,
            exploration_final_eps=self.config.dqn.exploration_final_eps,
            exploration_fraction=self.config.dqn.exploration_fraction,
            device=self.config.dqn.device,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.config.dqn.tensorboard_log,
            verbose=1,
        )

    def learn(self, total_timesteps: int | None = None) -> None:
        steps: int
        if total_timesteps is not None:
            steps = total_timesteps
        else:
            steps = self.config.dqn.total_timesteps
        self.model.learn(total_timesteps=steps)

    def act(self, observation: Observation, deterministic: bool = True) -> int:
        policy_obs = self.train_env.encode_observation(observation)
        action, _ = self.model.predict(policy_obs, deterministic=deterministic)
        return int(action)

    @property
    def device(self) -> torch.device:
        return self.model.device
