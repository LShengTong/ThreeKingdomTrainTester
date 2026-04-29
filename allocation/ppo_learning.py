from __future__ import annotations

from stable_baselines3 import PPO

from allocation.allocation_gym_env import AllocationGymEnv
from allocation.config import AllocationAgentConfig, DEFAULT_AGENT_CONFIG
from allocation.environment import Environment
from allocation.hero_develop_extractor import HeroDevelopExtractor
from allocation.observation import Observation


class ProximalPolicyOptimizationAgent:
    """PPO agent implemented with stable_baselines3."""

    def __init__(self, config: AllocationAgentConfig | None = None) -> None:
        self.config = config or DEFAULT_AGENT_CONFIG
        Environment.max_heroes = self.config.environment.max_heroes
        self.gamma = self.config.ppo.gamma
        self.device = self.config.ppo.device
        self.train_env = AllocationGymEnv()
        policy_kwargs = dict(
            net_arch=[],
            features_extractor_class=HeroDevelopExtractor,
        )
        self.model = PPO(
            policy="MultiInputPolicy",
            env=self.train_env,
            learning_rate=self.config.ppo.lr,
            n_steps=self.config.ppo.n_steps,
            batch_size=self.config.ppo.batch_size,
            n_epochs=self.config.ppo.n_epochs,
            gamma=self.config.ppo.gamma,
            gae_lambda=self.config.ppo.gae_lambda,
            clip_range=self.config.ppo.clip_range,
            ent_coef=self.config.ppo.ent_coef,
            vf_coef=self.config.ppo.vf_coef,
            max_grad_norm=self.config.ppo.max_grad_norm,
            device=self.config.ppo.device,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.config.ppo.tensorboard_log,
            verbose=1,
        )

    def learn(self, total_timesteps: int | None = None) -> None:
        steps: int
        if total_timesteps is not None:
            steps = total_timesteps
        else:
            steps = self.config.ppo.total_timesteps
        self.model.learn(total_timesteps=steps)

    def act(self, observation: Observation, deterministic: bool = True) -> int:
        policy_obs = self.train_env.encode_observation(observation)
        action, _ = self.model.predict(policy_obs, deterministic=deterministic)
        return int(action)
