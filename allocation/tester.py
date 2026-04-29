import copy
from typing import Any, List, Protocol

import numpy as np
import torch

from allocation.bias_calculator import BiasCalculator
from allocation.environment import Environment
from allocation.net_observation import NetObservation
from allocation.transition import NetTransition


class TrainableAgent(Protocol):
    gamma: float
    model: Any

    def act(self, observation: Any, deterministic: bool = True) -> int: ...

    @property
    def device(self) -> torch.device: ...


class Tester:
    def __init__(self, agent: TrainableAgent) -> None:
        self.agent = agent

    def test(self):
        envs: List[Environment] = []
        for _ in range(200):
            envs.append(Environment())

        bias_calculator = None
        if hasattr(self.agent.model.policy, "q_net"):
            bias_calculator = BiasCalculator(
                gamma=self.agent.gamma,
                policy=self.agent.model.policy,
                device=self.agent.device,
            )

        random_rewards = 0
        for env in envs:
            env = copy.deepcopy(env)
            done = False
            while not done:
                act = int(torch.randint(0, Environment.develop_shape[0], (1,))[0])
                _, reward, done = env.step(act, reward_shaping=False)
                random_rewards += reward
        print(f"random reward={random_rewards / len(envs)}")

        rewards = 0
        for env in envs:
            done = False
            env = copy.deepcopy(env)
            obs = env.get_observation()
            while not done:
                develop_index = self.agent.act(obs, deterministic=True)
                next_obs, reward, done = env.step(develop_index, reward_shaping=False)
                if bias_calculator is not None:
                    bias_calculator.add(
                        NetTransition(
                            obs=NetObservation.from_observation(obs),
                            action_id=develop_index,
                            reward=float(reward),
                            next_obs=NetObservation.from_observation(next_obs),
                            done=bool(done),
                            real_return=0.0,
                        )
                    )
                rewards += reward
                obs = next_obs
        print(f"test reward={rewards / len(envs)}")
        if bias_calculator is not None:
            bias_calculator.print_bias()

        # best_rewards = 0.0
        # for env in envs:
        #     best_rewards += self._best_reward_by_action_search(env)
        # print(f"best reward={best_rewards / len(envs)}")

    # @staticmethod
    # def _best_reward_by_action_search(env: Environment) -> float:
    #     num_develops = int(env.develop_shape[0])
    #     max_reward = -np.inf
    #     def dfs(dfs_env: Environment):
    #         nonlocal max_reward
    #         for i in range(num_develops):
    #             env_copy = copy.deepcopy(dfs_env)
    #             _, reward, done = env_copy.step(i, reward_shaping=False)
    #             if done:
    #                 max_reward = max(max_reward, reward)
    #                 continue
    #             dfs(env_copy)
    #     dfs(env)
    #     return max_reward