import copy

import torch

from allocation.bias_calculator import BiasCalculator
from allocation.deep_q_learning import DeepQLearningAgent
from allocation.environment import Environment
from allocation.transition import Transition

class Tester:
    def __init__(self, agent: DeepQLearningAgent):
        self.agent = agent

    def test(self):
        test_bias_calculator = BiasCalculator(
            self.agent.gamma, self.agent.online_net, self.agent.device
        )

        envs = []
        for i in range(200):
            envs.append(Environment())

        random_rewards = 0
        for env in envs:
            env = copy.deepcopy(env)
            done = False
            while not done:
                act = int(torch.randint(0, Environment.develop_shape[0], (1,))[0])
                _, reward, done = env.step(act)
                random_rewards += reward
        print(f"random reward={random_rewards / len(envs)}")

        rewards = 0
        for env in envs:
            done = False
            obs = env.get_observation()
            while not done:
                develop_index = self.agent.act(obs, False)
                next_obs, reward, done = env.step(develop_index)
                rewards += reward
                transition = Transition(obs, develop_index, reward, next_obs, done, 0)
                test_bias_calculator.add(transition)
                obs = next_obs
        print(f"test reward={rewards / len(envs)}")
        test_bias_calculator.print_bias()