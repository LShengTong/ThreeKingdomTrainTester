from allocation.config import DEFAULT_AGENT_CONFIG
from allocation.deep_q_learning import DeepQLearningAgent
from allocation.ppo_learning import ProximalPolicyOptimizationAgent
from allocation.tester import Tester


def main() -> None:
    if DEFAULT_AGENT_CONFIG.algorithm == "ppo":
        agent = ProximalPolicyOptimizationAgent(config=DEFAULT_AGENT_CONFIG)
    else:
        agent = DeepQLearningAgent(config=DEFAULT_AGENT_CONFIG)
    agent.learn()
    Tester(agent=agent).test()


if __name__ == "__main__":
    main()
