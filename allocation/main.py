from allocation.tester import Tester
from allocation.trainer import Trainer
from deep_q_learning import DeepQLearningAgent

if "__main__" == __name__:
    agent = DeepQLearningAgent()
    Trainer(agent=agent).train()
    Tester(agent=agent).test()