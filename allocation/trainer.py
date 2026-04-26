from allocation.bias_calculator import BiasCalculator
from allocation.deep_q_learning import DeepQLearningAgent
from allocation.environment import Environment
from allocation.transition import Transition


class Trainer:
    def __init__(self, agent: DeepQLearningAgent):
        self.agent = agent
        self.env = Environment()
        self.train_every = 16
        use_writer = False
        if use_writer:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir="../runs/hero_develop_dqn")
        else:
            self.writer = None

    def train(self):
        training_bias_calculator = BiasCalculator(
            self.agent.gamma, self.agent.online_net, self.agent.device
        )

        obs = self.env.reset()

        for step in range(20000):
            if step % 100 == 0:
                print(f"step={step}")

            develop_index = self.agent.act(obs)
            next_obs, reward, done = self.env.step(develop_index)
            transition = Transition(obs, develop_index, reward, next_obs, done, 0)
            self.agent.remember(transition)
            loss = None
            if step % self.train_every == 0:
                loss = self.agent.train_step()

            training_bias_calculator.add(transition)

            if self.writer is not None and loss is not None:
                self.writer.add_scalar("train/loss", loss, step)

            obs = next_obs
            if done:
                if self.writer is not None:
                    self.writer.add_scalar("train/reward", reward, step)
                obs = self.env.reset()

        training_bias_calculator.print_bias()

        if self.writer is not None:
            self.writer.close()