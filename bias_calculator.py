import torch
from torch import nn

from replay_buffer import ReplayBuffer
from transition import Transition


class BiasCalculator:
    def __init__(
        self,
        gamma: float,
        online_net: nn.Module,
        device: torch.device,
        *,
        prediction: str = "q",
    ) -> None:
        self.sample_set = ReplayBuffer(capacity=1000000)
        self.episode_steps: list[Transition] = []
        self.gamma = gamma
        self.online_net = online_net
        self.device = device
        self.prediction = prediction

    def add(self, transition: Transition):
        self.episode_steps.append(transition)
        if transition.done:
            returns = self._discounted_episode_returns()
            for episode_step, g_ret in zip(self.episode_steps, returns):
                episode_step.real_return = g_ret
                self.sample_set.add(episode_step)
            self.episode_steps.clear()

    def _discounted_episode_returns(self) -> list[float]:
        """Monte Carlo returns G_t = r_t + gamma * G_{t+1} along one episode (unbiased for finite episodes)."""
        g = 0.0
        backward: list[float] = []
        rewards = [t.reward for t in self.episode_steps]
        for r in reversed(rewards):
            g = r + self.gamma * g
            backward.append(g)
        return list(reversed(backward))

    def print_bias(self):
        batch_transition = self.sample_set.sample(len(self.sample_set), self.device)
        self.online_net.eval()
        with torch.no_grad():
            if self.prediction == "q":
                pred = self.online_net(batch_transition.obs).gather(1, batch_transition.action_id)
            elif self.prediction == "value":
                pred = self.online_net(batch_transition.obs)
            else:
                raise ValueError(f"Unknown prediction mode: {self.prediction}")
            bias = pred - batch_transition.real_return

        print(
            "sample set stats: "
            f"samples={len(self.sample_set)}, "
            f"abs_bias={bias.abs().mean().item():.6f}, "
            f"mse={(bias ** 2).mean().item():.6f}"
        )