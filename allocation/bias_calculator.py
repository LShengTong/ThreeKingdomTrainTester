import torch
from stable_baselines3.dqn.policies import DQNPolicy

from allocation.replay_buffer import ReplayBuffer
from allocation.transition import NetTransition


class BiasCalculator:
    def __init__(
        self,
        gamma: float,
        policy: DQNPolicy,
        device: torch.device
    ) -> None:
        self.sample_set = ReplayBuffer(capacity=1000000)
        self.episode_steps: list[NetTransition] = []
        self.gamma = gamma
        self.policy = policy
        self.device = device

    def add(self, transition: NetTransition):
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
        if len(self.sample_set) == 0:
            print("sample set stats: samples=0")
            return
        batch_transition = self.sample_set.sample(len(self.sample_set), self.device)
        self.policy.eval()
        with torch.no_grad():
            pred = self.policy.q_net(batch_transition.obs.to_dict()).gather(1, batch_transition.action_id)
            bias = pred - batch_transition.real_return

        print(
            "sample set stats: "
            f"samples={len(self.sample_set)}, "
            f"abs_bias={bias.abs().mean().item():.6f}, "
            f"mse={(bias ** 2).mean().item():.6f}"
        )