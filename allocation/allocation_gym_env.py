from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import Env, spaces

from allocation.environment import Environment
from allocation.net_observation import NetObservation
from allocation.observation import Observation


class AllocationGymEnv(Env[dict[str, np.ndarray], int]):
    """Wrap Environment and expose NetObservation-style fields."""

    def __init__(self) -> None:
        super().__init__()
        self.inner_env = Environment()
        self.observation_space = spaces.Dict(
            {
                "todo_heroes": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(Environment.max_heroes, Environment.hero_dim),
                    dtype=np.float32,
                ),
                "todo_hero_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(Environment.max_heroes,),
                    dtype=np.float32,
                ),
                "develops": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(Environment.develop_shape[0], 1),
                    dtype=np.float32,
                ),
                "curr_hero": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(Environment.develop_shape[0], 1),
                    dtype=np.float32,
                ),
                "working_heroes": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(Environment.develop_shape[0], Environment.max_heroes, 1),
                    dtype=np.float32,
                ),
                "working_heroes_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(Environment.develop_shape[0], Environment.max_heroes),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Discrete(int(Environment.develop_shape[0]))

    @staticmethod
    def encode_observation(obs: Observation) -> dict[str, np.ndarray]:
        net_obs = NetObservation.from_observation(obs)
        return net_obs.to_dict()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        obs = self.inner_env.reset()
        return self.encode_observation(obs), {}

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        next_obs, reward, done = self.inner_env.step(int(action))
        return self.encode_observation(next_obs), float(reward), bool(done), False, {}
