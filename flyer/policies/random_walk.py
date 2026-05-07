"""Random walk policy - GREEN moves randomly with occasional missiles."""

from __future__ import annotations

import numpy as np

from flyer.environment.environment import FlyerAction
from flyer.environment.flyer import Flyer
from flyer.policies.policy_flyer import PolicyFlyer
from flyer.policies.i_policy import IPolicy


class RandomWalkPolicy(IPolicy):
    """Difficulty 1: GREEN moves randomly with occasional missiles."""

    def __init__(self, flyer: Flyer, map_radius: float):
        self.map_radius = map_radius
        self.policy_flyer = PolicyFlyer(flyer)

    def get_action(self) -> FlyerAction:
        random_angle = np.random.uniform(0, 2 * np.pi)
        rand_target_position = np.array([np.cos(random_angle), np.sin(random_angle)]) * self.map_radius
        return FlyerAction(
            change_angle=self.policy_flyer.get_change_angle(rand_target_position),
            launch_missile=np.random.rand() < 0.001,
        )
