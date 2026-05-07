"""Defensive policy - Maintains preferred distance, fires when in range."""

from __future__ import annotations

import numpy as np

from flyer.environment.environment import FlyerAction
from flyer.environment.flyer import Flyer
from flyer.policies import RandomWalkPolicy
from flyer.policies.i_policy import IPolicy
from flyer.policies.policy_flyer import PolicyFlyer


class DefensivePolicy(IPolicy):
    """Bonus: Always tries to maintain distance."""

    def __init__(self, flyer: Flyer, map_radius: float):
        self.flyer = flyer
        self.map_radius = map_radius
        self.policy_flyer = PolicyFlyer(flyer)
        self.random_walk_policy = RandomWalkPolicy(flyer, map_radius)

    def get_action(self) -> FlyerAction:
        if self.flyer.target is None:
            return self.random_walk_policy.get_action()

        if np.linalg.norm(self.flyer.position) > self.map_radius * 0.9:
            direction = np.array([self.flyer.position[1], -self.flyer.position[0]])
            return FlyerAction(
                change_angle=self.policy_flyer.get_change_angle_by_direction(direction),
                launch_missile=False,
            )

        return FlyerAction(
            change_angle=self.policy_flyer.get_change_angle_by_direction(
                self.flyer.position - self.flyer.target.position),
            launch_missile=False,
        )