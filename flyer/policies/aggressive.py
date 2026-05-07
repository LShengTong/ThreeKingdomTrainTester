"""Aggressive policy - Always chase and attack, never retreats."""

from __future__ import annotations

from flyer.environment.attack import Attack
from flyer.environment.environment import FlyerAction
from flyer.environment.flyer import Flyer
from flyer.policies import RandomWalkPolicy
from flyer.policies.policy_flyer import PolicyFlyer
from flyer.policies.i_policy import IPolicy


class AggressivePolicy(IPolicy):
    """Bonus: Always chase and attack, never retreats."""

    def __init__(self, flyer: Flyer, map_radius: float):
        self.flyer = flyer
        self.map_radius = map_radius
        self.policy_flyer = PolicyFlyer(flyer)
        self.random_walk_policy = RandomWalkPolicy(flyer, map_radius)

    def get_action(self) -> FlyerAction:
        if self.flyer.target is None:
            return self.random_walk_policy.get_action()

        if self.flyer.missile_count <= 0:
            return FlyerAction(
                change_angle=self.policy_flyer.get_change_angle(self.flyer.target.position),
                launch_missile=False
            )
        # print(FlyerAction(
        #     change_angle=self.policy_flyer.get_change_angle(self.flyer.target.position),
        #     launch_missile=Attack(self.flyer, self.flyer.target).hit_on_probability() > 0.5,
        # ))
        return FlyerAction(
            change_angle=self.policy_flyer.get_change_angle(self.flyer.target.position),
            launch_missile=Attack(self.flyer, self.flyer.target).hit_on_probability() > 0.5,
        )
