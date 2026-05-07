"""Strategic policy - Full strategy with chase/attack/flee decisions."""

from __future__ import annotations

from flyer.environment.environment import FlyerAction
from flyer.environment.flyer import Flyer
from flyer.policies.random_walk import RandomWalkPolicy
from flyer.policies.aggressive import AggressivePolicy
from flyer.policies.defensive import DefensivePolicy
from flyer.policies.i_policy import IPolicy


class StrategicPolicy(IPolicy):
    """Difficulty 3: Full strategy - chase when have missile, run away when not have missile."""

    def __init__(
        self,
        flyer: Flyer,
        map_radius: float,
    ):
        self.flyer = flyer
        self.random_walk_policy = RandomWalkPolicy(flyer, map_radius)
        self.aggressive_policy = AggressivePolicy(flyer, map_radius)
        self.defensive_policy = DefensivePolicy(flyer, map_radius)

    def get_action(self) -> FlyerAction:
        if self.flyer.target is None:
            return self.random_walk_policy.get_action()

        if self.flyer.missile_count <= 0:
            return self.defensive_policy.get_action()

        return self.aggressive_policy.get_action()
