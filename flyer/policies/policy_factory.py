"""Factory for creating policy instances."""

from flyer.environment.flyer import Flyer
from flyer.policies.i_policy import IPolicy
from flyer.policies.policy_type import PolicyType

# Direct imports to avoid circular dependency
from flyer.policies.stationary import StationaryPolicy
from flyer.policies.random_walk import RandomWalkPolicy
from flyer.policies.aggressive import AggressivePolicy
from flyer.policies.defensive import DefensivePolicy
from flyer.policies.strategic import StrategicPolicy


class PolicyFactory:

    @staticmethod
    def create_policy_by_type(policy_type: PolicyType, flyer: Flyer, map_radius: float) -> IPolicy:
        if policy_type == PolicyType.Stationary:
            return StationaryPolicy()
        if policy_type == PolicyType.RandomWalk:
            return RandomWalkPolicy(flyer, map_radius)
        if policy_type == PolicyType.Aggressive:
            return AggressivePolicy(flyer, map_radius)
        if policy_type == PolicyType.Defensive:
            return DefensivePolicy(flyer, map_radius)
        if policy_type == PolicyType.Strategic:
            return StrategicPolicy(flyer, map_radius)
        raise Exception("invalid policy type: {}".format(policy_type))
