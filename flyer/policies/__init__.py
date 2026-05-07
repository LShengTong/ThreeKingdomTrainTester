"""GREEN opponent policies for curriculum learning."""

from flyer.policies.i_policy import IPolicy
from flyer.policies.stationary import StationaryPolicy
from flyer.policies.random_walk import RandomWalkPolicy
from flyer.policies.strategic import StrategicPolicy
from flyer.policies.aggressive import AggressivePolicy
from flyer.policies.defensive import DefensivePolicy
from flyer.policies.policy_factory import PolicyFactory

__all__ = [
    # Base class
    "IPolicy",
    # Policy implementations
    "StationaryPolicy",
    "RandomWalkPolicy",
    "StrategicPolicy",
    "AggressivePolicy",
    "DefensivePolicy",
    # Factory functions
    "PolicyFactory",
]
