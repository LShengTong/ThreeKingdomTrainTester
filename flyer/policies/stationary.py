"""Stationary policy - GREEN stays completely still."""

from __future__ import annotations

from flyer.environment.environment import FlyerAction
from flyer.policies.i_policy import IPolicy


class StationaryPolicy(IPolicy):
    """Difficulty 0: GREEN stays completely still."""

    def get_action(self) -> FlyerAction:
        return FlyerAction(
            change_angle=0.0,
            launch_missile=False,
        )
