from dataclasses import dataclass
from typing import List, Dict

from flyer.flyer import Flyer
from flyer.flyer_type import FlyerType

@dataclass
class FlyerAction:
    change_angle: float
    change_velocity: float
    launch_missile: bool


class Environment:

    def __init__(self):
        self.flyers: Dict[FlyerType, Flyer] = {
            FlyerType.RED: Flyer(FlyerType.RED),
            FlyerType.GREEN: Flyer(FlyerType.GREEN),
        }

    def reset(self):
        for flyer in self.flyers.values():
            flyer.reset()

    def step(self, actions: Dict[FlyerType, FlyerAction]) -> FlyerType | None:
        for (flyer_type, flyer) in self.flyers.items():
            action = actions[flyer_type]
            if action.launch_missile:
                if flyer.launch_missile():
                    return flyer_type
            flyer.change_angle(action.change_angle)
            flyer.change_velocity(action.change_velocity)
            flyer.move(10)
        return None