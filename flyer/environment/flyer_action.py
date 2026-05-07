from dataclasses import dataclass


@dataclass
class FlyerAction:
    change_angle: float
    launch_missile: bool