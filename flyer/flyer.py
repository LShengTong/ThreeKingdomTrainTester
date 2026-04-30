from __future__ import annotations

import numpy as np

from flyer.config import flyer_configs, EnvConfig
from flyer.flyer_type import FlyerType


class Flyer:

    def __init__(self, flyer_type: FlyerType):
        self.flyer_type = flyer_type
        self.config = flyer_configs[self.flyer_type]
        self.position: np.ndarray = np.zeros(3, dtype=np.float64)
        self.velocity: float = 0.0
        self.angle: np.ndarray = np.zeros(3, dtype=np.float64)
        self.missile_count: int = 0
        self.target: Flyer | None = None

    def reset(self):
        self.position = self.config.position
        self.velocity = self.config.velocity
        self.angle = self.config.angle
        self.missile_count = self.config.missile_count
        self.target = None

    def change_angle(self, ratio: float):
        if ratio < -1 or ratio > 1:
            raise Exception("change angle 比例范围应当在（-1， 1）之间")
        self.angle += ratio * self.config.max_angle_change_velocity * EnvConfig.delta_time

    def launch_missile(self):
        if self.missile_count <= 0:
            raise Exception("导弹数量不足")
        self.missile_count -= 1

    def change_velocity(self, ratio: float):
        if ratio < 0 or ratio > 1:
            raise Exception("change angle 比例范围应当在（0， 1）之间")
        self.velocity += self.config.max_accelerate * ratio * EnvConfig.delta_time
        self.velocity = (
            np.clip(self.velocity, self.config.min_velocity, self.config.max_velocity))

    def target_found(self, target: Flyer | None):
        self.target = target

    def move(self):
        self.position += EnvConfig.delta_time * self.velocity * self.angle
