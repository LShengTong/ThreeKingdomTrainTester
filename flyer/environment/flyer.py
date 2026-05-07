from __future__ import annotations

import numpy as np

from flyer.environment.config import EnvConfig, FlyerConfig
from flyer.environment.flyer_type import FlyerType


class Flyer:

    def __init__(self, flyer_type: FlyerType):
        self.flyer_type = flyer_type
        self.config: FlyerConfig | None = None
        self.map_radius: float = 0.
        self.position: np.ndarray = np.zeros((2, ), dtype=np.float64)
        self.velocity: float = 0.
        self.angle_degree: float = 0.
        self.missile_count: int = 0
        self.target: Flyer | None = None

    def set_config(self, flyer_config: FlyerConfig, map_radius: float):
        self.config = flyer_config
        self.map_radius = map_radius

    def reset(self):
        if self.config is None:
            raise Exception("Flyer config not set")
        # self.position = self.config.position.copy()
        rand_angle = np.deg2rad(np.random.rand() * 360)
        rand_length = np.random.rand() * self.map_radius * 0.8
        if self.config.position is None:
            self.position = rand_length * np.array([np.cos(rand_angle), np.sin(rand_angle)], dtype=np.float64)
        else:
            self.position = self.config.position.copy()
        self.velocity = self.config.velocity
        if self.config.angle_degree is None:
            self.angle_degree = float(np.random.rand()) * 360.
        else:
            self.angle_degree = self.config.angle_degree
        self.missile_count = self.config.missile_count
        self.target = None

    @property
    def direction(self) -> np.ndarray:
        angle_rad = np.deg2rad(self.angle_degree)
        return np.array([np.cos(angle_rad), np.sin(angle_rad)])

    def change_angle(self, ratio: float):
        if self.config is None:
            raise Exception("Flyer config not set")
        if ratio < -1 or ratio > 1:
            raise Exception("change angle 比例范围应当在（-1， 1）之间")
        self.angle_degree += ratio * self.config.max_angle_change_velocity * EnvConfig.delta_time
        self.angle_degree = self.angle_degree % 360

    def launch_missile(self) -> bool:
        if self.missile_count > 0:
            self.missile_count -= 1
            return True
        return False

    def target_found(self, target: Flyer | None):
        self.target = target

    def move(self):
        self.position += EnvConfig.delta_time * self.velocity * self.direction
