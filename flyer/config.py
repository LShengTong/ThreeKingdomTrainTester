from dataclasses import dataclass
from typing import Dict

import numpy as np

from flyer.flyer_type import FlyerType

@dataclass(frozen=True)
class EnvConfig:
    delta_time = 1


@dataclass
class FlyerConfig:
    max_attack_distance: float # km
    max_attack_angle_degree: float
    radar_radius: float # km
    min_velocity: float # km/s
    max_velocity: float # km/s
    velocity: float # km/s
    position: np.ndarray
    angle: np.ndarray
    missile_count: int
    max_angle_change_velocity: float # degree/s
    max_accelerate: float # km/s^2

flyer_configs: Dict[FlyerType, FlyerConfig] = {
    FlyerType.RED: FlyerConfig(
        max_attack_distance = 50,
        max_attack_angle_degree = 20,
        radar_radius = 150,
        min_velocity = 0.06,
        max_velocity = 0.3,
        velocity = 0.2,
        position = np.array([-100., -100.]),
        angle = np.array([1., 1.]),
        missile_count = 3,
        max_angle_change_velocity = 20,
        max_accelerate = 0.02
    ),
    FlyerType.GREEN: FlyerConfig(
        max_attack_distance=50,
        max_attack_angle_degree=20,
        radar_radius=150,
        min_velocity=0.06,
        max_velocity=0.3,
        velocity=0.2,
        position=np.array([100., 100.]),
        angle=np.array([-1., -1.]),
        missile_count=3,
        max_angle_change_velocity=20,
        max_accelerate=0.02
    )
}

