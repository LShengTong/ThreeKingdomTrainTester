from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from flyer.environment.flyer_type import FlyerType

@dataclass
class GameConfig:
    map_radius: float
    flyer_configs: Dict[FlyerType, FlyerConfig]

@dataclass(frozen=True)
class EnvConfig:
    delta_time: int = 1  # s

@dataclass(frozen=True)
class FlyerObservationUpperBound:
    velocity: float = 0.5
    missile_count: int = 10
    accelerate: float = 0.1
    map_radius: float = 500.

@dataclass
class FlyerConfig:
    max_attack_distance: float # km
    max_attack_angle_degree: float
    radar_radius: float # km
    velocity: float # km/s
    position: np.ndarray | None
    angle_degree: float | None
    missile_count: int
    max_angle_change_velocity: float # degree/s

# flyer_configs: Dict[FlyerType, FlyerConfig] = {
#     FlyerType.RED: FlyerConfig(
#         max_attack_distance = 50,
#         max_attack_angle_degree = 20,
#         radar_radius = 150,
#         min_velocity = 0.06,
#         max_velocity = 0.3,
#         velocity = 0.2,
#         position = np.array([-100., 0.]),
#         angle_degree = 0,
#         missile_count = 3,
#         max_angle_change_velocity = 20,
#         max_accelerate = 0.02
#     ),
#     FlyerType.GREEN: FlyerConfig(
#         max_attack_distance=50,
#         max_attack_angle_degree=20,
#         radar_radius=150,
#         min_velocity=0.06,
#         max_velocity=0.3,
#         velocity=0.2,
#         position=np.array([100., 0.]),
#         angle_degree=180,
#         missile_count=3,
#         max_angle_change_velocity=20,
#         max_accelerate=0.02
#     )
# }

