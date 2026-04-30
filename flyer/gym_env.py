import numpy as np
from gymnasium import Env

from flyer.config import flyer_configs
from flyer.environment import Environment
from flyer.flyer_type import FlyerType


class GymEnv(Env[dict[str, np.ndarray], int]):

    def __init__(self):
        self.inner_env = Environment()

    def _get_flyer_observation(self, flyer_type: FlyerType) -> np.ndarray:
        flyer = self.inner_env.flyers[flyer_type]
        config = flyer_configs[flyer_type]
        target = flyer.target
        return np.array([
            config.max_attack_distance,
            config.min_velocity,
            config.max_velocity,
            config.max_accelerate,
            config.max_angle_change_velocity,
            config.radar_radius,
            config.max_attack_angle_degree,
            flyer.position[0],
            flyer.position[1],
            flyer.missile_count,
            flyer.angle[0],
            flyer.angle[1],
            flyer.velocity,

            target.position[0],
            target.position[1],

            target.velocity,
            target.angle,
            target.position - flyer.position,
            
        ])
