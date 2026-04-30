import numpy as np

from flyer.config import Config
from flyer.flyer import Flyer


class Attack:

    def __init__(self, flyer: Flyer, target: Flyer):
        self.flyer = flyer
        self.target = target

    def attack(self) -> bool:
        delta_position = self.target.position - self.flyer.position
        distance = np.linalg.norm(delta_position)
        cos_theta = (np.dot(self.flyer.angle, delta_position) /
                     (np.linalg.norm(self.flyer.angle) * np.linalg.norm(delta_position)))
        theta = np.degrees(np.arccos(cos_theta))
        distance_factor = np.clip(1 - distance / Config.max_attack_distance, 0, 1)
        theta_factor = np.clip(1 - theta / Config.max_attack_angle_degree, 0, 1)
        return distance_factor * theta_factor > np.random.rand()
