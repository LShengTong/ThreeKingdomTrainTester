import numpy as np

from flyer.environment.flyer import Flyer


class Attack:

    def __init__(self, flyer: Flyer, target: Flyer):
        self.flyer = flyer
        self.target = target

    def attack(self) -> bool:
        # print("probability:", self.hit_on_probability())
        return self.hit_on_probability() > np.random.rand()

    def hit_on_probability(self) -> float:
        delta_position = self.target.position - self.flyer.position
        distance = np.linalg.norm(delta_position)
        if distance == 0.:
            return 1.
        delta_direction = delta_position / distance
        cos_theta = np.dot(self.flyer.direction, delta_direction)
        theta = np.degrees(np.arccos(cos_theta))
        config = self.flyer.config
        if config is None:
            raise Exception("attack but config is None")
        distance_factor = np.clip(1 - float(distance) / config.max_attack_distance, 0, 1)
        theta_factor = np.clip(1 - theta / config.max_attack_angle_degree, 0, 1)
        return distance_factor * theta_factor
