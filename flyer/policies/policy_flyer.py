import numpy as np

from flyer.environment.flyer import Flyer


class PolicyFlyer:

    def __init__(self, flyer: Flyer) -> None:
        self.flyer = flyer

    def get_change_angle(self, target_position: np.ndarray) -> float:
        delta_position = target_position - self.flyer.position
        delta_position_length = np.linalg.norm(delta_position)
        if delta_position_length == 0.:
            forward_dir = np.array([1., 0.])
        else:
            forward_dir = delta_position / delta_position_length
        return self.get_change_angle_by_direction(forward_dir)

    def get_change_angle_by_direction(self, direction: np.ndarray) -> float:
        if self.flyer.config is None:
            raise Exception("PolicyFlyer: flyer has no config")
        config = self.flyer.config
        target_angle = (np.atan2(direction[1], direction[0]) * 180 / np.pi) % 360
        delta_angle = target_angle - self.flyer.angle_degree
        if np.abs(delta_angle) < config.max_angle_change_velocity:
            change_angle = float(delta_angle / config.max_angle_change_velocity)
        elif delta_angle < 0:
            if delta_angle < -180:
                change_angle = 1.
            else:
                change_angle = -1.
        else:
            if delta_angle > 180:
                change_angle = -1.
            else:
                change_angle = 1.
        return change_angle