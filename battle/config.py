from typing import Tuple


class Config:
    turn_interval: float = 10
    turn_delta_time: float = 0.1
    attack_interval: float = 1
    attack_ratio: float = 0.01
    city_defence_ratio: float = 0.75
    attack_radius = 1
    army_speed = 0.5
    map_width: int = 10
    map_height: int = 10
    soldier_rand_range: Tuple[int, int] = (5000, 10000)
