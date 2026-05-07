from typing import Dict, Tuple

import numpy as np

from flyer.environment.attack import Attack
from flyer.environment.config import GameConfig
from flyer.environment.flyer import Flyer
from flyer.environment.flyer_action import FlyerAction
from flyer.environment.flyer_type import FlyerType
from flyer.environment.radar import Radar


class Environment:

    def __init__(self):
        self.flyers: Dict[FlyerType, Flyer] = {
            FlyerType.RED: Flyer(FlyerType.RED),
            FlyerType.GREEN: Flyer(FlyerType.GREEN),
        }
        self.radar: Radar = Radar(self.flyers)
        self.game_config: GameConfig | None = None

    def set_game_config(self, game_config: GameConfig):
        self.game_config = game_config
        for (flyer_type, flyer) in self.flyers.items():
            flyer.set_config(game_config.flyer_configs[flyer_type], game_config.map_radius)

    def reset(self):
        for flyer in self.flyers.values():
            flyer.reset()
        self.radar.re_detect()

    def step(self, actions: Dict[FlyerType, FlyerAction]) -> Tuple[bool, FlyerType | None]:
        if not self.game_config:
            raise Exception("Game config not set")
        self.radar.re_detect()
        has_missile = False
        for (flyer_type, flyer) in self.flyers.items():
            action = actions[flyer_type]
            if action.launch_missile:
                success = flyer.launch_missile()
                if success and (flyer.target is not None):
                    if Attack(flyer, flyer.target).attack():
                        return True, flyer_type
            flyer.change_angle(action.change_angle)
            flyer.move()
            if np.linalg.norm(flyer.position) > self.game_config.map_radius:
                return True, FlyerType.GREEN if flyer_type == FlyerType.RED else FlyerType.RED
            if flyer.missile_count > 0:
                has_missile = True
        if not has_missile:
            return True, None
        return False, None