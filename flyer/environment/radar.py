from typing import Dict

import numpy as np

from flyer.environment.flyer import Flyer
from flyer.environment.flyer_type import FlyerType


class Radar:

    def __init__(self, flyers: Dict[FlyerType, Flyer]):
        self.flyers = flyers

    def re_detect(self):
        for (type_i, flyer_i) in self.flyers.items():
            for (type_j, flyer_j) in self.flyers.items():
                if type_i == type_j:
                     continue
                distance = flyer_i.position - flyer_j.position
                if flyer_i.config is None:
                    raise Exception("radar system find flyer do not has config")
                if np.linalg.norm(distance) < flyer_i.config.radar_radius:
                    flyer_i.target_found(flyer_j)
                else:
                    flyer_i.target_found(None)
