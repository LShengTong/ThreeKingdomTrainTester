from typing import Dict

import numpy as np

from flyer.config import flyer_configs
from flyer.flyer import Flyer
from flyer.flyer_type import FlyerType


class Radar:

    def __init__(self, flyers: Dict[FlyerType, Flyer]):
        self.flyers = flyers

    def re_detect(self):
        for (type_i, flyer_i) in self.flyers.items():
            for (type_j, flyer_j) in self.flyers.items():
                if type_i == type_j:
                     continue
                distance = flyer_i.position - flyer_j.position
                if np.linalg.norm(distance) < flyer_configs[type_i].radar_radius:
                    flyer_i.target_found(flyer_j)
