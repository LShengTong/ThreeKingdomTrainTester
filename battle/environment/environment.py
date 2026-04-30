from typing import List, Dict, Tuple

import torch

from battle.environment.army import Army
from battle.environment.city import City
from battle.environment.config import Config
from battle.environment.i_attack import IAttack
from battle.environment.turn import Turn


class Environment:
    def __init__(self):
        self.cities: List[City] = []
        self.armies: Dict[int, Army] = {}
        self.next_army_id = 0
        self.reset()

    def reset(self):
        def rand_pos() -> torch.Tensor:
            return (torch.rand((2, )) *
                    torch.Tensor([Config.map_width, Config.map_height]))

        def rand_city(city_id: int) -> City:
            return City(city_id, rand_pos(),
                        int(torch.randint(
                            Config.soldier_rand_range[0],
                            Config.soldier_rand_range[1],
                            (1, )).item()))

        self.cities: List[City] = [rand_city(0), rand_city(1)]
        self.armies: Dict[int, Army] = {}
        self.next_army_id = 0

    def step(
        self,
        city_action: List[List[Tuple[IAttack, int]]],
        army_action: Dict[int, IAttack | None]
    ):
        for city_index in range(len(city_action)):
            armies = city_action[city_index]
            send_out_soldier = 0
            for (target, soldier) in armies:
                self.armies[self.next_army_id] = Army(
                    self.next_army_id,
                    target,
                    soldier,
                    self.cities[city_index],
                )
                self.next_army_id += 1
                send_out_soldier += soldier
            self.cities[city_index].send_out(send_out_soldier)

        for army_index in army_action:
            target = army_action[army_index]
            if target is not None:
                self.armies[army_index].change_target(target)

        Turn.exec(self.cities, list(self.armies.values()))