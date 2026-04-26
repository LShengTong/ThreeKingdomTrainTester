import torch

from battle.city import City
from battle.config import Config
from battle.i_attack import IAttack


class Army(IAttack):
    def attack(self) -> int:
        return int(self.soldier * Config.attack_ratio)

    def defense(self, value: int):
        self.soldier -= value

    def get_pos(self):
        return self.pos

    def __init__(self, army_id: int, target: IAttack, soldier: int, city: City):
        self.army_id = army_id
        self.target = target
        self.soldier = soldier
        self.city = city
        self.pos = self.city.pos.clone()

    def __repr__(self):
        return f'''
            target: {self.target.get_name()}
            soldier: {self.soldier}
            city: {self.city.get_city_id()}
            pos: {self.pos}
        '''

    def change_target(self, target: IAttack):
        self.target = target

    def move_to_target(self, delta_time: float):
        if not self.target.is_alive():
            self.target = self.city
        target_pos = self.target.get_pos()
        if torch.equal(target_pos, self.pos):
            return
        walk_distance = Config.army_speed * delta_time
        delta_pos = target_pos - self.pos
        distance = delta_pos.norm()
        if walk_distance >= distance:
            self.pos = target_pos
            if self.target == self.city:
                self.city.army_back(self.soldier)
        else:
            self.pos += delta_pos / distance * walk_distance

    def is_alive(self):
        return self.soldier > 0

    def get_city_id(self) -> int:
        return self.city.get_city_id()

    def get_name(self) -> str:
        return f"army {self.army_id}"