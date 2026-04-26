import torch

from battle.config import Config
from battle.i_attack import IAttack


class City(IAttack):
    def attack(self) -> int:
        return int(self.soldier * Config.attack_ratio)

    def defense(self, value: int):
        self.soldier -= value * Config.city_defence_ratio

    def get_pos(self) -> torch.Tensor:
        return self.pos

    def __init__(
            self,
            city_id: int,
            pos: torch.Tensor,
            soldier: int,
    ):
        self.city_id = city_id
        self.pos = pos
        self.soldier = soldier

    def __repr__(self):
        return f"""
        id: {self.city_id}
        pos: {self.pos}
        soldier: {self.soldier}
        """

    def send_out(self, soldier: int):
        self.soldier -= soldier

    def is_alive(self) -> bool:
        return self.soldier > 0

    def get_city_id(self) -> int:
        return self.city_id

    def get_name(self) -> str:
        return f"city {self.city_id}"

    def army_back(self, soldier: int):
        self.soldier += soldier