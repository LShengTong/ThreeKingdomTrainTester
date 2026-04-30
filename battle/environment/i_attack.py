from abc import abstractmethod, ABC

import torch


class IAttack(ABC):
    @abstractmethod
    def attack(self) -> int:
        pass

    @abstractmethod
    def defense(self, value: int):
        pass

    @abstractmethod
    def get_pos(self) -> torch.Tensor:
        pass

    @abstractmethod
    def is_alive(self) -> bool:
        pass

    @abstractmethod
    def get_city_id(self) -> int:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass