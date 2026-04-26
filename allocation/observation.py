from __future__ import annotations

from typing import Iterator

import torch


class Observation:
    def __init__(
        self,
        heroes: torch.Tensor,
        develops: torch.Tensor,
        working_actions: torch.Tensor,
    ) -> None:
        self.heroes = heroes
        self.develops = develops
        self.working_actions = working_actions

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.heroes
        yield self.develops
        yield self.working_actions
