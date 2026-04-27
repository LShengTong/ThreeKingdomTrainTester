from __future__ import annotations

from typing import Iterator

import numpy as np


class Observation:
    def __init__(
        self,
        heroes: np.ndarray,
        develops: np.ndarray,
        working_actions: np.ndarray,
    ) -> None:
        self.heroes = heroes
        self.develops = develops
        self.working_actions = working_actions

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.heroes
        yield self.develops
        yield self.working_actions
