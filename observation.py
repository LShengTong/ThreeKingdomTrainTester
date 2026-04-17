from __future__ import annotations

from typing import Sequence

import torch

class Observations:
    def __init__(
        self,
        heroes: torch.Tensor,
        hero_mask: torch.Tensor,
        develops: torch.Tensor,
        working_assignments: torch.Tensor,
    ) -> None:
        self.heroes = heroes
        self.hero_mask = hero_mask
        self.develops = develops
        self.working_assignments = working_assignments

    def to(
        self,
        device: torch.device,
        *,
        dtype: torch.dtype | None = None,
    ) -> Observations:
        if dtype is None:
            return Observations(
                self.heroes.to(device),
                self.hero_mask.to(device),
                self.develops.to(device),
                self.working_assignments.to(device)
            )
        return Observations(
            self.heroes.to(device, dtype=dtype),
            self.hero_mask.to(device, dtype=dtype),
            self.develops.to(device, dtype=dtype),
            self.working_assignments.to(device)
        )

    @classmethod
    def stack(cls, observations: Sequence[Observation]) -> Observations:
        return cls(
            torch.stack([o.heroes for o in observations], dim=0),
            torch.stack([o.hero_mask for o in observations], dim=0),
            torch.stack([o.develops for o in observations], dim=0),
            torch.stack([o.working_assignments for o in observations], dim=0),
        )


class Observation:
    def __init__(
        self,
        heroes: torch.Tensor,
        hero_mask: torch.Tensor,
        develops: torch.Tensor,
        working_assignments: torch.Tensor,
    ) -> None:
        self.heroes = heroes
        self.hero_mask = hero_mask
        self.develops = develops
        self.working_assignments = working_assignments

    def detach_cpu(self) -> Observation:
        return Observation(
            self.heroes.detach().cpu(),
            self.hero_mask.detach().cpu(),
            self.develops.detach().cpu(),
            self.working_assignments.detach().cpu()
        )

    def to(
        self,
        device: torch.device,
        *,
        dtype: torch.dtype | None = None,
    ) -> Observations:
        if dtype is None:
            return Observations(
                self.heroes.to(device),
                self.hero_mask.to(device),
                self.develops.to(device),
                self.working_assignments.to(device)
            )
        return Observations(
            self.heroes.to(device, dtype=dtype),
            self.hero_mask.to(device, dtype=dtype),
            self.develops.to(device, dtype=dtype),
            self.working_assignments.to(device)
        )
