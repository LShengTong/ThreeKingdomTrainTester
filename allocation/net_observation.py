from __future__ import annotations

from typing import Sequence

import torch

from allocation.environment import Environment
from observation import Observation

class NetObservations:
    @classmethod
    def from_observations(cls, net_obs: Sequence[NetObservation]):
        todo_heroes = torch.stack([o.todo_heroes for o in net_obs], dim=0)
        todo_hero_masks = torch.stack([o.todo_hero_mask for o in net_obs], dim=0)
        develops = torch.stack([o.develops for o in net_obs], dim=0)
        curr_heroes = torch.stack([o.curr_hero for o in net_obs], dim=0)
        working_heroes = torch.stack([o.working_heroes for o in net_obs], dim=0)
        working_heroes_masks = torch.stack([o.working_heroes_mask for o in net_obs], dim=0)
        return cls(
            todo_heroes=todo_heroes,
            todo_hero_mask=todo_hero_masks,
            develops=develops,
            curr_hero=curr_heroes,
            working_heroes=working_heroes,
            working_heroes_mask=working_heroes_masks
        )

    def __init__(
            self,
            todo_heroes: torch.Tensor,
            todo_hero_mask: torch.Tensor,
            develops: torch.Tensor,
            curr_hero: torch.Tensor,
            working_heroes: torch.Tensor,
            working_heroes_mask: torch.Tensor
    ) -> None:
        self.todo_heroes = todo_heroes
        self.todo_hero_mask = todo_hero_mask
        self.develops = develops
        self.curr_hero = curr_hero
        self.working_heroes = working_heroes
        self.working_heroes_mask = working_heroes_mask

    def to(
        self,
        device: torch.device
    ) -> NetObservations:
        return NetObservations(
            self.todo_heroes.to(device),
            self.todo_hero_mask.to(device),
            self.develops.to(device),
            self.curr_hero.to(device),
            self.working_heroes.to(device),
            self.working_heroes_mask.to(device),
        )

class NetObservation:
    def __init__(
            self,
            todo_heroes: torch.Tensor,
            todo_hero_mask: torch.Tensor,
            develops: torch.Tensor,
            curr_hero: torch.Tensor,
            working_heroes: torch.Tensor,
            working_heroes_mask: torch.Tensor
    ) -> None:
        self.todo_heroes = todo_heroes
        self.todo_hero_mask = todo_hero_mask
        self.develops = develops
        self.curr_hero = curr_hero
        self.working_heroes_mask = working_heroes_mask
        self.working_heroes = working_heroes

    @classmethod
    def from_observation(cls, ob: Observation):
        heroes, develops, working_actions = ob
        max_heroes = Environment.max_heroes
        hero_count = int(heroes.shape[0])
        working_count = int((working_actions != -1).sum().item())
        padding_heroes = torch.zeros(
            (max_heroes - hero_count, heroes.shape[1]),
            device=heroes.device,
            dtype=heroes.dtype,
        )
        padded_heroes = torch.cat((heroes, padding_heroes), dim=0)

        todo_heroes = padded_heroes.clone()

        todo_hero_mask = torch.zeros(
            (max_heroes,), device=heroes.device, dtype=heroes.dtype
        )
        if working_count + 1 < hero_count:
            todo_hero_mask[working_count + 1 : hero_count] = 1

        develops = ob.develops[:, 1:2] - ob.develops[:, 0:1]
        curr_hero = heroes[working_count].unsqueeze(0).repeat(develops.shape[0], 1)
        for i in range(curr_hero.shape[0]):
            curr_hero[i, 0] = curr_hero[i, i % 4]
        curr_hero = curr_hero[:, [0]]

        working_heroes = todo_heroes.unsqueeze(0).repeat(develops.shape[0], 1, 1)
        working_heroes_mask = torch.zeros(
            (develops.shape[0], max_heroes),
            device=heroes.device,
            dtype=heroes.dtype,
        )
        for i in range(working_actions.shape[0]):
            develop_index = int(working_actions[i].item())
            if develop_index < 0:
                continue
            working_heroes[develop_index, i, 0] = working_heroes[
                develop_index, i, develop_index % 4
            ]
            working_heroes_mask[develop_index, i] = 1
        working_heroes = working_heroes[:, :, [0]]

        return NetObservation(
            todo_heroes=todo_heroes,
            todo_hero_mask=todo_hero_mask,
            develops=develops,
            curr_hero=curr_hero,
            working_heroes=working_heroes,
            working_heroes_mask=working_heroes_mask
        )

    def detach_cpu(self) -> NetObservation:
        return NetObservation(
            self.todo_heroes.detach().cpu(),
            self.todo_hero_mask.detach().cpu(),
            self.develops.detach().cpu(),
            self.curr_hero.detach().cpu(),
            self.working_heroes.detach().cpu(),
            self.working_heroes_mask.detach().cpu()
        )