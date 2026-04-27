from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from allocation.environment import Environment
from allocation.observation import Observation


class NetObservations:

    @classmethod
    def from_observations(cls, net_obs: Sequence[NetObservation], device: torch.device)\
            -> NetObservations:
        todo_heroes = np.stack([o.todo_heroes for o in net_obs], axis=0)
        todo_hero_masks = np.stack([o.todo_hero_mask for o in net_obs], axis=0)
        develops = np.stack([o.develops for o in net_obs], axis=0)
        curr_heroes = np.stack([o.curr_hero for o in net_obs], axis=0)
        working_heroes = np.stack([o.working_heroes for o in net_obs], axis=0)
        working_heroes_masks = np.stack([o.working_heroes_mask for o in net_obs], axis=0)
        return cls(
            todo_heroes=torch.from_numpy(todo_heroes).to(device),
            todo_hero_mask=torch.from_numpy(todo_hero_masks).to(device),
            develops=torch.from_numpy(develops).to(device),
            curr_hero=torch.from_numpy(curr_heroes).to(device),
            working_heroes=torch.from_numpy(working_heroes).to(device),
            working_heroes_mask=torch.from_numpy(working_heroes_masks).to(device),
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

    def to_dict(self):
        return {
            "todo_heroes": self.todo_heroes,
            "todo_hero_mask": self.todo_hero_mask,
            "develops": self.develops,
            "curr_hero": self.curr_hero,
            "working_heroes": self.working_heroes,
            "working_heroes_mask": self.working_heroes_mask,
        }


class NetObservation:
    def __init__(
            self,
            todo_heroes: np.ndarray,
            todo_hero_mask: np.ndarray,
            develops: np.ndarray,
            curr_hero: np.ndarray,
            working_heroes: np.ndarray,
            working_heroes_mask: np.ndarray
    ) -> None:
        self.todo_heroes = todo_heroes
        self.todo_hero_mask = todo_hero_mask
        self.develops = develops
        self.curr_hero = curr_hero
        self.working_heroes_mask = working_heroes_mask
        self.working_heroes = working_heroes

    def to_dict(self):
        return {
            "todo_heroes": self.todo_heroes,
            "todo_hero_mask": self.todo_hero_mask,
            "develops": self.develops,
            "curr_hero": self.curr_hero,
            "working_heroes": self.working_heroes,
            "working_heroes_mask": self.working_heroes_mask,
        }

    @classmethod
    def from_observation(cls, ob: Observation):
        heroes, develops, working_actions = ob
        heroes = np.asarray(heroes, dtype=np.float32)
        develops = np.asarray(develops, dtype=np.float32)
        working_actions = np.asarray(working_actions, dtype=np.int64)
        max_heroes = Environment.max_heroes
        hero_count = int(heroes.shape[0])
        working_count = int(np.sum(working_actions != -1))
        padding_heroes = np.zeros(
            (max_heroes - hero_count, heroes.shape[1]),
            dtype=heroes.dtype,
        )
        padded_heroes = np.concatenate((heroes, padding_heroes), axis=0)

        todo_heroes = padded_heroes.copy()

        todo_hero_mask = np.zeros((max_heroes,), dtype=heroes.dtype)
        if working_count + 1 < hero_count:
            todo_hero_mask[working_count + 1 : hero_count] = 1

        develops = develops[:, 1:2] - develops[:, 0:1]
        curr_hero = np.repeat(heroes[working_count][None, :], develops.shape[0], axis=0)
        for i in range(curr_hero.shape[0]):
            curr_hero[i, 0] = curr_hero[i, i % 4]
        curr_hero = curr_hero[:, [0]]

        working_heroes = np.repeat(todo_heroes[None, :, :], develops.shape[0], axis=0)
        working_heroes_mask = np.zeros(
            (develops.shape[0], max_heroes),
            dtype=heroes.dtype,
        )
        for i in range(working_actions.shape[0]):
            develop_index = int(working_actions[i])
            if develop_index < 0:
                continue
            working_heroes[develop_index, i, 0] = working_heroes[
                develop_index, i, develop_index % 4
            ]
            working_heroes_mask[develop_index, i] = 1
        working_heroes = working_heroes[:, :, [0]]

        return NetObservation(
            todo_heroes=np.asarray(todo_heroes, dtype=np.float32),
            todo_hero_mask=np.asarray(todo_hero_mask, dtype=np.float32),
            develops=np.asarray(develops, dtype=np.float32),
            curr_hero=np.asarray(curr_hero, dtype=np.float32),
            working_heroes=np.asarray(working_heroes, dtype=np.float32),
            working_heroes_mask=np.asarray(working_heroes_mask, dtype=np.float32),
        )