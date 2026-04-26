import torch

from observation import Observation


class Environment:
    max_heroes = 20
    hero_dim = 4
    develop_shape = (20, 2)

    def __init__(self) -> None:
        self._heroAbilities = torch.empty(
            (Environment.max_heroes, Environment.hero_dim),
            dtype=torch.float32
        )
        self._develops = torch.empty(Environment.develop_shape, dtype=torch.float32)
        self._working_action = torch.full(
            (Environment.max_heroes,),
            fill_value=-1,
            dtype=torch.long,
        )
        self.reset()

    def _do_action(self, action: torch.Tensor) -> float:
        action = action[:self._heroAbilities.shape[0]]
        reward = 0.0
        for i in range(self._develops.shape[0]):
            hero_ids = torch.where(action == i)[0]
            if hero_ids.numel() == 0:
                abilities = torch.tensor(0.0, dtype=self._heroAbilities.dtype)
            else:
                abilities = self._heroAbilities[hero_ids, i % 4].sum()
            develop = self._develops[i]
            develop_delta = int(
                (abilities / 10 + (develop[1] - develop[0]) * abilities / 3000).item()
            )
            develop_delta = min(develop_delta, self._develops[i, 1] - self._develops[i, 0])
            self._develops[i, 0] += develop_delta
            reward += float(develop_delta)
        return reward

    def reset(self) -> Observation:
        hero_count = int(torch.randint(1, Environment.max_heroes, (1,))[0])
        self._heroAbilities = torch.randint(
            low=0,
            high=101,
            size=(hero_count, Environment.hero_dim),
            dtype=torch.int64,
        ).to(torch.float32)

        develop_start = torch.randint(
            low=0,
            high=1000,
            size=(Environment.develop_shape[0],),
            dtype=torch.int64,
        )
        max_delta = 1000 - develop_start
        develop_delta = (
            torch.floor(torch.rand(Environment.develop_shape[0]) * max_delta.to(torch.float32))
            .to(torch.int64)
            + 1
        )
        develop_end = develop_start + develop_delta
        self._develops = torch.stack((develop_start, develop_end), dim=1).to(torch.float32)
        self._working_action = torch.full(
            (hero_count,),
            fill_value=-1,
            dtype=torch.long,
        )
        return self.get_observation()

    def get_observation(self) -> Observation:
        num_heroes, hero_dim = self._heroAbilities.shape
        padded_heroes = torch.zeros((self.max_heroes, hero_dim), dtype=self._heroAbilities.dtype)
        if num_heroes > 0:
            padded_heroes[:num_heroes] = self._heroAbilities

        hero_mask = torch.zeros((self.max_heroes,), dtype=torch.float32)
        hero_mask[:num_heroes] = 1.0

        return Observation(
            self._heroAbilities / 100,
            self._develops / 1000,
            self._working_action.clone(),
        )

    def step(
        self,
        develop_index: int,
    ) -> tuple[Observation, float, bool]:
        if develop_index < 0 or develop_index >= self._develops.shape[0]:
            raise IndexError("develop_index out of range.")

        unassigned = torch.where(self._working_action < 0)[0]
        if unassigned.numel() > 0:
            first_idle_hero = int(unassigned[0].item())
            self._working_action[first_idle_hero] = int(develop_index)
        if (self._working_action != -1).sum() >= self._heroAbilities.shape[0]:
            reward = float(self._do_action(self._working_action))
            self._working_action.fill_(-1)
            done = True
        else:
            reward = 0
            done = False
        return self.get_observation(), reward, done
