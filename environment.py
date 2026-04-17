import torch

from observation import Observation


class Environment:

    def __init__(self) -> None:
        self.max_heroes = 10
        self._hero_shape = (1, 4)
        self._develop_shape = (20, 2)
        # self._heroAbilities = torch.tensor(
        #     [
        #         [75, 74, 73, 78],
        #         [93, 102, 77, 64],
        #         [83, 103, 35, 22],
        #         [88, 96, 78, 72],
        #         [98, 34, 100, 98],
        #     ],
        #     dtype=torch.float32,
        # )
        self._heroAbilities = torch.empty(self._hero_shape, dtype=torch.float32)

        # self._develops = torch.tensor(
        #     [
        #         [300, 1000],
        #         [300, 400],
        #         [300, 400],
        #         [300, 1000],
        #     ],
        #     dtype=torch.float32,
        # )
        self._develops = torch.empty(self._develop_shape, dtype=torch.float32)
        self._working_action = torch.full(
            (self.max_heroes,),
            fill_value=-1,
            dtype=torch.long,
        )
        self.reset()

    @property
    def action_n(self):
        return self._develops.shape[0]

    @property
    def hero_dim(self):
        return self._heroAbilities.shape[1]

    @property
    def develop_dim(self):
        return self._develops.shape[1]

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
            self._develops[i, 0] += develop_delta
            reward += float(develop_delta)
        return reward

    def reset(self) -> Observation:
        self._heroAbilities = torch.randint(
            low=0,
            high=101,
            size=self._hero_shape,
            dtype=torch.int64,
        ).to(torch.float32)

        develop_start = torch.randint(
            low=0,
            high=1000,
            size=(self._develop_shape[0],),
            dtype=torch.int64,
        )
        max_delta = 1000 - develop_start
        develop_delta = (
            torch.floor(torch.rand(self._develop_shape[0]) * max_delta.to(torch.float32))
            .to(torch.int64)
            + 1
        )
        develop_end = develop_start + develop_delta
        self._develops = torch.stack((develop_start, develop_end), dim=1).to(torch.float32)
        self._working_action.fill_(-1)
        return self.get_observation()

    def get_observation(self) -> Observation:
        num_heroes, hero_dim = self._heroAbilities.shape
        padded_heroes = torch.zeros((self.max_heroes, hero_dim), dtype=self._heroAbilities.dtype)
        if num_heroes > 0:
            padded_heroes[:num_heroes] = self._heroAbilities

        hero_mask = torch.zeros((self.max_heroes,), dtype=torch.float32)
        hero_mask[:num_heroes] = 1.0

        return Observation(
            padded_heroes / 100,
            hero_mask,
            self._develops.clone() / 1000,
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
        if (self._working_action != -1).sum() >= self._hero_shape[0]:
            reward = float(self._do_action(self._working_action))
            self._working_action.fill_(-1)
            done = True
        else:
            reward = 0
            done = False
        return self.get_observation(), reward, done
