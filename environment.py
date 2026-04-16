import torch

class Environment:

    def __init__(self) -> None:
        self.max_heroes = 10
        self._heroAbilities = torch.tensor(
            [
                [75, 74, 73, 78],
                # [93, 102, 77, 64],
                [83, 103, 35, 22],
                # [88, 96, 78, 72],
                # [98, 34, 100, 98],
            ],
            dtype=torch.float32,
        )

        self._develops = torch.tensor(
            [
                [300, 1000],
                [300, 400],
                [300, 400],
                [300, 1000],
            ],
            dtype=torch.float32,
        )
        self._initial_develops = self._develops.clone()
        self._working_action = torch.full(
            (self._heroAbilities.shape[0],),
            fill_value=-1,
            dtype=torch.long,
        )

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
                abilities = self._heroAbilities[hero_ids, i].sum()
            develop = self._develops[i]
            develop_delta = int(
                (abilities / 10 + (develop[1] - develop[0]) * abilities / 3000).item()
            )
            self._develops[i, 0] += develop_delta
            reward += float(develop_delta)
        return reward

    def reset(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._develops = self._initial_develops.clone()
        self._working_action.fill_(-1)
        return self.get_observation()

    def get_observation(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idle_hero_ids = torch.where(self._working_action < 0)[0]
        idle_heroes = self._heroAbilities[idle_hero_ids]
        num_heroes, hero_dim = idle_heroes.shape
        padded_heroes = torch.zeros((self.max_heroes, hero_dim), dtype=self._heroAbilities.dtype)
        if num_heroes > 0:
            padded_heroes[:num_heroes] = idle_heroes

        hero_mask = torch.zeros((self.max_heroes,), dtype=torch.float32)
        hero_mask[:num_heroes] = 1.0

        return padded_heroes / 100, hero_mask, self._develops.clone() / 1000

    def step(
        self,
        develop_index: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], float, bool]:
        if develop_index < 0 or develop_index >= self._develops.shape[0]:
            raise IndexError("develop_index out of range.")

        unassigned = torch.where(self._working_action < 0)[0]
        if unassigned.numel() > 0:
            first_idle_hero = int(unassigned[0].item())
            self._working_action[first_idle_hero] = int(develop_index)
        if torch.all(self._working_action >= 0):
            reward = float(self._do_action(self._working_action))
            self._working_action.fill_(-1)
            done = True
        else:
            reward = 0
            done = False
        return self.get_observation(), reward, done
