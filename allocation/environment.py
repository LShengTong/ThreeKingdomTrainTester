import numpy as np

from allocation.config import DEFAULT_AGENT_CONFIG
from allocation.observation import Observation


class Environment:
    max_heroes = DEFAULT_AGENT_CONFIG.environment.max_heroes
    hero_dim = 4
    develop_shape = (20, 2)

    def __init__(self) -> None:
        self._heroAbilities = np.empty(
            (Environment.max_heroes, Environment.hero_dim),
            dtype=np.int64
        )
        self._develops = np.empty(Environment.develop_shape, dtype=np.int64)
        self._working_action = np.full(
            (Environment.max_heroes,),
            fill_value=-1,
            dtype=np.int64,
        )
        self.reset()

    @property
    def hero_count(self) -> int:
        return self._heroAbilities.shape[0]

    def _calc_develop_delta(self) -> np.ndarray:
        develop_abilities = np.zeros((self._develops.shape[0],), dtype=np.int64)
        for i in range(len(self._working_action)):
            working_action = self._working_action[i]
            if working_action < 0:
                break
            develop_abilities[working_action] += self._heroAbilities[i, working_action % 4]
        develop_deltas = np.zeros_like(develop_abilities)
        for i in range(len(develop_abilities)):
            abilities = develop_abilities[i]
            develop_delta = self._develops[i][1] - self._develops[i][0]
            develop_deltas[i] = (
                abilities / 10 + develop_delta * abilities / 3000)
        return develop_deltas

    def _do_action(self) -> float:
        develop_deltas = self._calc_develop_delta()
        for i in range(self._develops.shape[0]):
            self._develops[i, 0] += develop_deltas[i]
        return develop_deltas.sum()

    def reset(self) -> Observation:
        if Environment.max_heroes == 1:
            hero_count = 1
        else:
            hero_count = int(np.random.randint(1, Environment.max_heroes))
        self._heroAbilities = np.random.randint(
            low=0, high=101, size=(hero_count, Environment.hero_dim), dtype=np.int64
        )

        develop_start = np.random.randint(
            low=0, high=1000, size=(Environment.develop_shape[0],), dtype=np.int64
        )
        max_delta = 1000 - develop_start
        develop_delta_float = np.random.random(Environment.develop_shape[0]) * max_delta
        develop_delta = np.array(develop_delta_float, dtype=np.int64) + 1
        develop_end = develop_start + develop_delta
        self._develops = np.stack((develop_start, develop_end), axis=1)
        self._working_action = np.full((hero_count,), fill_value=-1, dtype=np.int64)
        return self.get_observation()

    def get_observation(self) -> Observation:
        heroes = np.asarray(self._heroAbilities / 100.0, dtype=np.float32)
        develops = np.asarray(self._develops / 1000.0, dtype=np.float32)
        return Observation(
            heroes,
            develops,
            self._working_action.copy(),
        )

    def step(
        self,
        develop_index: int,
    ) -> tuple[Observation, float, bool]:
        if develop_index < 0 or develop_index >= self._develops.shape[0]:
            raise IndexError("develop_index out of range.")

        unassigned = np.where(self._working_action < 0)[0]
        if unassigned.size > 0:
            first_idle_hero = int(unassigned[0])
            self._working_action[first_idle_hero] = int(develop_index)
        if int((self._working_action != -1).sum()) >= self._heroAbilities.shape[0]:
            reward = float(self._do_action())
            self._working_action.fill(-1)
            done = True
        else:
            ability = self._heroAbilities[int(unassigned[0]), develop_index % 4]
            develop_delta = self._develops[develop_index][1] - self._develops[develop_index][0]
            reward = ability / 10 + develop_delta * ability / 3000
            done = False
        return self.get_observation(), reward, done
