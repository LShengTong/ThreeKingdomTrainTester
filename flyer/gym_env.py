from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import Env, spaces

from flyer.curriculum_config import CurriculumStage
from flyer.environment.attack import Attack
from flyer.environment.config import FlyerObservationUpperBound, GameConfig
from flyer.environment.environment import Environment, FlyerAction
from flyer.environment.flyer_type import FlyerType
from flyer.policies import PolicyFactory, IPolicy

FLYER_OBS_KEY = "flyer"
FLYER_SELF_DIM = 9
FLYER_TARGET_DIM = 8
ANGLE_ACTION_DIM = 5

MAX_STEPS = 1000

class GymEnv(Env[np.ndarray, np.ndarray]):

    def __init__(self) -> None:
        super().__init__()
        self.inner_env = Environment()
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(FLYER_SELF_DIM + FLYER_TARGET_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([ANGLE_ACTION_DIM, 2])
        # self.action_space = spaces.Box(
        #     low=-1.0,
        #     high=1.0,
        #     shape=(1, ),
        #     dtype=np.float32,
        # )
        self.game_config: GameConfig | None = None
        self.green_policy: IPolicy | None = None
        self.steps: int = 0

    def set_stage(self, curriculum_stage: CurriculumStage) -> None:
        self.inner_env.set_game_config(curriculum_stage.game_config)
        self.game_config = curriculum_stage.game_config
        self.green_policy = PolicyFactory.create_policy_by_type(curriculum_stage.green_policy, self.inner_env.flyers[FlyerType.GREEN], curriculum_stage.game_config.map_radius)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None)\
            -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        # print("reset")
        del options
        self.steps = 0
        self.inner_env.reset()
        return self._get_flyer_observation(), {}


    def _get_flyer_observation(self) -> np.ndarray:
        if self.game_config is None:
            raise Exception("gym env not set game config")
        flyer = self.inner_env.flyers[FlyerType.RED]
        config = self.game_config.flyer_configs[FlyerType.RED]
        upper_bound = FlyerObservationUpperBound
        max_distance = upper_bound.map_radius * 2
        self_info = np.array([
            config.max_attack_distance / max_distance,
            config.max_angle_change_velocity / 360,
            config.radar_radius / max_distance,
            config.max_attack_angle_degree / 180,
            flyer.position[0] / upper_bound.map_radius,
            flyer.position[1] / upper_bound.map_radius,
            flyer.missile_count / upper_bound.missile_count,
            flyer.angle_degree / 360,
            flyer.velocity / upper_bound.velocity,
        ], dtype=np.float32)
        target = flyer.target
        if target is not None:
            delta_angle = np.abs(target.angle_degree - flyer.angle_degree)
            delta_angle = delta_angle if delta_angle > 180 else 360 - delta_angle
            target_info = np.array([
                1,
                target.position[0] / upper_bound.map_radius,
                target.position[1] / upper_bound.map_radius,
                target.velocity,
                target.angle_degree / 360,
                float(np.linalg.norm(target.position - flyer.position)) / upper_bound.map_radius,
                delta_angle / 180,
                Attack(flyer, target).hit_on_probability()
            ], dtype=np.float32)
        else:
            target_info = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        return np.concatenate((self_info, target_info))

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.green_policy is None:
            raise Exception("gym env not set green policy")

        flyer = self.inner_env.flyers[FlyerType.RED]
        target = flyer.target
        self.steps += 1
        if self.steps > MAX_STEPS:
            return self._get_flyer_observation(), 0., True, False, {}

        old_attack_probability = 0.
        if target is not None:
            old_attack_probability = Attack(flyer, target).hit_on_probability()

        change_angle = 2. * action[0] / (ANGLE_ACTION_DIM - 1) - 1.
        done, winner = self.inner_env.step({
            FlyerType.RED: FlyerAction(change_angle=change_angle, launch_missile=(action[1] == 1)),
            FlyerType.GREEN: self.green_policy.get_action(),
        })
        if done:
            if winner is None:
                reward = 0.
            elif winner == FlyerType.RED:
                reward = 1.
            else:
                reward = -1.
        else:
            # if target is not None:
            #     reward = 0.
            #     hit_on = Attack(flyer, target).hit_on_probability()
            #     reward += hit_on - old_attack_probability
            #     if action[1] == 1:
            #         reward -= 1. - hit_on
            # elif action[1] == 1:
            #     reward = -10
            # else:
            #     reward = 0.
            reward = 0.
        # print(done, action, reward)
        return self._get_flyer_observation(), reward, done, False, {}
