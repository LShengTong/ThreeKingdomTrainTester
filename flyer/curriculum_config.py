from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyer.environment.config import GameConfig, FlyerConfig
from flyer.environment.flyer_type import FlyerType
from flyer.policies.policy_type import PolicyType


@dataclass(frozen=True)
class CurriculumStage:
    """Configuration for a single curriculum stage."""
    game_config: GameConfig
    green_policy: PolicyType

    # Curriculum advancement criteria
    target_win_rate: float  # win rate to advance to next stage
    test_episodes: int  # episodes to test win rate

    # Training parameters for this stage
    timesteps_per_iteration: int
    max_iterations: int  # max iterations before forced advancement

stages = curriculum_stages = [

    # region stage
    CurriculumStage(
        game_config=GameConfig(
            map_radius=10.,
            flyer_configs={
                FlyerType.RED: FlyerConfig(
                    max_attack_distance=10.,
                    max_attack_angle_degree=40.,
                    radar_radius=20.,
                    velocity=0.3,
                    position=np.array([-5., 0.]),
                    angle_degree=0.,
                    missile_count=3,
                    max_angle_change_velocity=20.,
                ),
                FlyerType.GREEN: FlyerConfig(
                    max_attack_distance=10.,
                    max_attack_angle_degree=40.,
                    radar_radius=20.,
                    velocity=0.,
                    position=np.array([5., 0.]),
                    angle_degree=180.,
                    missile_count=0,
                    max_angle_change_velocity=20.,
                )
            }
        ),
        green_policy=PolicyType.Stationary,
        target_win_rate=0.9,
        test_episodes=30,
        timesteps_per_iteration=10_000,
        max_iterations=20,
    ),
    #endregion

    # region stage
    CurriculumStage(
        game_config=GameConfig(
            map_radius=10.,
            flyer_configs={
                FlyerType.RED: FlyerConfig(
                    max_attack_distance=10.,
                    max_attack_angle_degree=40.,
                    radar_radius=20.,
                    velocity=0.3,
                    position=np.array([-5., 0.]),
                    angle_degree=180.,
                    missile_count=3,
                    max_angle_change_velocity=20.,
                ),
                FlyerType.GREEN: FlyerConfig(
                    max_attack_distance=10.,
                    max_attack_angle_degree=40.,
                    radar_radius=20.,
                    velocity=0.,
                    position=np.array([5., 0.]),
                    angle_degree=180.,
                    missile_count=0,
                    max_angle_change_velocity=20.,
                )
            }
        ),
        green_policy=PolicyType.Stationary,
        target_win_rate=0.9,
        test_episodes=30,
        timesteps_per_iteration=10_000,
        max_iterations=20,
    ),
    #endregion

    # region stage
    CurriculumStage(
        game_config=GameConfig(
            map_radius=10.,
            flyer_configs={
                FlyerType.RED: FlyerConfig(
                    max_attack_distance=10.,
                    max_attack_angle_degree=40.,
                    radar_radius=20.,
                    velocity=0.3,
                    position=np.array([-5., 0.]),
                    angle_degree=None,
                    missile_count=3,
                    max_angle_change_velocity=20.,
                ),
                FlyerType.GREEN: FlyerConfig(
                    max_attack_distance=10.,
                    max_attack_angle_degree=40.,
                    radar_radius=20.,
                    velocity=0.,
                    position=np.array([5., 0.]),
                    angle_degree=180.,
                    missile_count=0,
                    max_angle_change_velocity=20.,
                )
            }
        ),
        green_policy=PolicyType.Stationary,
        target_win_rate=0.9,
        test_episodes=30,
        timesteps_per_iteration=10_000,
        max_iterations=20,
    ),
    #endregion

    # region stage
    CurriculumStage(
        game_config=GameConfig(
            map_radius=10.,
            flyer_configs={
                FlyerType.RED: FlyerConfig(
                    max_attack_distance=10.,
                    max_attack_angle_degree=40.,
                    radar_radius=20.,
                    velocity=0.3,
                    position=None,
                    angle_degree=None,
                    missile_count=3,
                    max_angle_change_velocity=20.,
                ),
                FlyerType.GREEN: FlyerConfig(
                    max_attack_distance=10.,
                    max_attack_angle_degree=40.,
                    radar_radius=20.,
                    velocity=0.3,
                    position=np.array([5., 0.]),
                    angle_degree=180.,
                    missile_count=3,
                    max_angle_change_velocity=20.,
                )
            }
        ),
        green_policy=PolicyType.Aggressive,
        target_win_rate=0.9,
        test_episodes=30,
        timesteps_per_iteration=10_000,
        max_iterations=20,
    ),
    #endregion

    # # region stage
    # CurriculumStage(
    #     game_config=GameConfig(
    #         map_radius=10.,
    #         flyer_configs={
    #             FlyerType.RED: FlyerConfig(
    #                 max_attack_distance=10.,
    #                 max_attack_angle_degree=40.,
    #                 radar_radius=20.,
    #                 velocity=0.3,
    #                 position=np.array([-5., 0.]),
    #                 angle_degree=0.,
    #                 missile_count=3,
    #                 max_angle_change_velocity=20.,
    #             ),
    #             FlyerType.GREEN: FlyerConfig(
    #                 max_attack_distance=10.,
    #                 max_attack_angle_degree=40.,
    #                 radar_radius=20.,
    #                 velocity=0.3,
    #                 position=np.array([5., 0.]),
    #                 angle_degree=180.,
    #                 missile_count=3,
    #                 max_angle_change_velocity=20.,
    #             )
    #         }
    #     ),
    #     green_policy=PolicyType.Strategic,
    #     target_win_rate=0.75,
    #     test_episodes=30,
    #     timesteps_per_iteration=10_000,
    #     max_iterations=10,
    # ),
    # #endregion

    #
    # # region stage 3
    # CurriculumStage(
    #     game_config=GameConfig(
    #         map_radius=10.,
    #         flyer_configs={
    #             FlyerType.RED: FlyerConfig(
    #                 max_attack_distance=10.,
    #                 max_attack_angle_degree=40.,
    #                 radar_radius=8.,
    #                 velocity=0.3,
    #                 position=np.array([5., 0.]),
    #                 angle_degree=0.,
    #                 missile_count=3,
    #                 max_angle_change_velocity=20.,
    #             ),
    #             FlyerType.GREEN: FlyerConfig(
    #                 max_attack_distance=10.,
    #                 max_attack_angle_degree=40.,
    #                 radar_radius=8.,
    #                 velocity=0.,
    #                 position=np.array([-5., 0.]),
    #                 angle_degree=180.,
    #                 missile_count=0,
    #                 max_angle_change_velocity=20.,
    #             )
    #         }
    #     ),
    #     green_policy=PolicyType.Stationary,
    #     target_win_rate=0.75,
    #     test_episodes=30,
    #     timesteps_per_iteration=10_000,
    #     max_iterations=10,
    # ),
    # #endregion
    #
    # #region stage 4
    # CurriculumStage(
    #     game_config=GameConfig(
    #         map_radius=10.,
    #         flyer_configs={
    #             FlyerType.RED: FlyerConfig(
    #                 max_attack_distance=10.,
    #                 max_attack_angle_degree=40.,
    #                 radar_radius=8.,
    #                 velocity=0.3,
    #                 position=np.array([5., 0.]),
    #                 angle_degree=0.,
    #                 missile_count=3,
    #                 max_angle_change_velocity=20.,
    #             ),
    #             FlyerType.GREEN: FlyerConfig(
    #                 max_attack_distance=10.,
    #                 max_attack_angle_degree=40.,
    #                 radar_radius=8.,
    #                 velocity=0.3,
    #                 position=np.array([-5., 0.]),
    #                 angle_degree=180.,
    #                 missile_count=3,
    #                 max_angle_change_velocity=20.,
    #             )
    #         }
    #     ),
    #     green_policy=PolicyType.Strategic,
    #     target_win_rate=0.75,
    #     test_episodes=30,
    #     timesteps_per_iteration=10_000,
    #     max_iterations=10,
    # ),
    # #endregion
    #
    # #region stage 5
    # CurriculumStage(
    #     game_config=GameConfig(
    #         map_radius=20.,
    #         flyer_configs={
    #             FlyerType.RED: FlyerConfig(
    #                 max_attack_distance=10.,
    #                 max_attack_angle_degree=40.,
    #                 radar_radius=15.,
    #                 velocity=0.3,
    #                 position=np.array([-10., 0.]),
    #                 angle_degree=0.,
    #                 missile_count=3,
    #                 max_angle_change_velocity=20.,
    #             ),
    #             FlyerType.GREEN: FlyerConfig(
    #                 max_attack_distance=10.,
    #                 max_attack_angle_degree=40.,
    #                 radar_radius=15.,
    #                 velocity=0.3,
    #                 position=np.array([10., 0.]),
    #                 angle_degree=180.,
    #                 missile_count=3,
    #                 max_angle_change_velocity=20.,
    #             )
    #         }
    #     ),
    #     green_policy=PolicyType.Strategic,
    #     target_win_rate=0.75,
    #     test_episodes=30,
    #     timesteps_per_iteration=10_000,
    #     max_iterations=10,
    # ),
    # #endregion
    #
    # # region stage 6
    # CurriculumStage(
    #     game_config=GameConfig(
    #         map_radius=40.,
    #         flyer_configs={
    #             FlyerType.RED: FlyerConfig(
    #                 max_attack_distance=10.,
    #                 max_attack_angle_degree=40.,
    #                 radar_radius=30.,
    #                 velocity=0.3,
    #                 position=np.array([-20., 0.]),
    #                 angle_degree=0.,
    #                 missile_count=3,
    #                 max_angle_change_velocity=20.,
    #             ),
    #             FlyerType.GREEN: FlyerConfig(
    #                 max_attack_distance=10.,
    #                 max_attack_angle_degree=40.,
    #                 radar_radius=30.,
    #                 velocity=0.3,
    #                 position=np.array([20., 0.]),
    #                 angle_degree=180.,
    #                 missile_count=3,
    #                 max_angle_change_velocity=20.,
    #             )
    #         }
    #     ),
    #     green_policy=PolicyType.Strategic,
    #     target_win_rate=0.75,
    #     test_episodes=30,
    #     timesteps_per_iteration=10_000,
    #     max_iterations=10,
    # )
    # #endregion

]
