#!/usr/bin/env python3
"""Curriculum learning trainer for flyer PPO agent."""

from __future__ import annotations

import os

from stable_baselines3 import PPO

from flyer.config import FlyerPPOConfig, NetConfig
from flyer.curriculum_config import stages
from flyer.flyer_feature_extractor import FlyerFeatureExtractor
from flyer.gym_env import GymEnv


class CurriculumTrainer:
    """Train PPO agent with curriculum learning."""

    def __init__(
        self,
        model_dir: str = "./models",
        log_dir: str = "./logs",
    ):
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.current_stage_idx = 0

        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Initialize environment and model (will be updated as stages progress)
        self.env = GymEnv()
        self.model: PPO | None = None

    def _create_model(self) -> PPO:
        """Create or load PPO model with custom FlyerActorCriticPolicy."""
        os.makedirs(FlyerPPOConfig.tensorboard_log, exist_ok=True)

        return PPO(
            policy="MlpPolicy",
            policy_kwargs=dict(
                net_arch=NetConfig.arch,
                features_extractor_class = FlyerFeatureExtractor,
            ),
            env=self.env,
            learning_rate=FlyerPPOConfig.lr,
            n_steps=FlyerPPOConfig.n_steps,
            batch_size=FlyerPPOConfig.batch_size,
            n_epochs=FlyerPPOConfig.n_epochs,
            gamma=FlyerPPOConfig.gamma,
            gae_lambda=FlyerPPOConfig.gae_lambda,
            clip_range=FlyerPPOConfig.clip_range,
            ent_coef=FlyerPPOConfig.ent_coef,
            vf_coef=FlyerPPOConfig.vf_coef,
            max_grad_norm=FlyerPPOConfig.max_grad_norm,
            device=FlyerPPOConfig.device,
            tensorboard_log=FlyerPPOConfig.tensorboard_log,
            verbose=0,
        )

    def _get_model_path(self, stage_id: int) -> str:
        return os.path.join(self.model_dir, f"flyer_ppo_stage_{stage_id}.zip")

    def _save_model(self, stage_id: int) -> None:
        """Save current model."""
        if self.model is not None:
            model_path = self._get_model_path(stage_id)
            self.model.save(model_path)
            print(f"Model saved to {model_path}")

    def _test_win_rate(self, stage_id: int) -> float:
        """Test current win rate against curriculum stage."""
        if self.model is None:
            raise Exception("trainer has no model to test")
        stage = stages[stage_id]
        print(f"\n{'='*50}")
        print(f"Testing win rate for stage {stage_id}")
        print(f"Running {stage.test_episodes} test episodes...")

        wins = 0
        draws = 0
        losses = 0

        for episode in range(stage.test_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, _, _ = self.env.step(action)
                episode_reward = reward
                done = terminated
            if episode_reward == 0.:
                draws += 1
            elif episode_reward > 0:
                wins += 1
            else:
                losses += 1

        total = stage.test_episodes
        win_rate = wins / total

        print(f"Results: {wins} wins, {draws} draws, {losses} losses")
        print(f"Win rate: {win_rate:.2%} (target: {stage.target_win_rate:.2%})")
        print(f"{'='*50}\n")

        return win_rate

    def train_stage(self, stage_id: int) -> bool:
        """Train on a single curriculum stage until win rate target is met.
        
        Returns:
            True if stage passed (win rate达标), False if max iterations reached without passing
        """
        stage = stages[stage_id]
        print(f"\n{'#'*60}")
        print(f"# Starting Stage {stage_id}")
        print(f"# Map radius: {stage.game_config.map_radius}")
        print(f"# GREEN policy: {stage.green_policy}")
        print(f"# Target win rate: {stage.target_win_rate:.2%}")
        print(f"{'#'*60}\n")

        # Set environment stage
        self.env.set_stage(stage)

        # Load or create model
        if stage_id == 0:
            self.model = self._create_model()

        iteration = 0
        total_steps = 0

        while iteration < stage.max_iterations:
            iteration += 1
            print(f"\n--- Stage {stage_id}, Iteration {iteration}/{stage.max_iterations} ---")

            # Train
            timesteps = stage.timesteps_per_iteration
            if self.model is None:
                raise Exception("curriculum trainer: has no model")
            self.model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
            total_steps += timesteps

            # Save checkpoint
            self._save_model(stage_id)

            # Test win rate
            win_rate = self._test_win_rate(stage_id)

            # Check if passed
            if win_rate >= stage.target_win_rate:
                print(f"✓ Stage {stage_id} PASSED! Win rate {win_rate:.2%} >= {stage.target_win_rate:.2%}")
                return True

            print(f"✗ Stage {stage_id} not passed yet. Win rate {win_rate:.2%} < {stage.target_win_rate:.2%}")

        print(f"⚠ Stage {stage_id} max iterations reached without passing target win rate")
        return False

    def train(self) -> None:
        """Run full curriculum training."""
        print("=" * 70)
        print("Starting Curriculum Learning Training")
        print(f"Total stages: {len(stages)}")
        print("=" * 70)

        for stage_idx in range(len(stages)):
            self.current_stage_idx = stage_idx
            passed = self.train_stage(stage_idx)

            if not passed and stage_idx < len(stages) - 1:
                print(f"Stage {stage_idx} not passed, but continuing to next stage...")
            elif not passed:
                print(f"Final stage {stage_idx} not completed. Training finished.")
                break

        print("\n" + "=" * 70)
        print("Curriculum Training Complete!")
        print(f"Final model saved at: {self._get_model_path(self.current_stage_idx)}")
        print("=" * 70)

    def evaluate_final(self, num_episodes: int = 100) -> dict[str, float]:
        """Evaluate final model on all stages."""
        if self.model is None:
            raise Exception("trainer has no model to evaluate final")
        print("\n" + "=" * 70)
        print("Final Evaluation on All Stages")
        print("=" * 70)

        results = {}

        for stage_id in range(len(stages)):
            stage = stages[stage_id]
            print(f"\nEvaluating on Stage {stage_id}")
            self.env.set_stage(stage)

            wins = 0
            for _ in range(num_episodes):
                obs, _ = self.env.reset()
                done = False
                episode_reward = 0.0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, _, _ = self.env.step(action)
                    episode_reward = reward
                    done = terminated

                if episode_reward > 0:
                    wins += 1

            win_rate = wins / num_episodes
            results[f"stage_{stage_id}"] = win_rate
            print(f"Win rate: {win_rate:.2%}")

        print("\n" + "=" * 70)
        print("Final Evaluation Summary:")
        for stage_name, win_rate in results.items():
            print(f"  {stage_name}: {win_rate:.2%}")
        print("=" * 70)

        return results
