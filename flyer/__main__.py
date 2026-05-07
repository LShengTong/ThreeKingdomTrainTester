from flyer.curriculum_trainer import CurriculumTrainer


def main() -> None:
    trainer = CurriculumTrainer(
        model_dir="./models",
        log_dir="./logs",
    )

    # Run training
    trainer.train()

    # Final evaluation
    trainer.evaluate_final(num_episodes=100)


if __name__ == "__main__":
    main()
