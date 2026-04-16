from deep_q_learning import DeepQLearningAgent
from environment import Environment
from torch.utils.tensorboard import SummaryWriter

if "__main__" == __name__:
    env = Environment()
    obs = env.reset()
    writer = SummaryWriter(log_dir="runs/hero_develop_dqn")

    agent = DeepQLearningAgent(env=env)

    for step in range(5000):
        develop_index = agent.act(obs)
        next_obs, reward, done = env.step(develop_index)
        agent.remember(obs, develop_index, reward, next_obs, done)
        loss = agent.train_step(step == 4999)

        # print(
        #     f"step={step}, develop_index={develop_index}, "
        #     f"reward={reward:.2f}, loss={loss}"
        # )
        if loss is not None:
            writer.add_scalar("train/loss", loss, step)
            print(f"loss={loss}")

        obs = next_obs
        if done:
            writer.add_scalar("train/reward", reward, step)
            print(f"reward={reward}")
            print("Episode finished, reset environment.")
            obs = env.reset()

    writer.close()