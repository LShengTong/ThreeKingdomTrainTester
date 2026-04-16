from deep_q_learning import DeepQLearningAgent
from environment import Environment
from torch.utils.tensorboard import SummaryWriter

if "__main__" == __name__:
    use_writer = False

    env = Environment()
    obs = env.reset()

    writer = None
    if use_writer:
        writer = SummaryWriter(log_dir="runs/hero_develop_dqn")

    agent = DeepQLearningAgent(env=env)

    for step in range(2000):
        develop_index = agent.act(obs)
        next_obs, reward, done = env.step(develop_index)
        agent.remember(obs, develop_index, reward, next_obs, done)
        loss = agent.train_step()

        # print(
        #     f"step={step}, develop_index={develop_index}, "
        #     f"reward={reward:.2f}, loss={loss}"
        # )
        if loss is not None:
            if writer is not None:
                writer.add_scalar("train/loss", loss, step)
            print(f"loss={loss}")

        obs = next_obs
        if done:
            if writer is not None:
                writer.add_scalar("train/reward", reward, step)
            print(f"reward={reward}")
            print("Episode finished, reset environment.")
            obs = env.reset()

    obs = env.reset()
    reward = 0
    for i in range(5):
        act = agent.act(obs, False, True)
        obs, reward, _ = env.step(act)
        print(f"action={act}")
    print(f"reward={reward}")

    if writer is not None:
        writer.close()