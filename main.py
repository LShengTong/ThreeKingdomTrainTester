from bias_calculator import BiasCalculator
from deep_q_learning import DeepQLearningAgent
from environment import Environment
from torch.utils.tensorboard import SummaryWriter
import copy
import numpy as np
import torch

from transition import Transition

if "__main__" == __name__:
    use_writer = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_every = 8
    print(f"using device: {device}")

    env = Environment()
    obs = env.reset()

    if use_writer:
        writer = SummaryWriter(log_dir="runs/hero_develop_dqn")
    else:
        writer = None

    agent = DeepQLearningAgent(env=env, device=device)
    training_bias_calculator = BiasCalculator(agent.gamma, agent.online_net, agent.device)

    for step in range(50000):
        if step % 100 == 0:
            print(f"step={step}")

        develop_index = agent.act(obs)
        next_obs, reward, done = env.step(develop_index)
        transition = Transition(obs, develop_index, reward, next_obs, done, 0)
        agent.remember(transition)
        loss = None
        if step % train_every == 0:
            loss = agent.train_step()

        training_bias_calculator.add(transition)

        if writer is not None and loss is not None:
            writer.add_scalar("train/loss", loss, step)
        # print(f"loss={loss}")

        obs = next_obs
        if done:
            if writer is not None:
                writer.add_scalar("train/reward", reward, step)
            # print(f"reward={reward}")
            # print("Episode finished, reset environment.")
            obs = env.reset()

    training_bias_calculator.print_bias()

    test_bias_calculator = BiasCalculator(agent.gamma, agent.online_net, agent.device)
    obs = env.reset()
    for _ in range(2000):
        ep: list[Transition] = []
        develop_index = agent.act(obs)
        next_obs, reward, done = env.step(develop_index)
        transition = Transition(obs, develop_index, reward, next_obs, done, 0)
        test_bias_calculator.add(transition)
        obs = next_obs
        if done:
            obs = env.reset()
    test_bias_calculator.print_bias()

    env1 = Environment()
    env1.reset()
    env2 = copy.deepcopy(env1)

    real_rewards = np.zeros((env.action_n, env.action_n))
    for i in range(env1.action_n):
        envi = copy.deepcopy(env1)
        _, _, _ = envi.step(i)
        for j in range(env1.action_n):
            _, reward, _ = copy.deepcopy(envi).step(j)
            real_rewards[i, j] = reward
    print(f"real rewards={real_rewards}")

    reward = 0
    done = False
    while not done:
        act = int(torch.randint(0, 4, (1,))[0])
        _, reward, done = env1.step(act)
        print(f"random action={act}")
    print(f"random reward={reward}")

    obs = env2.get_observation()
    done = False
    while not done:
        act = agent.act(obs, False, True)
        obs, reward, done = env2.step(act)
        print(f"action={act}")
    print(f"reward={reward}")

    if writer is not None:
        writer.close()