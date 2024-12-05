import gym
import torch
import numpy as np
from torch.distributions import Categorical
from ppo_train import PolicyNetwork


def test():
    env = gym.make('ALE/Tetris-v5', render_mode="human")
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    model = PolicyNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load("tetris_ppo_gpu.pth"))
    model.eval()

    for episode in range(10):
        state = env.reset()[0].flatten()
        total_reward = 0

        while True:
            action_logits, _ = model(torch.FloatTensor(state))
            dist = Categorical(logits=action_logits)
            action = dist.sample().item()

            next_state, reward, done, _, _ = env.step(action)
            state = next_state.flatten()
            total_reward += reward

            if done:
                break

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")


if __name__ == "__main__":
    test()
