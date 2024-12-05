import gym
import torch
import numpy as np
from dqn_train import DQN


def test():
    env = gym.make('ALE/Tetris-v5', render_mode="human")
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    model = DQN(state_dim, action_dim)
    model.load_state_dict(torch.load("tetris_dqn_gpu.pth"))
    model.eval()

    for episode in range(10):
        state = env.reset()[0]
        state = state.flatten()
        total_reward = 0

        while True:
            q_values = model(torch.FloatTensor(state).unsqueeze(0))
            action = torch.argmax(q_values).item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = next_state.flatten()
            total_reward += reward
            state = next_state

            if done:
                break

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")


if __name__ == "__main__":
    test()
