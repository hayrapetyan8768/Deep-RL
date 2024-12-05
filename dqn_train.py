import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from tqdm import tqdm

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('ALE/Tetris-v5', render_mode=None)
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(10000)

    episodes = 500
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    update_target_steps = 10

    for episode in tqdm(range(episodes)):
        state = env.reset()[0]
        state = state.flatten()
        total_reward = 0

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = next_state.flatten()
            total_reward += reward

            buffer.add(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_model(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = loss_fn(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % update_target_steps == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    torch.save(model.state_dict(), "tetris_dqn_gpu.pth")
    print("Model saved as tetris_dqn_gpu.pth")


if __name__ == "__main__":
    train()
