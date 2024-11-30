import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray_frame, (84, 84)) / 255.0
    return np.expand_dims(resized_frame, axis=0)


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_)
        )

    def size(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, input_shape, n_actions, lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=100):
        self.model = DQN(input_shape, n_actions).to(device)
        self.target_model = DQN(input_shape, n_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.n_actions = n_actions

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = self.model(state)
        return torch.argmax(q_values, dim=1).item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.steps_done / self.epsilon_decay)


def train(env_name="ALE/Tetris-v5", total_episodes=2000, batch_size=128, buffer_capacity=50000, update_target_steps=1000):
    env = gym.make(env_name, render_mode="rgb_array")
    input_shape = (1, 84, 84)  # Grayscale frame shape
    n_actions = env.action_space.n

    agent = DQNAgent(input_shape, n_actions)
    buffer = ReplayBuffer(buffer_capacity)

    best_reward = -float('inf')
    model_save_path = "dqn_tetris_best_model"
    reward_window = deque(maxlen=100)

    for episode in range(total_episodes):
        state, _ = env.reset()
        state = preprocess_frame(state)
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocess_frame(next_state)

            buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if buffer.size() >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.tensor(states).to(device)
                actions = torch.tensor(actions).to(device)
                rewards = torch.tensor(rewards).to(device)
                next_states = torch.tensor(next_states).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                q_values = agent.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                next_q_values = agent.target_model(next_states).max(1)[0]
                target_q_values = rewards + agent.gamma * next_q_values * (1 - dones)

                loss = nn.functional.mse_loss(q_values, target_q_values)
                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()

            if agent.steps_done % update_target_steps == 0:
                agent.update_target_network()

            if done:
                agent.decay_epsilon()
                break

        reward_window.append(total_reward)
        avg_reward = np.mean(reward_window)

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.model.state_dict(), f"{model_save_path}_{episode}.pth")
            print(f"New best model saved with reward: {best_reward}")
            print(f"Episode {episode}, Total Reward: {total_reward}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
    torch.save(agent.model.state_dict(), f"{model_save_path}_last.pth")
    env.close()


if __name__ == "__main__":
    train()
