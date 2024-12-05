import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.action_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        action_logits = self.action_head(x)
        state_value = self.value_head(x)
        return action_logits, state_value


def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    next_value = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        delta = r + gamma * next_value - v
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = v
    return advantages


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('ALE/Tetris-v5', render_mode=None)  # Tetris environment
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    model = PolicyNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    gamma = 0.99
    lam = 0.95
    clip_ratio = 0.2
    epochs = 1000
    steps_per_epoch = 2000

    for epoch in tqdm(range(epochs)):
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        state = env.reset()[0].flatten()
        for _ in range(steps_per_epoch):
            state_tensor = torch.FloatTensor(state).to(device)
            action_logits, value = model(state_tensor)
            dist = Categorical(logits=action_logits)
            action = dist.sample().item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = next_state.flatten()

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(dist.log_prob(torch.tensor(action).to(device)).item())
            dones.append(done)

            state = next_state if not done else env.reset()[0].flatten()

            if done:
                break

        values = values + [0]
        advantages = compute_advantages(rewards, values, gamma, lam)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        log_probs = torch.FloatTensor(log_probs).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = advantages + torch.FloatTensor(values[:-1]).to(device)

        for _ in range(epochs):
            action_logits, state_values = model(states)
            dist = Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            policy_loss = torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            value_loss = ((state_values.squeeze() - returns) ** 2).mean()

            loss = -policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "tetris_ppo_gpu.pth")
    print("Model saved as tetris_ppo_gpu.pth")


if __name__ == "__main__":
    train()
