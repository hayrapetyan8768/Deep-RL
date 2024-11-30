import gym
import torch
from DQN_train import DQN, preprocess_frame, device


def demonstrate(env_name, model_path):
    env = gym.make(env_name, render_mode="human")
    input_shape = (1, 84, 84)
    n_actions = env.action_space.n

    model = DQN(input_shape, n_actions).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    state, _ = env.reset()
    state = preprocess_frame(state)
    total_reward = 0
    done = False

    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = torch.argmax(model(state_tensor), dim=1).item()

        next_state, reward, done, _, _ = env.step(action)
        next_state = preprocess_frame(next_state)
        total_reward += reward
        state = next_state
        if done:
            print(f"Demonstration complete, Total Reward: {total_reward}")
            break

    env.close()


if __name__ == "__main__":
    demonstrate("ALE/Tetris-v5", "dqn_tetris_best_model.pth")
