import torch
import numpy as np
from environment.elevation_sensor_env_dqn import ElevationSensorEnv
from agents.elevation_dqn import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
env = ElevationSensorEnv()
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = DQN(obs_dim, action_dim).to(device)
model.load_state_dict(torch.load("models/dqn_model.pth", map_location=device))
model.eval()

total_rewards = []
sensor_usages = []

for episode in range(50):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax().item()

        state, reward, done, _ = env.step(action)
        episode_reward += reward

    total_rewards.append(episode_reward)
    sensor_usages.append(np.sum(env.active_sensors))
    print(f"Episode {episode} | Total Reward: {episode_reward:.2f} | Sensors Used: {np.sum(env.active_sensors)}")

print("\n📊 Evaluation Summary")
print(f"Average Reward: {np.mean(total_rewards):.2f}")
print(f"Max Reward: {np.max(total_rewards):.2f}")
print(f"Min Reward: {np.min(total_rewards):.2f}")
print(f"Average Sensors Used: {np.mean(sensor_usages):.2f}")
