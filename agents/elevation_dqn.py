import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from environment.elevation_sensor_env_dqn import ElevationSensorEnv

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparameters
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 32
BUFFER_SIZE = 10000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.995
TARGET_UPDATE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN Model
class DQN(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.model(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(np.array(state), dtype=torch.float32).to(device),
            torch.tensor(action, dtype=torch.int64).to(device),
            torch.tensor(reward, dtype=torch.float32).to(device),
            torch.tensor(np.array(next_state), dtype=torch.float32).to(device),
            torch.tensor(done, dtype=torch.float32).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy_net = DQN(obs_dim, action_dim).to(device)
        self.target_net = DQN(obs_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPS_START

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + (1 - dones) * GAMMA * next_q_values

        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

# Training loop
def train_dqn(env, num_episodes=500):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(obs_dim, action_dim)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        if episode % 10 == 0:
            print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

    torch.save(agent.policy_net.state_dict(), "models/dqn_model.pth")
    print("✅ Training complete. Model saved to models/dqn_model.pth")

if __name__ == "__main__":
    env = ElevationSensorEnv()
    train_dqn(env)
