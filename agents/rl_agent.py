from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from environment.sensor_env import SensorEnv
import os

# Setup logging directory
os.makedirs("data/results", exist_ok=True)

# Create and wrap the environment
env = SensorEnv()
env = Monitor(env, filename="data/results/")

check_env(env)  # Optional: verifies Gym compliance

# Define PPO with tuned hyperparameters
model = PPO("MlpPolicy", env, verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.95,
            gae_lambda=0.9,
            ent_coef=0.005)

# Train for more timesteps
model.learn(total_timesteps=200_000)

# Save the model
save_path = os.path.join("agents", "ppo_sensor_model")
model.save(save_path)
print(f"Model saved to {save_path}")
