from stable_baselines3 import PPO
from environment.sensor_env import SensorEnv
import time

# Load the environment and trained model
env = SensorEnv()
model = PPO.load("agents/ppo_sensor_model")

obs, _ = env.reset()

for step in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    print(f"Step: {step} | Reward: {reward}")
    if terminated or truncated:
        print("Episode ended early.")
        break
    time.sleep(0.5)  # for nicer step-by-step viewing
