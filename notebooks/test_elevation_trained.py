import os
import time
import numpy as np
from stable_baselines3 import PPO
from environment.elevation_sensor_env import ElevationSensorEnv

# ✅ Corrected model path (pointing directly to the .zip file)
MODEL_PATH = os.path.join("ppo_models", "best_model", "best_model.zip")

# ✅ Load the trained PPO model
model = PPO.load(MODEL_PATH)
print("✅ Loaded trained model successfully.")

# ✅ Create the environment
env = ElevationSensorEnv()
obs, _ = env.reset()
done = False
total_reward = 0
step = 0

# ✅ Run the trained agent on the environment
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    step += 1

    print(f"Step {step} - Reward: {reward:.2f}")
    env.render()
    time.sleep(0.5)  # Optional: slow down to visualize clearly

print("\n🎯 Evaluation finished.")
print(f"📊 Total Reward: {total_reward:.2f}")
