import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from environment.elevation_sensor_env import ElevationSensorEnv

# ==== Configurations ====
ENV_ID = "ElevationSensorEnv-v0"
TOTAL_TIMESTEPS = 100_000
EVAL_FREQ = 5_000
LOG_DIR = "./ppo_logs/"
MODEL_DIR = "./ppo_models/"
BEST_MODEL_DIR = MODEL_DIR + "best_model/"
NUM_ENVS = 4  # Parallel environments

# ==== Register Custom Environment ====
import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id=ENV_ID,
    entry_point="environment.elevation_sensor_env:ElevationSensorEnv",
)

# ==== Setup Logging ====
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# ==== Environment Setup ====
train_env = make_vec_env(
    ENV_ID,
    n_envs=NUM_ENVS,
    monitor_dir=LOG_DIR
)

eval_env = Monitor(ElevationSensorEnv())

# ==== Callbacks ====
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=8.0, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,
    best_model_save_path=BEST_MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=EVAL_FREQ,
    deterministic=True,
    render=False,
    verbose=1
)

# ==== Model Setup ====
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    n_steps=1024,
    batch_size=256,
    gae_lambda=0.95,
    gamma=0.99,
    learning_rate=2.5e-4,
    ent_coef=0.01
)

# ==== Training ====
print("🚀 Training started...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback
)

# ==== Save Final Model ====
model.save(MODEL_DIR + "ppo_final_model")
print("✅ Training complete. Model saved.")
