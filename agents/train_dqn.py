import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from environment.elevation_sensor_env_dqn import ElevationSensorEnv

# ==== Configurations ====
ENV_ID = "ElevationSensorEnv-v0"
TOTAL_TIMESTEPS = 100_000
EVAL_FREQ = 5000
LOG_DIR = "./dqn_logs/"
MODEL_DIR = "./dqn_models/"
BEST_MODEL_DIR = MODEL_DIR + "best_model/"

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
train_env = Monitor(ElevationSensorEnv())
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
model = DQN(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=(4, "step"),
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
)

# ==== Training ====
print("🚀 DQN Training started...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback
)

# ==== Save Final Model ====
model.save(MODEL_DIR + "dqn_final_model")
print("✅ DQN Training complete. Model saved.")
