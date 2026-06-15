import os
import sys

# Add the project root to the Python path so it can find the 'environment' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from environment.multi_agent_sensor_env import MultiAgentSensorEnv
import supersuit as ss

def main():
    print("Setting up Multi-Agent PPO (IPPO) Training Pipeline...")
    
    # 1. Instantiate the PettingZoo Parallel Environment
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    geotiff_path = os.path.join(project_root, "data", "real_terrain.tif")
    env = MultiAgentSensorEnv(grid_size=50, num_sensors=50, max_steps=500, terrain_file=geotiff_path)
    
    # 3. Apply SuperSuit Wrappers
    # This allows all agents to share the same policy (Parameter Sharing)
    # and converts the multi-agent env into a standard single-agent Vectorized Env for SB3
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # We run 1 environment in parallel, but it contains 50 agents.
    # ConcatVecEnvs stacks them so SB3 sees a batch of 50.
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')

    # 4. Setup Logging directory
    log_dir = "./logs/marl/"
    os.makedirs(log_dir, exist_ok=True)
    
    # 5. Initialize PPO Model
    # We use Independent PPO (IPPO) where a single policy network is trained across all agents.
    print("Initializing IPPO Policy...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="auto" # Will use GPU if available
    )

    print("Starting MARL Training...")
    # Train for 100,000 steps (adjust as needed)
    model.learn(total_timesteps=100000)

    print("Training Complete. Saving Model...")
    model_dir = "./models/marl/"
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir + "ppo_shared_policy")
    print(f"[DONE] Training complete. Model saved to {model_dir}/ppo_shared_policy")

if __name__ == "__main__":
    main()
