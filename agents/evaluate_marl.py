import os
import sys
import time
import numpy as np

# Add project root to sys.path to prevent ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from environment.multi_agent_sensor_env import MultiAgentSensorEnv

def main():
    print("Loading Trained IPPO Model...")
    model_path = "./models/marl/ppo_shared_policy"
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model not found at {model_path}.zip")
        return

    # Load the trained model
    model = PPO.load(model_path)

    # Instantiate the base PettingZoo Parallel Environment
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    geotiff_path = os.path.join(project_root, "data", "real_terrain.tif")
    env = MultiAgentSensorEnv(grid_size=50, num_sensors=50, max_steps=500, render_mode="human", terrain_file=geotiff_path)
    
    print("\nStarting Evaluation Episode...")
    observations, infos = env.reset()
    
    total_network_reward = 0.0
    step = 0
    
    while env.agents:
        print(f"\n================ STEP {step+1} ================")
        actions = {}
        
        # In our IPPO setup, the same policy evaluates the observation for each agent
        for agent in env.agents:
            obs = observations[agent]
            # model.predict returns (action, state)
            action, _states = model.predict(obs, deterministic=True)
            actions[agent] = action

        # Step the environment forward
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Render to console
        env.render()
        
        # Calculate stats
        step_reward = sum(rewards.values())
        total_network_reward += step_reward
        
        # We can grab the coverage from any agent's info dict
        coverage = infos[env.agents[0]]["coverage"] * 100 if env.agents else 0.0
        print(f"Network Step Reward: {step_reward:.2f} | Global Coverage: {coverage:.1f}%")
        
        step += 1
        time.sleep(0.01) # Faster evaluation for analysis

    print("\n[SUCCESS] Evaluation Episode Complete!")
    print(f"Total Network Reward: {total_network_reward:.2f}")
    print(f"Total Steps Survived: {step}")

if __name__ == "__main__":
    main()
