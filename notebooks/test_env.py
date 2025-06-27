from environment.sensor_env import SensorEnv

env = SensorEnv()
obs = env.reset()

for _ in range(10):
    action = env.action_space.sample()  # Random ON/OFF decisions
    obs, reward, done, _ = env.step(action)
    env.render()
    print(f"Reward: {reward}\n")
    if done:
        print("Episode ended.")
        break
