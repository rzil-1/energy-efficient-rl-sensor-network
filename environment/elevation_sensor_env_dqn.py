
import numpy as np
import gym
from gym import spaces

class ElevationSensorEnv(gym.Env):
    def __init__(self, grid_size=5, num_sensors=5, max_steps=20):
        super(ElevationSensorEnv, self).__init__()
        self.grid_size = grid_size
        self.num_sensors = num_sensors
        self.max_steps = max_steps

        # Generate elevation map (e.g., like a terrain)
        self.elevation = np.random.rand(grid_size, grid_size)

        # Action space: choose 1 of num_sensors to activate
        self.action_space = spaces.Discrete(num_sensors)

        # Observation space: flattened elevation + sensor states + battery levels
        self.obs_dim = grid_size * grid_size + num_sensors + 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.sensor_positions = [  # Fixed sensor positions
            (0, 0),
            (0, self.grid_size - 1),
            (self.grid_size - 1, 0),
            (self.grid_size - 1, self.grid_size - 1),
            (self.grid_size // 2, self.grid_size // 2)
        ]
        self.active_sensors = np.zeros(self.num_sensors, dtype=np.float32)
        self.batteries = 10.0  # total energy
        self.steps = 0

        print(f"🔍 Elevation Map (for debug):\n{self.elevation}")

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.elevation.flatten(),        # Terrain
            self.active_sensors,             # Sensor status
            [self.batteries / 10.0]          # Normalize battery
        ]).astype(np.float32)

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action!"

        self.steps += 1
        done = self.steps >= self.max_steps or self.batteries <= 0

        if self.active_sensors[action] == 0:
            self.active_sensors[action] = 1
            self.batteries -= 1.0  # cost of activation

        reward = self._calculate_coverage_reward()

        obs = self._get_obs()
        return obs, reward, done, {}

    def _calculate_coverage_reward(self):
        reward = 0.0
        for i, active in enumerate(self.active_sensors):
            if active:
                x, y = self.sensor_positions[i]
                reward += self.elevation[x, y] * 2.0  # amplify elevation impact
        return reward

    def render(self, mode="human"):
        print(f"Step: {self.steps}, Battery: {self.batteries:.2f}")
        print(f"Active Sensors: {self.active_sensors}")
