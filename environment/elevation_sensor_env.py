import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ElevationSensorEnv(gym.Env):
    def __init__(self, num_nodes=10, grid_size=(10, 10), max_battery=100, max_steps=50, use_height=True):
        super(ElevationSensorEnv, self).__init__()
        self.num_nodes = num_nodes
        self.grid_size = grid_size
        self.max_battery = max_battery
        self.max_steps = max_steps
        self.coverage_threshold = 0.8
        self.use_height = use_height

        # Sensor node positions and terrain elevation
        self.positions = np.random.randint(0, grid_size[0], size=(num_nodes, 2))
        self.elevation = np.random.randint(0, 10, size=grid_size) if use_height else np.zeros(grid_size)

        # Gym spaces
        self.action_space = spaces.MultiBinary(num_nodes)
        self.observation_space = spaces.Box(low=0, high=max_battery, shape=(num_nodes * 2,), dtype=np.float32)

        self.reset(seed=42)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.batteries = np.full(self.num_nodes, self.max_battery, dtype=np.float32)
        self.states = np.zeros(self.num_nodes, dtype=np.int32)
        self.current_step = 0
        self.prev_coverage = 0.0
        self.total_reward = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.batteries, self.states]).astype(np.float32)

    def _calculate_coverage(self):
        grid_cells = np.zeros(self.grid_size, dtype=bool)
        for i in range(self.num_nodes):
            if self.states[i] == 1:
                x, y = self.positions[i]
                elevation_factor = max(1, self.elevation[x, y] // 3)
                radius = 1 + elevation_factor
                x_min, x_max = max(0, x - radius), min(self.grid_size[0], x + radius + 1)
                y_min, y_max = max(0, y - radius), min(self.grid_size[1], y + radius + 1)
                grid_cells[x_min:x_max, y_min:y_max] = True
        return np.sum(grid_cells) / (self.grid_size[0] * self.grid_size[1])

    def step(self, action):
        self.current_step += 1
        self.states = action
        self.batteries -= action.astype(np.float32)

        # Deactivate sensors with 0 battery
        for i in range(self.num_nodes):
            if self.batteries[i] <= 0:
                self.batteries[i] = 0
                self.states[i] = 0

        # Coverage and reward calculation
        coverage = self._calculate_coverage()
        coverage_gain = coverage - self.prev_coverage
        self.prev_coverage = coverage

        # Reward shaping
        active_sensors = np.count_nonzero(action)
        energy_used = np.sum(action)

        normalized_coverage = coverage
        reward = 2.0 * normalized_coverage  # direct incentive for high coverage
        reward += 5.0 * coverage_gain       # incentive for increasing coverage
        reward -= 0.2 * active_sensors      # discourage too many sensors
        reward -= 0.05 * energy_used        # slight energy penalty

        # Bonus for reaching thresholds
        if coverage > 0.8:
            reward += 1.0
        elif coverage > 0.6:
            reward += 0.5

        # Penalty for draining too many batteries
        if np.sum(self.batteries < 10) > self.num_nodes * 0.3:
            reward -= 1.0

        self.total_reward += reward

        terminated = (
            coverage < 0.2 or
            np.count_nonzero(self.batteries == 0) > self.num_nodes * 0.5 or
            self.current_step >= self.max_steps
        )
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        grid = np.full(self.grid_size, ".", dtype=str)
        for i in range(self.num_nodes):
            x, y = self.positions[i]
            grid[x, y] = "O" if self.states[i] == 1 else "x"
        print("\n".join(" ".join(row) for row in grid))
        print(f"Step: {self.current_step}")
        print(f"Batteries: {self.batteries}")
        print(f"Coverage: {self._calculate_coverage() * 100:.2f}%")
