import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

class MultiAgentSensorEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "multi_agent_sensor_v0"}

    def __init__(self, grid_size=10, num_sensors=5, max_steps=50, max_battery=100.0, render_mode=None, terrain_file=None):
        super().__init__()
        self.grid_size = grid_size
        self.num_sensors = num_sensors
        self.max_steps = max_steps
        self.max_battery = max_battery
        self.render_mode = render_mode
        self.terrain_file = terrain_file
        self.step_count = 0

        # Define agents
        self.possible_agents = [f"sensor_{i}" for i in range(num_sensors)]
        self.agents = self.possible_agents[:]

        # Environment layout
        # Terrain is loaded via the USGS ingestion pipeline
        from data.usgs_loader import TerrainLoader
        self.terrain_loader = TerrainLoader(target_grid_size=(grid_size, grid_size))
        # Loads a GeoTIFF if provided in kwargs, otherwise generates a realistic simulated terrain
        self.elevation = self.terrain_loader.get_terrain(file_path=self.terrain_file, seed=42)
        
        # Sensor fixed positions for simplicity (could be randomized)
        # Randomly place sensors
        np.random.seed(42)
        self.sensor_positions = {
            agent: (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
            for agent in self.possible_agents
        }

        # Spaces
        # Action: Continuous transmit power [0, 1]
        # Observation: Local elevation (3x3 patch around sensor) + own battery + active neighbor count
        self.action_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        # Obs = 9 (3x3 elevation) + 1 (battery) + 1 (active neighbors) = 11 dims
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.batteries = {agent: self.max_battery for agent in self.agents}
        self.active_power = {agent: 0.0 for agent in self.agents}

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _get_obs(self, agent):
        x, y = self.sensor_positions[agent]
        # Extract 3x3 local patch with padding if at edges
        padded_elevation = np.pad(self.elevation, pad_width=1, mode='edge')
        local_patch = padded_elevation[x:x+3, y:y+3].flatten()
        
        # Estimate active neighbors (within a simple radius)
        active_neighbors = 0
        for other in self.agents:
            if other != agent and self.active_power[other] > 0.1:
                ox, oy = self.sensor_positions[other]
                dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                if dist < 3.0: # arbitrary communication range
                    active_neighbors += 1

        obs = np.concatenate([
            local_patch, # already normalized to [0, 1] by TerrainLoader
            [self.batteries[agent] / self.max_battery],
            [active_neighbors / self.num_sensors]
        ]).astype(np.float32)
        
        return obs

    def step(self, actions):
        self.step_count += 1
        
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # 1. Update states based on actions
        for agent in self.agents:
            power = float(actions[agent][0])
            # Enforce bounds
            power = np.clip(power, 0.0, 1.0)
            
            # If battery is dead, power is 0
            if self.batteries[agent] <= 0:
                power = 0.0
                
            self.active_power[agent] = power
            # Drain battery relative to transmit power
            self.batteries[agent] = max(0.0, self.batteries[agent] - (power * 2.0))

        # 2. Calculate global coverage (continuous model to fix RL gradients)
        # Instead of a hard radius, power increases the "coverage strength" of the sensor's maximum zone
        coverage_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        max_radius = (self.grid_size / 10.0) * 3
        r_int = int(max_radius)
        
        for agent in self.agents:
            power = self.active_power[agent]
            if power > 0.0:
                x, y = self.sensor_positions[agent]
                for dx in range(-r_int, r_int + 1):
                    for dy in range(-r_int, r_int + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            if dx**2 + dy**2 <= max_radius**2:
                                # Linear continuous coverage avoids RL gradient step-function traps
                                coverage_grid[nx, ny] += power
                                
        # Cap maximum coverage of any single cell to 1.0 (prevents wasted overlap power)
        unclipped_coverage = coverage_grid.copy()
        clipped_coverage = np.clip(unclipped_coverage, 0.0, 1.0)
        global_coverage = np.sum(clipped_coverage) / (self.grid_size * self.grid_size)

        # 3. Calculate rewards and terminations
        network_dead = all(b <= 0 for b in self.batteries.values())
        truncated = self.step_count >= self.max_steps
        
        # Fully cooperative team reward prevents the "Free Rider" problem
        # (Where agents output 0.0 power to let others do the work and take the battery penalty)
        # We must solve the Multi-Agent Credit Assignment problem!
        # If we use a global team reward, a single agent's action is drowned out by the noise of 49 other agents.
        # We use Difference Rewards: Agent A's reward = (Global Coverage) - (Global Coverage if Agent A was removed)
        # This perfectly aligns individual gradients with the global objective, and prevents Free Riders!
        
        rewards = {}
        for agent in self.agents:
            power = self.active_power[agent]
            if power <= 0.0:
                marginal_coverage = 0.0
            else:
                # Reconstruct the agent's individual coverage footprint
                agent_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
                x, y = self.sensor_positions[agent]
                for dx in range(-r_int, r_int + 1):
                    for dy in range(-r_int, r_int + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            if dx**2 + dy**2 <= max_radius**2:
                                agent_grid[nx, ny] = power
                                
                # Calculate global coverage WITHOUT this agent
                unclipped_without_agent = unclipped_coverage - agent_grid
                coverage_without_agent = np.sum(np.clip(unclipped_without_agent, 0.0, 1.0)) / (self.grid_size * self.grid_size)
                marginal_coverage = global_coverage - coverage_without_agent

            # Individual reward: marginal contribution to coverage - individual power penalty
            r = (100.0 * marginal_coverage) - (0.1 * power)
                
            if self.batteries[agent] <= 0:
                # Agent is dead. Zero reward. The loss of future coverage is punishment enough.
                r = 0.0
                
            rewards[agent] = r
            terminations[agent] = self.batteries[agent] <= 0
            truncations[agent] = truncated
            infos[agent] = {"coverage": global_coverage}

        # 4. Generate new observations
        for agent in self.agents:
            observations[agent] = self._get_obs(agent)

        # PettingZoo ParallelEnv requires removing dead agents
        if network_dead or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode == "human":
            print(f"--- Step {self.step_count} ---")
            if self.num_sensors <= 10:
                for agent in self.possible_agents:
                    print(f"{agent}: Battery={self.batteries[agent]:.1f}, Power={self.active_power[agent]:.2f}")
            else:
                avg_batt = np.mean([self.batteries[a] for a in self.possible_agents])
                min_batt = np.min([self.batteries[a] for a in self.possible_agents])
                avg_pow = np.mean([self.active_power[a] for a in self.possible_agents])
                dead_sensors = sum(1 for a in self.possible_agents if self.batteries[a] <= 0)
                print(f"Sensors: {self.num_sensors} | Alive: {self.num_sensors - dead_sensors} | Avg Power: {avg_pow:.2f}")
                print(f"Avg Battery: {avg_batt:.1f} | Min Battery: {min_batt:.1f}")
