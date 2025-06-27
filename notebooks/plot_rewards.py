import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to your monitor logs
log_dir = "data/results/"
monitor_files = [f for f in os.listdir(log_dir) if f.startswith("monitor")]
log_file = os.path.join(log_dir, monitor_files[0])

# Load data (skip first commented row)
data = pd.read_csv(log_file, skiprows=1)

# Add episode numbers
data['Episode'] = range(1, len(data) + 1)

# Plot raw rewards
plt.figure(figsize=(10, 5))
plt.plot(data['Episode'], data['r'], label='Reward per Episode', alpha=0.6)

# Optional: Moving average for smoothing
window = 20
data['Smoothed Reward'] = data['r'].rolling(window).mean()
plt.plot(data['Episode'], data['Smoothed Reward'], label=f'{window}-Episode Moving Avg', linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward vs Episode")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
