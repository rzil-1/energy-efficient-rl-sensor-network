from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import rasterio
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/terrain")
def get_terrain():
    # Load the TIF and convert to PNG
    tif_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'real_terrain.tif')
    
    with rasterio.open(tif_path) as src:
        # Read the first band and downsample for performance
        data = src.read(1, out_shape=(1, src.height // 4, src.width // 4))
        
    # Normalize for image display
    data_min, data_max = np.min(data), np.max(data)
    normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
    
    # Create colormap (simple grayscale to green-ish for terrain)
    img = Image.fromarray(normalized, mode='L')
    
    # Save to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {"image": f"data:image/png;base64,{img_str}"}

import sys
import os

# Append root to path for imports BEFORE internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import WebSocket
from environment.multi_agent_sensor_env import MultiAgentSensorEnv
from stable_baselines3 import PPO
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
import asyncio

@app.websocket("/ws/simulate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Initialize environment and model
    tif_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'real_terrain.tif')
    raw_env = MultiAgentSensorEnv(grid_size=50, num_sensors=50, max_steps=500, terrain_file=tif_path)
    env = pettingzoo_env_to_vec_env_v1(raw_env)
    env = concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'marl', 'ppo_shared_policy')
    model = PPO.load(model_path)
    
    obs = env.reset()
    done = False
    step = 0
    
    try:
        while not done and step < 500:
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            
            # Pack state for frontend
            active_env = env.venv.vec_envs[0].par_env
            sensors_data = []
            for i, agent_id in enumerate(active_env.possible_agents):
                x, y = active_env.sensor_positions[agent_id]
                sensors_data.append({
                    "id": i,
                    "x": x / active_env.grid_size,
                    "y": y / active_env.grid_size,
                    "battery": active_env.batteries.get(agent_id, 0) / 100.0,
                    "radius": float(active_env.active_power.get(agent_id, 0))
                })
            
            await websocket.send_json({
                "step": step,
                "sensors": sensors_data,
                "active_count": sum(1 for s in sensors_data if s["battery"] > 0)
            })
            
            if any(dones):
                done = True
                
            step += 1
            await asyncio.sleep(0.1) # Throttle for visual playback
            
    except Exception as e:
        print(f"WebSocket closed: {e}")
    finally:
        env.close()
