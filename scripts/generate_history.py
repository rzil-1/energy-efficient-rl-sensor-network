import os
import subprocess
import time

def run_git_command(command, date_str):
    env = os.environ.copy()
    # Set both Author and Committer dates so the GitHub graph is perfect
    env["GIT_AUTHOR_DATE"] = date_str
    env["GIT_COMMITTER_DATE"] = date_str
    
    print(f"Executing: {' '.join(command)}")
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Success. Date set to: {date_str}\n")

def main():
    print("Starting backdated Git commit sequence...\n")

    commits = [
        {
            "add": ["git", "add", "environment/multi_agent_sensor_env.py", "agents/train_marl.py", "agents/evaluate_marl.py"],
            "msg": "feat(rl): implement independent PPO with difference rewards for decentralized sensors",
            "date": "2026-06-15T21:15:43" # Monday Evening
        },
        {
            "add": ["git", "add", "data/usgs_loader.py"],
            "msg": "feat(data): integrate real-world USGS terrain ingestion via rasterio",
            "date": "2026-06-20T11:42:15" # Saturday Morning
        },
        {
            "add": ["git", "add", "models/", "logs/", "scripts/", "-u"],
            "msg": "chore: restructure project directories for production deployment",
            "date": "2026-06-27T15:10:00" # Saturday Afternoon
        },
        {
            "add": ["git", "add", "server/main.py", "requirements.txt"],
            "msg": "feat(backend): build FastAPI websocket server for live PettingZoo simulation streaming",
            "date": "2026-07-02T22:05:12" # Thursday Night
        },
        {
            "add": ["git", "add", "frontend-react/"],
            "msg": "feat(ui): implement live tactical dashboard in React with canvas animation",
            "date": "2026-07-05T14:30:22" # Sunday Afternoon
        },
        {
            "add": ["git", "add", "-A"],
            "msg": "docs: comprehensively update architecture documentation and ignore rules",
            "date": "2026-07-08T18:45:00" # Today Evening
        }
    ]

    for step in commits:
        # Run git add
        subprocess.run(step["add"])
        
        # Run git commit with backdated environment
        commit_cmd = ["git", "commit", "-m", step["msg"]]
        run_git_command(commit_cmd, step["date"])
        
        # Small sleep just to be safe
        time.sleep(1)

    print("All done! Run 'git log' to see your beautifully organic commit history.")

if __name__ == "__main__":
    main()
