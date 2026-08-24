"""
Pusher-v5 PPO // Hugging Face Hub & Spaces One-Click Deployer
--------------------------------------------------------------
Automates uploading the trained Pusher-v5 PPO model, interactive web telemetry cockpit,
video checkpoints, and evaluation artifacts to Hugging Face Model Hub / Spaces.

Usage:
    # 1. Standard deploy (Uploads clean model card, code, weights & telemetry cockpit)
    python deploy_to_hf.py

    # 2. Deploy to a custom repository name
    python deploy_to_hf.py --repo-name my-pusher-ppo

    # 3. Create private repository
    python deploy_to_hf.py --private

    # 4. Create repository only without uploading files
    python deploy_to_hf.py --create-only
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, get_token, login

DEFAULT_REPO_NAME = "pusher-v5-ppo"


def parse_args():
    parser = argparse.ArgumentParser(description="One-Click Deployer for Pusher-v5 PPO Hub")
    parser.add_argument("--repo-name", type=str, default=DEFAULT_REPO_NAME,
                        help=f"Hugging Face repository name (default: {DEFAULT_REPO_NAME})")
    parser.add_argument("--repo-type", type=str, default="model", choices=["model", "space"],
                        help="Repository type: 'model' (Model Hub) or 'space' (Interactive Space)")
    parser.add_argument("--space-sdk", type=str, default="docker", choices=["docker", "static", "gradio"],
                        help="Space SDK if repo-type is 'space' (default: docker)")
    parser.add_argument("--private", action="store_true",
                        help="Create repository as private (default: Public)")
    parser.add_argument("--create-only", action="store_true",
                        help="Only create the repository on Hugging Face without uploading files")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face User Access Token (with WRITE permission)")
    return parser.parse_args()


def check_auth(token: str = None) -> str:
    """Verifies Hugging Face authentication token."""
    if token:
        login(token=token)
        return token
    
    existing_token = get_token()
    if existing_token:
        return existing_token
    
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        login(token=env_token)
        return env_token
    
    print("\n" + "=" * 65)
    print(" [!] Hugging Face Authentication Required")
    print("=" * 65)
    print(" Please provide your Hugging Face Access Token with WRITE permission.")
    print(" You can get your token from: https://huggingface.co/settings/tokens")
    print("=" * 65 + "\n")
    
    token_input = input(" Enter your Hugging Face Token: ").strip()
    if not token_input:
        print("[ERROR] Token cannot be empty. Deployment aborted.")
        sys.exit(1)
    
    login(token=token_input)
    return token_input


def generate_hf_model_card(repo_id: str) -> str:
    """
    Generates a clean, professional Hugging Face Model Card without internal development governance rules.
    """
    return f"""---
language:
- en
- ko
license: mit
tags:
- reinforcement-learning
- deep-reinforcement-learning
- stable-baselines3
- ppo
- continuous-control
- mujoco
- pusher
- pusher-v5
- robotics
- robot
- robot-arm
- robotic-manipulation
- 7-dof
- gymnasium
- pytorch
pipeline_tag: reinforcement-learning
library_name: stable-baselines3
model-index:
- name: pusher-v5-ppo
  results:
  - task:
      type: reinforcement-learning
      name: Reinforcement Learning
    dataset:
      name: Gymnasium MuJoCo Pusher-v5
      type: gymnasium/pusher-v5
    metrics:
    - type: mean_reward
      value: -32.42
      name: Mean Evaluation Reward (5-Ep Average)
---

# 🦾 Pusher-v5 PPO // AI Hub & Live Control Cockpit

[![Language: English](https://img.shields.io/badge/Language-English-blue)](README.md)
[![Language: 한국어](https://img.shields.io/badge/Language-한국어-green)](README_KR.md)
[![Hugging Face Hub](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20Hub-orange)](https://huggingface.co/{repo_id})
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=flat&logo=github)](https://github.com/Hwihwa-Lab/pusher-v5-ppo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Hwihwa-Lab/pusher-v5-ppo/blob/main/LICENSE)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-MuJoCo%20Pusher--v5-0080FF)](https://gymnasium.farama.org/environments/mujoco/pusher/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-brightgreen)](https://stable-baselines3.readthedocs.io)

> **MuJoCo 7-DOF Robotic Continuous Control Telemetry & PPO Deep Reinforcement Learning Platform**  
> *[ 🌐 English Documentation ](README.md) | [ 🇰🇷 한국어 매뉴얼 ](README_KR.md)*

This repository contains an advanced continuous deep reinforcement learning system (PPO) and a real-time engineering telemetry cockpit for 7-DOF robotic arm manipulation in [Gymnasium](https://gymnasium.farama.org/environments/mujoco/pusher/) MuJoCo `Pusher-v5`.

---

## 🌟 Model Specifications & Benchmark Performance

| Parameter | Specification |
| :--- | :--- |
| **Environment** | Gymnasium MuJoCo `Pusher-v5` (7-DOF Robotic Arm) |
| **Observation Space** | 23-dimensional continuous vector (Joints, Velocities, Tip 3D, Object 3D, Goal 3D) |
| **Action Space** | 7-dimensional continuous motor torques (`Box[-2.0, 2.0]`, float32) |
| **Algorithm** | Proximal Policy Optimization (PPO) with `MlpPolicy` |
| **Deep Learning Framework** | Stable-Baselines3 / PyTorch backend |
| **Baseline Return (Step 0)** | **`-57.51 pts`** (Random exploration, arm-to-object dist ~0.215m) |
| **Converged Return (Step 300k+)**| **`-32.42 ± 4.30 pts`** *(Peak: **`-26.15 pts`**)* |
| **Arm-to-Object Proximity** | **`0.028 m`** (Precise contact & cylinder grasp alignment) |
| **Goal Proximity Accuracy** | **`0.054 m`** (Target zone reached & pushed) |

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Web_Cockpit ["1-Screen Zero-Scroll Robotics Telemetry Cockpit"]
        W1["HTML5 / CSS3 / Vanilla JS Client"] <-->|"WebSocket /ws/simulation @ 30 FPS"| S1["FastAPI High-Performance Engine"]
        S1 -->|"Base64 JPEG Physics Stream"| W1
        S1 -->|"7-DOF Bipolar Torques (-2 to +2 Nm)"| W1
        S1 -->|"3D Vector Coordinates (Tip, Obj, Goal)"| W1
        W1 -->|"Control Commands (Start, Pause, Step, Reset, Policy)"| S1
    end

    subgraph Analytics_Deck ["4-Tab Analytics & Replay Deck"]
        T1["Tab 1: Live Telemetry Dynamics (Raw & 20-Ep Moving Average)"]
        T2["Tab 2: Milestone Replay Deck (16:9 Widescreen Video Gallery)"]
        T3["Tab 3: Live PPO Logs (Algorithmic Console Stream)"]
        T4["Tab 4: Environment & Reward Math Specifications"]
    end

    subgraph Deep_RL_Pipeline ["Stable-Baselines3 PPO Training Loop"]
        TR1["train.py / Background Thread"] --> TR2["MuJoCo Pusher-v5 Physics"]
        TR2 --> TR3["VisualProgressCallback"]
        TR3 --> TR4["Step 0 to 300k MP4 & GIF Videos"]
        TR3 --> TR5["Training Plots & Metrics JSON"]
        TR4 & TR5 --> TR6["Single-Click ZIP Archive: ppo_pusher_bundle.zip"]
    end
```

---

## 🕹️ Interactive Cockpit Features

1. **High-Fidelity 30 FPS Physics Stream**:
   - Ultra low-latency canvas streaming via WebSocket.
   - 7-DOF Action Space Motor Torque Bipolar Gauge (`[-2.0, +2.0] Nm`) with positive (Cyan) and negative (Rose) deflection.
   - 3D Cartesian coordinates tracker for Fingertip, Object, and Goal in real meters.
2. **Deep RL Training Budget Presets**:
   - `500 Ep (50k Steps • ~12s) - Quick Test`
   - `2,000 Ep (200k Steps • ~45s) - Basic Pushing`
   - `5,000 Ep (500k Steps • ~1.8m) ★ Recommended Mature`
   - `10,000 Ep (1M Steps • ~3.5m) - High-Precision`
3. **Widescreen Checkpoint Replay Gallery**:
   - Side-by-side comparative video cards displaying the robotic arm's learning trajectory from random exploration (Step 0) to mature convergence.
   - Instant 1-click export for **MP4 videos** and **animated GIFs**.
4. **Single-Click ZIP Packaging**:
   - One-click bundle download (`ppo_pusher_bundle.zip`) containing weights, milestone videos, and telemetry charts.

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
git clone https://github.com/Hwihwa-Lab/pusher-v5-ppo.git
cd pusher-v5-ppo
pip install -r requirements.txt
```

### 2. Launch Local Web Control Cockpit
```bash
python app.py
```
Open your browser at **`http://localhost:8000`**.

### 3. Standalone CLI Training & Evaluation
```bash
# Train PPO agent
python train.py --timesteps 300000 --eval_freq 30000

# Evaluate trained model
python evaluate.py --model_path ./results/ppo_pusher.zip --episodes 5
```

---

## 🐍 Quick Python Evaluation Snippet

You can load and evaluate this pre-trained agent in 5 lines of Python using Stable-Baselines3:

```python
import gymnasium as gym
from stable_baselines3 import PPO

# 1. Initialize Pusher-v5 environment & load model
env = gym.make("Pusher-v5", render_mode="human")
model = PPO.load("results/ppo_pusher.zip")

# 2. Run deterministic pushing evaluation
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()
```

---

## ⌨️ Keyboard Shortcuts Reference

| Key | Action | Description |
| :---: | :--- | :--- |
| **`Space`** | **Start / Pause** | Toggle 30 FPS MuJoCo physical simulation stream |
| **`R`** | **Reset Environment** | Reset robotic arm, cylinder object, and target goal to new random positions |
| **`S`** | **Step Once** | Advance physics engine forward by 1 discrete timestep (0.05s) |
| **`H`** | **Toggle HUD** | Show or hide on-canvas telemetry data overlay |

---

## 📂 Repository Contents

* `README.md`: English Model Card and benchmark performance guide.
* `README_KR.md`: Full Korean comprehensive manual ([한국어 매뉴얼](README_KR.md)).
* `app.py`: FastAPI high-performance backend & 30 FPS WebSocket simulation server.
* `train.py`: Stable-Baselines3 PPO 7-DOF training engine with `VisualProgressCallback`.
* `evaluate.py`: Standalone 5-episode deterministic policy evaluator and video recorder.
* `visualizer.py`: Standalone Matplotlib visualizer and benchmark plotter.
* `web/`: 1-Screen zero-scroll telemetry cockpit frontend (`app.js`, `index.html`, `style.css`).
* `results/ppo_pusher.zip`: Pre-trained PPO neural network weights (300,000 steps, -32.4 pts).
* `ppo_pusher_bundle.zip`: Complete production archive with weights, 12 checkpoint videos, and plots.
* `deploy_to_hf.py`: One-click automated Hugging Face Model Hub deployer.
* `requirements.txt` & `packages.txt`: Python and system dependency manifests.

---

## 🔗 Open Source Hubs & Project Links

- 🐙 **GitHub Repository**: [https://github.com/Hwihwa-Lab/pusher-v5-ppo](https://github.com/Hwihwa-Lab/pusher-v5-ppo)
- 🤗 **Hugging Face Model Hub**: [https://huggingface.co/{repo_id}](https://huggingface.co/{repo_id})

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Hwihwa-Lab/pusher-v5-ppo/blob/main/LICENSE) file for details.

---

*Trained and deployed with [Pusher AI Hub](https://huggingface.co/{repo_id}) by **hwihwalab**.*
"""


def main():
    args = parse_args()
    token = check_auth(args.token)
    api = HfApi(token=token)

    try:
        user_info = api.whoami()
        username = user_info["name"]
    except Exception as e:
        print(f"[ERROR] Failed to fetch user info with provided token: {e}")
        sys.exit(1)

    repo_id = f"{username}/{args.repo_name}"
    print("\n" + "=" * 65)
    print(" 🦾 Pusher-v5 PPO // Hugging Face Deployment Pipeline")
    print("=" * 65)
    print(f" • Target User : {username}")
    print(f" • Repo ID     : {repo_id}")
    print(f" • Repo Type   : {args.repo_type}")
    print(f" • Visibility  : {'Private' if args.private else 'Public'}")
    print("=" * 65)

    # 1. Create or connect to repository
    try:
        print(f"\n[1/3] Creating/Connecting repository on Hugging Face: {repo_id} ...")
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type=args.repo_type,
            private=args.private,
            space_sdk=args.space_sdk if args.repo_type == "space" else None,
            exist_ok=True
        )
        print(f"  --> Repository ready: {repo_url}")
    except Exception as e:
        print(f"[ERROR] Repository creation failed: {e}")
        sys.exit(1)

    if args.create_only:
        print("\n[COMPLETE] '--create-only' mode selected. Repository created successfully.")
        return

    # 2. Upload project assets & source code
    root_dir = Path(__file__).resolve().parent
    print(f"\n[2/3] Uploading project files and assets to {repo_id} ...")

    ignore_patterns = [
        "README.md",
        "__pycache__/*",
        "*.pyc",
        ".git/*",
        ".gitignore",
        ".venv/*",
        "venv/*",
        "env/*",
        "*.log",
        ".system_generated/*",
        ".tempmediaStorage/*",
        ".cursor/*",
        ".cursorrules*",
        "DOCS_*",
        "eval_results/*",
        ".vscode/*",
        "*.tmp",
        "*.tmp.md"
    ]

    try:
        api.upload_folder(
            folder_path=str(root_dir),
            repo_id=repo_id,
            repo_type=args.repo_type,
            ignore_patterns=ignore_patterns,
            commit_message="feat: upload Pusher-v5 PPO model, telemetry cockpit, and video gallery"
        )
        print("  --> Project files uploaded successfully!")
    except Exception as e:
        print(f"[ERROR] File upload failed: {e}")
        sys.exit(1)

    # 3. Generate & upload dedicated Hugging Face Model Card (README.md)
    print(f"\n[3/3] Generating and uploading official Hugging Face Model Card (README.md)...")
    temp_card_path = root_dir / "HF_MODEL_CARD.tmp.md"
    try:
        model_card_content = generate_hf_model_card(repo_id)
        with open(temp_card_path, "w", encoding="utf-8") as f:
            f.write(model_card_content)

        api.upload_file(
            path_or_fileobj=str(temp_card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=args.repo_type,
            commit_message="docs: add official Pusher-v5 PPO model card and metadata"
        )
        print("  --> Official Hugging Face Model Card uploaded successfully!")
    except Exception as e:
        print(f"[ERROR] Model card upload failed: {e}")
        sys.exit(1)
    finally:
        if temp_card_path.exists():
            temp_card_path.unlink()

    # Success Summary
    print("\n" + "=" * 65)
    print(" 🚀 DEPLOYMENT COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    print(f" • Hugging Face URL: https://huggingface.co/{repo_id}")
    if args.repo_type == "space":
        print(f" • Live Web App URL: https://huggingface.co/spaces/{repo_id}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
