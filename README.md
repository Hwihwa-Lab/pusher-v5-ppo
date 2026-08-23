---
language:
- en
- ko
tags:
- reinforcement-learning
- stable-baselines3
- ppo
- continuous-control
- mujoco
- pusher-v5
- robotics
- robot-arm
- 7-dof
- teleoperation
pipeline_tag: reinforcement-learning
library_name: stable-baselines3
---

# 🦾 Pusher-v5 PPO // AI Hub & Live Control Cockpit

[![Language: English](https://img.shields.io/badge/Language-English-blue)](README.md)
[![Language: 한국어](https://img.shields.io/badge/Language-한국어-green)](README_KR.md)
[![Hugging Face Hub](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20Hub-orange)](https://huggingface.co/hwihwalab/pusher-v5-ppo)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=flat&logo=github)](https://github.com/Hwihwa-Lab/pusher-v5-ppo)
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
| **Observation Normalization** | Raw MuJoCo coordinates & velocities |
| **Baseline Return (Step 0)** | **`-57.51 pts`** (Random exploration, arm-to-object dist ~0.215m) |
| **Converged Return (Step 300k+)**| **`-32.42 ± 4.30 pts`** *(Peak: **`-26.15 pts`**)* |
| **Arm-to-Object Proximity** | **`0.028 m`** (Precise contact & cylinder grasp alignment) |
| **Goal Proximity Accuracy** | **`0.054 m`** (Target zone reached & pushed) |

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Web_Cockpit [1-Screen Zero-Scroll Robotics Telemetry Cockpit]
        W1[HTML5 / CSS3 / Vanilla JS Client] <-->|WebSocket /ws/simulation @ 30 FPS| S1[FastAPI High-Performance Engine]
        S1 -->|Base64 JPEG Physics Stream| W1
        S1 -->|7-DOF Bipolar Torques [-2, +2] Nm| W1
        S1 -->|3D Vector Coordinates: Tip, Obj, Goal| W1
        W1 -->|Control Commands: Start, Pause, Step, Reset, Policy| S1
    end

    subgraph Analytics_Deck [4-Tab Analytics & Replay Deck]
        T1[Tab 1: Live Telemetry Dynamics - Raw & 20-Ep Moving Average]
        T2[Tab 2: Milestone Replay Deck - 16:9 Widescreen Video Gallery]
        T3[Tab 3: Live PPO Logs - Algorithmic Console Stream]
        T4[Tab 4: Environment & Reward Math Specifications]
    end

    subgraph Deep_RL_Pipeline [Stable-Baselines3 PPO Training Loop]
        TR1[train.py / Background Thread] --> TR2[MuJoCo Pusher-v5 Physics]
        TR2 --> TR3[VisualProgressCallback]
        TR3 --> TR4[Step 0, 10k, 20k, 30k MP4 & GIF Videos]
        TR3 --> TR5[Training Plots & Metrics JSON]
        TR4 & TR5 --> TR6[Single-Click ZIP Archive: ppo_pusher_bundle.zip]
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
   - Side-by-side comparative video cards displaying the robotic arm's learning trajectory from random exploration (Step 0) to mature convergence (Step 30.7k).
   - Instant 1-click export for **MP4 videos** and **animated GIFs**.

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

### 3. One-Click Deploy to Hugging Face
```bash
python deploy_to_hf.py
```

### 4. Standalone CLI Training & Evaluation
```bash
# Train PPO agent
python train.py --timesteps 50000 --eval_freq 10000

# Evaluate trained model
python evaluate.py --model_path ./results/ppo_pusher.zip --episodes 3
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
