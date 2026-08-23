"""
app.py
Pusher-v5 실시간 웹 시뮬레이션 및 학습 관제 FastAPI 백엔드 서버
"""

import os
import io
import json
import time
import base64
import asyncio
import threading
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image
import gymnasium as gym
from stable_baselines3 import PPO

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import visualizer
import train as train_module

app = FastAPI(title="Pusher-v5 Real-time Simulation & RL Control Center")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation state
class SimulationManager:
    def __init__(self):
        self.env: Optional[gym.Env] = None
        self.model: Optional[PPO] = None
        self.model_path = "./results/ppo_pusher.zip"
        self.is_running = False
        self.is_paused = False
        self.policy_type = "trained"  # 'trained' or 'random'
        self.show_hud = True
        self.speed = 1.0
        self.obs = None
        self.ep_reward = 0.0
        self.ep_step = 0
        self.episode_count = 1
        self.lock = threading.Lock()
        self.clients = set()
        
        # Load environment and model
        self.init_env_and_model()

    def init_env_and_model(self):
        with self.lock:
            try:
                if self.env is not None:
                    self.env.close()
                self.env = gym.make("Pusher-v5", render_mode="rgb_array")
                self.obs, _ = self.env.reset(seed=42)
                self.ep_reward = 0.0
                self.ep_step = 0
                
                if os.path.exists(self.model_path):
                    self.model = PPO.load(self.model_path, env=self.env)
                    print(f"[SimManager] Loaded trained model from {self.model_path}")
                else:
                    self.model = None
                    print("[SimManager] Trained model not found, using random policy.")
            except Exception as e:
                print(f"[SimManager Init Error] {e}")

    def reset_env(self, seed: Optional[int] = None):
        with self.lock:
            if self.env is None:
                self.env = gym.make("Pusher-v5", render_mode="rgb_array")
            seed_val = seed if seed is not None else int(time.time() * 1000) % 100000
            self.obs, _ = self.env.reset(seed=seed_val)
            self.ep_reward = 0.0
            self.ep_step = 0
            self.episode_count += 1

    def step(self) -> Dict[str, Any]:
        with self.lock:
            if self.env is None or self.obs is None:
                self.reset_env()

            try:
                if self.policy_type == "trained" and self.model is not None:
                    action, _ = self.model.predict(self.obs, deterministic=True)
                else:
                    action = self.env.action_space.sample()

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.ep_reward += float(reward)
                self.ep_step += 1
                done = terminated or truncated

                reward_dist = float(info.get("reward_dist", 0.0))
                reward_near = float(info.get("reward_near", 0.0))
                reward_ctrl = float(info.get("reward_ctrl", 0.0))

                dist_goal = abs(reward_dist) / 1.25 if reward_dist <= 0 else reward_dist
                dist_arm = abs(reward_near) if reward_near <= 0 else reward_near

                raw_frame = self.env.render()
                if raw_frame is not None:
                    if self.show_hud:
                        frame = visualizer.draw_hud_on_frame(
                            frame=raw_frame,
                            timestep=self.ep_step,
                            episode=self.episode_count,
                            step=self.ep_step,
                            step_reward=float(reward),
                            total_reward=self.ep_reward,
                            reward_dist=reward_dist,
                            reward_near=reward_near,
                            actions=action,
                            tag=f"Live ({'PPO' if self.policy_type=='trained' else 'Random'})",
                        )
                    else:
                        frame = raw_frame
                else:
                    frame = np.zeros((480, 480, 3), dtype=np.uint8)

                # JPEG Encode
                buf = io.BytesIO()
                Image.fromarray(frame).save(buf, format="JPEG", quality=75)
                frame_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                # Extract 3D telemetry
                joint_angles = [float(x) for x in self.obs[0:7]] if len(self.obs) >= 7 else []
                tip_pos = [float(x) for x in self.obs[14:17]] if len(self.obs) >= 17 else [0, 0, 0]
                obj_pos = [float(x) for x in self.obs[17:20]] if len(self.obs) >= 20 else [0, 0, 0]
                goal_pos = [float(x) for x in self.obs[20:23]] if len(self.obs) >= 23 else [0, 0, 0]

                packet = {
                    "type": "sim_frame",
                    "frame": frame_b64,
                    "episode": self.episode_count,
                    "step": self.ep_step,
                    "step_reward": float(reward),
                    "ep_reward": self.ep_reward,
                    "dist_goal": dist_goal,
                    "dist_arm": dist_arm,
                    "reward_ctrl": reward_ctrl,
                    "joints": joint_angles,
                    "tip_pos": tip_pos,
                    "obj_pos": obj_pos,
                    "goal_pos": goal_pos,
                    "actions": [float(a) for a in action],
                    "policy": self.policy_type,
                    "done": done,
                }

                self.obs = next_obs
                if done:
                    self.reset_env()

                return packet
            except Exception as e:
                print(f"[SimManager Step Error] {e}")
                self.reset_env()
                return {"type": "sim_frame", "frame": "", "episode": self.episode_count, "step": 0, "step_reward": 0.0, "ep_reward": 0.0, "dist_goal": 0.0, "dist_arm": 0.0, "reward_ctrl": 0.0, "joints": [], "tip_pos": [0,0,0], "obj_pos": [0,0,0], "goal_pos": [0,0,0], "actions": [], "policy": self.policy_type, "done": True}


# Global Background Training Manager
class TrainingManager:
    def __init__(self):
        self.is_training = False
        self.progress = 0
        self.current_timesteps = 0
        self.total_timesteps = 0
        self.status = "Idle"
        self.logs = []
        self.thread = None
        self.lock = threading.Lock()

    def add_log(self, message: str):
        timestamp = time.strftime("%I:%M:%S %p")
        with self.lock:
            self.logs.append(f"[{timestamp}] {message}")
            if len(self.logs) > 200:
                self.logs.pop(0)

    def start_training(self, timesteps: int = 20000, eval_freq: int = 5000):
        with self.lock:
            if self.is_training:
                return False, "Training already in progress."
            self.is_training = True
            self.progress = 0
            self.current_timesteps = 0
            self.total_timesteps = timesteps
            self.status = "Training in progress..."
            self.logs = []

        self.add_log(f"[PPO Engine] Initializing 7-DOF MuJoCo environment...")
        self.add_log(f"[PPO Setup] Target: {timesteps:,} Timesteps | Eval Frequency: Every {eval_freq:,} Steps")

        def run():
            try:
                # Custom callback to log progress periodically
                class LiveLogCallback(train_module.VisualProgressCallback):
                    def __init__(self, manager, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.manager = manager
                        self.start_t = time.time()
                        self.last_log_step = 0

                    def _on_step(self) -> bool:
                        res = super()._on_step()
                        step = self.num_timesteps
                        if step - self.last_log_step >= 1000:
                            self.last_log_step = step
                            elapsed = max(0.1, time.time() - self.start_t)
                            fps = int(step / elapsed)
                            pct = min(100.0, (step / max(1, self.manager.total_timesteps)) * 100)
                            with self.manager.lock:
                                self.manager.progress = int(pct)
                                self.manager.current_timesteps = step
                            self.manager.add_log(
                                f"[PPO Step {step:07d}] Progress: {pct:5.1f}% | Training Speed: {fps:4d} FPS"
                            )
                        return res

                    def _record_checkpoint(self, timestep: int, tag: str):
                        super()._record_checkpoint(timestep, tag)
                        last_r = self.metrics["eval_rewards"][-1] if self.metrics["eval_rewards"] else 0.0
                        last_d = self.metrics["eval_dist_goal"][-1] if self.metrics["eval_dist_goal"] else 0.0
                        self.manager.add_log(
                            f"[PPO Checkpoint] Step {timestep:07d} | Mean Return: {last_r:+.2f} | Goal Dist: {last_d:.3f}m | Rendered Video & GIF"
                        )

                train_env = gym.make("Pusher-v5")
                eval_env = gym.make("Pusher-v5", render_mode="rgb_array")

                model = PPO(
                    policy="MlpPolicy",
                    env=train_env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    verbose=0,
                )

                cb = LiveLogCallback(
                    manager=self,
                    eval_env=eval_env,
                    eval_freq=eval_freq,
                    output_dir="./results",
                    fps=30,
                )

                self.add_log("[PPO Train] Learning loop started...")
                model.learn(total_timesteps=timesteps, callback=cb)
                model.save("./results/ppo_pusher")
                self.add_log("[Model Saved] Model weights saved to ./results/ppo_pusher.zip")

                # Generate plots & HTML & ZIP
                plot_path = "./results/plots/training_metrics.png"
                visualizer.generate_training_plots(cb.metrics, plot_path)
                dashboard_path = "./results/dashboard.html"
                visualizer.generate_html_dashboard(
                    metrics=cb.metrics,
                    checkpoint_records=cb.records,
                    plot_rel_path="plots/training_metrics.png",
                    output_html_path=dashboard_path,
                )
                zip_out = os.path.abspath("./ppo_pusher_bundle.zip")
                train_module.bundle_into_zip("./results", zip_out)
                self.add_log(f"[Artifact Bundle] Packaged ppo_pusher_bundle.zip ({os.path.getsize(zip_out)/(1024*1024):.2f} MB)")

                # Reload model in sim manager
                sim_manager.init_env_and_model()
                self.add_log("[PPO Ready] Simulation manager refreshed with latest trained weights.")

                with self.lock:
                    self.status = "Completed successfully"
                    self.progress = 100
            except Exception as e:
                self.add_log(f"[PPO Error] {e}")
                with self.lock:
                    self.status = f"Error: {e}"
            finally:
                with self.lock:
                    self.is_training = False

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return True, "Training started in background."


sim_manager = SimulationManager()
training_manager = TrainingManager()


# WebSocket endpoint for real-time simulation streaming
@app.websocket("/ws/simulation")
async def websocket_simulation(websocket: WebSocket):
    await websocket.accept()
    sim_manager.clients.add(websocket)
    print(f"[WebSocket] Client connected: {websocket.client}")

    try:
        while True:
            # Handle incoming commands if any (non-blocking)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                data = json.loads(msg)
                cmd = data.get("command")

                if cmd == "start":
                    sim_manager.is_running = True
                    sim_manager.is_paused = False
                elif cmd == "pause":
                    sim_manager.is_paused = not sim_manager.is_paused
                elif cmd == "stop":
                    sim_manager.is_running = False
                elif cmd == "step":
                    sim_manager.is_paused = True
                    packet = sim_manager.step()
                    await websocket.send_json(packet)
                elif cmd == "reset":
                    sim_manager.reset_env()
                    packet = sim_manager.step()
                    await websocket.send_json(packet)
                elif cmd == "set_policy":
                    sim_manager.policy_type = data.get("policy", "trained")
                elif cmd == "toggle_hud":
                    sim_manager.show_hud = not sim_manager.show_hud
                elif cmd == "set_speed":
                    sim_manager.speed = float(data.get("speed", 1.0))
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass

            # If simulation is active and not paused, step and stream
            if sim_manager.is_running and not sim_manager.is_paused:
                packet = sim_manager.step()
                await websocket.send_json(packet)
                # Frame rate throttling: 30 FPS default
                sleep_time = max(0.01, (1.0 / 30.0) / max(0.2, sim_manager.speed))
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected: {websocket.client}")
    except Exception as e:
        print(f"[WebSocket Error] {e}")
    finally:
        sim_manager.clients.discard(websocket)


# REST Endpoints
@app.get("/api/metrics")
def get_metrics():
    metrics_path = "./results/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"eval_timesteps": [], "eval_rewards": [], "eval_dist_goal": []}


@app.get("/api/checkpoints")
def get_checkpoints():
    videos_dir = "./results/videos"
    metrics_path = "./results/metrics.json"
    metrics_map = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                m_data = json.load(f)
                timesteps = m_data.get("eval_timesteps", [])
                rewards = m_data.get("eval_rewards", [])
                dists = m_data.get("eval_dist_goal", [])
                for i, step in enumerate(timesteps):
                    metrics_map[step] = {
                        "reward": rewards[i] if i < len(rewards) else 0.0,
                        "dist_goal": dists[i] if i < len(dists) else 0.0,
                    }
        except Exception:
            pass

    checkpoints = []
    if os.path.exists(videos_dir):
        files = sorted(os.listdir(videos_dir))
        mp4s = [f for f in files if f.endswith(".mp4")]
        for mp4 in mp4s:
            base = mp4.replace(".mp4", "")
            gif = f"{base}.gif" if f"{base}.gif" in files else None
            try:
                step_num = int(base.split("_")[-1])
            except Exception:
                step_num = 0
            
            if metrics_map and step_num not in metrics_map:
                continue

            m_info = metrics_map.get(step_num, {})
            reward_val = m_info.get("reward", None)
            dist_val = m_info.get("dist_goal", None)

            checkpoints.append({
                "name": base,
                "step": step_num,
                "reward": reward_val,
                "dist_goal": dist_val,
                "mp4": f"/results/videos/{mp4}",
                "gif": f"/results/videos/{gif}" if gif else None,
            })
    return {"checkpoints": checkpoints}


@app.get("/api/bundle/download")
def download_bundle():
    zip_path = os.path.abspath("./ppo_pusher_bundle.zip")
    if os.path.exists(zip_path):
        return FileResponse(zip_path, filename="ppo_pusher_bundle.zip", media_type="application/zip")
    raise HTTPException(status_code=404, detail="Bundle file not found. Train the model first.")


@app.post("/api/train/start")
def start_train(timesteps: int = 20000, eval_freq: int = 5000):
    success, msg = training_manager.start_training(timesteps=timesteps, eval_freq=eval_freq)
    return {"success": success, "message": msg}


@app.get("/api/train/status")
def get_train_status():
    return {
        "is_training": training_manager.is_training,
        "progress": training_manager.progress,
        "current_timesteps": training_manager.current_timesteps,
        "total_timesteps": training_manager.total_timesteps,
        "status": training_manager.status,
        "logs": training_manager.logs,
    }


# Mount Static and Results directories
os.makedirs("web", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/videos", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/", StaticFiles(directory="web", html=True), name="web")


def main():
    print("=" * 65)
    print(" Pusher-v5 Real-time Simulation & Control Center Web Server")
    print(" -> Local Web URL: http://localhost:8000")
    print("=" * 65)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
