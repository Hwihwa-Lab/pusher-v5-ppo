# 📊 Pusher-v5 데이터 스키마 및 규격서

AI가 환경을 조작하거나 메트릭/로그/대시보드/웹소켓 API를 다룰 때 반드시 준수해야 하는 엄격한 데이터 규격입니다. 여기에 정의되지 않은 변수나 임의의 필드를 추가/삭제하지 마십시오.

---

## 1. Pusher-v5 환경 규격 (Environment Specification)

*   **Observation Space (23차원 연속 벡터, Box(-inf, inf, (23,), float64))**:
    *   `obs[0:7]`: 7개 조인트 각도 (Joint angles, rad)
    *   `obs[7:14]`: 7개 조인트 각속도 (Joint velocities)
    *   `obs[14:17]`: Fingertip / Pusher 끝단 3D 위치 (x, y, z)
    *   `obs[17:20]`: 물체(Object/Cylinder) 3D 위치 (x, y, z)
    *   `obs[20:23]`: 목표 지점(Goal/Target) 3D 위치 (x, y, z)
*   **Action Space (7차원 연속 벡터, Box(-2.0, 2.0, (7,), float32))**:
    *   7개 로봇 팔 조인트에 가해지는 토크(Torque) 제어값.
*   **Step Return `info` Dictionary**:
    ```python
    {
        "reward_dist": float,  # 물체와 목표 지점 간 거리 기반 음수 보상 (-norm(obj-goal) * 1.25)
        "reward_ctrl": float,  # 제어 토크 페널티 (-norm(action)^2 * 0.1)
        "reward_near": float   # 로봇 팔 끝단과 물체 간 거리 기반 음수 보상 (-norm(tip-obj))
    }
    ```

---

## 2. WebSocket 실시간 스트리밍 패킷 스키마 (`/ws/simulation`)

### Server &rarr; Client (`sim_frame`):
```json
{
  "type": "sim_frame",
  "frame": "<base64_encoded_jpeg_string>",
  "episode": 1,
  "step": 42,
  "step_reward": -0.452,
  "ep_reward": -18.24,
  "dist_goal": 0.220,
  "dist_arm": 0.185,
  "reward_ctrl": -0.082,
  "joints": [0.12, -0.45, 0.88, -1.20, 0.05, 0.32, -0.15],
  "tip_pos": [0.35, -0.12, 0.04],
  "obj_pos": [0.45, 0.05, -0.05],
  "goal_pos": [0.60, 0.10, -0.05],
  "actions": [0.1, -0.2, 0.5, -0.8, 0.0, 0.2, -0.1],
  "policy": "trained",
  "done": false
}
```

### Client &rarr; Server Command:
```json
{ "command": "start" }
{ "command": "pause" }
{ "command": "step" }
{ "command": "reset" }
{ "command": "set_policy", "policy": "trained" }
{ "command": "toggle_hud" }
{ "command": "set_speed", "speed": 1.0 }
```

---

## 3. 학습 메트릭 스키마 (`results/metrics.json`)

```json
{
  "eval_timesteps": [0, 10000, 20000, 30000],
  "eval_rewards": [-57.51, -50.12, -49.22, -49.25],
  "eval_lengths": [100, 100, 100, 100],
  "eval_dist_goal": [0.220, 0.220, 0.220, 0.220],
  "eval_dist_arm": [0.302, 0.245, 0.210, 0.187],
  "total_timesteps": 30720
}
```

## 4. 실행 설정 스키마 (`results/config.json`)

```json
{
  "timesteps": 500000,
  "eval_freq": 50000,
  "n_eval_episodes": 2,
  "seed": 42,
  "output_dir": "./results",
  "zip_name": "ppo_pusher_bundle.zip",
  "learning_rate": 0.0003
}
```

---

## 5. 학습 상태 및 실시간 로그 스키마 (`/api/train/status`)

```json
{
  "is_training": true,
  "progress": 45,
  "current_timesteps": 225000,
  "total_timesteps": 500000,
  "status": "Training in progress... (45%)",
  "logs": [
    "[11:50:12 AM] [PPO Step 050000] Progress:  10.0% | Training Speed: 2150 FPS",
    "[11:50:24 AM] [PPO Step 100000] Progress:  20.0% | Training Speed: 2180 FPS"
  ]
}
```

---

## 6. 체크포인트 갤러리 메타데이터 스키마 (`/api/checkpoints`)

```json
{
  "checkpoints": [
    {
      "name": "step_0000000",
      "step": 0,
      "reward": -57.51,
      "dist_goal": 0.220,
      "mp4": "/results/videos/step_0000000.mp4",
      "gif": "/results/videos/step_0000000.gif"
    }
  ]
}
```
