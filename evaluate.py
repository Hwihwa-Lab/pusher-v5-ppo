"""
evaluate.py
학습 완료된 Pusher-v5 PPO 모델을 로드하여 성능을 정밀 평가하고 고화질 비디오/GIF를 생성합니다.
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import visualizer


def evaluate_model(
    model_path: str,
    episodes: int = 5,
    output_dir: str = "./eval_results",
    seed: int = 100,
    fps: int = 30,
):
    """
    저장된 PPO 모델을 로드하여 테스트 에피소드를 수행하고 결과를 시각화합니다.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print(f" [Evaluation] 모델 로드 중: {model_path}")
    print(f" - 평가 에피소드 수: {episodes}")
    print(f" - 출력 디렉토리    : {output_dir}")
    print("=" * 65)

    # 환경 및 모델 로드
    env = gym.make("Pusher-v5", render_mode="rgb_array")
    model = PPO.load(model_path, env=env)

    all_rewards = []
    all_dists_goal = []
    all_dists_arm = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        ep_step = 0
        frames = []

        final_dist_goal = 0.0
        final_dist_arm = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += float(reward)
            ep_step += 1
            done = terminated or truncated

            reward_dist = step_info.get("reward_dist", 0.0)
            reward_near = step_info.get("reward_near", 0.0)
            final_dist_goal = abs(reward_dist) / 1.25 if reward_dist <= 0 else reward_dist
            final_dist_arm = abs(reward_near) if reward_near <= 0 else reward_near

            raw_frame = env.render()
            if raw_frame is not None:
                hud_frame = visualizer.draw_hud_on_frame(
                    frame=raw_frame,
                    timestep=-1,
                    episode=ep + 1,
                    step=ep_step,
                    step_reward=float(reward),
                    total_reward=ep_reward,
                    reward_dist=reward_dist,
                    reward_near=reward_near,
                    actions=action,
                    tag=f"Eval Ep {ep+1}",
                )
                frames.append(hud_frame)

            obs = next_obs

        all_rewards.append(ep_reward)
        all_dists_goal.append(final_dist_goal)
        all_dists_arm.append(final_dist_arm)

        # 에피소드별 비디오/GIF 저장
        out_base = os.path.join(output_dir, f"eval_episode_{ep+1}")
        gif_path, mp4_path = visualizer.save_video_and_gif(frames, out_base, fps=fps)

        print(f"Episode {ep+1:02d} | Return: {ep_reward:+.2f} | Final Goal Dist: {final_dist_goal:.3f}m | Arm Dist: {final_dist_arm:.3f}m")
        print(f"   -> 비디오: {mp4_path} | GIF: {gif_path}")

    env.close()

    print("\n" + "=" * 65)
    print(" [Evaluation Summary]")
    print(f" - 평균 보상 (Mean Return)          : {np.mean(all_rewards):+.2f} ± {np.std(all_rewards):.2f}")
    print(f" - 평균 목표 거리 (Mean Goal Dist)   : {np.mean(all_dists_goal):.3f}m")
    print(f" - 평균 팔-물체 거리 (Mean Arm Dist): {np.mean(all_dists_arm):.3f}m")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Pusher-v5 PPO 모델 독립 평가 및 비디오 생성")
    parser.add_argument("--model_path", type=str, default="./results/ppo_pusher.zip", help="평가할 모델 파일 경로 (.zip)")
    parser.add_argument("--episodes", type=int, default=3, help="평가할 에피소드 수 (기본: 3)")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="평가 결과물 저장 디렉토리")
    parser.add_argument("--seed", type=int, default=100, help="평가 랜덤 시드")
    parser.add_argument("--fps", type=int, default=30, help="비디오 재생 FPS")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        episodes=args.episodes,
        output_dir=args.output_dir,
        seed=args.seed,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
