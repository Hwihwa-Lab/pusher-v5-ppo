"""
train.py
Gymnasium Pusher-v5 환경에서 PPO 강화학습을 수행하고,
전체 학습 과정을 실시간 시각화, 비디오/GIF 생성, 인터랙티브 대시보드 및 압축 파일로 패키징합니다.
"""

import os
import sys
import json
import time
import shutil
import zipfile
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import visualizer


class VisualProgressCallback(BaseCallback):
    """
    학습 과정(Step 0, 주기적 체크포인트, 최종 단계)마다
    에피소드 주행 영상을 HUD와 함께 렌더링하여 비디오/GIF로 저장하고
    성능 지표(보상, 목표 거리, 팔-물체 거리 등)를 추적하는 콜백입니다.
    """
    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 2,
        output_dir: str = "./results",
        fps: int = 30,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.output_dir = output_dir
        self.fps = fps
        self.video_dir = os.path.join(output_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        self.records = []
        self.metrics = {
            "eval_timesteps": [],
            "eval_rewards": [],
            "eval_lengths": [],
            "eval_dist_goal": [],
            "eval_dist_arm": [],
            "total_timesteps": 0,
        }
        self.last_eval_step = -1

    def _on_training_start(self) -> None:
        """학습 시작 직후 Step 0 (초기 무작위 상태) 기준선 평가 및 비디오 녹화"""
        print("\n" + "=" * 65)
        print(" [Visual Callback] Step 0 초기 기준선(Baseline) 시각화 렌더링 시작...")
        print("=" * 65)
        self._record_checkpoint(timestep=0, tag="Baseline (Step 0)")

    def _on_step(self) -> bool:
        """지정된 eval_freq마다 정기 평가 및 비디오/GIF 생성"""
        if self.n_calls % self.eval_freq == 0 and self.n_calls != self.last_eval_step:
            self.last_eval_step = self.n_calls
            tag = f"Checkpoint ({self.num_timesteps:,} steps)"
            print("\n" + "-" * 65)
            print(f" [Visual Callback] {tag} 평가 및 시각화 렌더링...")
            print("-" * 65)
            self._record_checkpoint(timestep=self.num_timesteps, tag=tag)
        return True

    def _on_training_end(self) -> None:
        """학습 완료 후 최종 모델 시각화 렌더링"""
        if self.last_eval_step != self.num_timesteps:
            print("\n" + "=" * 65)
            print(f" [Visual Callback] 최종(Final) 완료 모델 시각화 렌더링 (Step {self.num_timesteps:,})...")
            print("=" * 65)
            self._record_checkpoint(timestep=self.num_timesteps, tag=f"Final ({self.num_timesteps:,} steps)")

    def _record_checkpoint(self, timestep: int, tag: str):
        episode_rewards = []
        episode_lengths = []
        episode_dists_goal = []
        episode_dists_arm = []

        all_rendered_frames = []

        for ep in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset(seed=42 + ep)
            done = False
            ep_reward = 0.0
            ep_step = 0
            ep_frames = []

            final_dist_goal = 0.0
            final_dist_arm = 0.0

            while not done:
                # Deterministic action if trained, else stochastic/policy prediction
                action, _ = self.model.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, step_info = self.eval_env.step(action)
                ep_reward += float(reward)
                ep_step += 1
                done = terminated or truncated

                # Distance metrics extraction
                reward_dist = step_info.get("reward_dist", 0.0)
                reward_near = step_info.get("reward_near", 0.0)
                final_dist_goal = abs(reward_dist) / 1.25 if reward_dist <= 0 else reward_dist
                final_dist_arm = abs(reward_near) if reward_near <= 0 else reward_near

                # 첫 번째 에피소드만 고화질 비디오/GIF로 렌더링하여 디스크 절약
                if ep == 0:
                    raw_frame = self.eval_env.render()
                    if raw_frame is not None:
                        hud_frame = visualizer.draw_hud_on_frame(
                            frame=raw_frame,
                            timestep=timestep,
                            episode=ep + 1,
                            step=ep_step,
                            step_reward=float(reward),
                            total_reward=ep_reward,
                            reward_dist=reward_dist,
                            reward_near=reward_near,
                            actions=action,
                            tag=tag,
                        )
                        ep_frames.append(hud_frame)

                obs = next_obs

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_step)
            episode_dists_goal.append(final_dist_goal)
            episode_dists_arm.append(final_dist_arm)

            if ep == 0:
                all_rendered_frames = ep_frames

        mean_reward = float(np.mean(episode_rewards))
        mean_len = float(np.mean(episode_lengths))
        mean_dist_goal = float(np.mean(episode_dists_goal))
        mean_dist_arm = float(np.mean(episode_dists_arm))

        # 메트릭 누적
        self.metrics["eval_timesteps"].append(timestep)
        self.metrics["eval_rewards"].append(mean_reward)
        self.metrics["eval_lengths"].append(mean_len)
        self.metrics["eval_dist_goal"].append(mean_dist_goal)
        self.metrics["eval_dist_arm"].append(mean_dist_arm)
        self.metrics["total_timesteps"] = timestep

        # 비디오 및 GIF 저장
        file_prefix = f"step_{timestep:07d}"
        base_path = os.path.join(self.video_dir, file_prefix)
        visualizer.save_video_and_gif(all_rendered_frames, base_path, fps=self.fps)

        rel_gif = f"videos/{file_prefix}.gif"
        rel_mp4 = f"videos/{file_prefix}.mp4"

        record_entry = {
            "tag": tag,
            "timestep": timestep,
            "reward": mean_reward,
            "dist_goal": mean_dist_goal,
            "dist_arm": mean_dist_arm,
            "gif_rel": rel_gif,
            "mp4_rel": rel_mp4,
        }
        self.records.append(record_entry)

        print(f" -> [결과] Return: {mean_reward:+.2f} | Obj-to-Goal Dist: {mean_dist_goal:.3f}m | Arm-to-Obj Dist: {mean_dist_arm:.3f}m")
        print(f" -> [저장] 비디오: {rel_mp4} | GIF: {rel_gif}\n")


def bundle_into_zip(source_dir: str, output_zip_path: str):
    """
    학습 산출물 디렉토리 전체를 ZIP 압축 파일로 패키징합니다.
    """
    print(f"\n[Packaging] 전체 결과물 압축 중... -> {output_zip_path}")
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # ZIP 내부 상대 경로 계산
                arcname = os.path.relpath(file_path, start=source_dir)
                zf.write(file_path, arcname)
    print(f"[Packaging 완료] 압축 파일 생성됨: {output_zip_path} ({os.path.getsize(output_zip_path) / (1024*1024):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Pusher-v5 PPO 강화학습 및 시각화 극대화")
    parser.add_argument("--timesteps", type=int, default=50000, help="총 학습 타임스텝 수 (기본: 50,000)")
    parser.add_argument("--eval_freq", type=int, default=10000, help="시각화 및 평가 주기 (기본: 10,000)")
    parser.add_argument("--n_eval_episodes", type=int, default=2, help="각 체크포인트 평가 에피소드 수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--output_dir", type=str, default="./results", help="결과물 저장 폴더")
    parser.add_argument("--zip_name", type=str, default="ppo_pusher_bundle.zip", help="최종 압축 파일 이름")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="PPO 학습률")
    args = parser.parse_args()

    # 출력 폴더 초기화
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print(" Pusher-v5 PPO 강화학습 및 시각화 파이프라인 시작")
    print(f" - 총 Timesteps    : {args.timesteps:,}")
    print(f" - 시각화 체크포인트: 매 {args.eval_freq:,} steps")
    print(f" - 결과물 디렉토리  : {os.path.abspath(args.output_dir)}")
    print(f" - 압축 패키지 이름 : {args.zip_name}")
    print("=" * 65)

    # 1. 학습 및 평가 환경 생성
    train_env = gym.make("Pusher-v5")
    eval_env = gym.make("Pusher-v5", render_mode="rgb_array")

    # 2. PPO 모델 초기화 (MlpPolicy)
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        seed=args.seed,
    )

    # 3. 시각화 및 평가 콜백 설정
    visual_callback = VisualProgressCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        output_dir=args.output_dir,
        fps=30,
    )

    # 4. 학습 시작
    start_time = time.time()
    print("\n[Training] PPO 학습을 시작합니다...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=visual_callback,
        progress_bar=True,
    )
    elapsed = time.time() - start_time
    print(f"\n[Training 완료] 총 소요 시간: {elapsed:.2f}초 ({elapsed/60:.2f}분)")

    # 5. SB3 모델 저장
    model_save_path = os.path.join(args.output_dir, "ppo_pusher")
    model.save(model_save_path)
    print(f"[Model Saved] SB3 모델 저장 완료: {model_save_path}.zip")

    # 6. 메트릭 JSON 및 설정 저장
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(visual_callback.metrics, f, indent=2)

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # 7. 종합 분석 차트 생성
    plot_path = os.path.join(args.output_dir, "plots", "training_metrics.png")
    visualizer.generate_training_plots(visual_callback.metrics, plot_path)
    print(f"[Plots Generated] 학습 메트릭 차트 생성 완료: {plot_path}")

    # 8. 인터랙티브 HTML 대시보드 생성
    dashboard_path = os.path.join(args.output_dir, "dashboard.html")
    visualizer.generate_html_dashboard(
        metrics=visual_callback.metrics,
        checkpoint_records=visual_callback.records,
        plot_rel_path="plots/training_metrics.png",
        output_html_path=dashboard_path,
    )
    print(f"[Dashboard Generated] 웹 시각화 대시보드 생성 완료: {dashboard_path}")

    # 9. 전체 결과물 ZIP 압축 패키징
    output_zip_file = os.path.join(args.output_dir, "..", args.zip_name)
    output_zip_file = os.path.abspath(output_zip_file)
    bundle_into_zip(args.output_dir, output_zip_file)

    # 환경 종료
    train_env.close()
    eval_env.close()

    print("\n" + "=" * 65)
    print(" 모든 학습, 시각화, 대시보드 생성 및 압축 패키징이 성공적으로 완료되었습니다!")
    print(f" -> 압축 파일 경로 : {output_zip_file}")
    print(f" -> 웹 대시보드   : {dashboard_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
