"""
visualizer.py
Pusher-v5 시각화 및 대시보드 생성 유틸리티 모듈
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw_hud_on_frame(
    frame: np.ndarray,
    timestep: int,
    episode: int,
    step: int,
    step_reward: float,
    total_reward: float,
    reward_dist: float = None,
    reward_near: float = None,
    actions: np.ndarray = None,
    tag: str = ""
) -> np.ndarray:
    """
    MuJoCo 렌더링 프레임 상단에 실시간 상태 정보 HUD(Heads-Up Display)를 오버레이합니다.
    """
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 기본 폰트 로드
    try:
        font_large = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
        font_bold = ImageFont.truetype("arialbd.ttf", 14)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_bold = ImageFont.load_default()

    # 상단 HUD 패널 (반투명 블랙 글래스모피즘 박스)
    panel_w = 260
    panel_h = 135
    margin = 12
    draw.rounded_rectangle(
        [margin, margin, margin + panel_w, margin + panel_h],
        radius=10,
        fill=(15, 23, 42, 210),  # Slate-900 semi-transparent
        outline=(56, 189, 248, 180),  # Cyan accent border
        width=2,
    )

    # 타이틀 / 태그
    title_text = f"Pusher-v5 PPO  [{tag or f'Step {timestep:,}'}]"
    draw.text((margin + 12, margin + 8), title_text, fill=(56, 189, 248, 255), font=font_bold)

    # 정보 텍스트 라인
    lines = [
        (f"Global Step : {timestep:,}", (226, 232, 240, 255)),
        (f"Episode / Step : #{episode} (Step {step})", (203, 213, 225, 255)),
        (f"Cumulative Rwd : {total_reward:+.2f} (Step: {step_reward:+.3f})", (74, 222, 128, 255)),
    ]

    if reward_dist is not None:
        # Pusher-v5 distance reward is approx -norm(obj-goal)*1.25
        dist_goal = abs(reward_dist) / 1.25 if reward_dist <= 0 else reward_dist
        lines.append((f"Dist (Obj -> Goal) : {dist_goal:.3f} m", (251, 191, 36, 255)))

    if reward_near is not None:
        dist_arm = abs(reward_near) if reward_near <= 0 else reward_near
        lines.append((f"Dist (Arm -> Obj)  : {dist_arm:.3f} m", (168, 85, 247, 255)))

    y_offset = margin + 28
    for line_text, color in lines:
        draw.text((margin + 12, y_offset), line_text, fill=color, font=font_small)
        y_offset += 18

    # 하단 액션 바 게이지 오버레이 (옵션)
    if actions is not None and len(actions) > 0:
        act_panel_y = img.height - 32
        draw.rounded_rectangle(
            [margin, act_panel_y, img.width - margin, img.height - 8],
            radius=6,
            fill=(15, 23, 42, 190),
            outline=(100, 116, 139, 120),
            width=1,
        )
        act_text = "Act: " + " ".join([f"{a:+.1f}" for a in actions[:7]])
        draw.text((margin + 10, act_panel_y + 4), act_text, fill=(148, 163, 184, 255), font=font_small)

    combined = Image.alpha_composite(img, overlay).convert("RGB")
    return np.array(combined)


def save_video_and_gif(frames: list, output_base_path: str, fps: int = 30):
    """
    프레임 리스트를 MP4 비디오 및 GIF 애니메이션으로 저장합니다.
    """
    if not frames:
        return

    os.makedirs(os.path.dirname(output_base_path), exist_ok=True)
    gif_path = f"{output_base_path}.gif"
    mp4_path = f"{output_base_path}.mp4"

    # Save GIF
    try:
        # 프레임이 너무 많을 경우 GIF 용량 최적화 (2프레임마다 1개 샘플링)
        sampled_frames = frames[::2] if len(frames) > 100 else frames
        imageio.mimsave(gif_path, sampled_frames, fps=max(15, fps // 2), loop=0)
    except Exception as e:
        print(f"[Warning] GIF 저장 실패: {e}")

    # Save MP4
    try:
        imageio.mimsave(mp4_path, frames, fps=fps, quality=8)
    except Exception as e:
        # Fallback to default writer
        try:
            writer = imageio.get_writer(mp4_path, fps=fps)
            for f in frames:
                writer.append_data(f)
            writer.close()
        except Exception as e2:
            print(f"[Warning] MP4 저장 실패: {e2}")

    return gif_path, mp4_path


def generate_training_plots(metrics: dict, output_path: str):
    """
    학습 과정의 지표들을 미려한 다크 테마 4분할 차트로 시각화하여 저장합니다.
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    fig.patch.set_facecolor("#0f172a")

    for row in axes:
        for ax in row:
            ax.set_facecolor("#1e293b")
            ax.grid(True, linestyle="--", alpha=0.3, color="#94a3b8")
            ax.tick_params(colors="#cbd5e1")
            for spine in ax.spines.values():
                spine.set_color("#475569")

    # 1. Episode Returns
    eval_timesteps = metrics.get("eval_timesteps", [])
    eval_rewards = metrics.get("eval_rewards", [])
    ax1 = axes[0, 0]
    ax1.set_title("Evaluation Episode Return over Timesteps", color="#38bdf8", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Timesteps", color="#cbd5e1")
    ax1.set_ylabel("Mean Return", color="#cbd5e1")
    if eval_timesteps and eval_rewards:
        ax1.plot(eval_timesteps, eval_rewards, color="#38bdf8", marker="o", linewidth=2.5, label="Mean Reward")
        # Moving average
        if len(eval_rewards) >= 3:
            window = min(5, len(eval_rewards))
            smooth = np.convolve(eval_rewards, np.ones(window)/window, mode='valid')
            smooth_x = eval_timesteps[window-1:]
            ax1.plot(smooth_x, smooth, color="#f43f5e", linestyle="--", linewidth=2, label="Trend (MA)")
        ax1.legend(loc="lower right", facecolor="#1e293b", edgecolor="#475569")

    # 2. Distance to Goal (Object -> Target)
    eval_dist_goal = metrics.get("eval_dist_goal", [])
    ax2 = axes[0, 1]
    ax2.set_title("Final Object-to-Goal Distance (Lower is Better)", color="#34d399", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Timesteps", color="#cbd5e1")
    ax2.set_ylabel("Final Distance (meters)", color="#cbd5e1")
    if eval_timesteps and eval_dist_goal:
        ax2.plot(eval_timesteps, eval_dist_goal, color="#34d399", marker="s", linewidth=2.5, label="Distance to Goal")
        ax2.axhline(0.05, color="#f59e0b", linestyle=":", label="Success Threshold (~0.05m)")
        ax2.legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569")

    # 3. Distance Arm to Object
    eval_dist_arm = metrics.get("eval_dist_arm", [])
    ax3 = axes[1, 0]
    ax3.set_title("Arm-to-Object Distance (Reaching Efficiency)", color="#a855f7", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Timesteps", color="#cbd5e1")
    ax3.set_ylabel("Distance (meters)", color="#cbd5e1")
    if eval_timesteps and eval_dist_arm:
        ax3.plot(eval_timesteps, eval_dist_arm, color="#a855f7", marker="^", linewidth=2.5, label="Arm-to-Object")
        ax3.legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569")

    # 4. Episode Lengths or Loss
    eval_lens = metrics.get("eval_lengths", [])
    ax4 = axes[1, 1]
    ax4.set_title("Evaluation Episode Length", color="#fbbf24", fontsize=13, fontweight="bold", pad=10)
    ax4.set_xlabel("Timesteps", color="#cbd5e1")
    ax4.set_ylabel("Steps", color="#cbd5e1")
    if eval_timesteps and eval_lens:
        ax4.bar(eval_timesteps, eval_lens, width=max(500, (eval_timesteps[-1] - eval_timesteps[0])//(len(eval_timesteps)*2) if len(eval_timesteps)>1 else 500), color="#fbbf24", alpha=0.7, label="Ep Length")
        ax4.legend(loc="lower right", facecolor="#1e293b", edgecolor="#475569")

    plt.tight_layout(pad=2.5)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return output_path


def generate_html_dashboard(
    metrics: dict,
    checkpoint_records: list,
    plot_rel_path: str,
    output_html_path: str
):
    """
    모든 비디오/GIF 및 메트릭 차트를 한눈에 탐색할 수 있는 인터랙티브 HTML 대시보드를 생성합니다.
    """
    best_reward = max(metrics.get("eval_rewards", [0])) if metrics.get("eval_rewards") else 0
    final_reward = metrics.get("eval_rewards", [0])[-1] if metrics.get("eval_rewards") else 0
    best_dist = min(metrics.get("eval_dist_goal", [0])) if metrics.get("eval_dist_goal") else 0
    total_timesteps = metrics.get("total_timesteps", 0)

    # HTML 템플릿 생성
    cards_html = ""
    for rec in checkpoint_records:
        tag = rec.get("tag", "")
        step = rec.get("timestep", 0)
        reward = rec.get("reward", 0.0)
        dist = rec.get("dist_goal", 0.0)
        gif_rel = rec.get("gif_rel", "")
        mp4_rel = rec.get("mp4_rel", "")

        cards_html += f"""
        <div class="checkpoint-card">
            <div class="card-header">
                <span class="badge">{tag}</span>
                <span class="step-label">Step: {step:,}</span>
            </div>
            <div class="video-container">
                <video autoplay loop muted playsinline poster="{gif_rel}">
                    <source src="{mp4_rel}" type="video/mp4">
                    <img src="{gif_rel}" alt="{tag} GIF preview">
                </video>
            </div>
            <div class="card-stats">
                <div class="stat-item">
                    <span class="stat-name">Mean Reward</span>
                    <span class="stat-val {'text-green' if reward > -30 else 'text-yellow'}">{reward:+.2f}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-name">Obj-Goal Dist</span>
                    <span class="stat-val text-cyan">{dist:.3f} m</span>
                </div>
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pusher-v5 PPO 강화학습 시각화 대시보드</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-base: #090d16;
            --bg-surface: #0f172a;
            --bg-card: #1e293b;
            --border-color: rgba(148, 163, 184, 0.15);
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.35);
            --accent-green: #34d399;
            --accent-purple: #a855f7;
            --accent-amber: #fbbf24;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            padding: 32px 24px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 32px;
        }}
        .header-title h1 {{
            font-size: 28px;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }}
        .header-title p {{
            color: var(--text-muted);
            font-size: 14px;
            margin-top: 4px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 36px;
        }}
        .stat-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
            position: relative;
            overflow: hidden;
        }}
        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 3px;
            background: linear-gradient(90deg, var(--primary), transparent);
        }}
        .stat-card.green::before {{ background: linear-gradient(90deg, var(--accent-green), transparent); }}
        .stat-card.purple::before {{ background: linear-gradient(90deg, var(--accent-purple), transparent); }}
        .stat-card.amber::before {{ background: linear-gradient(90deg, var(--accent-amber), transparent); }}
        .stat-card-title {{
            font-size: 13px;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        .stat-card-value {{
            font-size: 28px;
            font-weight: 700;
            margin-top: 8px;
            font-family: 'JetBrains Mono', monospace;
        }}
        .section-title {{
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section-title::before {{
            content: '';
            display: inline-block;
            width: 4px;
            height: 20px;
            background: var(--primary);
            border-radius: 2px;
        }}
        .checkpoints-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 24px;
            margin-bottom: 48px;
        }}
        .checkpoint-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .checkpoint-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 20px 30px -10px var(--primary-glow);
            border-color: rgba(56, 189, 248, 0.4);
        }}
        .card-header {{
            padding: 14px 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(30, 41, 59, 0.5);
            border-bottom: 1px solid var(--border-color);
        }}
        .badge {{
            background: rgba(56, 189, 248, 0.15);
            color: var(--primary);
            font-size: 12px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 20px;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }}
        .step-label {{
            font-size: 12px;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }}
        .video-container {{
            width: 100%;
            background: #000;
            aspect-ratio: 1 / 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .video-container video, .video-container img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        .card-stats {{
            padding: 14px 18px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            background: var(--bg-card);
        }}
        .stat-item {{
            display: flex;
            flex-direction: column;
        }}
        .stat-name {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
        }}
        .stat-val {{
            font-size: 16px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            margin-top: 2px;
        }}
        .text-green {{ color: var(--accent-green); }}
        .text-yellow {{ color: var(--accent-amber); }}
        .text-cyan {{ color: var(--primary); }}
        .text-purple {{ color: var(--accent-purple); }}
        
        .charts-section {{
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 48px;
        }}
        .charts-container img {{
            width: 100%;
            height: auto;
            border-radius: 12px;
            display: block;
        }}
        footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 13px;
            padding-top: 24px;
            border-top: 1px solid var(--border-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">
                <h1>Pusher-v5 PPO 학습 시각화 리포트</h1>
                <p>Gymnasium MuJoCo Robotic Arm Manipulation &bull; Stable-Baselines3 PPO</p>
            </div>
            <div>
                <span class="badge" style="font-size: 14px; padding: 6px 14px;">Total Steps: {total_timesteps:,}</span>
            </div>
        </header>

        <!-- 핵심 요약 지표 카드 -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-card-title">Total Timesteps</div>
                <div class="stat-card-value text-cyan">{total_timesteps:,}</div>
            </div>
            <div class="stat-card green">
                <div class="stat-card-title">Best Return</div>
                <div class="stat-card-value text-green">{best_reward:+.2f}</div>
            </div>
            <div class="stat-card purple">
                <div class="stat-card-title">Final Return</div>
                <div class="stat-card-value text-purple">{final_reward:+.2f}</div>
            </div>
            <div class="stat-card amber">
                <div class="stat-card-title">Best Obj-to-Goal Dist</div>
                <div class="stat-card-value text-yellow">{best_dist:.3f} m</div>
            </div>
        </div>

        <!-- 단계별 학습 진행 비디오/GIF -->
        <div class="section-title">학습 단계별 행동 변화 (Visual Progressions)</div>
        <div class="checkpoints-grid">
            {cards_html}
        </div>

        <!-- 종합 분석 차트 -->
        <div class="section-title">학습 수렴 및 메트릭 분석 차트</div>
        <div class="charts-section">
            <div class="charts-container">
                <img src="{plot_rel_path}" alt="PPO Training Metrics Plots">
            </div>
        </div>

        <footer>
            Pusher-v5 PPO Visualizer &bull; Generated automatically with Full Artifact Package
        </footer>
    </div>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html_path
