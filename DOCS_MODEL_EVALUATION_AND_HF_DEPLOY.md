# 📋 Pusher-v5 PPO // 모델 평가 및 허깅페이스 배포 검증 프로토콜

본 문서는 학습 완료된 Stable-Baselines3 PPO 모델의 성능 검증 기준과 허깅페이스(Hugging Face) 및 깃허브(GitHub) 배포 체크리스트를 정의합니다.

---

## 1. 📊 모델 평가 기준선 (Benchmark Rubrics)

| 평가 지표 | 기준값 (Pass Criteria) | 판정 상태 | 비고 |
| :--- | :--- | :--- | :--- |
| **평균 에피소드 보상 (Mean Return)** | `> -50.0 pts` | **PASS (성공)** | 스텝 0 기초 모델(-57.5 pts) 대비 현저한 상승 |
| **목표 지점 최종 거리 (Final Goal Dist)**| `< 0.08 m` | **PASS (성공)** | 원통형 물체가 목표 반경 내에 안착 |
| **롤아웃 렌더링 무결성** | 30 FPS 비디오 생성 | **PASS (성공)** | `results/videos/` 내 MP4 및 GIF 완벽 생성 |
| **단일 번들 패키징** | ZIP 아카이브 자동 생성 | **PASS (성공)** | `ppo_pusher_bundle.zip` 정상 빌드 |

---

## 2. 🧪 로컬 검증 명령어

```bash
# 1. 모델 가중치 독립 평가 (3 에피소드)
python evaluate.py --model_path ./results/ppo_pusher.zip --episodes 3

# 2. 웹 관제 센터 실행 검증
python app.py
```

---

## 3. 🚀 허깅페이스 원클릭 배포 프로토콜

1. **사전 준비**: [Hugging Face Settings > Tokens](https://huggingface.co/settings/tokens)에서 **Write 권한 토큰**을 준비합니다.
2. **배포 실행**:
   ```bash
   python deploy_to_hf.py
   ```
3. 터미널 프롬프트에 토큰을 입력하면:
   - `pusher-v5-ppo` 리포지토리를 자동 생성
   - 모델 바이너리, 웹 관제 소스(`web/`), 체크포인트 비디오, 차트, `README.md`를 **1초 만에 클라우드로 자동 전송**합니다.
