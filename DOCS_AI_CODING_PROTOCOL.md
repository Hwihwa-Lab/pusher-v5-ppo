# 🛡️ AI Coding Protocol (Pusher-v5 강화학습 마스터 지침서)

본 문서는 AI가 `pusherppotest` 프로젝트에서 작업할 때 '바이브 코딩(Vibe Coding)'과 강화학습 환경 규격 왜곡을 방지하기 위한 마스터 가이드(목차)입니다.

---

## 📂 문서 맵핑 (Document Mapping)
AI는 코드 작성 및 디버깅 전, 작업 내용에 맞춰 아래의 문서를 **반드시 선행 로드**하고 숙지해야 합니다.

*   **시스템 아키텍처 및 실행 파이프라인 변경 시:** `DOCS_SYSTEM_ARCHITECTURE.md` 필수 확인
*   **환경 규격, 보상 함수, 로그/메트릭 JSON 스키마 변경 시:** `DOCS_DATA_SCHEMA.md` 필수 확인

---

## 🛑 핵심 방어 수칙 (Guardrails)
1. **MuJoCo 환경 규격 왜곡 금지:** `Pusher-v5` 환경의 관측(23차원), 행동(7차원), `info` 딕셔너리 키(`reward_dist`, `reward_near`, `reward_ctrl`)를 임의로 변경하지 마십시오.
2. **시각화 파이프라인 보존:** 학습 스크립트 수정 시 실시간 HUD 프레임 렌더링, 단계별 MP4/GIF 생성, 4분할 분석 차트(`plots/training_metrics.png`), 대시보드(`dashboard.html`), 자동 압축 패키징(`ppo_pusher_bundle.zip`)이 상시 작동해야 합니다.
3. **독립 평가 일관성 유지:** `evaluate.py`는 저장된 모든 SB3 PPO 모델(`.zip`)과 완벽히 호환되어야 하며, 독립적으로 고화질 영상과 통계를 추출할 수 있어야 합니다.
4. **문서 자동 동기화:** CLI 옵션(`--timesteps`, `--eval_freq` 등)이나 메트릭 키를 추가/수정할 경우 즉시 `DOCS_*.md` 문서를 갱신하십시오.
