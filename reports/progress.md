# 슈퍼마리오 DQN 프로젝트 진행 현황

> 최초 작성일: 2026-06-25 | 최종 수정일: 2026-06-26

---

## 프로젝트 기본 정보

| 항목 | 내용 |
|---|---|
| 주제 | Deep Q-Network(DQN)으로 슈퍼마리오 자율 플레이 학습 |
| 기간 | 2026-06-25 ~ 2026-07-01 (7일) |
| 인원 | 1인 |
| 학습 환경 | Google Colab (T4 GPU) + Google Drive 연동 |
| 시연 | Streamlit 웹앱 (4탭 구성) |
| GitHub | https://github.com/HyunDove/supermario-dqn |

---

## 완료된 작업

### 프로젝트 구조 설계 및 코드 구현

| 파일 | 내용 |
|---|---|
| `src/env/wrappers.py` | 환경 전처리 파이프라인 (SkipFrame·GrayScale·Resize·FrameStack) |
| `src/model/dqn.py` | CNN 기반 DQN 신경망 (Conv2d×3 + FC×2) |
| `src/agent/dqn_agent.py` | DQN 에이전트 (ε-greedy·학습·저장/로드) |
| `src/utils/replay_buffer.py` | Experience Replay Buffer (deque 기반) |
| `src/utils/recorder.py` | 에피소드 플레이 MP4 녹화 (2배속·30초) |
| `train.py` | 로컬 학습 실행 진입점 |
| `app/streamlit_app.py` | Streamlit 시연 웹앱 (4탭 구성) |
| `colab/train.ipynb` | Google Colab 학습 노트북 |
| `requirements.txt` | Python 패키지 의존성 |
| `README.md` | 프로젝트 소개 + 에피소드별 GIF 섹션 |
| `reports/DQN_수행내역서.md` | AI개발 수행내역서 |

### Colab 노트북 구성 (colab/train.ipynb 최종 셀 순서)

| 셀 | 내용 |
|---|---|
| 1 | Google Drive 마운트 및 폴더 생성 (models·videos·results) |
| 2 | 패키지 설치 + numpy·gym·nes-py 호환성 패치 4종 |
| 3 | 가상 디스플레이 시작 |
| 4 | GitHub 프로젝트 클론 |
| 5 | GPU 확인 |
| 6 | TensorBoard 실행 (학습 전 먼저 실행) |
| 7 | **학습 실행** (영상·그래프 자동 기록 + TensorBoard 로깅) |
| 8 | 학습 곡선 전체 그래프 저장 |
| 9 | 저장 파일 목록 확인 |
| 10 | MP4 → GIF 변환 (README 삽입용) |
| 11 | 세션 끊긴 후 이어서 학습하기 |

### 영상·그래프 기록 계획

| 기록 시점 | epsilon | 파일명 | 목적 |
|---|---|---|---|
| EP 0 | 1.0 (완전 랜덤) | mario_ep0000.mp4 | 학습 전 기준점 |
| EP 2000 | 0.05 | mario_ep2000.mp4 | 초기 학습 결과 |
| EP 5000 | 0.05 | mario_ep5000.mp4 | 중기 학습 결과 |
| EP 7000 | 0.05 | mario_ep7000.mp4 | 후기 학습 결과 |
| EP 10000 | 0.05 | mario_ep10000.mp4 | 최종 학습 결과 |

### TensorBoard 로깅 항목

| 지표 | 설명 |
|---|---|
| `Reward/episode` | 에피소드별 총 보상 |
| `Reward/avg100` | 최근 100회 이동 평균 보상 |
| `Agent/epsilon` | 현재 탐험 확률 |
| `Agent/learn_step` | 누적 신경망 업데이트 횟수 |

### Google Drive 저장 구조

```
supermario_dl_project/      ← Google Drive
├── models/                 ← 체크포인트 (.pth, 500 에피소드마다 + best)
├── videos/                 ← 플레이 영상 5개 (.mp4)
├── results/                ← 학습 곡선 그래프 (.png, 에피소드별 + 전체)
├── gifs/                   ← MP4 변환 GIF (README 삽입용)
└── logs/                   ← TensorBoard 로그
```

### GitHub 커밋 이력

| 날짜 | 커밋 내용 |
|---|---|
| 2026-06-25 | `feat: 슈퍼마리오 DQN 프로젝트 초기 구조` |
| 2026-06-25 | `feat: 에피소드별 영상 기록 및 학습 곡선 그래프 추가` |
| 2026-06-25 | `docs: 프로젝트 진행 현황 보고서 추가` |
| 2026-06-25 | `feat: 에피소드 비교 탭 추가 (EP 0/1000/5000/7000 영상 비교)` |
| 2026-06-25 | `fix: nes-py 빌드 오류 해결 (setuptools 다운그레이드)` |
| 2026-06-25 | `fix: nes-py 빌드 오류 해결 (no-build-isolation + cmake 추가)` |
| 2026-06-25 | `docs: 진행 현황 보고서 업데이트` |
| 2026-06-26 | `feat: EPISODES=10000, TensorBoard, 체크포인트 그래프 EP별 저장` |
| 2026-06-26 | `feat: RECORD_AT 2000 에피소드 적용 및 README GIF 섹션 추가` |
| 2026-06-26 | `feat: README GIF 섹션 HTML 테이블 고정 크기(300px) 적용` |

---

## 진행 중인 작업

- [ ] Colab T4 GPU 학습 실행 중 (EPISODES=10000)
- [ ] EP 2000 GIF README 업로드 완료, 나머지 GIF 대기 중

---

## 앞으로 진행해야 할 작업

### 학습 완료 후

- [ ] EP 5000 / 7000 / 10000 GIF 변환 후 `reports/gif/` 업로드 및 커밋
- [ ] DQN_수행내역서.md 실제 지표값 기입 (평균보상·최고보상·epsilon 등)
- [ ] 학습된 모델(.pth) GitHub Releases 업로드
- [ ] Streamlit 에피소드 비교 탭 EP 0/2000/5000/7000/10000으로 업데이트
- [ ] README 결과 섹션 실제 수치 업데이트
- [ ] 최종 커밋 및 푸시

---

## 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---|---|---|
| EPISODES | 10,000 | 총 학습 에피소드 수 |
| SAVE_EVERY | 500 | 체크포인트 저장 주기 |
| RECORD_AT | [0, 2000, 5000, 7000, 10000] | 영상·그래프 기록 시점 |
| RECORD_STEPS | 450 | 녹화 스텝 수 (30초 @ 15fps) |
| RECORD_SPEED | 2.0 | 영상 재생 배속 |
| lr | 0.0001 | 학습률 (Adam) |
| gamma | 0.9 | 할인율 |
| epsilon_start | 1.0 | 초기 탐험율 |
| epsilon_min | 0.1 | 최소 탐험율 |
| epsilon_decay | 0.99999 | 탐험율 감소율 (스텝마다) |
| batch_size | 32 | 학습 배치 크기 |
| buffer_size | 50,000 | Replay Buffer 크기 |
| target_update_freq | 1,000 스텝 | Target Network 동기화 주기 |
| frame_skip | 4 | 프레임 스킵 수 |
| frame_stack | 4 | 프레임 스택 수 |

---

## 참고 사항

- **세션 끊긴 후 재개**: `CHECKPOINT_PATH`를 마지막 저장 모델 경로로 변경 후 1번 셀부터 재실행
- **모델 가중치**: `.gitignore`로 git 제외 → Google Drive에만 저장
- **GIF 파일**: `reports/gif/`는 `.gitignore` 예외 처리 → git 추적됨
- **Colab 제한**: 무료 T4 기준 최대 12시간 연속 실행, 10,000 에피소드 약 14~20시간 예상 (2일 분할)
- **코드 수정 시**: 로컬 수정 → git push → Colab GitHub 클론 셀 재실행
