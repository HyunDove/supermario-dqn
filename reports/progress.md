# 슈퍼마리오 DQN 프로젝트 진행 현황

> 작성일: 2026-06-25

---

## 프로젝트 기본 정보

| 항목 | 내용 |
|---|---|
| 주제 | Deep Q-Network(DQN)으로 슈퍼마리오 자율 플레이 학습 |
| 기간 | 2026-06-25 ~ 2026-07-01 (7일) |
| 인원 | 1인 |
| 학습 환경 | Google Colab (T4 GPU) + Google Drive 연동 |
| 시연 | Streamlit 웹앱 |
| GitHub | https://github.com/HyunDove/supermario-dqn |

---

## 완료된 작업

### 프로젝트 구조 설계 및 코드 구현

| 파일 | 내용 |
|---|---|
| `src/env/wrappers.py` | 환경 전처리 파이프라인 (그레이스케일, 리사이즈, 프레임스킵, 프레임스택) |
| `src/model/dqn.py` | CNN 기반 DQN 신경망 구조 (Conv2d x3 + FC x2) |
| `src/agent/dqn_agent.py` | DQN 에이전트 (epsilon-greedy, 학습, 저장/로드) |
| `src/utils/replay_buffer.py` | Experience Replay Buffer |
| `src/utils/recorder.py` | 에피소드별 플레이 영상 녹화 유틸 |
| `train.py` | 로컬 학습 실행 진입점 |
| `app/streamlit_app.py` | Streamlit 시연 웹앱 (시연/학습곡선/모델구조 3탭) |
| `colab/train.ipynb` | Google Colab 학습 노트북 (Drive 연동, 영상 기록, 그래프 포함) |
| `requirements.txt` | Python 패키지 의존성 |
| `README.md` | 프로젝트 소개 문서 (ml_project 스타일 참고) |

### Colab 노트북 구성 (train.ipynb 셀 순서)

| 셀 | 내용 |
|---|---|
| 1 | Google Drive 마운트 및 폴더 생성 |
| 2 | 패키지 설치 (gym, nes-py, opencv 등) |
| 3 | 가상 디스플레이 시작 (헤드리스 렌더링용) |
| 4 | GitHub에서 프로젝트 클론 |
| 5 | GPU 확인 |
| 6 | 학습 실행 (영상 자동 기록 포함) |
| 7 | 학습 곡선 그래프 저장 |
| 8 | 저장 파일 목록 확인 |
| 9 | 세션 끊긴 후 이어서 학습하기 |

### 영상 기록 계획

| 기록 시점 | epsilon | 파일명 | 목적 |
|---|---|---|---|
| EP 0 | 1.0 (완전 랜덤) | mario_ep0000.mp4 | 학습 전 기준점 |
| EP 1000 | 0.05 | mario_ep1000.mp4 | 초기 학습 결과 |
| EP 5000 | 0.05 | mario_ep5000.mp4 | 중간 학습 결과 |
| EP 7000 | 0.05 | mario_ep7000.mp4 | 최종 학습 결과 |

### Google Drive 저장 구조

```
supermario_dl_project/      <- Google Drive
├── models/                 <- 체크포인트 (.pth, 200 에피소드마다)
├── videos/                 <- 플레이 영상 4개 (.mp4)
└── results/                <- 학습 곡선 그래프 (.png)
```

### GitHub

- 레포 생성: https://github.com/HyunDove/supermario-dqn
- 초기 커밋: `feat: 슈퍼마리오 DQN 프로젝트 초기 구조`
- 2차 커밋: `feat: 에피소드별 영상 기록 및 학습 곡선 그래프 추가`

---

## 앞으로 진행해야 할 작업

### 1일차 남은 작업 (오늘)

- [ ] 소스 파일 전체 주석 추가 (wrappers, dqn, dqn_agent, replay_buffer, recorder, train.py)
- [ ] Colab에서 환경 설치 및 테스트 실행 (1~2 에피소드 정상 동작 확인)

### 2일차

- [ ] Colab T4 GPU로 본격 학습 시작 (EPISODES=7000)
- [ ] EP 0 영상 기록 확인
- [ ] 200 에피소드 체크포인트 저장 확인

### 3~5일차

- [ ] Colab 야간 학습 실행 (세션 끊김 시 체크포인트에서 재개)
- [ ] EP 1000 / EP 5000 영상 기록 확인
- [ ] 학습 곡선 중간 점검 (보상이 올라가는지 확인)

### 6일차

- [ ] EP 7000 학습 완료 및 최종 영상 기록
- [ ] 학습 곡선 그래프 최종 저장
- [ ] 영상 4개 비교 (0 -> 1000 -> 5000 -> 7000 성장 과정 확인)
- [ ] 학습된 모델(.pth) GitHub Releases 업로드
- [ ] Streamlit 앱에 영상 4개 비교 탭 추가 (선택)

### 7일차

- [ ] Streamlit 시연 앱 완성 및 최종 테스트
- [ ] README 결과 섹션 업데이트 (실제 학습 결과, 스크린샷 추가)
- [ ] 진행 현황 체크리스트 최종 업데이트
- [ ] 최종 커밋 및 푸시

---

## 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---|---|---|
| EPISODES | 7,000 | 총 학습 에피소드 수 |
| lr | 0.0001 | 학습률 |
| gamma | 0.9 | 할인율 |
| epsilon_start | 1.0 | 초기 탐험율 |
| epsilon_min | 0.1 | 최소 탐험율 |
| epsilon_decay | 0.99999 | 탐험율 감소율 |
| batch_size | 32 | 학습 배치 크기 |
| buffer_size | 50,000 | Replay Buffer 크기 |
| target_update_freq | 1,000 스텝 | Target Network 동기화 주기 |
| frame_skip | 4 | 프레임 스킵 수 |
| frame_stack | 4 | 프레임 스택 수 |

---

## 참고 사항

- **세션 끊긴 후 재개**: `CHECKPOINT_PATH` 를 마지막 저장 모델 경로로 변경 후 Colab 1번 셀부터 재실행
- **모델 가중치**: `.gitignore`로 git 제외 -> Google Drive에만 저장
- **Colab 제한**: 무료 T4 기준 최대 12시간 연속 실행 가능, 유휴 90분 초과 시 세션 끊김
