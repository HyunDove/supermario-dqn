import streamlit as st
import torch
import numpy as np
import imageio
import os
import glob

from src.env.wrappers import make_env
from src.agent.dqn_agent import DQNAgent

st.set_page_config(page_title="슈퍼마리오 DQN", layout="wide")
st.title("🍄 슈퍼마리오 DQN 강화학습 시연")

# 사이드바
st.sidebar.header("설정")
world = st.sidebar.selectbox("월드", [1, 2, 3], index=0)
stage = st.sidebar.selectbox("스테이지", [1, 2, 3, 4], index=0)

model_files = sorted(glob.glob("models/*.pth"))
if not model_files:
    st.warning("학습된 모델이 없습니다. 먼저 Colab에서 학습을 완료하세요.")
    st.stop()

selected_model = st.sidebar.selectbox("모델 선택", model_files)

# 탭 구성 (에피소드 비교 탭 추가)
tab1, tab2, tab3, tab4 = st.tabs(["🎮 시연", "🎬 에피소드 비교", "📈 학습 곡선", "🧠 모델 구조"])

# ── 탭 1: 시연 ──────────────────────────────────────────
with tab1:
    st.subheader("학습된 에이전트 플레이 영상")
    st.caption("선택한 모델로 에이전트가 직접 플레이한 영상을 생성합니다.")

    if st.button("▶ 플레이 시작"):
        with st.spinner("에이전트 플레이 중..."):
            device = torch.device("cpu")
            env = make_env(world=world, stage=stage)
            agent = DQNAgent(n_actions=env.action_space.n, device=device)
            agent.load(selected_model)
            agent.epsilon = 0.0

            frames = []
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                raw_frame = env.render(mode="rgb_array")
                frames.append(raw_frame)
                action = agent.select_action(state)
                state, reward, done, info = env.step(action)
                total_reward += reward

            env.close()

            video_path = "results/demo.mp4"
            os.makedirs("results", exist_ok=True)
            imageio.mimsave(video_path, frames, fps=30)

        st.success(f"총 보상: {total_reward:.1f}")
        st.video(video_path)

# ── 탭 2: 에피소드 비교 ──────────────────────────────────
with tab2:
    st.subheader("에피소드별 성장 비교")
    st.caption("학습이 진행될수록 마리오가 얼마나 발전했는지 확인합니다.")

    # 기록된 영상 목록 정의
    episodes = [
        {"ep": 0,    "label": "EP 0",    "desc": "학습 전 (완전 랜덤)", "file": "videos/mario_ep0000.mp4"},
        {"ep": 1000, "label": "EP 1000", "desc": "초기 학습",           "file": "videos/mario_ep1000.mp4"},
        {"ep": 5000, "label": "EP 5000", "desc": "중간 학습",           "file": "videos/mario_ep5000.mp4"},
        {"ep": 7000, "label": "EP 7000", "desc": "최종 학습",           "file": "videos/mario_ep7000.mp4"},
    ]

    # 영상 존재 여부 확인
    available = [e for e in episodes if os.path.exists(e["file"])]
    missing   = [e for e in episodes if not os.path.exists(e["file"])]

    if not available:
        st.info("아직 기록된 영상이 없습니다. Colab 학습 중 자동으로 저장됩니다.")
    else:
        # 있는 영상만 2열 그리드로 표시
        cols = st.columns(2)
        for i, ep in enumerate(available):
            with cols[i % 2]:
                st.markdown(f"#### {ep['label']} — {ep['desc']}")
                st.video(ep["file"])

        # 아직 없는 영상 안내
        if missing:
            st.divider()
            st.caption("📌 아직 기록되지 않은 에피소드: " + ", ".join([e["label"] for e in missing]))

# ── 탭 3: 학습 곡선 ──────────────────────────────────────
with tab3:
    st.subheader("학습 곡선")
    st.caption("에피소드가 쌓일수록 평균 보상이 올라가면 학습이 잘 되고 있다는 뜻입니다.")

    curve_path = "results/training_curve.png"
    if os.path.exists(curve_path):
        st.image(curve_path)
    else:
        st.info("학습 완료 후 그래프가 표시됩니다.")

# ── 탭 4: 모델 구조 ──────────────────────────────────────
with tab4:
    st.subheader("DQN 네트워크 구조")
    st.code("""
입력: (4, 84, 84) - 4프레임 스택 그레이스케일

Conv2d(4 → 32, kernel=8, stride=4)  → ReLU
Conv2d(32 → 64, kernel=4, stride=2) → ReLU
Conv2d(64 → 64, kernel=3, stride=1) → ReLU

Flatten → Linear(3136 → 512) → ReLU → Linear(512 → 7)

출력: 7개 행동의 Q값
    """)

    st.subheader("하이퍼파라미터")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("학습률 (lr)", "0.0001")
        st.metric("할인율 (γ)", "0.9")
        st.metric("배치 크기", "32")
    with col2:
        st.metric("버퍼 크기", "50,000")
        st.metric("epsilon 최솟값", "0.1")
        st.metric("타깃 네트워크 업데이트", "1,000 스텝")
