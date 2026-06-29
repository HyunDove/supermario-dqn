import streamlit as st
import json
import numpy as np
import os
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 경로 설정 ──────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

def rp(*parts):
    return os.path.join(ROOT, *parts)

# ── 페이지 설정 ────────────────────────────────────
st.set_page_config(
    page_title="Super Mario DQN",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 마리오 테마 CSS ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

/* 배경 */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #0a0a1a 0%, #0d1b4b 55%, #1a1e6b 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent !important; }

/* 폰트 */
h1 {
    font-family: 'Press Start 2P', monospace !important;
    color: #F8B800 !important;
    text-shadow: 4px 4px 0px #9b4e00, 6px 6px 12px rgba(0,0,0,0.5);
}
h2 {
    font-family: 'Press Start 2P', monospace !important;
    color: #F8B800 !important;
    font-size: 1.1rem !important;
    text-shadow: 2px 2px 0px #9b4e00;
}
h3 {
    font-family: 'Press Start 2P', monospace !important;
    color: #ffffff !important;
    font-size: 0.8rem !important;
}

/* 일반 텍스트 */
p, li, span, [data-testid="stMarkdownContainer"] p {
    color: #cccccc !important;
}

/* 히어로 영역 */
.hero-title {
    font-family: 'Press Start 2P', monospace;
    font-size: 2rem;
    color: #F8B800;
    text-shadow: 4px 4px 0px #9b4e00, 7px 7px 18px rgba(0,0,0,0.5);
    text-align: center;
    margin: 24px 0 10px 0;
    line-height: 1.8;
    letter-spacing: 0.05em;
}
.hero-subtitle {
    text-align: center;
    color: #888888 !important;
    font-size: 0.88rem;
    margin-bottom: 28px;
    letter-spacing: 0.02em;
}
.hero-stars {
    text-align: center;
    font-size: 1.6rem;
    margin-bottom: 8px;
    letter-spacing: 0.3em;
}

/* 에피소드 카드 배지 */
.ep-badge {
    font-family: 'Press Start 2P', monospace;
    font-size: 0.48rem;
    padding: 5px 10px;
    border-radius: 4px;
    color: white;
    display: inline-block;
    margin-bottom: 10px;
    letter-spacing: 0.08em;
}
.ep-label {
    font-family: 'Press Start 2P', monospace;
    font-size: 0.68rem;
    color: #F8B800;
    margin: 10px 0 4px 0;
    text-align: center;
}
.ep-desc {
    font-size: 0.72rem;
    color: #999999 !important;
    margin: 2px 0 10px 0;
    text-align: center;
}

/* 탭 */
button[data-baseweb="tab"] {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.58rem !important;
    color: #777777 !important;
    padding: 10px 18px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #F8B800 !important;
}
[data-testid="stTabContent"] { padding-top: 24px; }

/* 메트릭 */
[data-testid="stMetricLabel"] > div {
    color: #888888 !important;
    font-size: 0.72rem !important;
}
[data-testid="stMetricValue"] > div {
    color: #F8B800 !important;
    font-size: 1.25rem !important;
    font-weight: bold !important;
}
[data-testid="stMetricDelta"] > div { font-size: 0.72rem !important; }

/* 구분선 */
hr { border-color: rgba(248, 184, 0, 0.18) !important; }

/* 테이블 */
table {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    width: 100% !important;
}
th {
    background: rgba(248,184,0,0.12) !important;
    color: #F8B800 !important;
    font-size: 0.82rem !important;
    padding: 10px !important;
}
td {
    color: #cccccc !important;
    font-size: 0.82rem !important;
    padding: 8px 10px !important;
}
tr:nth-child(even) { background: rgba(255,255,255,0.02) !important; }

/* 코드 블록 */
code, pre { background: rgba(0,0,0,0.5) !important; color: #7ec8e3 !important; }

/* 이미지 */
div[data-testid="stImage"] img { border-radius: 8px; }

/* 캡션 */
[data-testid="stCaptionContainer"] p {
    color: #777777 !important;
    font-size: 0.78rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── 데이터 로드 ────────────────────────────────────
@st.cache_data
def load_rewards():
    fp = rp("reports", "screenshot", "rewards_history.json")
    if os.path.exists(fp):
        with open(fp, encoding="utf-8") as f:
            return json.load(f)
    return []


def read_gif(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None


rewards = load_rewards()

# ── 체크포인트 정의 ────────────────────────────────
CHECKPOINTS = [
    {
        "ep":       0,
        "label":    "EP 0",
        "tag":      "BEFORE TRAINING",
        "desc":     "100% 무작위 행동",
        "detail":   "학습 전 베이스라인. 랜덤 행동으로 방향 없이 움직인다.",
        "gif":      rp("reports", "gif", "mario_ep0000.gif"),
        "curve":    rp("reports", "screenshot", "curve_ep0000.png"),
        "epsilon":  1.0,
        "steps":    0,
        "avg100":   592.0,
        "max100":   592.0,
        "best":     592.0,
        "border":   "#555555",
        "badge_bg": "#444444",
    },
    {
        "ep":       2000,
        "label":    "EP 2000",
        "tag":      "EARLY STAGE",
        "desc":     "우측 전진 학습 시작",
        "detail":   "ε이 최솟값(0.1)에 도달. 오른쪽으로 달리는 패턴이 자리를 잡기 시작.",
        "gif":      rp("reports", "gif", "mario_ep2000.gif"),
        "curve":    rp("reports", "screenshot", "curve_ep2000.png"),
        "epsilon":  0.1,
        "steps":    269649,
        "avg100":   860.7,
        "max100":   1904.0,
        "best":     2874.0,
        "border":   "#43B047",
        "badge_bg": "#2a6e2d",
    },
    {
        "ep":       5000,
        "label":    "EP 5000",
        "tag":      "MID STAGE",
        "desc":     "장애물 회피 패턴 등장",
        "detail":   "점프 타이밍을 학습하여 장애물을 넘는 행동이 관찰된다.",
        "gif":      rp("reports", "gif", "mario_ep5000.gif"),
        "curve":    rp("reports", "screenshot", "curve_ep5000.png"),
        "epsilon":  0.1,
        "steps":    686248,
        "avg100":   1222.6,
        "max100":   2916.0,
        "best":     3059.0,
        "border":   "#049CD8",
        "badge_bg": "#026e99",
    },
    {
        "ep":       7000,
        "label":    "EP 7000",
        "tag":      "LATE STAGE",
        "desc":     "전진 전략 고도화",
        "detail":   "안정적인 전진 전략이 완성되어 일관성 있는 고득점을 기록. EP 10000 클리어로 이어지는 단계.",
        "gif":      rp("reports", "gif", "mario_ep7000.gif"),
        "curve":    rp("reports", "screenshot", "curve_ep7000.png"),
        "epsilon":  0.1,
        "steps":    1019666,
        "avg100":   1387.2,
        "max100":   3047.0,
        "best":     3059.0,
        "border":   "#E52521",
        "badge_bg": "#a81b19",
    },
    {
        "ep":       10000,
        "label":    "EP 10000",
        "tag":      "FINAL CHECKPOINT",
        "desc":     "EP 10000 클리어 이후 최종 학습",
        "detail":   "EP 10000 스테이지 클리어 달성 이후 추가 학습된 최종 체크포인트. 100회 평균 보상 1,590 기록.",
        "gif":      rp("reports", "gif", "mario_ep10000.gif"),
        "curve":    rp("reports", "screenshot", "curve_ep10000.png"),
        "epsilon":  0.1,
        "steps":    1593659,
        "avg100":   1590.0,
        "max100":   3059.0,
        "best":     3062.0,
        "border":   "#F8B800",
        "badge_bg": "#7A5C00",
    },
]


# ════════════════════════════════════════════════════
# 헤더
# ════════════════════════════════════════════════════
st.markdown('<div class="hero-stars">⭐ 🍄 ⭐</div>', unsafe_allow_html=True)
st.markdown('<p class="hero-title">SUPER MARIO DQN</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    'Deep Q-Network 강화학습으로 마리오를 자율 플레이 &nbsp;·&nbsp; EP 10000에서 1-1 스테이지 클리어 달성!'
    '</p>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("🎮 학습 에피소드", "10,000")
with m2: st.metric("⚡ 총 학습 스텝", "1,593,659")
with m3: st.metric("🏆 스테이지 클리어", "EP 10000")
with m4: st.metric("📈 보상 성장", "+169%", delta="EP 0 → EP 10000 평균 기준")

st.divider()


# ════════════════════════════════════════════════════
# 탭
# ════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🎮  에피소드 비교", "📈  학습 성과", "🧠  모델 구조"])


# ──────────────────────────────────────────────────
# 탭 1 · 에피소드 비교 (GIF)
# ──────────────────────────────────────────────────
with tab1:
    st.markdown("## 학습 단계별 마리오 성장")
    st.caption("EP 0 → EP 10000 성장 과정 · EP 10000에서 1-1 스테이지 클리어 달성")
    st.markdown("")

    cols = st.columns(4, gap="medium")
    for i, cp in enumerate(CHECKPOINTS[:4]):
        with cols[i]:
            st.markdown(
                f'<div style="text-align:center; margin-bottom:8px;">'
                f'<span class="ep-badge" style="background:{cp["badge_bg"]};">'
                f'{cp["tag"]}</span></div>',
                unsafe_allow_html=True,
            )

            gif_data = read_gif(cp["gif"])
            if gif_data:
                st.image(gif_data, use_container_width=True)
            else:
                st.markdown(
                    '<div style="background:rgba(255,255,255,0.04); border:2px dashed #444;'
                    'border-radius:8px; padding:50px 10px; text-align:center; color:#666;">'
                    '🎮<br><small>준비 중</small></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<p class="ep-label">{cp["label"]}</p>'
                f'<p class="ep-desc">{cp["desc"]}</p>',
                unsafe_allow_html=True,
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("평균 보상", f"{cp['avg100']:,.0f}")
            with col_b:
                st.metric("최고 보상", f"{cp['best']:,.0f}")

    # ── EP 10000 + DQN 한계 ────────────────────────
    st.divider()
    st.markdown("## EP 10000 — 최종 학습 결과 · DQN 한계")

    cp10k = CHECKPOINTS[4]
    gif_data = read_gif(cp10k["gif"])
    if gif_data:
        gif_b64 = base64.b64encode(gif_data).decode()
        gif_html = f'<img src="data:image/gif;base64,{gif_b64}" style="width:100%; border-radius:8px;" />'
    else:
        gif_html = ('<div style="background:rgba(255,255,255,0.04); border:2px dashed #444;'
                    'border-radius:8px; padding:50px 10px; text-align:center; color:#666;">'
                    '🎮<br><small>준비 중</small></div>')

    st.markdown(f"""
<div style="display:flex; gap:28px; align-items:flex-start;">
  <div style="flex:1; min-width:0;">
    <div style="text-align:center; margin-bottom:8px;">
      <span class="ep-badge" style="background:{cp10k['badge_bg']};">{cp10k['tag']}</span>
    </div>
    {gif_html}
    <p class="ep-label">{cp10k['label']}</p>
    <p class="ep-desc">{cp10k['desc']}</p>
    <div style="display:flex; gap:12px; justify-content:center; margin-top:6px;">
      <div style="text-align:center;">
        <div style="color:#888888; font-size:0.72rem;">100회 평균</div>
        <div style="color:#F8B800; font-size:1.25rem; font-weight:bold;">{cp10k['avg100']:,.0f}</div>
      </div>
      <div style="text-align:center;">
        <div style="color:#888888; font-size:0.72rem;">전체 최고</div>
        <div style="color:#F8B800; font-size:1.25rem; font-weight:bold;">{cp10k['best']:,.0f}</div>
      </div>
    </div>
  </div>
  <div style="flex:2; min-width:0;">
    <div style="background:rgba(67,176,71,0.10); border-left:4px solid #43B047;
                border-radius:6px; padding:18px 20px; margin-bottom:16px;">
      <p style="color:#43B047 !important; font-weight:bold; margin:0 0 10px 0;">🏆 EP 10000 · 1-1 스테이지 클리어 달성</p>
      <p style="color:#cccccc !important; font-size:0.88rem; line-height:1.7; margin:0;">
        총 10,000 에피소드 학습 끝에 <strong style="color:#F8B800;">EP 10000에서 슈퍼마리오 1-1 스테이지 클리어</strong>를 달성하였습니다.<br><br>
        픽셀(84×84 흑백 화면)만을 입력으로 받는 CNN DQN 에이전트가 별도의 사전 지식 없이 보상 신호만을 통해 스테이지를 완주하였습니다.<br><br>
        <strong style="color:#ffffff;">학습 과정 요약:</strong><br>
        EP 0~2000 — 랜덤 탐험에서 우측 전진 패턴 발견<br>
        EP 2000~5000 — 장애물 회피 및 점프 타이밍 학습<br>
        EP 5000~7000 — 전진 전략 고도화, 일관성 향상<br>
        EP 7000~9000 — 전략 정교화 → <strong style="color:#43B047;">스테이지 완주 달성</strong>
      </p>
    </div>
    <div style="background:rgba(4,156,216,0.08); border-left:4px solid #049CD8;
                border-radius:6px; padding:14px 20px;">
      <p style="color:#049CD8 !important; font-weight:bold; margin:0 0 8px 0;">💡 향후 발전 방향</p>
      <p style="color:#cccccc !important; font-size:0.85rem; line-height:1.7; margin:0;">
        <strong>Double DQN</strong> — Q값 과대추정 억제로 학습 안정화<br>
        <strong>Prioritized Experience Replay</strong> — 중요 경험 우선 학습으로 샘플 효율 향상<br>
        <strong>다중 스테이지 학습</strong> — 1-2, 1-3 등 추가 스테이지로 일반화 능력 확장
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 성장 요약 ──────────────────────────────────
    st.divider()
    st.markdown("## 체크포인트별 평균 보상 변화")

    base = CHECKPOINTS[0]["avg100"]
    sc = st.columns(5)
    for col, cp in zip(sc, CHECKPOINTS):
        with col:
            delta_str = f"+{cp['avg100'] - base:.0f}" if cp["ep"] > 0 else None
            st.metric(cp["label"], f"{cp['avg100']:,.0f}", delta=delta_str)

    st.markdown("")
    for cp in CHECKPOINTS:
        growth = ((cp["avg100"] / base) - 1) * 100 if cp["ep"] > 0 else 0
        growth_txt = f" (+{growth:.0f}%)" if cp["ep"] > 0 else " (베이스라인)"
        st.markdown(
            f"**{cp['label']}** &nbsp;—&nbsp; {cp['detail']}"
            f"<span style='color:#666; font-size:0.78rem;'>{growth_txt}</span>",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────
# 탭 2 · 학습 성과
# ──────────────────────────────────────────────────
with tab2:

    if rewards:
        st.markdown("## 전체 학습 곡선")

        fig, ax = plt.subplots(figsize=(13, 4))
        fig.patch.set_facecolor("#0d1b4b")
        ax.set_facecolor("#070d28")

        x = np.arange(len(rewards))
        ax.plot(x, rewards, alpha=0.18, color="#7ec8e3", linewidth=0.6, label="Episode Reward")

        window = 100
        if len(rewards) >= window:
            avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window - 1, len(rewards)), avg,
                    color="#F8B800", linewidth=2.2, label="100-ep Moving Avg")

        y_top = max(rewards) * 1.05
        for cp in CHECKPOINTS[1:]:
            ep_idx = cp["ep"] - 1
            ax.axvline(x=ep_idx, color=cp["border"], linestyle="--", alpha=0.75, linewidth=1.3)
            ax.text(ep_idx + 60, y_top * 0.88, cp["label"],
                    color=cp["border"], fontsize=7.5, ha="left", fontweight="bold")

        ax.set_xlabel("Episode", color="#aaa", fontsize=10)
        ax.set_ylabel("Total Reward", color="#aaa", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.tick_params(colors="#888", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(facecolor="#0a0f2a", edgecolor="#333", labelcolor="#ccc", fontsize=9)
        ax.grid(alpha=0.12, color="#444")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("")
    st.markdown("## 체크포인트별 성능 비교")
    st.markdown("""
| 체크포인트 | 총 학습 스텝 | ε | 100회 평균 보상 | 100회 최고 보상 | 전체 최고 보상 | 비고 |
|:---:|---:|:---:|---:|---:|---:|---|
| EP 0 | 0 | 1.0000 | 592.0 | 592.0 | 592.0 | 베이스라인 |
| EP 2000 | 269,649 | 0.1000 | 860.7 | 1,904.0 | 2,874.0 | 초기 학습 |
| EP 5000 | 686,248 | 0.1000 | 1,222.6 | 2,916.0 | 3,059.0 | 중기 학습 |
| EP 7000 | 1,019,666 | 0.1000 | 1,387.2 | 3,047.0 | 3,059.0 | 후기 학습 |
| **EP 10000** | 1,593,659 | 0.1000 | 1,590.0 | 3,059.0 | 3,062.0 | **🏆 1-1 스테이지 클리어** |
""")
    st.caption("EP 10000에서 슈퍼마리오 1-1 스테이지 클리어 달성 · 총 10,000 에피소드 학습 완료")

    st.markdown("""
<div style="background:rgba(67,176,71,0.08); border:1px solid rgba(67,176,71,0.3);
            border-radius:8px; padding:14px 18px; margin-top:8px;">
<p style="color:#43B047 !important; font-size:0.85rem; margin:0; line-height:1.7;">
<strong>🏆 스테이지 클리어 달성</strong> — EP 10000에서 슈퍼마리오 1-1 스테이지 클리어.
픽셀 입력만으로 동작하는 CNN DQN 에이전트가 보상 신호만을 통해 스테이지를 완주하였습니다.
</p>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("## 체크포인트별 학습 곡선 이미지")

    for row_cps in [CHECKPOINTS[:2], CHECKPOINTS[2:4], [CHECKPOINTS[4]]]:
        rcols = st.columns(2, gap="medium")
        for col, cp in zip(rcols, row_cps):
            with col:
                st.caption(f"{cp['label']} — {cp['desc']}")
                if os.path.exists(cp["curve"]):
                    st.image(cp["curve"], use_container_width=True)
                else:
                    st.info("이미지 준비 중")
        st.markdown("")


# ──────────────────────────────────────────────────
# 탭 3 · 모델 구조
# ──────────────────────────────────────────────────
with tab3:
    col_arch, col_param = st.columns([3, 2], gap="large")

    with col_arch:
        st.markdown("## CNN DQN 신경망 구조")
        st.code("""입력: (4, 84, 84)  —  4프레임 스택 그레이스케일

┌─ 합성곱 블록 ──────────────────────────────────┐
│  Conv2d(  4 → 32, kernel=8, stride=4) → ReLU  │  출력: (32, 20, 20)
│  Conv2d( 32 → 64, kernel=4, stride=2) → ReLU  │  출력: (64,  9,  9)
│  Conv2d( 64 → 64, kernel=3, stride=1) → ReLU  │  출력: (64,  7,  7)
└────────────────────────────────────────────────┘

               Flatten  →  3,136 차원

┌─ 완전연결 블록 ─────────────────────────────────┐
│  Linear(3,136 → 512) → ReLU                    │
│  Linear(  512 →   7)                           │
└────────────────────────────────────────────────┘

출력: 7개 행동의 Q값 (SIMPLE_MOVEMENT)""", language="text")

        st.markdown("")
        st.markdown("## 핵심 기법")
        st.markdown("""
| 기법 | 구현 | 역할 |
|---|---|---|
| **Experience Replay** | Buffer 50,000건 | 랜덤 샘플링으로 시간 상관관계 제거 |
| **Target Network** | 1,000 스텝마다 동기화 | 목표 Q값 안정화 |
| **ε-greedy** | 1.0 → 0.1 감소 | 탐험 / 활용 균형 |
| **Huber Loss** | SmoothL1Loss | 이상치에 강건한 손실함수 |
| **Gradient Clipping** | max_norm=10 | 기울기 폭발 방지 |
""")

        st.markdown("")
        st.markdown("## 행동 공간 (SIMPLE_MOVEMENT)")
        st.markdown("""
| 인덱스 | 행동 | 설명 |
|:---:|---|---|
| 0 | NOOP | 정지 |
| 1 | right | 오른쪽 이동 |
| 2 | right + A | 오른쪽 + 점프 |
| 3 | right + B | 오른쪽 + 달리기 |
| 4 | right + A + B | 오른쪽 + 달리기 + 점프 |
| 5 | left | 왼쪽 이동 |
| 6 | A | 제자리 점프 |
""")

    with col_param:
        st.markdown("## 하이퍼파라미터")
        params = [
            ("학습률 (lr)",      "0.0001"),
            ("할인율 (γ)",       "0.9"),
            ("ε 시작값",         "1.0"),
            ("ε 최솟값",         "0.1"),
            ("ε 감소율",         "0.99999 / step"),
            ("배치 크기",        "32"),
            ("Replay Buffer",   "50,000"),
            ("Target 업데이트",  "1,000 steps"),
            ("프레임 스킵",       "4"),
            ("입력 해상도",       "4 × 84 × 84"),
        ]
        for label, val in params:
            st.metric(label, val)

        st.markdown("")
        st.markdown("## 학습 환경")
        st.markdown("""
| 항목 | 내용 |
|---|---|
| 플랫폼 | Google Colab |
| GPU | NVIDIA T4 |
| Python | 3.12 |
| PyTorch | 2.x (CUDA 12) |
| 총 학습 에피소드 | 10,000 (완료) |
| 기록 시점 | EP 0 / 2000 / 5000 / 7000 / 10000 |
""")
