import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.env.wrappers import make_env
from src.agent.dqn_agent import DQNAgent

SAVE_DIR = "models"
RESULT_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

EPISODES = 5000
SAVE_EVERY = 200
CHECKPOINT_PATH = None  # 이어서 학습할 경우 경로 지정 예: "models/mario_ep1000.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {device}")

env = make_env(world=1, stage=1)
n_actions = env.action_space.n
agent = DQNAgent(n_actions=n_actions, device=device)

start_episode = 0
if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    agent.load(CHECKPOINT_PATH)
    start_episode = int(CHECKPOINT_PATH.split("ep")[-1].split(".")[0])
    print(f"체크포인트 로드: {CHECKPOINT_PATH} (에피소드 {start_episode}부터 재개)")

rewards_history = []
best_reward = -float('inf')

for episode in tqdm(range(start_episode, start_episode + EPISODES)):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.learn()
        state = next_state
        total_reward += reward

    rewards_history.append(total_reward)

    if (episode + 1) % SAVE_EVERY == 0:
        path = f"{SAVE_DIR}/mario_ep{episode + 1}.pth"
        agent.save(path)
        avg = np.mean(rewards_history[-100:])
        print(f"\n[EP {episode+1}] 평균 보상: {avg:.1f} | epsilon: {agent.epsilon:.3f} | 저장: {path}")

    if total_reward > best_reward:
        best_reward = total_reward
        agent.save(f"{SAVE_DIR}/mario_best.pth")

env.close()

plt.figure(figsize=(12, 4))
plt.plot(rewards_history, alpha=0.4, label='에피소드 보상')
window = min(100, len(rewards_history))
avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
plt.plot(range(window-1, len(rewards_history)), avg, label=f'{window}회 평균')
plt.xlabel('에피소드')
plt.ylabel('보상')
plt.title('슈퍼마리오 DQN 학습 곡선')
plt.legend()
plt.savefig(f"{RESULT_DIR}/training_curve.png", dpi=150)
plt.show()
print("학습 완료!")
