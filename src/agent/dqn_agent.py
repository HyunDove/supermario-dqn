import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.model.dqn import DQN
from src.utils.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN 에이전트 - 마리오를 학습시키는 핵심 클래스
    - Policy Network: 실제 행동 결정에 사용하는 신경망
    - Target Network: Q값 목표 계산용 신경망 (학습 안정화 목적으로 별도 유지)
    - Replay Buffer: 과거 경험 저장소 (랜덤 샘플링으로 상관관계 제거)
    """
    def __init__(
        self,
        n_actions,          # 행동 공간 크기 (7)
        device,             # 학습 장치 (cuda / cpu)
        lr=1e-4,            # 학습률 - 가중치 업데이트 보폭
        gamma=0.9,          # 할인율 - 미래 보상을 현재 가치로 환산하는 비율
        epsilon_start=1.0,  # 탐험율 초기값 - 1.0이면 100% 랜덤 행동
        epsilon_min=0.1,    # 탐험율 최솟값 - 0.1이면 10%는 항상 탐험
        epsilon_decay=0.99999,  # 탐험율 감소율 - 매 스텝 epsilon에 곱함
        buffer_size=50000,  # Replay Buffer 최대 크기
        batch_size=32,      # 한 번에 학습할 경험 샘플 수
        target_update_freq=1000,  # Target Network 동기화 주기 (스텝 단위)
    ):
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step = 0  # 총 학습 횟수 카운터

        # Policy Network: 매 스텝 업데이트되는 메인 신경망
        self.policy_net = DQN(n_actions).to(device)
        # Target Network: 일정 주기로만 동기화되는 안정화용 신경망
        self.target_net = DQN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target Network는 추론 모드 고정

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # SmoothL1Loss: MSE보다 이상치에 강건한 손실함수 (Huber Loss)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, state):
        """
        epsilon-greedy 정책으로 행동 선택
        - epsilon 확률로 랜덤 행동 (탐험: 새로운 상황 경험)
        - (1-epsilon) 확률로 Q값 최대 행동 선택 (활용: 배운 것 사용)
        - 학습이 진행될수록 epsilon이 감소해 점점 최적 행동 위주로 변함
        """
        if np.random.rand() < self.epsilon:
            # 탐험: 무작위 행동
            return np.random.randint(self.n_actions)
        # 활용: 신경망이 예측한 Q값이 가장 높은 행동 선택
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state_t).argmax().item()

    def store(self, state, action, reward, next_state, done):
        """
        경험을 Replay Buffer에 저장
        - (현재상태, 행동, 보상, 다음상태, 종료여부) 튜플을 저장
        - 버퍼가 가득 차면 오래된 경험부터 자동 삭제
        """
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """
        Replay Buffer에서 샘플링해 신경망 가중치 업데이트
        - 벨만 방정식으로 목표 Q값 계산: Q_target = r + gamma * max(Q_next)
        - Policy Network 예측값과 목표값의 차이(Loss)를 최소화하도록 역전파
        - 일정 주기마다 Target Network를 Policy Network와 동기화
        """
        # 버퍼에 데이터가 충분히 쌓일 때까지 학습 대기
        if len(self.buffer) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        # numpy 배열 -> GPU 텐서 변환
        state_t      = torch.tensor(state).to(self.device)
        action_t     = torch.tensor(action).to(self.device)
        reward_t     = torch.tensor(reward).to(self.device)
        next_state_t = torch.tensor(next_state).to(self.device)
        done_t       = torch.tensor(done).to(self.device)

        # Policy Network로 현재 상태에서 선택한 행동의 Q값 계산
        q_values = self.policy_net(state_t).gather(1, action_t.unsqueeze(1)).squeeze(1)

        # Target Network로 다음 상태의 최대 Q값 계산 (그래디언트 계산 제외)
        with torch.no_grad():
            next_q = self.target_net(next_state_t).max(1)[0]
            # 벨만 방정식: 목표 Q값 = 보상 + 할인율 * 다음 최대Q값 (종료 시 0)
            target = reward_t + self.gamma * next_q * (1 - done_t)

        # 손실 계산 -> 역전파 -> 가중치 업데이트
        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping: 기울기 폭발 방지 (최대 10으로 제한)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.learn_step += 1
        # Target Network 동기화: 일정 주기마다 Policy Network 가중치 복사
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # epsilon 감소: 학습이 쌓일수록 랜덤 행동 비율 줄임
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

    def save(self, path):
        """
        모델 체크포인트 저장
        - Policy/Target Network 가중치, 옵티마이저 상태, epsilon, 학습 스텝 저장
        - 이후 load()로 불러와 학습을 이어서 진행 가능
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step': self.learn_step,
        }, path)

    def load(self, path):
        """
        저장된 체크포인트 불러오기
        - map_location으로 GPU/CPU 환경이 달라도 호환 가능
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.learn_step = checkpoint['learn_step']
