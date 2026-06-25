import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    """
    CNN 기반 DQN 신경망
    - 게임 화면(픽셀)에서 각 행동의 가치(Q값)를 예측하는 신경망
    - 입력: (batch, 4, 84, 84) - 4프레임 스택 그레이스케일 이미지
    - 출력: (batch, n_actions) - 각 행동에 대한 Q값 (높을수록 좋은 행동)

    구조:
      CNN 레이어 (특징 추출) -> FC 레이어 (Q값 계산)
    """
    def __init__(self, n_actions):
        """
        n_actions: 행동 공간 크기 (SIMPLE_MOVEMENT = 7)
        """
        super().__init__()

        # --- CNN 레이어: 게임 화면에서 패턴/특징 추출 ---
        self.conv = nn.Sequential(
            # 1번째 컨볼루션: 큰 커널로 전체적인 구조 파악 (84x84 -> 20x20)
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # 2번째 컨볼루션: 세부 특징 추출 (20x20 -> 9x9)
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # 3번째 컨볼루션: 정밀한 특징 추출 (9x9 -> 7x7)
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # --- FC 레이어: 추출된 특징 -> Q값 변환 ---
        self.fc = nn.Sequential(
            nn.Flatten(),                    # (64, 7, 7) -> 3136 벡터로 펼치기
            nn.Linear(64 * 7 * 7, 512),     # 3136 -> 512 차원 압축
            nn.ReLU(),
            nn.Linear(512, n_actions),       # 512 -> 행동 수만큼 Q값 출력
        )

    def forward(self, x):
        """
        순전파 (입력 -> Q값 예측)
        - 픽셀값을 0~1 범위로 정규화 후 CNN -> FC 순으로 통과
        """
        x = x.float() / 255.0  # 픽셀값 0~255 -> 0~1 정규화 (학습 안정화)
        return self.fc(self.conv(x))
