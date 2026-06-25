import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    """
    CNN 기반 DQN 네트워크
    입력: (batch, 4, 84, 84) - 4프레임 스택
    출력: (batch, n_actions) - 각 행동의 Q값
    """
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # (64, 7, 7)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = x.float() / 255.0  # 정규화
        return self.fc(self.conv(x))
