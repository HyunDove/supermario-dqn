import gym
import numpy as np
import cv2
from collections import deque


class GrayScaleObservation(gym.Wrapper):
    """
    [전처리 1단계] RGB 컬러 이미지 -> 그레이스케일 변환
    - 컬러 정보는 마리오 학습에 불필요하므로 채널을 3 -> 1로 줄여 연산량 감소
    - 입력: (H, W, 3) / 출력: (H, W)
    - gym.Wrapper 직접 상속: gym 0.26의 ObservationWrapper.step()은 5-return을
      기대하지만 gym-super-mario-bros JoypadSpace는 4-return을 반환해 충돌함
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, obs):
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        return self.observation(obs)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        return self.observation(obs), reward, done, info


class ResizeObservation(gym.Wrapper):
    """
    [전처리 2단계] 이미지 크기를 84x84로 축소
    - 원본 해상도(240x256)를 84x84로 줄여 신경망 입력 크기 통일 및 연산량 감소
    - 입력: (H, W) / 출력: (84, 84)
    """
    def __init__(self, env, shape=84):
        super().__init__(env)
        self.shape = (shape, shape)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8
        )

    def observation(self, obs):
        return cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        return self.observation(obs)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        return self.observation(obs), reward, done, info


class SkipFrame(gym.Wrapper):
    """
    [전처리 3단계] 프레임 스킵 - 매 n프레임마다 한 번만 행동 결정
    - 연속된 프레임은 거의 동일하므로 4프레임을 건너뛰며 같은 행동 반복
    - 학습 속도 4배 향상 / 건너뛴 프레임의 보상은 모두 합산
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip  # 건너뛸 프레임 수

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class FrameStack(gym.Wrapper):
    """
    [전처리 4단계] 연속 n프레임을 쌓아 하나의 관측값으로 만들기
    - 단일 프레임만으로는 마리오의 이동 방향/속도를 알 수 없음
    - 4프레임을 채널처럼 쌓아 움직임 정보를 신경망에 전달
    - 입력: (84, 84) x 4회 / 출력: (4, 84, 84)
    """
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        # maxlen 설정으로 가장 오래된 프레임 자동 제거
        self.frames = deque(maxlen=n_stack)
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(n_stack, *obs_shape),
            dtype=np.uint8
        )

    def reset(self):
        obs = self.env.reset()
        # 초기화 시 동일 프레임으로 버퍼를 채워 크기 맞춤
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return np.array(self.frames)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        self.frames.append(obs)
        return np.array(self.frames), reward, done, info


def make_env(world=1, stage=1):
    """
    슈퍼마리오 환경 생성 + 전처리 파이프라인 조립
    - 전처리 순서: 프레임스킵 -> 그레이스케일 -> 리사이즈 -> 프레임스택
    - 최종 관측값 shape: (4, 84, 84) - CNN 입력으로 바로 사용 가능
    """
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

    # 월드-스테이지 환경 생성 (예: 1-1, 1-2 등)
    env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0")
    # 7가지 단순 이동 행동만 사용 (SIMPLE_MOVEMENT)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, n_stack=4)
    return env
