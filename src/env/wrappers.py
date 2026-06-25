import gym
import numpy as np
import cv2
from collections import deque


class GrayScaleObservation(gym.ObservationWrapper):
    """RGB 이미지 → 그레이스케일 변환"""
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, obs):
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)


class ResizeObservation(gym.ObservationWrapper):
    """이미지 크기 조정 (84x84)"""
    def __init__(self, env, shape=84):
        super().__init__(env)
        self.shape = (shape, shape)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8
        )

    def observation(self, obs):
        return cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)


class SkipFrame(gym.Wrapper):
    """n프레임마다 행동 결정 (중간 프레임 스킵)"""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class FrameStack(gym.Wrapper):
    """연속 n프레임을 쌓아서 움직임 정보 포함"""
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(n_stack, *obs_shape),
            dtype=np.uint8
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return np.array(self.frames)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return np.array(self.frames), reward, done, info


def make_env(world=1, stage=1):
    """슈퍼마리오 환경 생성 + 전처리 파이프라인"""
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

    env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, n_stack=4)
    return env
