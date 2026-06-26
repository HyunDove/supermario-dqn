import numpy as np
import imageio
import os

# NES는 ~60fps, SkipFrame(4)로 1스텝 = 4게임프레임
# → 1스텝당 프레임 1개 캡처 시 올바른 재생 속도: 60/4 = 15fps
_RECORD_FPS = 15


def record_episode(env_fn, agent, save_path, epsilon=0.05,
                   target_steps=300, playback_speed=1.0):
    """
    에이전트 플레이 영상 녹화

    target_steps   : 녹화할 스텝 수 (15fps 기준: 300 = 20초, 450 = 30초)
    playback_speed : 재생 배속 (1.0 = 실제 게임 속도, 2.0 = 2배속, 0.5 = 0.5배속)
    """
    env = env_fn()
    agent.epsilon = epsilon

    frames = []
    total_reward = 0
    total_steps = 0

    while total_steps < target_steps:
        state = env.reset()
        done = False

        while not done and total_steps < target_steps:
            # nes-py render()는 C++ 버퍼 view를 반환 → 즉시 복사해야 close() 후 깨지지 않음
            frame = env.render(mode="rgb_array")
            frames.append(np.ascontiguousarray(frame, dtype=np.uint8))
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            total_steps += 1

    env.close()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, frames, fps=_RECORD_FPS * playback_speed)
    return total_reward, len(frames)
