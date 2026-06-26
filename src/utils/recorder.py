import numpy as np
import imageio
import os

# NES는 ~60fps, SkipFrame(4)로 1스텝 = 4게임프레임
# → 1스텝당 프레임 1개 캡처 시 올바른 재생 속도: 60/4 = 15fps
_RECORD_FPS = 15


def record_episode(env_fn, agent, save_path, epsilon=0.05, target_steps=300):
    """
    에이전트 플레이 영상 녹화

    target_steps: 녹화할 스텝 수. 에피소드가 일찍 끝나면 재시작해 채우고,
                  에피소드가 길어도 이 스텝에서 즉시 중단 → 항상 고정 길이
                  (15fps 기준: 300스텝 = 20초, 450스텝 = 30초)
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
    imageio.mimsave(save_path, frames, fps=_RECORD_FPS)
    return total_reward, len(frames)
