import numpy as np
import imageio
import os

# NES는 ~60fps, SkipFrame(4)로 1스텝 = 4게임프레임
# → 1스텝당 프레임 1개 캡처 시 올바른 재생 속도: 60/4 = 15fps
_RECORD_FPS = 15


def record_episode(env_fn, agent, save_path, epsilon=0.05,
                   min_steps=300, max_steps=2000):
    """
    에이전트 플레이 영상 녹화

    min_steps: 에피소드가 일찍 끝나도 이 스텝 수만큼 재시작해 이어 녹화
    max_steps: 최대 녹화 스텝 상한 (무한 루프 방지)
    """
    env = env_fn()
    agent.epsilon = epsilon

    frames = []
    total_reward = 0
    total_steps = 0

    while total_steps < max_steps:
        state = env.reset()
        done = False

        while not done and total_steps < max_steps:
            # nes-py render()는 C++ 버퍼 view를 반환 → 즉시 복사해야 close() 후 깨지지 않음
            frame = env.render(mode="rgb_array")
            frames.append(np.ascontiguousarray(frame, dtype=np.uint8))
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            total_steps += 1

        # min_steps 이상 찍었으면 추가 재시작 없이 종료
        if total_steps >= min_steps:
            break

    env.close()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, frames, fps=_RECORD_FPS)
    return total_reward, len(frames)
