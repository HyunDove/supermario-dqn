import numpy as np
import imageio
import os


def record_episode(env_fn, agent, save_path, epsilon=0.05):
    """에이전트 플레이 영상 녹화"""
    env = env_fn()
    agent.epsilon = epsilon

    frames = []
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # nes-py의 render(rgb_array)는 C++ 화면 버퍼를 가리키는 numpy view를 반환한다.
        # 복사하지 않으면 모든 프레임이 같은 메모리를 공유하고, env.close() 후
        # mimsave가 해제된 버퍼를 읽어 영상이 깨진다 → np.ascontiguousarray로 즉시 복사
        frame = env.render(mode="rgb_array")
        frames.append(np.ascontiguousarray(frame, dtype=np.uint8))
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward

    env.close()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, frames, fps=30)
    return total_reward, len(frames)
