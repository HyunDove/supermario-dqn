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
        frame = env.render(mode="rgb_array")
        frames.append(frame)
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward

    env.close()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, frames, fps=30)
    return total_reward, len(frames)
