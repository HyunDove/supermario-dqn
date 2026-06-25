import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.model.dqn import DQN
from src.utils.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        n_actions,
        device,
        lr=1e-4,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.99999,
        buffer_size=50000,
        batch_size=32,
        target_update_freq=1000,
    ):
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step = 0

        self.policy_net = DQN(n_actions).to(device)
        self.target_net = DQN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state_t).argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        state_t = torch.tensor(state).to(self.device)
        action_t = torch.tensor(action).to(self.device)
        reward_t = torch.tensor(reward).to(self.device)
        next_state_t = torch.tensor(next_state).to(self.device)
        done_t = torch.tensor(done).to(self.device)

        q_values = self.policy_net(state_t).gather(1, action_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_state_t).max(1)[0]
            target = reward_t + self.gamma * next_q * (1 - done_t)

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step': self.learn_step,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.learn_step = checkpoint['learn_step']
