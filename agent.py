import torch
import random
import copy
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


class ReplayBuffer:
    def __init__(self, capacity=100000, device=None):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device

    def insert(self, transition):
        transition = [item.to('cpu') for item in transition]

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory.remove(self.memory[0])
            self.memory.append(transition)

    def sample(self, batch_size):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [torch.cat(items).to(self.device) for items in batch]

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10

    def __len__(self):
        return len(self.memory)


class Agent:

    def __init__(self, model, device=None, gamma=0.99, epsilon=1.0, min_epsilon=0.05, exploration_episodes=7500,
                 number_of_actions=4,
                 memory_capacity=100000, batch_size=32, lr=0.00025):
        self.memory = ReplayBuffer(device=device, capacity=memory_capacity)
        self.model = model.to(device)
        self.target_model = copy.deepcopy(model).to(device)
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = 1 - (((epsilon - min_epsilon) / exploration_episodes) * 2)
        self.batch_size = batch_size
        self.gamma = gamma
        self.number_of_actions = number_of_actions
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        if not os.path.exists("./tensorboard_logdir"):
            os.makedirs("./tensorboard_logdir")

    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.number_of_actions, (1, 1))
        else:
            av = self.model(state).detach()
            return torch.argmax(av, dim=-1, keepdim=True)

    def train(self, env, epochs, batch_identifier=0):
        stats = {'Returns': [], 'AvgReturns': [], 'EpsilonCheckpoint': []}
        writer = SummaryWriter(log_dir=f"./tensorboard_logdir/{datetime.now().strftime('%Y-%m-%d')}")
        for epoch in range(epochs):
            state = env.reset()
            done = False
            ep_return = 0
            while not done:
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                self.memory.insert([state, action, reward, done, next_state])


                if self.memory.can_sample(self.batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = self.memory.sample(self.batch_size)
                    qsa_b = self.model(state_b).gather(1, action_b)
                    next_qsa_b = self.target_model(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]
                    target_b = reward_b + self.gamma * next_qsa_b * ~done_b

                    # Compute Huber loss
                    loss = F.huber_loss(qsa_b, target_b.detach())

                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                state = next_state
                ep_return += reward.item()

            writer.add_scalar(f'Returns: {batch_identifier}', ep_return, epoch)

            stats['Returns'].append(ep_return)

            if self.epsilon > self.min_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay

            if epoch % 10 == 0:
                self.model.save_the_model()
                print("")

                average_returns = np.mean(stats['Returns'][-100:])

                stats['AvgReturns'].append(average_returns)
                stats['EpsilonCheckpoint'].append(self.epsilon)

                if (len(stats['Returns'])) > 100:
                    print(
                        f"Epoch: {epoch} - Average Return Last 100 Episodes: {np.mean(stats['Returns'][-100:])} - Epsilon: {self.epsilon}")
                else:
                    print(
                        f"Epoch: {epoch} - Episode Return: {np.mean(stats['Returns'][-1:])} - Epsilon: {self.epsilon}")

            if epoch % 500 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if epoch % 1000 == 0:
                self.model.save_the_model(f"models/model_iter_{epoch}.pt")

        return stats

    def test(self, env):
        for epoch in range(1, 3):
            state = env.reset()
            done = False
            for _ in range(1000):
                time.sleep(0.01)
                action = self.get_action(state)
                state, reward, done, info = env.step(action)
                if done:
                    break