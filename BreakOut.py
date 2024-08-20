import collections
import cv2
import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import cv2


class DQNEnvironment(gym.Wrapper):
    def __init__(self, render_mode='rgb_array', repeat=4, no_ops=0,
                 fire_first=False,device=None):
        env = gym.make("BreakoutNoFrameskip-v4", render_mode=render_mode)

        super(DQNEnvironment, self).__init__(env)
        self.repeat = repeat
        self.frame_buffer = []
        self.no_ops = no_ops
        self.fire_first = fire_first
        self.device = device

    def step(self, action):
        total_reward = 0
        done = False
        for i in range(self.repeat):
            observation, reward, done, trash, info = self.env.step(action)
            total_reward += reward
            self.frame_buffer.append(observation)
            if done:
                break

        max_frame = np.max(self.frame_buffer[-2:], axis=0)
        max_frame = self.process_observation(max_frame)
        max_frame = max_frame.to(self.device)

        total_reward = torch.tensor(total_reward).view(1, -1).float()
        total_reward = total_reward.to(self.device)

        done = torch.tensor(done).view(1, -1)
        done = done.to(self.device)

        return max_frame, total_reward, done, info

    def reset(self):
        self.frame_buffer = []
        observation, _ = self.env.reset()
        observation = self.process_observation(observation)
        return observation

    # same as PreprocessAtariObs in the ipynb file
    def process_observation(self, observation):
        # the size that openai used
        img = cv2.resize(observation, (84, 84))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.array(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0).unsqueeze(0)
        img = img / 255.0
        img = img.to(self.device)
        return img
